'''Construct kernel model with EigenPro optimizer.'''
import collections
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.linalg import eigh

def asm_eigen_fn(samples, map_fn, top_q=None, alpha=0.95, seed=1, verbose=True):
    """
    Compute standard eigendecomposition of the kernel matrix without EigenPro.
    """
    np.random.seed(seed)
    start = time.time()

    # Compute kernel matrix
    kernel_matrix = map_fn(samples, samples)

    # Eigendecomposition using scipy
    eigvals, eigvecs = eigh(kernel_matrix)

    # Reorder eigenvalues and eigenvectors (largest to smallest)
    eigvals = torch.from_numpy(eigvals[::-1]).float()
    eigvecs = torch.from_numpy(eigvecs[:, ::-1]).float()

    if verbose:
        print("Eigendecomposition time: %.2f seconds" % (time.time() - start))
        print(f"Largest eigenvalue: {eigvals[0].item()}")

    return eigvals, eigvecs


class KernelModel(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''
    def __init__(self, kernel_fn, centers, y_dim, device="cuda"):
        super(KernelModel, self).__init__()
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape
        self.device = device
        self.pinned_list = []

        self.centers = self.tensor(centers, release=True, dtype=centers.dtype)
        self.weight = self.tensor(torch.zeros(
            self.n_centers, y_dim), release=True, dtype=centers.dtype)
        
        self.save_kernel_matrix = self.n_centers <= 85000
        self.kernel_matrix = [] if self.save_kernel_matrix else None

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")

    def tensor(self, data, dtype=None, release=False):
        if torch.is_tensor(data):
            tensor = data.clone().detach().to(self.device)
        else:
            tensor = torch.tensor(data, requires_grad=False, device=self.device)

        if release:
            self.pinned_list.append(tensor)
        return tensor

    def get_kernel_matrix(self, batch, batch_ids, samples=None, sample_ids=None):
        if batch_ids is not None and self.save_kernel_matrix and isinstance(self.kernel_matrix, torch.Tensor):
            if samples is None or sample_ids is None:
                kmat = self.kernel_matrix[batch_ids]
            else:
                kmat = self.kernel_matrix[batch_ids][:, sample_ids]
        else:
            if samples is None or sample_ids is None:
                kmat = self.kernel_fn(batch, self.centers)
            else:
                kmat = self.kernel_fn(batch, samples)
        return kmat

    def forward(self, batch, batch_ids=None, weight=None, save_kernel_matrix=False):
        if weight is None:
            weight = self.weight
        kmat = self.get_kernel_matrix(batch, batch_ids)
        if save_kernel_matrix: # only call if self.kernel_matrix is a list
            self.kernel_matrix.append((batch_ids.cpu(), kmat.cpu()))
        pred = kmat.mm(weight)
        return pred

    def primal_gradient(self, batch, labels, weight, batch_ids, save_kernel_matrix=False):
        pred = self.forward(batch, batch_ids, weight, save_kernel_matrix)
        grad = pred - labels
        return grad

    @staticmethod
    def _compute_opt_params(bs, bs_gpu, beta, top_eigval):
        if bs is None:
            bs = min(np.int32(beta / top_eigval + 1), bs_gpu)

        if bs < beta / top_eigval + 1:
            eta = bs / beta
        else:
            eta = 0.99 * 2 * bs / (beta + (bs - 1) * top_eigval)
        return bs, float(eta)

    def gradient_iterate(self, x_batch, y_batch, eta, batch_ids):
        grad = self.primal_gradient(x_batch, y_batch, self.weight, batch_ids)
        self.weight.index_add_(0, batch_ids, -eta * grad)
        return

    def evaluate(self, X_all, y_all, n_eval, bs, metrics=('mse')):
        p_list = []
        n_sample, _ = X_all.shape
        n_eval = n_sample if n_eval is None else n_eval
        eval_ids = np.random.choice(n_sample,
                               min(n_sample, n_eval),
                               replace=False)
        n_batch = n_sample // min(n_sample, bs)
        for batch_ids in np.array_split(eval_ids, n_batch):
            x_batch = self.tensor(X_all[batch_ids], dtype=X_all.dtype)
            p_batch = self.forward(x_batch).cpu()
            p_list.append(p_batch)
        p_eval = torch.concat(p_list, dim=0).to(self.device)
        y_eval = y_all[eval_ids].to(self.device)

        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            eval_metrics['mse'] = torch.mean(torch.square(p_eval - y_eval)).item()
        if 'multiclass-acc' in metrics:
            y_class = torch.argmax(y_eval, dim=-1)
            p_class = torch.argmax(p_eval, dim=-1)
            eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / len(eval_ids)
        if 'binary-acc' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['binary-acc'] = torch.sum(y_class == p_class).item() / len(eval_ids)
        if 'f1' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['f1'] = torch.mean(2 * (y_class * p_class) / (y_class + p_class + 1e-8)).item()
        if 'auc' in metrics:
            eval_metrics['auc'] = roc_auc_score(y_eval.cpu().flatten(), p_eval.cpu().flatten())

        return eval_metrics

    def fit(self, X_train, y_train, X_val, y_val, epochs, mem_gb,
        n_subsamples=None, top_q=None, bs=None, eta=None,
        n_train_eval=5000, run_epoch_eval=True, lr_scale=1,
        verbose=True, seed=1, classification=False, threshold=1e-5,
        early_stopping_window_size=6):
            
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        assert(len(X_train)==len(y_train))
        assert(len(X_val)==len(y_val))
        
        if bs is not None:
            n_train_eval = min(bs, n_train_eval)
            
        metrics = ('mse',)
        if classification:
            if y_train.shape[-1] == 1:
                metrics += ('binary-acc', 'f1', 'auc')
            else:
                metrics += ('multiclass-acc',)
        
        n_samples, n_labels = y_train.shape
        
        if n_subsamples is None:
            n_subsamples = min(n_samples, 12000)
        n_subsamples = min(n_subsamples, n_samples)
        
        mem_bytes = (mem_gb - 1) * 1024**3 # preserve 1GB
        bsizes = np.arange(n_subsamples)
        mem_usages = ((self.x_dim + 3 * n_labels + bsizes + 1)
                    * self.n_centers + n_subsamples * 1000) * 4
        bs_gpu = np.sum(mem_usages < mem_bytes) # device-dependent batch size
        
        # Calculate batch size / learning rate
        np.random.seed(seed)
        sample_ids = np.random.choice(n_samples, n_subsamples, replace=False)
        sample_ids = self.tensor(sample_ids)
        print(f"sample_ids: {sample_ids}")
        samples = self.centers[sample_ids]
        
        # Compute eigendecomposition
        _, scale, top_eigval, beta = asm_eigen_fn(
            samples, self.kernel_fn, top_q, bs_gpu, alpha=.95, seed=seed, verbose=verbose)
        
        # Calculate learning rate
        if eta is None:
            eta = 1.0 / top_eigval  # Simplify learning rate based on largest eigenvalue
            bs = min(bs_gpu, n_samples) if bs is None else min(bs, bs_gpu, n_samples)
        
        if verbose:
            print("n_subsamples=%d, bs_gpu=%d, eta=%.2f, bs=%d, top_eigval=%.2e" %
                  (n_subsamples, bs_gpu, eta, bs, top_eigval))
        
        eta = self.tensor(lr_scale * eta / bs, dtype=torch.float)
        
        res = dict()
        initial_epoch = 0
        train_sec = 0  # training time in seconds
        best_weights = None
        if classification:
            best_metric = 0
        else:
            best_metric = float('inf')
        
        # Add early stopping variables
        val_loss_history = []
        prev_val_metric = 0 if classification else float('inf')

        for epoch in range(epochs):
            start = time.time()
            for _ in range(epoch - initial_epoch):
                # Create a permutation of all indices
                epoch_ids = np.random.permutation(n_samples)

                save_kernel_matrix = epoch==1 and self.save_kernel_matrix

                for batch_ids in tqdm(np.array_split(epoch_ids, n_samples // bs)):
                    batch_ids = self.tensor(batch_ids)
                    x_batch = self.tensor(X_train[batch_ids], dtype=X_train.dtype)
                    y_batch = self.tensor(y_train[batch_ids], dtype=y_train.dtype)
                    self.gradient_iterate(x_batch, y_batch, eta, batch_ids)

                    del x_batch, y_batch, batch_ids

                if save_kernel_matrix:
                    print(f"Storing kernel matrix")
                    # First concatenate all rows
                    concat_matrix = torch.cat([pair[1] for pair in self.kernel_matrix], dim=0)
                    # Get all batch indices and their positions
                    all_batch_ids = torch.cat([pair[0] for pair in self.kernel_matrix])
                    # Get sorting indices and reorder the matrix
                    _, sort_indices = torch.sort(all_batch_ids)
                    self.kernel_matrix = concat_matrix[sort_indices]
                    self.kernel_matrix = self.kernel_matrix.to(self.device)

            if run_epoch_eval:
                train_sec += time.time() - start
                tr_score = self.evaluate(X_train, y_train, n_eval=n_train_eval, bs=bs, metrics=metrics)
                tv_score = self.evaluate(X_val, y_val, n_eval=None, bs=bs, metrics=metrics)
                if verbose:
                    out_str = f"({epoch} epochs, {train_sec} seconds)\t train l2: {tr_score['mse']} \tval l2: {tv_score['mse']}"
                    if classification:
                        if 'binary-acc' in tr_score:
                            out_str += f"\ttrain binary acc: {tr_score['binary-acc']} \tval binary acc: {tv_score['binary-acc']}"
                        else:
                            out_str += f"\ttrain multiclass acc: {tr_score['multiclass-acc']} \tval multiclass acc: {tv_score['multiclass-acc']}"
                        if 'f1' in tr_score:
                            out_str += f"\ttrain f1: {tr_score['f1']} \tval f1: {tv_score['f1']}"
                        if 'auc' in tr_score:
                            out_str += f"\ttrain auc: {tr_score['auc']} \tval auc: {tv_score['auc']}"
                    print(out_str)

                res[epoch] = (tr_score, tv_score, train_sec)
                if classification:
                    if 'auc' in tv_score:
                        if tv_score['auc'] > best_metric:
                            best_metric = tv_score['auc']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best auc: {best_metric}")
                    elif 'binary-acc' in tv_score:
                        if tv_score['binary-acc'] > best_metric:
                            best_metric = tv_score['binary-acc']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best binary-acc: {best_metric}")
                    elif 'multiclass-acc' in tv_score:
                        if tv_score['multiclass-acc'] > best_metric:
                            best_metric = tv_score['multiclass-acc']
                            best_weights = self.weight.cpu().clone()
                            print(f"New best multiclass-acc: {best_metric}")
                else:
                    if tv_score['mse'] < best_metric:
                        best_metric = tv_score['mse']
                        best_weights = self.weight.cpu().clone()
                        print(f"New best mse: {best_metric}")

                # Track validation loss changes
                if 'binary-acc' in tv_score:
                    val_loss_history.append(tv_score['binary-acc'] <= prev_val_metric)
                elif 'multiclass-acc' in tv_score:
                    val_loss_history.append(tv_score['multiclass-acc'] <= prev_val_metric)
                else:
                    val_loss_history.append(tv_score['mse'] >= prev_val_metric)
                if len(val_loss_history) > early_stopping_window_size:
                    val_loss_history.pop(0)
                    # Check if validation loss increased in majority of recent iterations
                    if sum(val_loss_history) / len(val_loss_history) >= 0.6:  # 60% of recent iterations showed increase
                        if verbose:
                            print(f"Early stopping triggered: validation loss increased in majority of last {early_stopping_window_size} epochs")
                        break
                
                if classification:
                    prev_val_metric = tv_score['multiclass-acc'] if 'multiclass-acc' in tv_score else tv_score['binary-acc']
                else:
                    prev_val_metric = tv_score['mse']

                if tr_score['mse'] < threshold:
                    break

            initial_epoch = epoch

        self.weight = best_weights.to(self.device)

        if self.kernel_matrix is not None:
            del self.kernel_matrix

        return res
