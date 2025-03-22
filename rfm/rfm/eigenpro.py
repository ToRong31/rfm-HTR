import collections
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

# Assume multiplicative_update, random_initialization, nndsvd_initialization are imported

def asm_nmf_fn_custom(samples, map_fn, rank=10, max_iter=100, init_mode='nndsvd', verbose=True):
    """
    Approximate kernel matrix using custom NMF with multiplicative update.
    """
    # kernel_matrix = map_fn(samples, samples)
    kernel_matrix = map_fn(samples, samples).cpu().numpy()
    kernel_matrix = np.maximum(kernel_matrix, 0)  # Ensure non-negativity

    W, H, norms = multiplicative_update(kernel_matrix, k=rank, max_iter=max_iter, init_mode=init_mode)

    if verbose:
        print(f"NMF with init='{init_mode}' completed.")
        print(f"Final reconstruction error (Frobenius norm): {norms[-1]:.4f}")

    return torch.from_numpy(W).float(), torch.from_numpy(H).float(), norms

class KernelModel(nn.Module):
    '''Fast Kernel Regression using NMF iteration.'''
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
        if save_kernel_matrix:  # only call if self.kernel_matrix is a list
            self.kernel_matrix.append((batch_ids.cpu(), kmat.cpu()))
        pred = kmat.mm(weight)
        return pred

    def primal_gradient(self, batch, labels, weight, batch_ids, save_kernel_matrix=False):
        pred = self.forward(batch, batch_ids, weight, save_kernel_matrix)
        grad = pred - labels
        return grad

    def evaluate(self, X_all, y_all, n_eval, bs,
                 metrics=('mse')):
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
            n_subsamples=None, bs=None,
            n_train_eval=5000, run_epoch_eval=True, lr_scale=1, 
            verbose=True, seed=1, classification=False, threshold=1e-5,
            early_stopping_window_size=6, nmf_rank=10, nmf_max_iter=100, nmf_init_mode='nndsvd'):

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        assert (len(X_train) == len(y_train))
        assert (len(X_val) == len(y_val))

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

        np.random.seed(seed)
        sample_ids = np.random.choice(n_samples, n_subsamples, replace=False)
        sample_ids = self.tensor(sample_ids)
        # print(f"sample_ids: {sample_ids}")
        samples = self.centers[sample_ids]

        # Prepare NMF function
        # nmf_W, nmf_H, nmf_norms = asm_nmf_fn_custom(
        #     samples.cpu().numpy(),
        #     lambda x, y: self.kernel_fn(self.tensor(x), self.tensor(y)).cpu().numpy(),
        #     rank=nmf_rank, max_iter=nmf_max_iter, init_mode=nmf_init_mode, verbose=verbose
        # )
        nmf_W, nmf_H, nmf_norms = asm_nmf_fn_custom(
            samples,
            lambda x, y: self.kernel_fn(x, y),
            rank=nmf_rank, max_iter=nmf_max_iter, init_mode=nmf_init_mode, verbose=verbose
        )


        nmf_W = nmf_W.to(self.device)
        nmf_H = nmf_H.to(self.device)

        if verbose:
            print(f"NMF with rank={nmf_rank}, max_iter={nmf_max_iter}, init_mode='{nmf_init_mode}' completed.")

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
            epoch_ids = np.random.permutation(n_samples)

            save_kernel_matrix = epoch == 1 and self.save_kernel_matrix

            for batch_ids in tqdm(np.array_split(epoch_ids, n_samples // bs)):
                batch_ids = self.tensor(batch_ids)
                x_batch = self.tensor(X_train[batch_ids], dtype=X_train.dtype)
                y_batch = self.tensor(y_train[batch_ids], dtype=y_train.dtype)

                # Update weight using precomputed NMF factors
                kmat = self.get_kernel_matrix(x_batch, None) # Shape: (batch_size, n_centers)
                # self.weight.data = torch.mm(kmat, nmf_H.T)
                self.weight.data = torch.mm(nmf_W, nmf_H)
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
                    # Check if validation loss increases too many times...
                    if sum(val_loss_history) >= early_stopping_window_size - 1:
                        print(f"Early stopping at epoch {epoch}")
                        break
                prev_val_metric = tv_score['binary-acc'] if 'binary-acc' in tv_score else \
                    tv_score['multiclass-acc'] if 'multiclass-acc' in tv_score else tv_score['mse']

        res['best_weights'] = best_weights if best_weights is not None else self.weight.cpu().clone()
        self.weight = self.tensor(res['best_weights'])
        print("DONE")
        return res
