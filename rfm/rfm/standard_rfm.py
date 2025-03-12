# standard_rfm.py
import torch
import numpy as np
from rfm import LaplaceRFM

# Định nghĩa StandardKernelModel
class StandardKernelModel(torch.nn.Module):
    def __init__(self, kernel_fn, centers, y_dim, device="cuda"):
        super(StandardKernelModel, self).__init__()
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape
        self.device = device
        self.centers = centers.to(device)
        self.weight = torch.zeros(self.n_centers, y_dim, device=device)
        
    def forward(self, batch):
        kmat = self.kernel_fn(batch, self.centers)
        pred = kmat @ self.weight
        return pred
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, reg=1e-3, verbose=True):
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Tính toán ma trận kernel
        kernel_matrix = self.kernel_fn(X_train, self.centers)
        
        # Thêm regularization
        kernel_matrix.diagonal().add_(reg)
        
        # Giải hệ phương trình tuyến tính
        if verbose:
            print(f"Đang giải hệ phương trình tuyến tính...")
        
        self.weight = torch.linalg.solve(kernel_matrix, y_train)
        
        if verbose:
            print(f"Hoàn thành!")
            
        return self.weight

# Định nghĩa StandardLaplaceRFM
class StandardLaplaceRFM(LaplaceRFM):
    def __init__(self, bandwidth=1., **kwargs):
        super().__init__(bandwidth=bandwidth, **kwargs)
    
    def fit_predictor_standard(self, centers, targets, reg=1e-3, verbose=True, **kwargs):
        # Sử dụng mô hình kernel tiêu chuẩn
        model = StandardKernelModel(self.kernel, centers, targets.shape[-1], device=self.device)
        self.weights = model.fit(centers, targets, None, None, reg=reg, verbose=verbose)
        return self.weights
    
    def fit(self, train_data, test_data, **kwargs):
        # Ghi đè phương thức fit để luôn sử dụng phương pháp tiêu chuẩn
        kwargs['method'] = 'lstsq'  # Luôn sử dụng lstsq
        return super().fit(train_data, test_data, **kwargs)
