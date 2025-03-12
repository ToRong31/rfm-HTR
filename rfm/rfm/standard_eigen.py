# Tạo file mới tên là standard_eigen.py
import torch
import numpy as np
import collections
import time
from tqdm import tqdm
import torch.nn as nn

class StandardKernelModel(nn.Module):
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
    
    def fit(self, X_train, y_train, X_val, y_val, reg=1e-3, verbose=True):
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Tính toán ma trận kernel
        kernel_matrix = self.kernel_fn(X_train, self.centers)
        
        # Thêm regularization
        kernel_matrix.diagonal().add_(reg)
        
        # Giải hệ phương trình tuyến tính
        start_time = time.time()
        self.weight = torch.linalg.solve(kernel_matrix, y_train)
        
        if verbose:
            print(f"Thời gian giải hệ phương trình: {time.time() - start_time:.2f}s")
            
        # Đánh giá trên tập kiểm định
        if X_val is not None and y_val is not None:
            y_pred = self.forward(X_val.to(self.device))
            mse = torch.mean((y_pred - y_val.to(self.device))**2).item()
            if verbose:
                print(f"Validation MSE: {mse:.4f}")
        
        return self.weight
