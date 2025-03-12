
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from rfm import LaplaceRFM, GeneralizedLaplaceRFM, GaussRFM, NTKModel
import torch.nn.functional as F
import logging

# Cấu hình logging: ghi log vào file và in ra console
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')

# File handler: ghi log vào output.txt
file_handler = logging.FileHandler("output.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler: in log ra console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Hàm chuyển đổi batch: chuyển đổi nhãn thành one-hot encoding với 10 lớp
def one_hot_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    # Chuyển đổi nhãn sang one-hot (float)
    labels = F.one_hot(labels, num_classes=10).float()
    return images, labels

# Thiết lập thiết bị và bộ nhớ GPU (nếu có)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM_GB = torch.cuda.get_device_properties(DEVICE).total_memory // 1024**3 - 1 
else:
    DEVICE = torch.device("cpu")
    DEV_MEM_GB = 8

# Định nghĩa transform: chuyển đổi ảnh MNIST thành tensor và làm phẳng thành vector 784 chiều
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Tải dataset MNIST cho training và testing
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Chia tập train ra thành một phần nhỏ để tránh làm đầy bộ nhớ
subset_size = 10000  # Có thể điều chỉnh theo khả năng của máy
train_subset, _ = random_split(full_train_dataset, [subset_size, len(full_train_dataset) - subset_size])

# Tạo DataLoader cho tập train và test, sử dụng collate_fn để chuyển đổi nhãn
batch_size = 16
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=one_hot_collate)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=one_hot_collate)

# Set a default p_batch_size value that will be used when None is passed
default_p_batch_size = 8

# Khởi tạo các mô hình với cùng tham số cơ bản
laplace_model = LaplaceRFM(bandwidth=1., device=DEVICE, mem_gb=DEV_MEM_GB, diag=False, p_batch_size=default_p_batch_size)
generalized_model = GeneralizedLaplaceRFM(bandwidth=1., device=DEVICE, mem_gb=DEV_MEM_GB, diag=False, p_batch_size=default_p_batch_size)
gauss_model = GaussRFM(bandwidth=1., device=DEVICE, mem_gb=DEV_MEM_GB, diag=False, p_batch_size=default_p_batch_size)
ntk_model = NTKModel(device=DEVICE, mem_gb=DEV_MEM_GB, diag=False, p_batch_size=default_p_batch_size)

# Huấn luyện từng mô hình và ghi log, truyền thêm M_batch_size và p_batch_size để tránh lỗi
logger.info("Training LaplaceRFM")
laplace_model.fit(
    train_loader, 
    test_loader, 
    loader=True, 
    iters=3,
    classification=True,  # Use the proper parameter name (classification instead of classif)
    total_points_to_sample=subset_size,
    M_batch_size=batch_size,
    p_batch_size=default_p_batch_size
)

logger.info("Training GeneralizedLaplaceRFM")
generalized_model.fit(
    train_loader, 
    test_loader, 
    loader=True, 
    iters=3,
    classification=True,
    total_points_to_sample=subset_size,
    M_batch_size=batch_size,
    p_batch_size=default_p_batch_size,
    return_best_params=True
)

logger.info("Training GaussRFM")
gauss_model.fit(
    train_loader, 
    test_loader, 
    loader=True, 
    iters=3,
    classification=True,
    total_points_to_sample=subset_size,
    M_batch_size=batch_size,
    p_batch_size=default_p_batch_size,
    return_best_params=True
)

logger.info("Training NTKModel")
ntk_model.fit(
    train_loader, 
    test_loader, 
    loader=True, 
    iters=3,
    classification=True,
    total_points_to_sample=subset_size,
    M_batch_size=batch_size,
    p_batch_size=default_p_batch_size
)