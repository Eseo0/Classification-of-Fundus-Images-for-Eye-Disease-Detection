from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

DATA_DIR = "./eye_fundus_image_dataset"

def get_data_loaders(batch_size=4, train_ratio=0.7, valid_ratio=0.2):
    """
    데이터셋을 학습, 검증, 테스트 세트로 나눕니다.

    Args:
        batch_size (int): 배치 크기.
        train_ratio (float): 학습 데이터 비율.
        valid_ratio (float): 검증 데이터 비율.

    Returns:
        train_loader, valid_loader, test_loader, class_names
    """
    # 데이터 전처리 (학습 및 예측 단계에서 동일하게 사용)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 전체 데이터셋 로드
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transform)
    total_size = len(full_dataset)
    indices = np.random.permutation(total_size)

    # 데이터셋 분할
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    train_indices = indices[:train_size]
    valid_indices = indices[train_size:train_size + valid_size]
    test_indices = indices[train_size + valid_size:]

    # Subset 생성
    train_dataset = Subset(full_dataset, train_indices)
    valid_dataset = Subset(full_dataset, valid_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, full_dataset.classes
