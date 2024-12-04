import torch
from utils import get_data_loaders
from model import CNN
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd


def load_model(model_path, class_names, device):
    """
    저장된 모델을 로드합니다.
    """
    model = CNN(output_size=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def evaluate_model_with_confusion_matrix(model, test_loader, class_names, device):
    """
    테스트 데이터셋에서 모델의 성능을 평가하고 혼동 행렬을 생성합니다.
    """
    model.eval()
    all_labels = []
    all_predictions = []
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.exp(outputs)
            predicted = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # 전체 정확도 계산
    overall_accuracy = (total_correct / total_samples) * 100

    # 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_predictions, labels=np.arange(len(class_names)))

    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return overall_accuracy, cm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to the trained model")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for test data loader")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 데이터 로드 (테스트 데이터만 사용)
    _, _, test_loader, class_names = get_data_loaders(batch_size=args.batch_size)

    # 모델 로드
    model = load_model(args.model_path, class_names, device)

    # 테스트 데이터 평가 및 혼동 행렬 생성
    overall_accuracy, cm = evaluate_model_with_confusion_matrix(model, test_loader, class_names, device)

    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")


