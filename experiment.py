import os
import subprocess
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data_loaders
from model import CNN
from predict import evaluate_model_with_confusion_matrix

# 하이퍼파라미터 설정
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [8, 16, 32]
epochs = [10, 15, 20]

# 저장 경로
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)

def train_models():
    """
    모든 하이퍼파라미터 조합으로 모델을 학습하고 저장.
    """
    for lr in learning_rates:
        for bs in batch_sizes:
            for ep in epochs:
                # 모델 파일 이름 정의
                model_name = f"model_lr_{lr}_bs_{bs}_ep_{ep}.pth"
                model_path = os.path.join(models_dir, model_name)

                # train.py 실행 명령어
                command = (
                    f"python train.py --model_fn {model_path} "
                    f"--learning_rate {lr} --batch_size {bs} --n_epochs {ep}"
                )

                # 학습 실행
                print(f"Running: {command}")
                subprocess.run(command, shell=True)

    print("All models have been trained and saved!")

def parse_model_filename(filename):
    """
    모델 파일명에서 파라미터 값 추출.
    예: 'model_lr_0.01_bs_32_ep_20.pth' -> {'learning_rate': 0.01, 'batch_size': 32, 'n_epochs': 20}
    """
    pattern = r"model_lr_(.+?)_bs_(.+?)_ep_(.+?)\.pth"
    match = re.match(pattern, filename)
    if match:
        return {
            'learning_rate': float(match.group(1)),
            'batch_size': int(match.group(2)),
            'n_epochs': int(match.group(3))
        }
    return None

def evaluate_models(batch_size=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    저장된 모델들을 평가하고 결과를 CSV에 저장.
    """
    # 테스트 데이터 로드
    _, _, test_loader, class_names = get_data_loaders(batch_size=batch_size)

    # 결과 저장용 리스트
    results = []

    # 모델 디렉토리의 모든 파일 평가
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pth"):
            model_path = os.path.join(models_dir, model_file)

            # 파일명에서 파라미터 추출
            params = parse_model_filename(model_file)
            if params is None:
                print(f"Skipping file {model_file}: Unable to parse parameters.")
                continue

            # 모델 로드
            model = CNN(output_size=len(class_names)).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            # 모델 평가
            overall_accuracy, _ = evaluate_model_with_confusion_matrix(model, test_loader, class_names, device)

            # 결과 저장
            results.append({
                'Learning Rate': params['learning_rate'],
                'Batch Size': params['batch_size'],
                'Epochs': params['n_epochs'],
                'Accuracy': overall_accuracy
            })
            print(f"Evaluated model: {model_file}, Accuracy = {overall_accuracy:.2f}%")

    # 결과를 DataFrame으로 변환 및 CSV 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_accuracies_with_params.csv', index=False)
    print("All results saved to model_accuracies_with_params.csv.")

def plot_heatmaps_by_epochs(csv_file_path):
    """
    Reads a CSV file and creates heatmaps for each unique epoch value.
    """
    # Read CSV data
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found at {csv_file_path}")
        return

    data = pd.read_csv(csv_file_path)

    # Check for required columns
    required_columns = {'Epochs', 'Learning Rate', 'Batch Size', 'Accuracy'}
    if not required_columns.issubset(data.columns):
        print(f"CSV file does not contain required columns: {required_columns}")
        return

    # Extract unique epochs
    unique_epochs = data['Epochs'].unique()

    # Create heatmaps for each epoch
    for epoch in unique_epochs:
        epoch_data = data[data['Epochs'] == epoch]
        
        # Create a pivot table
        pivot_table = epoch_data.pivot_table(
            index="Batch Size", columns="Learning Rate", values="Accuracy", aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        plt.title(f"Accuracy Heatmap (Epochs={epoch})")
        plt.xlabel("Learning Rate")
        plt.ylabel("Batch Size")
        heatmap = plt.imshow(pivot_table, cmap='viridis', aspect='auto')
        plt.colorbar(heatmap, label="Accuracy (%)")
        plt.xticks(ticks=range(len(pivot_table.columns)), labels=pivot_table.columns, rotation=45)
        plt.yticks(ticks=range(len(pivot_table.index)), labels=pivot_table.index)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 학습 단계
    train_models()

    # 평가 단계
    evaluate_models(batch_size=16)

    # 히트맵 생성
    plot_heatmaps_by_epochs('./model_accuracies_with_params.csv')


