from copy import deepcopy
import torch
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, loss_function, device):
        """
        Trainer 클래스 초기화

        Args:
            model: PyTorch 모델.
            optimizer: PyTorch 옵티마이저.
            loss_function: 손실 함수.
            device: 학습에 사용할 장치 (cuda 또는 cpu).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def train_one_epoch(self, train_loader):
        """
        단일 에폭 학습

        Args:
            train_loader: 학습 데이터 로더.

        Returns:
            avg_loss: 에폭당 평균 손실.
            accuracy: 에폭당 정확도.
        """
        self.model.train()
        total_loss = 0
        total_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # 라벨이 원-핫 인코딩이라면 정수형으로 변환
            if labels.dim() > 1:  # 2차원 텐서 (원-핫 라벨)
                labels = torch.argmax(labels, dim=1)

            # Forward pass
            outputs = self.model(images)
            loss = self.loss_function(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / len(train_loader.dataset)

        return avg_loss, accuracy

    def validate(self, valid_loader):
        """
        검증 데이터셋 평가

        Args:
            valid_loader: 검증 데이터 로더.

        Returns:
            avg_loss: 검증 데이터 평균 손실.
            accuracy: 검증 데이터 정확도.
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 라벨이 원-핫 인코딩이라면 정수형으로 변환
                if labels.dim() > 1:  # 2차원 텐서 (원-핫 라벨)
                    labels = torch.argmax(labels, dim=1)

                # Forward pass
                outputs = self.model(images)
                loss = self.loss_function(outputs, labels)

                total_loss += loss.item()
                total_correct += (outputs.argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(valid_loader)
        accuracy = total_correct / len(valid_loader.dataset)

        return avg_loss, accuracy

    def train(self, train_data, valid_data, config, patience=3):
        """
        학습 및 검증 반복 수행

        Args:
            train_data: 학습 데이터 로더.
            valid_data: 검증 데이터 로더.
            config: 학습 설정 (에폭 수 등).
            patience: Early Stopping 기준 에폭 수.

        Returns:
            None
        """
        lowest_loss = np.inf
        best_model = None
        no_improvement = 0

        for epoch in range(config.n_epochs):
            train_loss, train_acc = self.train_one_epoch(train_data)
            valid_loss, valid_acc = self.validate(valid_data)

            print(f"Epoch {epoch + 1}/{config.n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

            # 검증 손실이 낮아지면 모델 저장
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                no_improvement = 0
                print("Validation loss improved, saving best model...")
            else:
                no_improvement += 1

            # Early Stopping
            if no_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # 최적 모델 로드
        self.model.load_state_dict(best_model)
        print("Restored best model.")
