import torch
from torch import optim
from model import CNN
from trainer import Trainer
from utils import get_data_loaders
from torch.nn import NLLLoss

def define_argparser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_fn', required=True, help="Path to save the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--n_epochs', type=int, default=20, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Ratio of training data to total data")
    parser.add_argument('--valid_ratio', type=float, default=0.2, help="Ratio of validation data to total data")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Device (cuda or cpu)")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")

    return parser.parse_args()

def main(args):
    device = torch.device(args.device)

    train_loader, valid_loader, _, class_names = get_data_loaders(
        batch_size=args.batch_size, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio
    )

    print(f"Class names: {class_names}, Number of classes: {len(class_names)}")

    # 모델 초기화
    model = CNN(output_size=len(class_names)).to(device)

    # 손실 함수 정의
    loss_function = NLLLoss()  # 클래스 가중치를 추가하고 싶다면 NLLLoss(weight=weights) 사용 가능

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(model, optimizer, loss_function, device)

    trainer.train(
        train_data=train_loader,
        valid_data=valid_loader,
        config=args,
        patience=args.patience
    )

    torch.save(model.state_dict(), args.model_fn)
    print(f"Model saved to {args.model_fn}")

if __name__ == "__main__":
    args = define_argparser()
    main(args)
