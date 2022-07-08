import torch
from torch import nn
from torch.utils.data import DataLoader

from train import train
from eval import test
from datasets import training_data, test_data


class FizzBuzzNet(nn.Module):
    def __init__(self) -> None:
        super(FizzBuzzNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 1000), nn.ReLU(), nn.Linear(1000, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear_relu_stack(x)
        return logits


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=101)

    model = FizzBuzzNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epochs = 200

    for t in range(epochs):
        print(f"Epoch {t+1}", end=" - ")
        loss = train(train_dataloader, model, loss_fn, optimizer, device)
        print(f"loss: {loss:>7f}")

    test_loss, acc = test(test_dataloader, model, loss_fn, device)
    print(f"Test Result: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    main()
