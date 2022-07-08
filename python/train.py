from torch import nn


def train(dataloader, model, loss_fn, optimizer, device) -> float:
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = nn.functional.one_hot(y, num_classes=4).float()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()
