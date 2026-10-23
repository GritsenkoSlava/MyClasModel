import torch
from torch import nn

def learn(epochs):
    x = torch.tensor([[1], [2], [3], [5], [6], [8]], dtype=float)
    y = torch.tensor([[0], [0], [0], [1], [1], [1]], dtype=float)

    model = nn.Linear(1, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'||  Loss: {loss}  ||')

    hours = torch.tensor([[4.0]])

    with torch.no_grad():
        logits = model(hours)
        probability = torch.sigmoid(logits)
        prediction = (probability >= 0.5).int()

    print()
    print("Вероятность сдать:", probability.item())
    print("Предсказанный класс:", prediction.item())                        

learn(2000)
