
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Dict
import mlflow
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_epoch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct

def train_func(config: Dict):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)


    model = NeuralNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_loss, correct = test_epoch(test_dataloader, model, loss_fn)
        mlflow.log_metrics(dict(loss=test_loss), step=epoch)
        mlflow.log_metrics(dict(accuracy=correct), step=epoch)
        mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path=f"checkpoint_{epoch:06d}")
    return model

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def train_fashion_mnist():
    import os
    user = os.environ.get("USER", "default")
    cluster = os.environ.get("HOSTNAME", "raycluster").split("-")[0]

    tags = { "mlflow.user" : user,
         "experiment name" : "fashion minst", "ray cluster": cluster }
    name  = f"fashion minst-without_ray-{user}"
    exp_id = mlflow.set_experiment(experiment_name=name)
    with mlflow.start_run() as run:
        mlflow.set_tags(tags)
        params = {"lr": 1e-3, "batch_size": 64, "epochs": 5, "inference_classes": classes, "dataset": "torchvision.FashionMNIST"}
        mlflow.log_params(params)
        model = train_func(params)
        torch.save(model.state_dict(), "model.pth")
        print("Saved PyTorch Model State to model.pth")
        input_example = test_data[0][0].detach().cpu().numpy()
        mlflow.pytorch.log_model(model,f"model", input_example=input_example)

if __name__ == "__main__":
    train_fashion_mnist()


