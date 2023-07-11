import argparse
from typing import Dict
from ray.air import session

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import ray
import ray.train as train
from ray.train.torch import TorchTrainer, TorchCheckpoint
from ray.air import Checkpoint
from ray.air.config import ScalingConfig, RunConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback, setup_mlflow
import pickle
import mlflow
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="~/data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="~/data",
    train=False,
    download=True,
    transform=ToTensor(),
)


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
    size = len(dataloader.dataset) // session.get_world_size()
    model.train()
    for batch, (X, y) in enumerate(dataloader):
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
    size = len(dataloader.dataset) // session.get_world_size()
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


def train_func(config: Dict):
    import os
    user = os.environ.get("USER", "default")
    cluster = os.environ.get("HOSTNAME", "raycluster").split("-")[0]

    job_id = ray.get_runtime_context().get_job_id()
    #run_id = active_run().info.run_id
    tags = { "mlflow.user" : user,
         "experiment name" : "fashion minst", "job_id":job_id, "ray cluster": cluster }
    name = f"fashion minst-without-callback-{user}"
    setup_mlflow(config= config, create_experiment_if_not_exists=True, experiment_name=name, tags=tags )

    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["epochs"]

    worker_batch_size = batch_size // session.get_world_size()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=worker_batch_size)
    test_dataloader = DataLoader(test_data, batch_size=worker_batch_size)

    train_dataloader = train.torch.prepare_data_loader(train_dataloader)
    test_dataloader = train.torch.prepare_data_loader(test_dataloader)

    # Create model.
    model = NeuralNetwork()
    model = train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        train_epoch(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_epoch(test_dataloader, model, loss_fn)
        checkpoint = Checkpoint.from_dict(
            dict(epoch=epoch, model=model.state_dict())
        )
        mlflow.log_metrics(dict(loss=test_loss), step=epoch)
        mlflow.pytorch.log_model(model,f"model")
        session.report(dict(loss=test_loss), checkpoint=checkpoint)



def train_fashion_mnist(num_workers=2, use_gpu=False):
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 5},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config= RunConfig(
            name="mlflow",
            )
    )
    result = trainer.fit()
    print(result)
    #model = TorchCheckpoint.from_checkpoint(result.checkpoint).get_model(NeuralNetwork())
    #mlflow.pytorch.log_model(model,f"model")
    print(f"Last result: {result.metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=4,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args, _ = parser.parse_known_args()

    import ray

    if args.smoke_test:
        # 2 workers + 1 for trainer.
        ray.init(num_cpus=3)
        train_fashion_mnist()
    else:
        ray.init(address=args.address)
        train_fashion_mnist(num_workers=args.num_workers, use_gpu=args.use_gpu)
