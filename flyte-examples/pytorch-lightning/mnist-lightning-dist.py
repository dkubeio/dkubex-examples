import flytekit
from flytekit import PodTemplate, Resources, task, workflow
from flytekit.core.base_task import IgnoreOutputs
from flytekitplugins.kfpytorch.task import Elastic
from kubernetes.client.models import (
    V1Container,
    V1PodSpec,
    V1EnvVar,
)

NUM_NODES = 2
NUM_DEVICES = 1

CONTAINER_IMAGE = (
    "dkubex123/flyte-pt:lightning-dist"
)

# Define the environment variables
env_vars = [
    V1EnvVar(
        name="MLFLOW_TRACKING_TOKEN",
        value="MLFLOW_TRACKING_TOKEN",
    ),
    V1EnvVar(
        name="MLFLOW_TRACKING_URI",
        value="MLFLOW_TRACKING_URI",
    ),
    V1EnvVar(name="MLFLOW_TRACKING_INSECURE_TLS", value="true"),
]

experiment_name = "lightning_logs"

container = V1Container(name="primary", env=env_vars)
custom_pod_template = PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(containers=[container]),
)

@task(
    container_image=CONTAINER_IMAGE,
    task_config=Elastic(
        nnodes=NUM_NODES,
        nproc_per_node=NUM_DEVICES,
        rdzv_configs={"timeout": 36000, "join_timeout": 36000},
        max_restarts=3,
    ),
    # accelerator=T4,
    requests=Resources(mem="32Gi", cpu="1", gpu="1", ephemeral_storage="100Gi"),
    pod_template=custom_pod_template,
)
def pl_train_task(
    batch_size: int,
    epochs: int,
    lr: float,
    data_dir: str,
) -> str:
    import os
    import wget
    import pytorch_lightning as pl
    from lightning.pytorch.loggers import MLFlowLogger
    import torch
    import torch.nn.functional as F
    import multiprocessing
    from torch import nn
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms
    from torchmetrics.functional import accuracy
    import tarfile

    class LitMNIST(pl.LightningModule):
        def __init__(
            self, data_dir="./", hidden_size=64, learning_rate=2e-4, batch_size=32
        ):
            super().__init__()
            # Set our init args as class attributes
            self.data_dir = data_dir
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            # Hardcode some dataset specific attributes
            self.num_classes = 10
            self.dims = (1, 28, 28)
            channels, width, height = self.dims
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            # Define PyTorch model
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * width * height, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, self.num_classes),
            )

        def forward(self, x):
            x = self.model(x)
            return F.log_softmax(x, dim=1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, y, task="multiclass", num_classes=10)

            # Calling self.log will surface up scalars for you in TensorBoard
            # https://pytorch-lightning.readthedocs.io/en/1.1.2/multi_gpu.html#synchronize-validation-and-test-logging
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)
            # import pdb
            # pdb.set_trace()
            self.logger.experiment.log_metric(
                self.logger.run_id, key="loss", value=loss
            )
            self.logger.experiment.log_metric(self.logger.run_id, key="acc", value=acc)
            return loss

        def test_step(self, batch, batch_idx):
            # Here we just reuse the validation_step for testing
            return self.validation_step(batch, batch_idx)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

        def prepare_data(self):
            # download
            if not os.path.exists("MNIST"):
                wget.download(
                    "https://activeeon-public.s3.eu-west-2.amazonaws.com/datasets/MNIST.new.tar.gz",
                    "MNIST.tar.gz",
                )
                tar = tarfile.open("MNIST.tar.gz", "r:gz")
                tar.extractall()
                tar.close()
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)

        def setup(self, stage=None):
            # Assign train/val datasets for use in dataloaders
            global experiment_name
            if self.logger.run_id != None:
                print(
                    f"mlflow experiment:{experiment_name}\nmlflow run id: {self.logger.run_id}"
                )
            if stage == "fit" or stage is None:
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                self.mnist_train, self.mnist_val = random_split(
                    mnist_full, [55000, 5000]
                )
            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.mnist_test = MNIST(
                    self.data_dir, train=False, transform=self.transform
                )

        def train_dataloader(self):
            if torch.cuda.is_available():
                return DataLoader(
                    self.mnist_train,
                    batch_size=self.batch_size,
                    num_workers=multiprocessing.cpu_count(),
                    pin_memory=True,
                    persistent_workers=True,
                )
            else:
                return DataLoader(
                    self.mnist_train,
                    batch_size=self.batch_size,
                    num_workers=multiprocessing.cpu_count(),
                    persistent_workers=True,
                )

        def val_dataloader(self):
            if torch.cuda.is_available():
                return DataLoader(
                    self.mnist_val,
                    batch_size=self.batch_size,
                    num_workers=multiprocessing.cpu_count(),
                    pin_memory=True,
                    persistent_workers=True,
                )
            else:
                return DataLoader(
                    self.mnist_val,
                    batch_size=self.batch_size,
                    num_workers=multiprocessing.cpu_count(),
                    persistent_workers=True,
                )

        def test_dataloader(self):
            if torch.cuda.is_available():
                return DataLoader(
                    self.mnist_test,
                    batch_size=self.batch_size,
                    num_workers=multiprocessing.cpu_count(),
                    pin_memory=True,
                    persistent_workers=True,
                )
            else:
                return DataLoader(
                    self.mnist_test,
                    batch_size=self.batch_size,
                    num_workers=multiprocessing.cpu_count(),
                    persistent_workers=True,
                )

    # Build CNN
    model = LitMNIST(
        data_dir=data_dir, hidden_size=64, learning_rate=lr, batch_size=batch_size
    )

    # Use GPUs if they are available otherwise use cpu
    devices = 0
    if torch.cuda.is_available():
        devices = torch.cuda.device_count()
    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", log_model="all")

    # trainer = pl.Trainer(num_nodes=NUM_NODES, devices=NUM_DEVICES, max_epochs=epochs,  accelerator="auto", strategy="ddp", logger=mlf_logger)
    trainer = pl.Trainer(
        num_nodes=NUM_NODES,
        devices=NUM_DEVICES,
        fast_dev_run=True,
        accelerator="auto",
        strategy="ddp",
        logger=mlf_logger,
    )

    trainer.fit(model)
    if model.logger.run_id != None:
        print(
            f"mlflow experiment:{experiment_name}\nmlflow run id: {model.logger.run_id}"
        )
    return "Completed"


@workflow
def pl_training_wf(
    batch_size: int = 10,
    epochs: int = 10,
    lr: float = 0.01,
    data_dir: str = "/share/songole",
) -> str:
    return pl_train_task(
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    model = pl_training_wf()
    #
    # print(f"Model: {model}, Tensorboard Log Dir: {logs}")

