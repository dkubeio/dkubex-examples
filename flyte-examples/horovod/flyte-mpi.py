import os
import pathlib

import argparse
import os
from packaging import version

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from filelock import FileLock
from torchvision import datasets, transforms

import flytekit
from flytekit import Resources, task, workflow
from flytekit.core.base_task import IgnoreOutputs
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.kfmpi import Launcher, MPIJob, Worker
from flytekit.types.directory import TensorboardLogs

from tensorboardX import SummaryWriter
import typing
from typing import Tuple


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


@task(
    task_config=MPIJob(
        launcher=Launcher(
            replicas=1,
            requests=Resources(cpu="1", mem="2Gi"),
            limits=Resources(cpu="2", mem="3Gi"),
        ),
        worker=Worker(
            replicas=2,
            requests=Resources(cpu="1", mem="2Gi", gpu="1"),
            limits=Resources(cpu="2", mem="3Gi", gpu="1"),
        ),
    ),
    retries=0,
)
def horovod_train_task(
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    seed: int,
    log_interval: int,
    gradient_predivide_factor: float,
    data_dir: str,
) -> FlyteDirectory:
    import horovod.torch as hvd

    log_dir = os.path.join(flytekit.current_context().working_directory, "logs")
    writer = SummaryWriter(log_dir)

    fp16_allreduce = False
    use_mixed_precision = True
    use_adasum = False

    cuda = torch.cuda.is_available()
    print(f"Use cuda {cuda}")
    hvd.init()
    torch.manual_seed(seed)
    if cuda:
        # Horovod: pin GPU to local rank.
        print(hvd.local_rank())
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(seed)
    else:
        if use_mixed_precision:
            raise ValueError("Mixed precision is only supported with cuda enabled.")
    device = torch.device("cuda" if cuda else "cpu")

    if use_mixed_precision and version.parse(torch.__version__) < version.parse(
        "1.6.0"
    ):
        raise ValueError(
            """Mixed precision is using torch.cuda.amp.autocast(),
                            which requires torch >= 1.6.0"""
        )

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (
        kwargs.get("num_workers", 0) > 0
        and hasattr(mp, "_supports_context")
        and mp._supports_context
        and "forkserver" in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = "forkserver"

    data_dir = data_dir or "./data"
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs
    )

    model = Net().to(device)

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not use_adasum else 1

    if cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(model.parameters(), lr=lr * lr_scaler, momentum=momentum)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if use_adasum else hvd.Average,
        gradient_predivide_factor=gradient_predivide_factor,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                # Horovod: use train_sampler to determine the number of examples in
                # this worker's partition.
                print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar("loss", loss.item(), niter)

    # Save the model
    working_dir = flytekit.current_context().working_directory
    model_file = os.path.join(
        flytekit.current_context().working_directory, "mnist_cnn.pt"
    )
    torch.save(model.state_dict(), model_file)

    torch.cuda.empty_cache()

    return FlyteDirectory(path=working_dir)


@workflow
def horovod_training_wf(
    batch_size: int = 10,
    epochs: int = 10,
    lr: float = 0.01,
    momentum: float = 0.5,
    seed: int = 42,
    log_interval: int = 10,
    gradient_predivide_factor: float = 1.0,
    data_dir: str = "./data",
) -> FlyteDirectory:
    return horovod_train_task(
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        seed=seed,
        log_interval=log_interval,
        gradient_predivide_factor=gradient_predivide_factor,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    model, plot, logs = horovod_training_wf()
    print(f"Model: {model}, Tensorboard Log Dir: {logs}")
