FROM nvcr.io/nvidia/pytorch:24.03-py3
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONPATH /workspace
RUN pip3 install\
    "flytekit==1.10.2" \
    "flytekitplugins-kfpytorch"

RUN pip3 install pytorch-lightning torchvision wget lightning mlflow kubernetes torchmetrics

COPY mnist-lightning-dist.py mnist-lightning-dist.py.py
