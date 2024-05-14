ARG CUDA_DOCKER_VERSION=11.3.1-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ARG CUDA_DOCKER_VERSION=11.3.1-devel-ubuntu20.04
ARG TENSORFLOW_VERSION=2.9.2
ARG PYTORCH_VERSION=1.12.1+cu113
ARG PYTORCH_LIGHTNING_VERSION=1.5.9
ARG TORCHVISION_VERSION=0.13.1+cu113
ARG CUDNN_VERSION=8.2.1.32-1+cuda11.3
ARG NCCL_VERSION=2.9.9-1+cuda11.3
ARG MXNET_VERSION=1.9.1

ARG PYSPARK_PACKAGE=pyspark==3.3.0
ARG SPARK_PACKAGE=spark-3.3.0/spark-3.3.0-bin-hadoop2.tgz

ARG PYTHON_VERSION=3.10

# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root
ENV VENV /opt/venv
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Extract ubuntu distribution version and download the corresponding key.
# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub

RUN apt-get update \
	&& apt-get install -y software-properties-common \
	&& add-apt-repository ppa:ubuntu-toolchain-r/test \
	&& add-apt-repository -y ppa:deadsnakes/ppa \
	&& apt update \
	&& apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn8=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openjdk-8-jdk-headless \
        openssh-client \
        openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Virtual environment
ENV VENV /opt/venv
RUN python3.10 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Install wheel after venv is activated
RUN pip install wheel

# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-4.1.4-bin.tar.gz https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz && \
    cd /tmp && tar -zxf /tmp/openmpi-4.1.4-bin.tar.gz && \
    mkdir openmpi-4.1.4/build && cd openmpi-4.1.4/build && ../configure --prefix=/usr/local && \
    make -j all && make install && ldconfig && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

#RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

#RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

# Install PyTorch, TensorFlow, Keras and MXNet
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html
RUN pip install --no-cache-dir pytorch_lightning==${PYTORCH_LIGHTNING_VERSION}


COPY requirements.in /root
RUN pip install -r /root/requirements.in

RUN pip install --no-cache-dir future typing packaging
RUN pip install --no-cache-dir \
    tensorflow==${TENSORFLOW_VERSION} \
    keras \
    h5py

RUN pip install --no-cache-dir mxnet-cu112==${MXNET_VERSION} "numpy<1.24.0"

# Install Spark stand-alone cluster.
RUN wget --progress=dot:giga "https://www.apache.org/dyn/closer.lua/spark/${SPARK_PACKAGE}?action=download" -O - | tar -xzC /tmp; \
    archive=$(basename "${SPARK_PACKAGE}") bash -c "mv -v /tmp/\${archive/%.tgz/} /spark"

# Install PySpark.
RUN pip install --no-cache-dir ${PYSPARK_PACKAGE}

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
         pip install -v --no-cache-dir horovod[all-frameworks] && \
    ldconfig
# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.tensorflow as hvd; hvd.init()" && \
    python -c "import horovod.torch as hvd; hvd.init()" && \
    python -c "import horovod.mxnet as hvd; hvd.init()" && \
    ldconfig

# Copy the actual code
COPY . /root/
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN pip install protobuf==4.25.3 grpcio==1.63.0
