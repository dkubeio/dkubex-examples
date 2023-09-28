# dkubex-examples
This repo contains the files required for the examples for the DKubeX platform.

## apps
The apps folder contains the required files for creating custom user applications on DKubeX.

## ray training

The ray folder contains example training program files for Ray training in Dkubex.

## rayserve
The rayserve folder contains the required files to deploy models on the DKubeX platform. Currently Hugging-face and MLFlow models can be deployed using DKubeX.

### HF
The [HF folder](rayserve/HF) contains the files to deploy Huggging-face models on DKubeX. For specific and detailed instructions for how to deploy, please check out the [README.md file](rayserve/HF/README.md) .

### MLFLOW
The [MLFLOW folder](rayserve/MLFLOW) contains the files to deploy MLFlow models on DKubeX. For specific and detailed instructions for how to deploy, please check out the [README.md file](rayserve/MLFLOW/README.md) .

### FLYTE
The flyte folder contains the mnist.py and Dockerfile where the mnist.py contains the python code for running the workflow for flyte and it also contains the tasks. The Dockerfile is as an image spec required to run the imported libraries inside each task. Please build the image and put it to a repository for docker images. Example command to run the mnist workflow is "pyflyte run --remote --project flytetester --domain development --envs '{"MLFLOW_TRACKING_URI":"http://d3x-controller.d3x.svc.cluster.local:5000"}' --image 'image-name' mnist.py mnist_workflow"
