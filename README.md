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
The flyte folder contains the mnist.py, iris_ray.py, project.yaml and Dockerfile. The Dockerfile is as an image spec required to run the imported libraries inside each task. Please build the image and put it to a repository for docker images. 
mnist.py, iris_ray.py are examples which contain flyte tasks and the workflow on how to run them, iris_ray.py contains integration of flyte tasks with optuna for hyper parameter tuning, mlflow for logging the parameters and metrics and ray for inference. Below is the command for running iris_ray.py


**Command** 

**pyflyte run --remote --image dkubex123/my_flyte_image:latest --envs '{"MLFLOW_EXP_NAME":"flyte_optuna","MLFLOW_TRACKING_URI":"http://d3x-controller.d3x.svc.cluster.local:5000"}' iris_ray.py optimize_model**

Instead of flyte_optuna, replace with your own mlflow experiment name.

Example command to run the mnist workflow is "pyflyte run --remote --project flytetester --domain development --envs '{"MLFLOW_TRACKING_URI":"http://d3x-controller.d3x.svc.cluster.local:5000"}' --image dkubex123/my_flyte_image:latest mnist.py mnist_workflow"


**Note**
One can use project.yaml to create a new project in flyte with the following command **flytectl create project --file project.yaml**
Inside project.yaml you can configure the name and id for the project.
