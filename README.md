# dkubex-examples
This repo contains the files required for the examples for the DKubeX platform.

## apps
The [apps](apps) folder contains the required files for creating custom user applications on DKubeX.

## flyte-examples
The [flyte-examples](flyte-examples) folder contains the mnist.py, iris_ray.py, project.yaml and Dockerfile. The Dockerfile is as an image spec required to run the imported libraries inside each task. Please build the image and put it to a repository for docker images. 
mnist.py, iris_ray.py are examples which contain flyte tasks and the workflow on how to run them, iris_ray.py contains integration of flyte tasks with optuna for hyper parameter tuning, mlflow for logging the parameters and metrics and ray for inference.

## kserve
[Folder](kserve) containing example file for kserve.

## local
The [local](local) folder contains an example training which can be executed locally on your laptop and track experiments in DKubeX.

## rag
The [rag folder](rag) folder contains files and instructions regarding RAG pipeline workflow with Llamaindex. The data ingestion, query and evaluation pipelines offer users the ability to ingest large datasets, perform RAG based queries on datasets and run evaluations using ground truth or synthetic datasets generated using language models (LLMs). Users can customize each pipeline or further extend pipelines to fit their specific needs.

## ray
The [ray folder](ray) contains example training program files for Ray training in DKubeX.

## rayserve
The [rayserve folder](rayserve) contains the required files to deploy models on the DKubeX platform. Currently Hugging-face and MLFlow models can be deployed using DKubeX.

### hf
The [hf folder](rayserve/hf) contains the files to deploy Huggging-face models on DKubeX. For specific and detailed instructions for how to deploy, please check out the [README.md file](rayserve/hf/README.md) .

### mlflow
The [mlflow folder](rayserve/mlflow) contains the files to deploy MLFlow models on DKubeX. For specific and detailed instructions for how to deploy, please check out the [README.md file](rayserve/mlflow/README.md) .

## sky
The [sky folder](sky) folder contains examples to run SkyPilot jobs.