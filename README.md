# dkubex-examples
This repo contains the files required for the examples for the DKubeX platform.

## apps
The apps folder contains the required files for creating custom user applications on DKubeX.

## ray training
The ray folder contains example training program files for Ray training in Dkubex.

## rayserve
The rayserve folder contains the required files to deploy models on the DKubeX platform. Currently Hugging-face and MLFlow models can be deployed using DKubeX.

### hf
The [HF folder](rayserve/HF) contains the files to deploy Huggging-face models on DKubeX. For specific and detailed instructions for how to deploy, please check out the [README.md file](rayserve/HF/README.md) .

### mlflow
The [MLFLOW folder](rayserve/MLFLOW) contains the files to deploy MLFlow models on DKubeX. For specific and detailed instructions for how to deploy, please check out the [README.md file](rayserve/MLFLOW/README.md) .

## flyte
The flyte folder contains the mnist.py, iris_ray.py, project.yaml and Dockerfile. The Dockerfile is as an image spec required to run the imported libraries inside each task. Please build the image and put it to a repository for docker images. 
mnist.py, iris_ray.py are examples which contain flyte tasks and the workflow on how to run them, iris_ray.py contains integration of flyte tasks with optuna for hyper parameter tuning, mlflow for logging the parameters and metrics and ray for inference. 

## sky
This folder contains examples to run skypilot jobs

## local
This folder contains an example training which can be executed locally on your laptop and track experiments in dkubex

## rag
This folder contains files and instructions regarding RAG pipeline workflow with Llamaindex. The data ingestion, query and evaluation pipelines offer users the ability to ingest large datasets, perform RAG based queries on datasets and run evaluations using ground truth or synthetic datasets generated using language models (LLMs). Users can customize each pipeline or further extend pipelines to fit their specific needs.
