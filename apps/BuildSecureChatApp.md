# Examples app

## Step 1: Ingest data

Run tools/fm-ingest.py and specify the following parameters
  - source: Path where documents are located
  - dataset: Name for the dataset created out of these documents. 

Prerequisites
- Export openai api token, dkubex api token
- Fix weaviate URLs depending on whether you are running inside or outside the cluster.

run tools/fm-ingest.py to ingest corpus and remember the dataset name. This needs to be used in the App

Example: `python tools/fm-ingest.py --source /path/to/my/files --dataset <mydataset>`

Unit test using fm-query.py. Just specify the dataset used. it runs in interactive mode

Example: `python tools/fm-query.py --dataset <mydataset>`


## Step 2: Build App and push container image. 

Example:

  `cd apps`

  `echo $DOCKER_PASSWORD | docker login -u $DOCKER_USER --password-stdin`

  `docker build -t $DOCKER_REGISTRY/llmapp:qna -f securechat/Dockerfile --build-arg APPDIR=./qna  .`

  `docker push $DOCKER_REGISTRY/llmapp:qna`

Note: The same docker image works for many qna type apps. You would just need to specify a few run time tunables. Please see `d3x appa create` command below

## Step 3: Deploy application behind ingress 

Example: Uses d3x CLI and specifies the \<dataset\> created above using fm-ingest.py

  `d3x apps create --name=<name> -e OPENAI_API_KEY=$OPENAI_API_KEY -e DATASET=<dataset> -e APP_TITLE="DatasetQnAAgent" -p 3000   --dockeruser=$DOCKER_USER --dockerpsw=$DOCKER_TOKEN -rt false -ip /<prefix> --publish=true --memory=4 --cpu=1 $DOCKER_REGISTRY/llmapp:qna`

  Note: APP_TITLE can't have spaces

## Details
Apps can be placed inside `apps` directory
**(If you wish to change to something custom, then change the securechat/Dockerfile accordingly)**

- apps/\<your app\> => Should have the `requirements.txt` and all the *.py* and any other files.
- Each app should have `main.py` file inside `apps/\<your app\>/` .

Look at this file main.py
Each app must implement two APIs - one for title and one for streaming request.
Inside the `/stream` api - you can plugin any langchain implementation. You can follow the logic with paperqa in that file.

## Other useful d3x apps commands
`d3x apps list`
`d3x apps delete <app name>`
