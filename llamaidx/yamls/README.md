File : ReadMe
Description : Steps and commands for RAG-workflow uisng llama-index.

# Ingestion of data-corpus from various sources.

d3x fm docs llamaidx ingest -d <dataset_name> --config <absolute path to your yaml-config file>

Ex. d3x fm docs llamaidx ingest -d mydataset --config /home/john

    Note:
    A. Any of the data loaders are supported and here are some examples of the different types of loaders
        1. txt/simple-pdf file from local directory.
        2. simple web pages using url.
        3. wikipedia pages
        4. documents from MS Share Point
        5. confluence page
    B. You can modify the yaml files as per your usecase

# Ingestion using sky-cluster

d3x fm docs llamaidx ingest -d <dataset_name> --config /home/data/ingest.yaml --remote-sky --dkubex-url <dkubex_url> --dkubex-apikey <dkubex_api_key>

Ex.

    Note:
    A. Make sure that ingest.yaml file and data-corpus directory are inside "/home/data/"

# Querying the dataset using RAG-flow.

There are 3 modes of querying - Single question, Batch-question, Interactive mode

    A. Single question mode

    d3x fm query llamaidx rag -d <dataset_name> --config <absolute path to your yaml-config file> -q "<question>"

    Ex.

    B. Batch question mode

    d3x fm query llamaidx rag -d <dataset_name> -b <path to your batch-que json file> --config <absolute path to your yaml-config file>

    Ex.

    C. Interactive mode

    d3x fm query llamaidx evaluate -d <dataset-name> --config <absolute path to your yaml-config file>

    Ex.

# Evaluation

The Eval facilitates the creation of a Synthesia dataset leveraging either open-source language models (LLMs) or OpenAI's models. This dataset will then undergo evaluation against other LLMs, utilizing metrics such as (MRR), similarity score, and hit rate.

    d3x fm query llamaidx evaluate -d <dataset-name> --config <absolute path to your yaml-config file>

# Secure chat app
  d3x apps create <securechat.yaml>
