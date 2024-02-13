# RAG Pipeline
The ingestion pipeline provides flexibility to end-users to perform Retrieval-Augmented Generation (RAG) on their ingested datasets using the language model (LLM) of their own choice. Different LLMs are supported, allowing users to deploy them based on their preferences.

```
Input

    Question: The input question to be answered by the RAG system.
    Mode: The mode of interaction with the pipeline (e.g., command-line interface).
```
```
Vector Store Retriever

    Kind: Specifies the type of vector store retriever. Currently, only WeaviateVectorStore is supported.
    Provider: Provider for the vector store retriever.
    Embedding Class: Class of embedding used for retrieval (e.g., HuggingFaceEmbedding).
    Embedding Model: Name of the embedding model from Hugging Face.
    Dataset: Name of the dataset to be ingested.
    Text Key: Key identifying the text data within the dataset.
    Top K: The number of top results to retrieve.
```
```
Prompt Builder

    Prompt String: The prompt string used for generation.
    Prompt File: The file containing the prompt string.
```
```
Nodes Sorter

    Max Sources: Maximum number of sources to consider during sorting.
```
```
Re-ranker

    Model: Name of the re-ranker model from Hugging Face.
    Top N: The number of top results to re-rank.
```
```
Contexts Joiner

    Separator: Separator used for joining different contexts.
```
```
Chat Engine

    LLM: Specifies the LLM to be used for generation. Use "dkubex" for dkube deployments.
    URL: Service URL for the LLM deployment to be used. 
    LLM Key: Authentication key for accessing the LLM service.
    Window Size: Size of the window for context generation.
    Max Tokens: Maximum number of tokens for generation.
```
```
Tracking

    Experiment: MLflow experiment name for tracking.
```


  ## Querying the dataset 

There are 3 modes of querying - Single question, Batch-question, Interactive mode

**A. Single question mode:**

```
d3x fm query llamaidx rag -d <dataset_name> --config <absolute path to your yaml-config file> -q "<question>"
```

**B. Batch question mode**

```
d3x fm query llamaidx rag -d <dataset_name> -b <path to your batch-que json file> --config <absolute path to your yaml-config file>
```

**C. Interactive mode**

```
d3x fm query llamaidx evaluate -d <dataset-name> --config <absolute path to your yaml-config file>
```