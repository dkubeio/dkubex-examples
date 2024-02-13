# RAG Pipeline
The ingestion pipeline provides flexibility to end-users to perform Retrieval-Augmented Generation (RAG) on their ingested datasets using the language model (LLM) of their own choice. Different LLMs are supported, allowing users to deploy them based on their preferences.

## Pipeline Description:

- **`input`**:
    - `question`: The input question to be answered by the RAG system.
    - `mode`: The mode of interaction with the pipeline (e.g., command-line interface).

- **`vectorstore_retriever`:**
    - `kind`: Specifies the type of vector store retriever. Currently, only `WeaviateVectorStore` is supported.
    - `provider`: Provider for the vector store retriever.
    - `embedding_class`: Class of embedding used for retrieval (e.g., `HuggingFaceEmbedding`).
    - `embedding_model`: Name of the embedding model from HuggingFace.
    - `dataset`: Name of the dataset to be ingested.
    - `textkey`: Key identifying the text data within the dataset.
    - `top_k`: The number of top results to retrieve.

- **`prompt_builder`:**
    - `prompt_str`: The prompt string used for generation.
    - `prompt_file`: The file containing the prompt string.

- **`nodes_sorter`:**
    - `max_sources`: Maximum number of sources to consider during sorting.

- **`reranker`:**
    - `model`: Name of the re-ranker model from Hugging Face.
    - `top_n`: The number of top results to re-rank.

- **`contexts_joiner`:**
    - `separator`: Separator used for joining different contexts.

- **`chat_engine`:**
    - `llm`: Specifies the LLM to be used for generation. Use `dkubex` for dkube deployments.
    - `url`: Service URL for the LLM deployment to be used. 
    - `llmkey`: Authentication key for accessing the LLM service.
    - `window_size`: Size of the window for context generation.
    - `max_tokens`: Maximum number of tokens for generation.

- **`tracking`:**
    - `experiment`: MLflow experiment name for tracking.

## Querying the dataset 
There are 3 modes of querying - Single question, Batch-question, and Interactive mode.

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
