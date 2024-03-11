# Query Pipeline
Once you've ingested your data and created a dataset, you can now run the query pipeline over your dataset. This pipeline facilitates retrival, post processing, response synthesis and finally generates a summarized response from a language model (LLM). Users can configure the pipeline to use locally deployed LLMs on DKubeX or custom external endpoints including Open AI. The pipeline has additional configurable parameters including ability to pass custom prompts, use a reranker model and specify top k for retrival.

## Pipeline Description:

- **`input`**:
    - `question`: The input question to be answered by the RAG system.
    - `mode`: The mode of interaction with the pipeline (e.g., command-line interface).

- **`vectorstore_retriever`:**
    - `kind`: Specifies the type of vector store retriever. Currently, only `WeaviateVectorStore` is supported.
    - `provider`: Provider for the vector store retriever.
    - `embedding_class`: Class of embedding used for retrieval (e.g., `HuggingFaceEmbedding`).
    - `embedding_model`: Name of the embedding model from HuggingFace.
    - `dataset`: Name of the ingested dataset.
    - `textkey`: Key identifying the text data within the dataset.
    - `top_k`: The number of results to retrieve per query.

- **`prompt_builder`:**
    - `prompt_str`: The prompt string used for generation.
    - `prompt_file`: The file containing the prompt string.

- **`nodes_sorter`:**
    - `max_sources`: Maximum number of sources to consider during sorting.

- **`reranker`:**
    - `model`: Name of the re-ranker model from Hugging Face.
    - `top_n`: The number of results to re-rank.

- **`contexts_joiner`:**
    - `separator`: Separator used for joining different contexts.

- **`chat_engine`:**
    - `llm`: Specifies the LLM to be used for generation. Use `dkubex` for DKubeX deployments.
    - `url`: Service URL for the LLM deployment to be used. 
    - `llmkey`: Authentication key for accessing the LLM service.
    - `window_size`: Size of the window for context generation.
    - `max_tokens`: Maximum number of tokens for generation.

- **`tracking`:**
    - `experiment`: MLflow experiment name for tracking.

## Querying the dataset 
There are 3 ways to query your dataset - Single question, Batch mode, and Interactive mode.

**A. Single question:**

```
d3x rag query llamaidx rag -d <dataset_name> --config <absolute path to your yaml-config file> -q "<question>"
```

**B. Batch mode**

```
d3x rag query llamaidx rag -d <dataset_name> -b <path to your batch-que json file> --config <absolute path to your yaml-config file>
```

**C. Interactive mode**

```
d3x rag query llamaidx rag -d <dataset_name> --config <absolute path to your yaml-config file> --cli
```
