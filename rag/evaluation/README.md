# Eval Pipeline
This pipeline is used for evaluating synthetic datasets generated using language models (LLMs), either open-source or utilizing OpenAI's models. The evaluation process involves comparing the generated dataset against other LLMs using metrics such as Mean Reciprocal Rank (MRR), similarity score, and hit rate.

## Pipeline Description:

- **Vectorstore Reader (Weaviate):**
  - **Kind:** Specifies the type of vectorstore reader, which in this case is `Weaviate`.
  - **Provider:** Indicates the provider of the vectorstore reader, which is `dkubex`.
  - **Properties:** Lists the properties of the Weaviate vectorstore reader, including `paperchunks` and `dkubexfm`.

- **Questions Generator:**
  - **Prompt Strategy:** Defines the strategy for generating prompts. In this configuration, the default strategy is used.
  - **Number of Questions per Chunk:** Specifies the number of questions to generate per data chunk.
  - **Maximum Chunks:** Sets the maximum number of data chunks to generate questions for.
  - **LLM:** Determines the language model (LLM) to use for generating questions. The default is `openai` but can be switched to `dkubex`.
  - **LLM Key:** Requires providing the API key for the chosen LLM (OpenAI in this case).
  - **Maximum Tokens:** Specifies the maximum number of tokens allowed in each question prompt.

- **Retrieval Evaluator:**
  - **Vector Retriever:**
    - **Kind:** Indicates the type of vector retriever, which is `Weaviate`.
    - **Provider:** Specifies the provider of the vector retriever, which is `dkubex`.
    - **Text Key:** Refers to the key used to access the text data within the vector retriever, which is `paperchunks`.
    - **Embedding Model:** Specifies the name of the embedding model used for text representation, in this case, `BAAI/bge-large-en-v1.5` from Huggingface.
    - **Similarity Top K:** Sets the number of similar items to retrieve for each query.
  - **Metrics:** Specifies the evaluation metrics used for retrieval evaluation, which include `mrr` (Mean Reciprocal Rank) and `hit_rate`.

- **Semantic Similarity Evaluator:**
  - **Prompt Strategy:** Similar to the Questions Generator, it defines the strategy for generating prompts.
  - **LLM:** Specifies the language model (LLM) to use for semantic similarity evaluation, which is `dkubex` in this configuration.
  - **LLM Key:** Although labeled as `dummy`, it might be used for authentication or identification purposes.
  - **LLM URL:** Indicates the URL where the chosen LLM service is deployed. It's specified as `http://mistral-serve-svc.username:8000`, where `username` needs to be replaced with the workspace name where the deployment was created.
  - **Maximum Tokens:** Specifies the maximum number of tokens allowed in each semantic similarity evaluation prompt.
  - **Metrics:** Specifies the evaluation metric used for semantic similarity evaluation, which is the `similarity_score`.

- **Tracking:**
  - **Experiment Name:** Provides a unique name for the MLFlow experiment, allowing for tracking and comparison of different runs of the pipeline. 

## Command for Evaluation

```
d3x fm query llamaidx evaluate -d <dataset-name>  --config <path to your config file>
```

> [!TIP]
> You can track the evaluation flow under the experiments tab in MLFlow.
