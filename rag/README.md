# Retrieval-Augmented Generation with DKubeX
The data ingestion, query and evaluation pipelines offer users the ability to ingest large datasets, perform RAG based queries on datasets and run evaluations using ground truth or synthetic datasets generated using language models (LLMs). Users can customize each pipeline or further extend pipelines to fit their specific needs.

## Key Features
- **Flexible Data Ingestion:** Users can customize various parameters in the ingestion pipeline including choice of embedding model, vector store and various data loaders.
- **Retrieval-Augmented Generation (RAG):** The query pipeline facilitates interactive and batch mode RAG on ingested datasets with customization including ability to use locally deployed LLMs or external hosted LLM end points including Open AI. Various other parameters including use of reranking models, top k and window size can be configured to achieve the best performance for different use cases. 
- **Evaluation Metrics:** The evaluation pipeline computes various metrics including MRR, Hit Rate and Similarity scores between user defined ground truth or generated synthetic data and responses from an LLM candidate the user wishes to evaluate.

## Usage
- **Data Ingestion:**
    - Configure the pipeline by selecting a preferred embedding model, vector store and chunk size etc.
    - Choose between 100 of data loaders already supported via llamahub.
      
- **Retrieval-Augmented Generation (RAG):**
    - Configure locally hosted or external LLMs, reranker model, top k, etc.
    - Run queries in interactive or batch mode.
    - Build a simple chat app to test and collect feedback from larger groups. 


- **Evaluation:**
    - Generate sythetic data from your data corpus using any LLM of choice. 
    - Compare your ground truth with an LLM candidate of choice with metrics like MRR, similarity score, and hit rate.
    - Analyze the results on MLFlow UI to assess the performance of LLMs and the quality of generated datasets.
