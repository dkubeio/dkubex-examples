# Ingestion Pipeline
Data ingestion is the first step of the RAG workflow. The ingestion pipeline loads and transforms raw data (including but not limited to TXT, PDF, HTML, etc) into embeddings and stores these embeddings in a vector database. The user can load data directly from a local directory within DKubeX or use one or more of the data loaders available to load data from external sources such as Websites, Cloud Storage Drives or Databases. The user can also tweak various parameters relating to transformation and ingestion including choosing a different chunking strategy, embeddings model, vector store etc to achieve the best results for their specific use case. 


## Pipeline Description:

- **Token Text Splitter(`splitter`):**
  
  The TokenTextSplitter divides the input text into smaller chunks based on tokens. Users can adjust the chunk size and chunk overlap to suit their use case.
  
  - **Class:** `TokenTextSplitter`
  - **Parameters:**
    - `chunk_size`: Specifies the size of each text chunk.
    - `chunk_overlap`: Determines the overlap between consecutive chunks.

- **Embedding Model Selection (`embedder`):**

  Users can specify the type of embedding model to be used for generating embeddings. Currently supported sources include any HuggingFace embedding model or OpenAI based embedding model.

  - **Class:** `HuggingFaceEmbedding`
  - **Parameters:**
    - `model`: Specifies the name or identifier of the embedding model to be used. For HuggingFace embeddings, any model from the HuggingFace model hub can be used.

- **Metadata Extraction (`metadata`):**

  The metadata extractor component extracts metadata information from the input data. Currently, only `DKubexFMMetadataExtractor` is supported.

  - **Class:** `DKubexFMMetadataExtractor`

- **Vector Store Configuration (`vectorstore`)**:

  The vector store component stores the generated embeddings in a database. Currently, only WeaviateVectorStore is supported. Users can specify the provider and URI for the vector store.

  - **Class:** `WeaviateVectorStore`
  - **Parameters:**
    - `provider`: Specifies the provider of the vector store.
    - `uri`: Specifies the URI of the vector store. If the provider is not `dkubex`, this URI will be used.

- **Document Store Configuration (`docstore`):**

  The document store component stores the original documents along with their associated metadata. Currently, only `WeaviateDocumentStore` is supported. Users can specify the provider and URI for the document store.

  - **Class:** `WeaviateDocumentStore`
  - **Parameters:**
    - `provider`: Specifies the provider of the document store.
    - `uri`: Specifies the URI of the document store. If the provider is not `dkubex`, this URI will be used.

- **Data Loader Configuration (`reader`):**

  The data loader component loads the input data from various sources. Users can specify the source and configuration parameters for the data loader. 

  - **Class:** `FileDirectoryReader`
  - **Description:** Reads texts and PDFs from a file directory.
  - **Parameters:**
    - `input_dir`: Specifies the absolute path to the data corpus directory.
    - `recursive`: Indicates whether subdirectories should be recursively searched.
    - `exclude_hidden`: Specifies whether hidden files should be excluded from the loading process.

> [!NOTE]  
> For additional data loader options and configurations, please refer to the yaml files in [dataloaders](./dataloaders) subdirectory.

## Ingestion using local cluster

```
d3x dataset ingest -d <dataset_name> --config <absolute path to your yaml-config file>
```

## Ingestion using remote cluster
For very large workloads, users can burst to the cloud to leverage accelerators from AWS, GCP, Azure etc. The default configuration (`default.yaml`) uses a T4 accelerator and can easily be configured by the user to use any other accelerator type from across supported clouds.

```
d3x dataset ingest -d contracts --remote-sky --sky-cluster=contracts --sky-accelerator="A10G:1" --dkubex-apikey <api_key> --dkubex-url <dkubex_url> --config <absolute path to your yaml config file
```

> [!NOTE]  
> Make sure that [ingest.yaml](./ingest.yaml) file and data-corpus directory are inside `/home/data/` directory on your DKubeX workspace.

## Listing Datasets
This command will list your datasets from the vector store.

```
d3x dataset list
```

> [!TIP]
> Data ingestion pipelines automatically record metadata and key metrics under `Experiments` tab in MLFlow within DKubeX. 
