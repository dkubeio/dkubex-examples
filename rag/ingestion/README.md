# Ingestion Pipeline
The data ingestion pipeline offers users the flexibility to incorporate their custom data seamlessly. Users can tailor the pipeline to their specific needs by selecting their preferred embedding model for generating embeddings and a vector store for storing the resulting vector database. Furthermore, the pipeline supports various data loaders, enabling users to customize the ingestion process according to their chosen data loader.

## Pipeline Description:

- **Token Text Splitter(`splitter`):**
  
  The TokenTextSplitter divides the input text into smaller chunks based on tokens. Users can adjust the chunk size and overlap to suit their needs.
  
  - **Class:** `TokenTextSplitter`
  - **Parameters:**
    - `chunk_size`: Specifies the size of each text chunk.
    - `chunk_overlap`: Determines the overlap between consecutive chunks.

- **Embedding Model Selection (`embedder`):**

  Users can specify the type of embedding model to be used for generating embeddings. Currently supported embedding models include HuggingFace and OpenAI embeddings.

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
> For additional data loader options and configurations, please refer to the yaml files in `dataloaders` subdirectory.

## Ingestion of data-corpus from various sources

```
d3x fm docs llamaidx ingest -d <dataset_name> --config <absolute path to your yaml-config file>
```

## Ingestion using sky-cluster
This command facilitates data ingestion via Sky, where the default configuration (`default.yaml`) supports T4 accelerator utilization. Users can initiate ingestion through Sky, automatically allocating the T4 accelerator for efficient processing. Tracked datasets are managed using DKubeX, providing seamless monitoring. 

```
d3x fm docs llamaidx ingest -d <dataset_name> --config /home/data/ingest.yaml --remote-sky --dkubex-url <dkubex_url> --dkubex-apikey <dkubex_api_key>
```

> [!NOTE]  
> Make sure that `ingest.yaml` file and data-corpus directory are inside `/home/data/` directory.

## Tracking the Dataset
This command will list the dataset the user have ingested.

```
d3x fm docs show datasets
```

> [!TIP]
> The user can also track their datasets under `Experiments` tab on MLFlow. 
