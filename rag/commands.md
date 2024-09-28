Usage: d3x dataset ingest [OPTIONS]

  Ingest the documents from provided source into the vectorstore identified by
  given dataset name.

  SUPPORTED EXTRA OPTIONS (make use of the options below to overwrite the
  individual values without need to copy config file

  --embedder.class=[HuggingfaceEmbedding/OpenAIEmbedding]
  --embedder.model=<embedding-model-name-for-the-above-class>

Options:
  -d, --dataset TEXT         A name to represent the ingested docs  [required]
  -p, --pipeline TEXT        Ingestion pipeline to be used.
  -c, --config TEXT          User configuration for pipeline stages
  -s, --remote-sky           If the ingestion should be scheduled on remote
                             sky cluster.
  -r, --remote-ray           If the ingestion should be scheduled on remote
                             ray cluster.
  -rc, --ray-config TEXT     configuration for running remote ray job in json
                             format
  -m, --remote-command TEXT  sky/ray job command to run.
  --dkubex-url TEXT          URL of the dkubex for the remote clusters to
                             reach.
  -k, --dkubex-apikey TEXT   API key for remote clusters to reach dkubex.
  -w, --num-workers INTEGER  number of process to use for parallelization.
  --type TEXT                node selector
  --faq                      Add this option to enable creation of DATASET and
                             also set a flag on the parent dataset that faq is
                             enabled for it
  --help                     Show this message and exit.

  --------------------------------------------------------------------------------------------------

  Usage: d3x dataset query [OPTIONS]

  Query over documents using the selected pipeline.  SUPPORTED EXTRA OPTIONS
  (make use of the options below to overwrite the individual values without
  need to copy config file       --vectorstore_retriever.embedding_class=[Hugg
  ingfaceEmbedding/OpenAIEmbedding]
  --vectorstore_retriever.embedding_model=
  --vectorstore_retriever.top_k=         --chat_engine.llm=[openai/dkubex]
  --chat_engine.url=<dkubex-dep-url>         --chat_engine.llmkey=<llm-access-
  key>         --chat_engine.max_tokens"

Options:
  -d, --dataset TEXT       Dataset on which the question should be applied.
  -p, --pipeline TEXT      RAG pipeline to be used.
  -c, --config TEXT        User configuration for pipeline stages
  -q, --question TEXT      User query to run through the pipeline
  -b, --batch TEXT         Batch of questions to run through the pipeline
  -i, --interactive        Run this tool in cli mode
  -f, --filters TEXT       Chunk-Metadata filter expression in string format.
  -k, --conversation TEXT  Name of the conversation to track all the questions
                           for.
  --help                   Show this message and exit.

  --------------------------------------------------------------------------------------------------

  Usage: d3x dataset evaluate [OPTIONS]

  Run evaluation pipeline over the given dataset.

Options:
  -d, --dataset TEXT         Dataset on which the question should be applied.
  -p, --pipeline TEXT        RAG pipeline to be used.
  -c, --config TEXT          User configuration for pipeline stages
  -s, --remote-sky           If the ingestion should be scheduled on remote
                             sky cluster.
  -m, --remote-command TEXT  sky/ray job command to run.
  -r, --remote-ray           If the ingestion should be scheduled on remote
                             ray cluster.
  -rc, --ray-config TEXT     configuration for running remote ray job in json
                             format
  --dkubex-url TEXT          URL of the dkubex for the remote clusters to
                             reach.
  -k, --dkubex-apikey TEXT   API key for remote clusters to reach dkubex.
  --help                     Show this message and exit.

  --------------------------------------------------------------------------------------------------

  