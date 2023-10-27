# import ray
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from typing import Tuple, Literal
import multiprocessing
import itertools
import logging

import click
import weaviate  # weaviate-python client
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Weaviate
from langchain.vectorstores.base import VectorStore
from joblib import Parallel, delayed
from paperqa import Doc, Docs, Text, utils
from langchain.embeddings import HuggingFaceBgeEmbeddings

#__package__ = "docs"
"""
Weaviate URL from inside the cluster
--- "http://weaviate.d3x.svc.cluster.local/"
Weaviate URL from outside the cluster 
--- NodePort - "http://<ip>:30716/api/vectordb/" 
--- Loadbalancer - "http://<ip>:80/api/vectordb/"
"""

WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)
DATASET_PAT = r"[A-Za-z][A-Za-z0-9_-]+"

def get_embeddings(embeddings_model_name="mpnet-v2") -> Embeddings:
    if embeddings_model_name == "mpnet-v2":
        from langchain.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
        )
        return hf

    elif embeddings_model_name == "openai":
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(client=None)
        return embeddings

    elif embeddings_model_name == "bert":
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)


def get_vectordb_client() -> VectorStore:
    DKUBEX_API_KEY = os.getenv("DKUBEX_API_KEY", "deadbeef")

    # Use Weaviate VectorDB
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"Authorization": DKUBEX_API_KEY},
    )

    return weaviate_client


def create_paperqa_vector_indexes(
    client: VectorStore, embeddings: Embeddings, dataset: str
) -> Tuple[VectorStore, VectorStore]:
    # Create 2 classes (or DBs) in Weaviate
    # One for docs and one for chunks
    docs = Weaviate(
        client=client,
        index_name="D" + dataset + "docs",
        text_key="paperdoc",
        attributes=["dockey"],
        embedding=embeddings,
    )

    chunks = Weaviate(
        client=client,
        index_name="D" + dataset + "chunks",
        text_key="paperchunks",
        embedding=embeddings,
        attributes=["doc", "name"],
    )

    return docs, chunks


"""
@ray.remote
"""

def adddoc(docslist, chunks_dir, chunk_size, job_id):
    from paperqa import Docs
    import json

    docs_store = Docs()

    counter = 0
    total_count = {'img': 0, "table":0, "text": 0, "doc":0}
    for doc in docslist:
        if doc is not None:
            print("Extracting doc: ", doc)
            try:
                filename = doc.split("/")[-1]
                base_dir = "%s/%s"%(chunks_dir, filename)
                finetune_chunks_base_dir = f"{chunks_dir.rstrip('/')}_for_finetuning"
                os.makedirs(base_dir, exist_ok=True)
                data_path = "%s/data"%base_dir
                chunks_path = "%s/chunks"%base_dir
                os.mkdir(data_path)
                os.mkdir(chunks_path)

                _, chunks = docs_store.generate_chunks(doc, citation=filename, docname=filename, chunk_chars=chunk_size)
                text_count = len(chunks)
                if text_count > 0:
                    numbered_dir = f"{job_id}-{counter:06d}"
                    os.makedirs(os.path.join(finetune_chunks_base_dir, numbered_dir), exist_ok=True)
                    new_chunks = []
                    for chunk in chunks:
                        new_chunks.append({"chunks": list(chunk.values())[0]})
                    with open(os.path.join(finetune_chunks_base_dir, numbered_dir, "text_chunks.json"),"w") as f:
                        json.dump(new_chunks, f, indent=2)
                    counter += 1
                count = {}
                count["text"] = text_count
                with open(os.path.join(chunks_path, "text_chunks.json"), "w") as outfile:
                    json.dump(chunks, outfile, indent=2)
                with open(os.path.join(base_dir, "metadata.json"), "w") as outfile:
                    metadata = {}
                    metadata["chunks_count"] = count
                    json.dump(metadata, outfile, indent=2)
                docs_store.clear_docs()
                total_count["img"] = 0
                total_count["text"] += text_count
                total_count["table"] = 0
                total_count["doc"] += 1

            except Exception as ingest_except:
                with open(os.path.join(chunks_path, "text_chunks.json"), "w") as outfile:
                    json.dump([], outfile, indent=2)
                with open(os.path.join(base_dir, "metadata.json"), "w") as outfile:
                    metadata = {}
                    metadata["chunks_count"] = 0
                    json.dump(metadata, outfile, indent=2)
                #traceback.print_stack(ingest_except)
                #import sys
                #sys.exit()
                print("xxxxx extraction failed, error: ", ingest_except)
    return total_count

def extract(source, chunks_dir, chunk_size):
    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()//2

    docsdir = [source]

    docs_list = []
    for docdir in docsdir:
        for extension in ["*.pdf", "*.html", "*.txt"]:
        #for extension in ["*.pdf", "*.txt"]:
            docs_list.extend(
                glob.glob(os.path.join(docdir, "**/" + extension), recursive=True)
            )

    def grouper(iterable, n=1000, fillvalue=None):
        import itertools
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    results = Parallel(n_jobs=num_cores)(
        delayed(adddoc)(list(groupp), chunks_dir, chunk_size, job_id) for job_id,groupp in enumerate(grouper(docs_list))
    )
    count_doc = count_img = count_table = count_text = 0
    for result in results:
        count_doc += result['doc']
        count_img += result['img']
        count_table += result['table']
        count_text += result['text']

    print("no. of docs = %s"%count_doc)
    print("no. of images = %s"%count_img)
    print("no. of tables = %s"%count_table)
    print("no. of text chunks = %s"%count_text)

    print("=== Chunks creation completed...")

    ########################################################################
def ingest(source, dataset, chunks_path, no_citation=True, embeddings_model_name:str="mpnet-v2"):
    from paperqa import Docs

    embeddings = get_embeddings(embeddings_model_name)
    weaviate = get_vectordb_client()

    # Does this dataset already exist?
    schema = weaviate.schema.get()
    all_classes = [c["class"] for c in schema["classes"]]

    docs, chunks = create_paperqa_vector_indexes(weaviate, embeddings, dataset)
    docs_store = Docs(doc_index=docs, texts_index=chunks, embeddings=embeddings)
    if "D" + dataset + "docs" in all_classes:
        # raise Exception(f"The specified dataset {dataset}  already exists")
        docs_store.build_doc_index()

    docs_list = []
    for extension in ["*.pdf", "*.html", "*.txt"]:
        docs_list.extend(
            glob.glob(os.path.join(source, "**/" + extension), recursive=True)
        )
    count = total = 0
    all_chunks = []

    for doc in docs_list:
        # print("Ingesting doc: ", doc, end="  ")
        print("Ingesting doc: ", doc)
        total += 1
        try:
            if no_citation:
                filename = doc.split('/')[-1]
                _, chunks = docs_store.add(doc, citation=filename, docname=filename)
            else:
                _, chunks = docs_store.add(doc)
            if chunks == None:
                print("----- Doc already ingested in the dataset, Skipping this doc")
            else:
                count += 1
                chunk_json = [{"text": x} for x in chunks]
                all_chunks.extend(chunk_json)
                print("----- Ingestion succeded. Number of chunks:", len(chunk_json))
        except Exception as ingest_except:
            print("xxxxx Ingestion failed, error: ", ingest_except)

    if not chunks_path is None and len(all_chunks) != 0:
        with open(
            chunks_path
            + f"/{dataset}-chunks-"
            + datetime.now().strftime("%m-%d-%Y-%H-%M")
            + ".json",
            "w",
        ) as fp:
            # If json.dumps  is going to be expensive with large lists, do it for each element separately
            fp.write(json.dumps(all_chunks, indent=4))

    print(f"\n ------ Ingest {count} of {total} documents ------")

    ########################################################################

@click.group()
def main(args=None):
    pass


def create_embeddings(file_name_dirs:str, dataset_path:str, dataset_name:str, embeddings_model_name:str):
    '''
    Creates embeddings from chunks read from files and stores in weaviate vector database.

    Parameters:
    fil_name_dirs (list):This is list of directory names associated with file names, inside each directories
    we have chunks directory, From this chunks directory code is reading text_chunks.json file.

    dataset_path (str):Used to create weaviate dataset name, to create chunks json file path and acesss it.

    use_openai_embeddings (bool):Flag to decide if we want to use openai embeddings model.
    '''
    emb_obj = get_embeddings(embeddings_model_name)
    weaviate = get_vectordb_client()
    # dataset = "GIdatasetBeCautious_new"
    # dataset = "gi_pipeline_v0_0_1_oct_11_2023_external_mpnet"
    # print("dataset {}".format(dataset))
    # exit(0)

    docs, chunks = create_paperqa_vector_indexes(weaviate, emb_obj, dataset_name)
    docs_store = Docs(doc_index=docs, texts_index=chunks, embeddings=emb_obj)

    # Reading chunk file from chunk file path defined above line.
    for file_name_dir in file_name_dirs:
        try:
            if file_name_dir != None: 
                # print("file name is : ",file_name_dir)
                chunk_file_path = dataset_path + file_name_dir + '/chunks/text_chunks.json'

                # Reading chunk file from chunk file path defined above line.
                with open(chunk_file_path, "r") as chunk_file:
                    chunk_data = json.load(chunk_file)

                if len(chunk_data) > 0:
                    
                    dockey = utils.md5sum(chunk_file_path)
                    doc = Doc(docname=file_name_dir, citation=file_name_dir, dockey=dockey)

                    texts = [
                        Text(text=list(t.values())[0], name=f"{doc.docname}_chunk_{i}", doc=doc)
                        for i, t in enumerate(chunk_data)
                        ]

                    docs_store.add_texts(texts=texts, doc=doc)

                    docs_store.clear_docs()

                    print(f"filename : {file_name_dir}, chunks : " , len(chunk_data))
                else:
                    logging.error(f"NO CHUNKS : {file_name_dir} has no chunks")
        except Exception as exc: 
            logging.error(exc)
            logging.error(f"ERROR : Embedding creation failed for {file_name_dir}")

@main.command()
@click.option("-d", "--dataset_path", "dataset_path", required=True, help="The path where docs are sourced from for ingestion",)
@click.option("--dataset_name", "dataset_name", required=True, help="The path where chunks are created",)
@click.option('-emb', '--embeddings', "embeddings_model_name", type=click.Choice(['mpnet-v2', 'openai', "bert"]), 
              required=False, default="mpnet-v2", help="Use on of OpenAI, mpnet-v2 or Bert")
def create_dataset(dataset_path:str, dataset_name:str, embeddings_model_name:Literal['mpnet-v2', 'openai', "bert"]):

    '''
    Create dataset out of chunks.
    Creates lists (groupes) using grouper function, Where each list will have 'n' directory names.
    Use these lists for multiprocessing in embedding creation process.

    Parameters:
    dataset_path (str):The directory path where directories associated with each file is there.
    dataset_name (str): Name of the dataset.
    embeddings_model_name (str): 

    '''
    num_cores = multiprocessing.cpu_count()
    no_of_files = len(os.listdir(dataset_path))
    file_name_dirs = os.listdir(dataset_path)

    def grouper(iterable, n=100, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    results = Parallel(n_jobs=num_cores)(
        delayed(create_embeddings)(list(groupp), dataset_path, dataset_name, embeddings_model_name) for groupp in grouper(file_name_dirs)
    )

    logging.info(f"Successfully embedded chunks from {dataset_path.split('/')[-2]} directory")


@main.command()
@click.option(
    "-s",
    "--source",
    "source",
    required=True,
    help="The path where docs are sourced from for ingestion",
)
@click.option(
    "-d",
    "--destination",
    "destination",
    required=True,
    help="The path where chunks are created",
)
@click.option(
    "-c",
    "--chunk-size",
    "chunk_size",
    required=False,
    default=3000,
    type=click.INT,
    help="chunk size",
)
def create_chunks(source, destination, chunk_size):
    """Create chunks of documents"""

    if not os.path.exists(source) or not os.path.isdir(source):
        print(f"Error: {source} doesn't exist or is not a directory")
        exit(1)

    # Workaround for openai-python bug #140. Doesn't close connections
    import warnings

    warnings.simplefilter("ignore", ResourceWarning)

    from datetime import datetime

    start_time = datetime.now()
    extract(source, destination, int(chunk_size))
    print(" ------ Duration(hh:mm:ss) {} ------".format(datetime.now() - start_time))

@main.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
@click.option("-s", "--source", "source", required=True, help="The path where docs are sourced from for ingestion",
    )
@click.option("-c", "--chunks", "chunks", required=False, default=None, help="The path where document chunks are saved for training",)
@click.option('-emb', '--embeddings', "embddings_model_name", type=click.Choice(['mpnet-v2', 'openai', "bert"]), 
              required=False, default="mpnet-v2", help="Use on of OpenAI, mpnet-v2 or Bert")
@click.option('-nc', '--no-citation', is_flag=True, default=True, help="Don't derive citation. Use file name [True]")
def add(dataset, source, chunks, embddings_model_name, no_citation):
    """ Ingest documents"""

    if re.fullmatch(DATASET_PAT, dataset) is None:
        print(
            f"Error: {dataset} should start with an alpabet and remaining should be alphanumeric"
        )
        exit(1)

    if not os.path.exists(source) or not os.path.isdir(source):
        print(f"Error: {source} doesn't exist or is not a directory")
        exit(1)

    if (
        chunks is not None
        and not os.path.exists(chunks)
        and not os.path.isdir(args.chunks)
    ):
        print(f"Error: {args.chunks} doesn't exist or is not a directory")
        exit(1)

    """
    # Automatically connect to the running Ray cluster.
    ray.init()
    print(ray.get(ingest(source, dataset).remote()))
    """
    # Workaround for openai-python bug #140. Doesn't close connections
    import warnings

    warnings.simplefilter("ignore", ResourceWarning)

    from datetime import datetime

    start_time = datetime.now()
    ingest(source, dataset, chunks, no_citation, embddings_model_name)
    print(" ------ Duration(hh:mm:ss) {} ------".format(datetime.now() - start_time))

@click.group()
def show():
    """Group for list commands"""
    pass

def list_datasets():
    client = get_vectordb_client()
    schema = client.schema.get()

    all_classes = [c["class"] for c in schema["classes"]]

    # Created a dictionary to group datasets without "ch" suffix
    dataset_groups = {}

    for c in all_classes:
        r = client.query.aggregate(class_name=c).with_meta_count().do()

        if "docs" in c:
            dataset = c[1:-4]
            ctype = "Documents"
        elif "chunks" in c:
            dataset = c[1:-6]
            ctype = "Chunks"
        else:
            dataset = c
            ctype = "unknown"

        if dataset not in dataset_groups:
            dataset_groups[dataset] = {"Documents": 0, "Chunks": 0}

        if ctype == "Documents":
            dataset_groups[dataset]["Documents"] = r["data"]["Aggregate"][c][0]["meta"]["count"]
        elif ctype == "Chunks":
            dataset_groups[dataset]["Chunks"] = r["data"]["Aggregate"][c][0]["meta"]["count"]

    for dataset in dataset_groups:
        print(f'Dataset: {dataset} - Documents: {dataset_groups[dataset]["Documents"]}, Chunks: {dataset_groups[dataset]["Chunks"]}')

def list_dataset_docs(dataset):
    weaviate = get_vectordb_client()
    schema = weaviate.schema.get()
    all_classes = [c["class"] for c in schema["classes"]]
    if not "D" + dataset + "docs" in all_classes:
        print(f"{dataset} doesn't exist")
        return

    try:
        embeddings = get_embeddings()
        docs, chunks = create_paperqa_vector_indexes(weaviate, embeddings, dataset)
        cursor = None
        batch_size = 1000 
        while True:
            docss, cursor = docs.get_objects(
                properties=["dockey"], limit=batch_size, cursor=cursor
            )

            for doc in docss:
                new_doc = json.loads(doc.page_content)
                print(new_doc["docname"])

            if len(docss) < batch_size:
                break

    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

@show.command()
def datasets():
    """ Lists all datasets in the vectordb """

    from datetime import datetime

    start_time = datetime.now()
    list_datasets()
    #print(" ------ Duration(hh:mm:ss) {} ------".format(datetime.now() - start_time))

@show.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
def docs(dataset):
    """ Lists all documents in a dataset """

    from datetime import datetime

    start_time = datetime.now()
    list_dataset_docs(dataset)
    #print(" ------ Duration(hh:mm:ss) {} ------".format(datetime.now() - start_time))

@click.group()
def delete():
    """Group for delete commands"""
    pass

def delete_docs(source, dataset):
    from paperqa import Docs

    weaviate = get_vectordb_client()

   # Does this dataset already exist?
    schema = weaviate.schema.get()
    all_classes = [c["class"] for c in schema["classes"]]

    if not "D" + dataset + "docs" in all_classes:
        print(f"{dataset} doesn't exist")
        return

    docs, chunks = create_paperqa_vector_indexes(
        weaviate, None, dataset
    )  # No embeddings needed for deletion
    docs_store = Docs(doc_index=docs, texts_index=chunks)
    docs_store.build_doc_index()

    file_extensions = [".pdf", ".html", ".txt"]
    docs_list = []
    for extension in ["*.pdf", "*.html", "*.txt"]:
        docs_list.extend(
            glob.glob(os.path.join(source, "**/" + extension), recursive=True)
        )
    count = total = 0

    for doc in docs_list:
        print("Deleting doc:", doc)
        total += 1
        try:
            docs_store.delete(doc)
            print("----- Deletion succeeded")
            count += 1
        except Exception as delete_except:
            print("xxxxx Deletion failed, error:", delete_except)

    print(f"\n------ Deleted {count} of {total} documents ------")

    # Verification of deleted docs

    verify_deleted_doc_count = 0
    try:
        for doc in docs_list:
            try:
                response = weaviate.data_object.get(f"D{dataset}docs", str(hash(doc)))
                if response.status_code == 404:
                    verify_deleted_doc_count += 1
            except Exception as search_error:
                print("Error during search:", search_error)
    except Exception as conn_error:
        print("Error connecting to Weaviate:", conn_error)

def delete_dataset(dataset):
    WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)

    if WEAVIATE_URL is None:
        print(
            "WEAVIATE_URI environment variable is not set. Please set it to the Weaviate server URL."
        )
        return

    client = weaviate.Client(WEAVIATE_URL)
    schema = client.schema.get()
    all_classes = [c["class"] for c in schema["classes"]]

    if not "D" + dataset + "docs" in all_classes:
        print(f"{dataset} doesn't exist")
        return

    # Derive class names for documents and chunks
    docs_class_name = f"D{dataset}docs"
    chunks_class_name = f"D{dataset}chunks"

    # Delete document class
    try:
        client.schema.delete_class(docs_class_name)
        #print(f"Deleted document class: {docs_class_name}")
    except weaviate.exceptions.RequestsConnectionError as e:
        print(f"Error deleting dataset {dataset}: Failed connecting to VectorStore: {e}")
    except weaviate.exceptions.ObjectNotFoundError:
        print(f"Error deleting dataset {dataset}: Document class {docs_class_name} does not exist.")
    except Exception as e:
        print(f"Error deleting dataset {dataset}: Failed to delete document class: {e}")

    # Delete chunk class
    try:
        client.schema.delete_class(chunks_class_name)
        #print(f"Deleted chunk class: {chunks_class_name}")
    except weaviate.exceptions.RequestsConnectionError as e:
        print(f"Error deleting dataset {dataset}: Failed connecting to VectorStore: {e}")
    except weaviate.exceptions.ObjectNotFoundError:
        print(f"Error deleting dataset {dataset}: Chunk class {chunks_class_name} does not exist.")
    except Exception as e:
        print(f"Error deleting dataset {dataset}: Failed to delete chunk class: {e}")

@delete.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
@click.option("-s", "--source", "source", required=True, help="The path where docs are sourced from for ingestion",
    )
def docs(dataset, source):
    """ Deletes specified docs from a dataset """

    if re.fullmatch(DATASET_PAT, dataset) is None:
        print(
            f"Error: {dataset} should start with an alpabet and remaining should be alphanumeric"
        )
        exit(1)

    if not os.path.exists(source) or not os.path.isdir(source):
        print(f"Error: {source} doesn't exist or is not a directory")
        exit(1)

    """
    # Automatically connect to the running Ray cluster.
    ray.init()
    print(ray.get(ingest(source, dataset).remote()))
    """
    # Workaround for openai-python bug #140. Doesn't close connections
    import warnings
    warnings.simplefilter("ignore", ResourceWarning)

    from datetime import datetime

    start_time = datetime.now()
    delete_ingested_docs(source, dataset)
    #print(" ------ Duration(hh:mm:ss) {} ------".format(datetime.now() - start_time))

@delete.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
def dataset(dataset):
    """ Deletes all docs for a given dataset """

    from datetime import datetime

    #start_time = datetime.now()
    delete_dataset(dataset)
    #print(" ------ Duration(hh:mm:ss) {} ------".format(datetime.now() - start_time))


@click.group()
def backup():
    """Group for backup commands"""
    pass

def backup_create(dataset, backup_id, backend):

    if "WEAVIATE_URI" not in os.environ:
        print(
            "WEAVIATE_URL environment variable is not set. Please set it to the Weaviate server URL."
        )

        return

    WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)
    client = weaviate.Client(WEAVIATE_URL)

    # Derive class names for documents and chunks
    docs_class_name = f"D{dataset}docs"
    chunks_class_name = f"D{dataset}chunks"

    try:
        if backend == "filesystem":
            # Create a backup to the local filesystem
            result = client.backup.create(
                backup_id=backup_id,
                backend=backend,
                include_classes=[docs_class_name, chunks_class_name],
                wait_for_completion=True,
            )
            print(
                f"Backup created successfully in filesystem. Backup ID: {result['id']}"
            )

        elif backend == "s3":
            # Create a backup to Amazon S3
            result = client.backup.create(
                backup_id=backup_id,
                backend=backend,
                include_classes=[docs_class_name, chunks_class_name],
                wait_for_completion=True,
            )
            print(
                f"Backup created successfully in s3-bucket. Backup ID: {result['id']}"
            )
        else:
            print(f"Invalid backend specified: {backend}")
            return

    except Exception as e:
        print(f"Error creating backup: {e}")


def backup_restore(backup_id, backend):

    if "WEAVIATE_URI" not in os.environ:
        print(
            "WEAVIATE_URL environment variable is not set. Please set it to the Weaviate server URL."
        )
        return

    WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)
    client = weaviate.Client(WEAVIATE_URL)

    # Restore parameters
    restore_params = {
        "backup_id": backup_id,
        "backend": backend,
        "wait_for_completion": True,
    }

    # if exclude_classes:
    #     restore_params["exclude_classes"] = exclude_classes

    try:
        if backend == "filesystem":
            # Restore from the local filesystem
            result = client.backup.restore(**restore_params)
            print(f"Restore completed successfully from filesystem. Result: {result}")

        elif backend == "s3":
            # Restore from Amazon S3
            result = client.backup.restore(
                **restore_params,
            )
            print(f"Restore completed successfully from S3 bucket. Result: {result}")

        else:
            print(f"Invalid backend specified: {backend}")
            return

    except Exception as e:
        print(f"Error during restore: {e}")

@backup.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
@click.option("-i", "--id", "id", required=True, help="An ID for the backup",
    )
@click.option("-b", "--backend", "backend", required=True, type=click.Choice(['filesystem', 's3']),help="s3 or filesystem to save the backup",
    )
def create(dataset, id, backend):
    """ Creates backup of embeddings. Separate backups for each dataset """

    backup_create(dataset, id, backend)

@backup.command()
@click.option("-i", "--id", "id", required=True, help="An ID for the backup",
    )
@click.option("-b", "--backend", "backend", required=True, type=click.Choice(['filesystem', 's3']),help="s3 or filesystem to save the backup",
    )
def restore(id, backend):
    """ restores embeddings from backup. """

    backup_restore(id, backend)

main.add_command(delete)
main.add_command(show)
main.add_command(backup)
if __name__ == '__main__':
    main()
