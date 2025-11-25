# script.py
# import ray
import csv
import glob
import logging
import os
import re
import sys
import uuid
from typing import Literal, Tuple

import click
import openai
import weaviate  # weaviate-python client
from langchain.chat_models import ChatOpenAI

#sys.path.append('/usr/local/lib/python3.10/dist-packages')
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Weaviate
from langchain.vectorstores.base import VectorStore
from langchain.embeddings import HuggingFaceBgeEmbeddings

'''
Weaviate URL from inside the cluster
--- "http://weaviate.d3x.svc.cluster.local/"
Weaviate URL from outside the cluster
--- NodePort - "http://<ip>:30716/api/vectordb/"
--- Loadbalancer - "http://<ip>:80/api/vectordb/"
'''

WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)

def get_embeddings(embeddings_model_name: Literal['mpnet-v2', 'openai', "bert"]) -> Embeddings:
    logging.info(f"Embeddings model name: {embeddings_model_name}")
    if embeddings_model_name == "mpnet-v2":
        from langchain.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
        )
        return hf
    
    elif embeddings_model_name == "openai":
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            client=None, openai_api_base="https://api.openai.com/v1",
            openai_api_type="open_ai")
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
    DKUBEX_API_KEY = os.getenv( "DKUBEX_API_KEY", "deadbeef")

    # Use Weaviate VectorDB
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={"Authorization": DKUBEX_API_KEY},
    )

    return weaviate_client

def create_paperqa_vector_indexes(client:VectorStore, embeddings:Embeddings, dataset:str) -> Tuple[VectorStore, VectorStore]:

    # Create 2 classes (or DBs) in Weaviate
    # One for docs and one for chunks
    docs = Weaviate(
        client=client,
        index_name="D"+ dataset + "docs",
        text_key="paperdoc",
        attributes=['dockey'],
        embedding=embeddings,
        by_text=False,
    )


    chunks = Weaviate(
        client=client,
        index_name="D"+ dataset + "chunks",
        text_key="paperchunks",
        embedding=embeddings,
        attributes=['doc', 'name'],
        by_text=False,
    )

    return docs, chunks


'''
@ray.remote
'''

@click.group()
def main(args=None):
    pass

@main.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
@click.option("-g", "--securellm", "securellm", is_flag=True, default=False, help="Use SecureLLM as the Gateway",
    )
@click.option('-emb', '--embeddings', "embddings_model_name", type=click.Choice(['mpnet-v2', 'openai', "bert"]), 
              required=False, default="mpnet-v2", help="Use on of OpenAI, mpnet-v2 or Bert")
@click.option("-f", "--input", "input_file", required=False, help="Input file for batch Q&A",
    )
@click.option("-o", "--output", "output_file", required=False, help="Output file for batch Q&A",
    )
@click.option("-k", "--k-neighbors", "k", required=False, default=5, help="K nearest neighbors from vectordb",
    )
@click.option("-ms", "--max-sources", "max_sources", required=False, default=3, help="Number of chunks/summarizations to use for final answer",
    )
@click.option('-mmr', '--marginal-relevance', 'marginal_relevance', is_flag=True, default=False, help="Use marginal relevance default:similarity_search")
@click.option('-ds', '--disable-summarization', 'disable_summarization', is_flag=True, default=False, help="Disable LLM Summarization of K chunks")
@click.option("-v", "--log-level", "log_level", type=click.Choice(['DEBUG', "INFO"]), required=False, default="INFO", help="log level DEBUG, INFO")
def openAI(dataset,
           securellm, 
           embddings_model_name, 
           input_file, output_file,
           k, max_sources,
           marginal_relevance,
           disable_summarization,
           log_level):

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.getLevelName(log_level), datefmt='%Y-%m-%d %H:%M:%S')
    return query(dataset = dataset, 
                 securellm=securellm, 
                 embddings_model_name=embddings_model_name, 
                 deployment=None, 
                 namespace=None, 
                 input_file=input_file, 
                 output_file=output_file,
                 k=k,
                 max_sources=max_sources,
                 marginal_relevance=marginal_relevance,
                 disable_summarization=disable_summarization
                 )


@main.command()
@click.option("-d", "--dataset", "dataset", required=True, help="A name to represent the ingested docs",
    )
@click.option("-g", "--securellm", "securellm", is_flag=True, default=False, help="Use SecureLLM as the Gateway",
    )
@click.option('-emb', '--embeddings', "embddings_model_name", type=click.Choice(['mpnet-v2', 'openai', "bert"]), 
              required=False, default="mpnet-v2", help="Use on of OpenAI, mpnet-v2 or Bert")
@click.option("-dep", "--deployment", "deployment", required=True, help="LLM deployment name",
    )
@click.option("-n", "--namespace", "namespace", required=False, help="Deployment namespace. Default: user namespace",
    )
@click.option("-f", "--input", "input_file", required=False, help="Input file for batch Q&A",
    )
@click.option("-o", "--output", "output_file", required=False, help="Output file for batch Q&A",
    )
@click.option("-k", "--k-neighbors", "k", required=False, default=5, help="K nearest neighbors from vectordb",
    )
@click.option("-ms", "--max-sources", "max_sources", required=False, default=3, help="Number of chunks/summarizations to use for final answer",
    )
@click.option('-mmr', '--marginal-relevance', 'marginal_relevance', is_flag=True, default=False, help="Use marginal relevance default:similarity_search")
@click.option('-ds', '--disable-summarization', 'disable_summarization', is_flag=True, default=False, help="Disable LLM Summarization of K chunks")
@click.option("-v", "--log-level", "log_level", type=click.Choice(['DEBUG', "INFO"]), required=False, default="INFO", help="log level DEBUG, INFO")
def llm(dataset, 
        securellm, 
        embddings_model_name, 
        deployment, namespace, 
        input_file, output_file, 
        k, max_sources,
        marginal_relevance,
        disable_summarization,
        log_level):

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.getLevelName(log_level), datefmt='%Y-%m-%d %H:%M:%S')
    return query(dataset = dataset, 
                 securellm=securellm, 
                 embddings_model_name=embddings_model_name, 
                 deployment=deployment, 
                 namespace=namespace, 
                 input_file=input_file, 
                 output_file=output_file,
                 k=k,
                 max_sources=max_sources,
                 marginal_relevance=marginal_relevance,
                 disable_summarization=disable_summarization
                 )

def query(dataset, 
          securellm, 
          embddings_model_name, 
          deployment, namespace, 
          input_file, output_file, 
          k,
          max_sources,
          marginal_relevance,
          disable_summarization
          ):

    from paperqa import Docs

    embeddings = get_embeddings(embddings_model_name)
    weaviate = get_vectordb_client()
    docs, chunks = create_paperqa_vector_indexes(weaviate, embeddings, dataset)
    docs_store = Docs(doc_index=docs, texts_index = chunks, embeddings=embeddings)

    model_name = None
    headers = {}
    openai_api_base=None
    openai_api_host=None
    user = os.getenv("USER")
    if deployment != None and namespace == None:
        namespace = user

    if deployment != None:
        openai_api_base = f"http://{deployment}-serve-svc.{namespace}:8000/v1"
        openai_api_host = f"http://{deployment}-serve-svc.{namespace}:8000"

        openai.api_base = openai_api_base
        openai.api_host = openai_api_host

        models = openai.Model.list()
        # GPT3.5 is much faster. This picks up GPT 4
        model_name = models["data"][0]["id"]


    if securellm == True:
        if deployment != None:
            dkubexdepep = f"http://{deployment}-serve-svc.{user}:8000"
            headers = {"x-sgpt-flow-id": str(uuid.uuid4()), "X-Auth-Request-Email": user, "llm-provider": dkubexdepep}
        else:
            headers = {"x-sgpt-flow-id": str(uuid.uuid4()), "X-Auth-Request-Email": user}

        openai_api_base = "http://securellm-be.securellm:3005/v1"
        openai_api_host = "http://securellm-be.securellm:3005"


    def update_llm():
        if model_name != None:
            chatllm = ChatOpenAI(
                model_name=model_name,
                headers=headers,
                openai_api_base=openai_api_base,
                openai_api_host=openai_api_host
            )
        else:
            chatllm = ChatOpenAI(
                headers=headers,
                openai_api_base=openai_api_base,
            )
        docs_store.update_llm(llm=chatllm)
    
   
    docs_store.build_doc_index()


    if input_file or output_file:
        if input_file == None or output_file == None:
            print("Input and Output files need to be specified for batch Q&A")
            exit(1)
        if not os.path.isfile(input_file):
            print("Input file does not exist")
            exit(1)
        if not 'csv' in output_file.split('.')[-1]:
            print("Output file is not a csv file")
            exit(1)

        write_header = False
        if not os.path.isfile(output_file):
            write_header=True

        header = ['id', 'question', 'answer']
        qid = 0 # query id
        with open(input_file) as i_file:
            queries = i_file.readlines()

        if len(output_file.split('/')) > 1:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if not write_header:
            with open(output_file, "r") as o_file:
                csv_reader = csv.reader(o_file, dialect='excel')
                if header != list(csv_reader)[0]:
                    print("Output file has invalid headers")
                    exit(1)
        o_file = open(output_file, "a", newline='')
        csv_writer = csv.writer(o_file, dialect='excel')

        if write_header:
            csv_writer.writerow(header)


        for query in queries:
            update_llm()
            query=query.strip()
            print('  --- generating answer for "%s"'%query)
            answer = docs_store.query(query, k=k, max_sources=max_sources,
                                      marginal_relevance=marginal_relevance,
                                      disable_summarization=disable_summarization)

            csv_writer.writerow([qid, query, '\n'.join(str(answer).split('\n')[1:]).strip()])
            qid += 1
        o_file.close()
        return
    
    # query from terminal
    while True:
        update_llm()
        query = input("Question>: ")
        if query.lower() in ['exit','stop','quit']:
            exit(0)

        answer = docs_store.query(query, k=k, max_sources=max_sources, 
                                  marginal_relevance=marginal_relevance, 
                                  disable_summarization=disable_summarization)
        print(answer)


if __name__ == '__main__':
    main()
