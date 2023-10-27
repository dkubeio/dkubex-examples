# script.py
# import ray
import glob
import os
import re
import uuid
import weaviate  # weaviate-python client
import openai
from langchain.chat_models import ChatOpenAI
from typing import Tuple

import sys
#sys.path.append('/usr/local/lib/python3.10/dist-packages')


from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Weaviate

'''
Weaviate URL from inside the cluster
--- "http://weaviate.d3x.svc.cluster.local/"
Weaviate URL from outside the cluster
--- NodePort - "http://<ip>:30716/api/vectordb/"
--- Loadbalancer - "http://<ip>:80/api/vectordb/"
'''

WEAVIATE_URL = os.getenv("WEAVIATE_URI", None)


def get_embeddings(use_openai_embeddings=False) -> Embeddings:
    if not use_openai_embeddings:
        from langchain.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        hf = HuggingFaceEmbeddings(
            model_name=model_name,
        )
        return hf
    else:
        from langchain.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(client=None)
        return embeddings

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
    )


    chunks = Weaviate(
        client=client,
        index_name="D"+ dataset + "chunks",
        text_key="paperchunks",
        embedding=embeddings,
        attributes=['doc', 'name'],
    )

    return docs, chunks


'''
@ray.remote
'''
def query(dataset, securellm=False, use_openai_embeddings=False, deployment=None):

    from paperqa import Docs

    embeddings = get_embeddings(use_openai_embeddings)
    weaviate = get_vectordb_client()
    docs, chunks = create_paperqa_vector_indexes(weaviate, embeddings, dataset)
    docs_store = Docs(doc_index=docs, texts_index = chunks, embeddings=embeddings)

    model_name = None
    headers = {}
    openai_api_base=None
    openai_api_host=None
    user = os.getenv("USER")
    if deployment != None:
        openai_api_base = f"http://{deployment}-serve-svc:8000/v1"
        openai_api_host = f"http://{deployment}-serve-svc:8000"

        openai.api_base = openai_api_base
        openai.api_host = openai_api_host

        models = openai.Model.list()
        model_name = models["data"][0]["id"]


    if securellm == True:
        if deployment != None:
            dkubexdepep = f"http://{deployment}-serve-svc.{user}:8000"
            headers = {"x-sgpt-flow-id": str(uuid.uuid4()), "X-Auth-Request-Email": user, "llm-provider": dkubexdepep}
        else:
            headers = {"x-sgpt-flow-id": str(uuid.uuid4()), "X-Auth-Request-Email": user}

        openai_api_base = "http://securellm-be.securellm:3005/v1"
        openai_api_host = "http://securellm-be.securellm:3005"


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

    while True:
        query = input("Question>: ")
        if query.lower() in ['exit','stop','quit']:
            exit(1)
        answer = docs_store.query(query, k=5)
        print(answer)



import fire

def cli(dataset=None, securellm=False, use_openai_embeddings=False, deployment=None):
    dataset_re = r"[A-Za-z0-9]+"
    if re.fullmatch(dataset_re, dataset) == None:
        print(f"{args.dataset} should be alphanumeric")
        exit(1)

    query(dataset, securellm=securellm, use_openai_embeddings=use_openai_embeddings, deployment=deployment)

if __name__ == '__main__':
  fire.Fire(cli)
