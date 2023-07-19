
from ast import List
import asyncio
import threading
import queue
import os
from typing import Any, Dict
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response,StreamingResponse

from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from fastapi.middleware.cors import CORSMiddleware
import json
from paperqa import Docs
import pickle
import uuid
import os
import weaviate  # weaviate-python client

from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings


# Weaviate URL from inside the cluster
WEAVIATE_URL = "http://weaviate.d3x.svc.cluster.local/" ## Todo Get it from Environment

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


def load_vectorstore():
    global vectorstore

    embeddings = OpenAIEmbeddings(client=None)
    dataset = os.getenv("DATASET", "Ankitgroup")  ## Todo Get it from Environment

    # Use Weaviate VectorDB
    auth_headers = {}
    DKUBEX_API_KEY = os.getenv("DKUBEX_API_KEY", None)
    if DKUBEX_API_KEY is not None:
        auth_headers["Authorization"] = DKUBEX_API_KEY
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers=auth_headers)

    weaviatedb_docs = Weaviate(
        client=weaviate_client,
        # The first letter needs to be Capitalized. Prefixing 'D'
        index_name="D" + dataset + "docs",
        text_key="paperdoc",
        attributes=['dockey'],
        embedding=embeddings,
    )


    weaviatedb_chunks = Weaviate(
        client=weaviate_client,
        # The first letter needs to be Capitalized. Prefixing 'D'
        index_name="D" + dataset + "chunks",
        text_key="paperchunks",
        embedding=embeddings,
        attributes=['doc', 'name'],
    )

    vectorstore = Docs(doc_index=weaviatedb_docs, texts_index = weaviatedb_chunks)
    vectorstore.build_doc_index()


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, tag, gen):
        super().__init__()
        self.gen = gen
        self.tag = tag

    def on_llm_new_token(self, token, **kwargs):
        # print("on llm new token",token)
        self.gen.send(token)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        message = self.tag
        message = f'\n\n **{message}**  \n\n'
        self.gen.send(message)


def llm_thread(g, prompt, model, oaikey, flowid, username):
    global vectorstore
    print(f"Passed flowid {flowid}")
    flowid = flowid or str(uuid.uuid4())
    chatllm = ChatOpenAI(temperature=0.1, 
            model_name=model,
            streaming=True, 
            openai_api_key=oaikey, 
            headers={
                "x-sgpt-flow-id": flowid, 
                "X-Auth-Request-Email": username
                }
        )
    vectorstore.update_llm(llm=chatllm)
    def get_callbacks(arg):
        #Enable this line to capture only the answer and not other chain calls made by paperQA
        if arg.lower() == "answer":
        #if type(arg) == str:
            return [ChainStreamHandler(arg, g)]
        return []
    async def query():
        try:
            global vectorstore
            result = await vectorstore.aquery(prompt, get_callbacks=get_callbacks)
            sdocs = result.references
            smessage = "\n\n" + "**References:**" + "\n\n" + ''.join(sdocs)
            g.send(smessage)
        finally:
            g.close()
    
    asyncio.run(query())

def chat(prompt, model, oaikey, flowid, username):
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, prompt, model, oaikey, flowid, username)).start()
    return g

#--------------------------------------------------------------------------
# Each Custom application must implement the below routes.
# This agent implementation is based on paperQA
#--------------------------------------------------------------------------

app = FastAPI(
    title="Langchain AI API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global vectorstore

    load_vectorstore()

@app.get("/title")
async def getTitle():
    title = os.getenv("APP_TITLE", "QnA Agent")
    return {"title": title} 

@app.post("/stream")
async def stream(request: Request):
    body = await request.body()
    body = body.decode()
    body = json.loads(body)
    message = body['messages'][-1]['content']
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    oai_base = os.getenv("OPENAI_API_BASE", None)
    oai_key = os.getenv("OPENAI_API_KEY", None)
    flowid = request.headers.get("x-sgpt-request-id", None)
    username = request.headers.get("X-Auth-Request-Email", "anonymous")
    return StreamingResponse(chat(message, model, oai_key, flowid, username))

