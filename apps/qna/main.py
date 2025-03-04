from ast import List
import asyncio
import threading
import queue
import os
from typing import Any, Dict
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

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
WEAVIATE_URL = "http://weaviate.d3x.svc.cluster.local/"  ## Todo Get it from Environment

from langchain.schema import AIMessage, HumanMessage, SystemMessage

lock = threading.Lock()


def load_vectorstore():
    global vectorstore

    if os.getenv("embeddings", "openai") == "mpnet":
        from langchain.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
        )

    else:
        embeddings = OpenAIEmbeddings(client=None)
    dataset = os.getenv("DATASET", "Ankitgroup")  ## Todo Get it from Environment

    # Use Weaviate VectorDB
    auth_headers = {}
    DKUBEX_API_KEY = os.getenv("DKUBEX_API_KEY", None)
    if DKUBEX_API_KEY is not None:
        auth_headers["Authorization"] = DKUBEX_API_KEY
    weaviate_client = weaviate.Client(url=WEAVIATE_URL, additional_headers=auth_headers)

    weaviatedb_docs = Weaviate(
        client=weaviate_client,
        # The first letter needs to be Capitalized. Prefixing 'D'
        index_name="D" + dataset + "docs",
        text_key="paperdoc",
        attributes=["dockey"],
        embedding=embeddings,
	by_text=False,
    )

    weaviatedb_chunks = Weaviate(
        client=weaviate_client,
        # The first letter needs to be Capitalized. Prefixing 'D'
        index_name="D" + dataset + "chunks",
        text_key="paperchunks",
        embedding=embeddings,
        attributes=["doc", "name"],
	by_text=False,
    )

    vectorstore = Docs(doc_index=weaviatedb_docs, texts_index=weaviatedb_chunks, embeddings=embeddings)
    vectorstore.build_doc_index()


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()
        self.lock = lock

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get(block=True)
        if item is StopIteration:
            raise item
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
        if "answer" in self.tag.lower():
            #print(f"sending token.....")
            self.gen.send(token)
        else:
            self.gen.send("\r")

    def on_llm_start(self, serialized, prompts, **kwargs):
        message = self.tag
        message = f"\n\n **{message}** \n\n"
        self.gen.send(message)


async def ask(work):
    g = work["generator"]
    prompt = work["prompt"]
    model = work["model"]
    oaikey = work["oaikey"]
    flowid = work["flowid"]
    username = work["username"]

    global vectorstore
    print(f"Passed flowid {flowid}")
    flowid = flowid or str(uuid.uuid4())

    if os.getenv("DKUBEX_DEPLOYMENT", None) != None:
        dkubex_depep = f"http://{os.environ['DKUBEX_DEPLOYMENT']}-serve-svc.{os.environ['USER']}:8000"

        chatllm = ChatOpenAI(
            temperature=0.1,
            model_name=model,
            streaming=True,
            #openai_api_key=oaikey,
            headers={"x-sgpt-flow-id": flowid, "X-Auth-Request-Email": username, "llm-provider": dkubex_depep},
        )
    else:
        chatllm = ChatOpenAI(
            temperature=0.1,
            model_name="gpt-3.5-turbo",
            streaming=True,
            openai_api_key=oaikey,
            headers={"x-sgpt-flow-id": flowid, "X-Auth-Request-Email": username},
        )

    vectorstore.update_llm(llm=chatllm)

    def get_callbacks(arg):
        return [ChainStreamHandler(arg, g)]

    async def query():
        try:
            global vectorstore
            global MATCH_K, MATCH_MS, MATCH_MR, MATCH_DS
            
            result = await vectorstore.aquery(prompt, get_callbacks=get_callbacks,
                                              k=MATCH_K, max_sources=MATCH_MS,
                                              marginal_relevance=MATCH_MR,
                                              disable_summarization=MATCH_DS)
            sdocs = result.references
            smessage = "\n\n" + "**References:**" + "\n\n" + "".join(sdocs)
            g.send(smessage)
        finally:
            g.close()

    await query()


def task_thread(loop, work):
    worker = ask(work)
    future = asyncio.run_coroutine_threadsafe(worker, loop)
    future.result()


# --------------------------------------------------------------------------
# Each Custom application must implement the below routes.
# This agent implementation is based on paperQA
# --------------------------------------------------------------------------

app = FastAPI(
    title="Langchain AI API",
)

app.fifo_queue = asyncio.Queue(maxsize=-1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def waiter():
    loop = asyncio.get_running_loop()
    while True:
        work = await app.fifo_queue.get()
        # start a new thread
        threading.Thread(
            target=task_thread,
            args=(
                loop,
                work,
            ),
        ).start()


@app.on_event("startup")
async def startup_event():
    global vectorstore
    global MATCH_K, MATCH_MS, MATCH_MR, MATCH_DS

    load_vectorstore()

    true_values = ["1", "True", "Yes"]
    MATCH_K = int(os.getenv('MATCH_K', 4))
    MATCH_MS = int(os.getenv('MATCH_MS', 3))
    MATCH_MR = os.getenv('MATCH_MR', 'True') in true_values
    MATCH_DS = os.getenv('MATCH_DS', 'False') in true_values
    print("k = {}, max_sources = {}, marginal_relevance = {}, \
          disable_summarization = {}".format(MATCH_K, MATCH_MS, MATCH_MR, MATCH_DS))
    
    asyncio.create_task(waiter())


@app.get("/title")
async def getTitle():
    title = os.getenv("APP_TITLE", "QnA Agent")
    return {"title": title}

async def eos():
    print("End of stream reached...")

@app.post("/stream")
async def stream(request: Request):
    body = await request.body()
    body = body.decode()
    body = json.loads(body)
    message = body["messages"][-1]["content"]
    model = body.get('model', {"id": None}).get("id")
    if model == None:
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    oai_base = os.getenv("OPENAI_API_BASE", None)
    oai_key = os.getenv("OPENAI_API_KEY", None)
    flowid = request.headers.get("x-sgpt-request-id", None)
    username = request.headers.get("X-Auth-Request-Email", "anonymous")
    g = ThreadedGenerator()
    work = {
        "generator": g,
        "prompt": message,
        "model": model,
        "oaikey": oai_key,
        "flowid": flowid,
        "username": username,
    }
    await app.fifo_queue.put(work)
    return StreamingResponse(g, background=BackgroundTask(eos))

