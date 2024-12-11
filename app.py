from FlagEmbedding import BGEM3FlagModel
from typing import List, Tuple, Union, cast
import asyncio
import os
from fastapi import FastAPI, Request, Response, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
# from starlette.status import HTTP_504_GATEWAY_TIMEOUT
# from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
# import time
from concurrent.futures import ThreadPoolExecutor
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from sentence_transformers import SentenceTransformer
import torch

# Configuration
class Config:
    API_KEY = os.getenv("API_KEY", "secure-api-key")
    RERANKER_TYPE = os.getenv("RERANKER_TYPE", "bge-m3")  # Default to bge-m3
    DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU
    BATCH_SIZE = 1
    MAX_REQUEST = 5
    MAX_QUERY_LENGTH = 256
    REQUEST_TIMEOUT = 120
    GPU_TIMEOUT = 15
    RERANK_WEIGHTS = [0.4, 0.2, 0.4]
    PORT = int(os.getenv("PORT", 8000))


class M3ModelWrapper:
    def __init__(self, model_name: str):
        self.model = BGEM3FlagModel(model_name, device=Config.DEVICE, use_fp16=False)
        self.lock = threading.Lock()  # Ensure thread-safety

    def embed(self, sentences: List[str]) -> List[List[float]]:
        with self.lock:  # Ensure thread-safe access
            with torch.no_grad():  # Disable gradient computation
                embeddings = self.model.encode(sentences, batch_size=Config.BATCH_SIZE)["dense_vecs"]
        return embeddings.tolist()
    
    def rerank(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        with self.lock:  # Ensure thread-safe access
            scores = self.model.compute_score(
                sentence_pairs,
                batch_size=Config.BATCH_SIZE,
                max_query_length=Config.MAX_QUERY_LENGTH,
                weights_for_different_modes=Config.RERANK_WEIGHTS,
            )["colbert+sparse+dense"]
        return scores


# Reranker Model Wrapper for Cross-Encoder
class RerankerModelWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(Config.DEVICE)

    def rerank(self, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
        inputs = [f"{q} [SEP] {p}" for q, p in sentence_pairs]
        tokenized = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            logits = self.model(**tokenized).logits.squeeze(-1)
        return logits.tolist()


# Reranker Factory
def get_reranker():
    if Config.RERANKER_TYPE == "bge-m3":
        print("Using BGE-M3 Reranker")
        return M3ModelWrapper("BAAI/bge-m3")
    elif Config.RERANKER_TYPE == "cross-encoder":
        print("Using Cross-Encoder Reranker")
        return RerankerModelWrapper("cross-encoder/ms-marco-MiniLM-L-12-v2")
    else:
        raise ValueError(f"Unknown Reranker Type: {Config.RERANKER_TYPE}")


class EmbedRequest(BaseModel):
    sentences: List[str]

class RerankRequest(BaseModel):
    sentence_pairs: List[List[str]]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class RerankResponse(BaseModel):
    scores: List[float]


# Request Processor
class RequestProcessor:
    def __init__(self, model, max_request_to_flush: int, accumulation_timeout: float):
        self.model = model
        self.max_batch_size = max_request_to_flush
        self.accumulation_timeout = accumulation_timeout
        self.queue = asyncio.Queue()
        self.response_futures = {}
        self.executor = ThreadPoolExecutor()
        self.gpu_lock = asyncio.Semaphore(1)

    async def ensure_processing_loop_started(self):
        if not hasattr(self, "processing_loop_task") or self.processing_loop_task.done():
            self.processing_loop_task = asyncio.create_task(self.processing_loop())

    async def processing_loop(self):
        while True:
            requests, request_types, request_ids = [], [], []
            start_time = asyncio.get_event_loop().time()

            while len(requests) < self.max_batch_size:
                timeout = self.accumulation_timeout - (asyncio.get_event_loop().time() - start_time)
                if timeout <= 0:
                    break
                try:
                    req_data, req_type, req_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    requests.append(req_data)
                    request_types.append(req_type)
                    request_ids.append(req_id)
                except asyncio.TimeoutError:
                    break

            if requests:
                await self.process_requests_by_type(requests, request_types, request_ids)

    async def process_requests_by_type(self, requests, request_types, request_ids):
        tasks = []
        for request_data, request_type, request_id in zip(requests, request_types, request_ids):
            if request_type == "embed":
                task = asyncio.create_task(self.run_with_semaphore(self.model.embed, request_data.sentences, request_id))
            else:  # 'rerank'
                task = asyncio.create_task(self.run_with_semaphore(self.model.rerank, request_data.sentence_pairs, request_id))
            tasks.append(task)
        await asyncio.gather(*tasks)

    async def run_with_semaphore(self, func, data, request_id):
        async with self.gpu_lock:
            future = self.executor.submit(func, data)
            try:
                result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=Config.GPU_TIMEOUT)
                self.response_futures[request_id].set_result(result)
            except asyncio.TimeoutError:
                self.response_futures[request_id].set_exception(TimeoutError("Processing timeout"))
            except Exception as e:
                self.response_futures[request_id].set_exception(e)

    async def process_request(self, request_data: Union[EmbedRequest, RerankRequest], request_type: str):
        await self.ensure_processing_loop_started()
        request_id = str(uuid4())
        self.response_futures[request_id] = asyncio.Future()
        await self.queue.put((request_data, request_type, request_id))
        return await self.response_futures[request_id]


# FastAPI App
app = FastAPI()

# Initialize Embedding and Reranking Models
embedding_model = M3ModelWrapper("BAAI/bge-m3")
if Config.RERANKER_TYPE == "bge-m3":
    reranker_model = embedding_model
else:
    reranker_model = get_reranker()

embedding_processor = RequestProcessor(embedding_model, Config.MAX_REQUEST, Config.REQUEST_TIMEOUT)
reranker_processor = RequestProcessor(reranker_model, Config.MAX_REQUEST, Config.REQUEST_TIMEOUT)

# Define the API Key Header (default: "X-API-Key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

print("config:", Config.API_KEY)
def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != Config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )

@app.post("/embeddings/", response_model=EmbedResponse, dependencies=[Depends(validate_api_key)])
async def get_embeddings(request: EmbedRequest):
    embeddings = await embedding_processor.process_request(request, 'embed')
    return EmbedResponse(embeddings=embeddings)

@app.post("/rerank/", response_model=RerankResponse, dependencies=[Depends(validate_api_key)])
async def rerank(request: RerankRequest):
    scores = await reranker_processor.process_request(request, 'rerank')
    return RerankResponse(scores=scores)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)