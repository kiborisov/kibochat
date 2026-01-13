#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

This module now exposes a build_chat_web_app() helper so it can be used both as a
standalone CLI and inside Modal web endpoints.
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096


@dataclass
class ChatWebConfig:
    num_gpus: int = 1
    source: str = "sft"
    temperature: float = 0.8
    top_k: int = 50
    max_tokens: int = 512
    model_tag: Optional[str] = None
    step: Optional[int] = None
    dtype: str = "bfloat16"
    device_type: str = ""
    host: str = "0.0.0.0"
    port: int = 8000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NanoChat Web Server")
    parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
    parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: sft|mid|rl")
    parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Default temperature for generation")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Default top-k sampling parameter")
    parser.add_argument("-m", "--max-tokens", type=int, default=512, help="Default max tokens for generation")
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps"],
        help="Device type for evaluation: cuda|cpu|mps. empty => autodetect",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    return parser


@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""

    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast


class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int], device_type: str, ptdtype: torch.dtype):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1  # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()
        self.device_type = device_type
        self.ptdtype = ptdtype

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert self.device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):
            if self.device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(self.device_type)  # e.g. cpu|mps
                print(f"Loading model on {self.device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            autocast_ctx = (
                torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
                if self.device_type == "cuda"
                else nullcontext()
            )

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400, detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message",
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed",
        )

    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'",
            )

    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}",
            )

    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")

    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}",
            )


async def generate_stream(
    worker: Worker,
    tokens,
    config: ChatWebConfig,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else config.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else config.max_tokens
    top_k = top_k if top_k is not None else config.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    accumulated_tokens = []
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1),
        ):
            token = token_column[0]

            if token == assistant_end or token == bos:
                break

            accumulated_tokens.append(token)
            current_text = worker.tokenizer.decode(accumulated_tokens)
            if not current_text.endswith("�"):
                new_text = current_text[len(last_clean_text) :]
                if new_text:
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


def build_chat_web_app(config: ChatWebConfig) -> FastAPI:
    """Construct the FastAPI app with the supplied configuration."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device_type = autodetect_device_type() if config.device_type == "" else config.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if config.dtype == "float32" else torch.bfloat16

    worker_pool = WorkerPool(config.num_gpus, device_type, ptdtype)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Loading nanochat models across GPUs...")
        await worker_pool.initialize(config.source, model_tag=config.model_tag, step=config.step)
        app.state.worker_pool = worker_pool
        print(f"Server ready at http://{config.host}:{config.port}")
        try:
            yield
        finally:
            # No explicit cleanup needed; workers share process lifetime.
            pass

    app = FastAPI(lifespan=lifespan)
    app.state.config = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        ui_html_path = os.path.join("nanochat", "ui.html")
        with open(ui_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace(
            "const API_URL = `http://${window.location.hostname}:8000`;", "const API_URL = '';"
        )
        return HTMLResponse(content=html_content)

    @app.get("/logo.svg")
    async def logo():
        logo_path = os.path.join("nanochat", "logo.svg")
        return FileResponse(logo_path, media_type="image/svg+xml")

    @app.post("/chat/completions")
    async def chat_completions(request: ChatRequest):
        validate_chat_request(request)

        logger.info("=" * 20)
        for i, message in enumerate(request.messages):
            logger.info(f"[{message.role.upper()}]: {message.content}")
        logger.info("-" * 20)

        worker = await app.state.worker_pool.acquire_worker()

        try:
            bos = worker.tokenizer.get_bos_token_id()
            user_start = worker.tokenizer.encode_special("<|user_start|>")
            user_end = worker.tokenizer.encode_special("<|user_end|>")
            assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
            assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

            conversation_tokens = [bos]
            for message in request.messages:
                if message.role == "user":
                    conversation_tokens.append(user_start)
                    conversation_tokens.extend(worker.tokenizer.encode(message.content))
                    conversation_tokens.append(user_end)
                elif message.role == "assistant":
                    conversation_tokens.append(assistant_start)
                    conversation_tokens.extend(worker.tokenizer.encode(message.content))
                    conversation_tokens.append(assistant_end)
            conversation_tokens.append(assistant_start)

            response_tokens = []

            async def stream_and_release():
                try:
                    async for chunk in generate_stream(
                        worker,
                        conversation_tokens,
                        config,
                        temperature=request.temperature,
                        max_new_tokens=request.max_tokens,
                        top_k=request.top_k,
                    ):
                        chunk_data = json.loads(chunk.replace("data: ", "").strip())
                        if "token" in chunk_data:
                            response_tokens.append(chunk_data["token"])
                        yield chunk
                finally:
                    full_response = "".join(response_tokens)
                    logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                    logger.info("=" * 20)
                    await app.state.worker_pool.release_worker(worker)

            return StreamingResponse(stream_and_release(), media_type="text/event-stream")
        except Exception:
            await app.state.worker_pool.release_worker(worker)
            raise

    @app.get("/health")
    async def health():
        worker_pool_state = getattr(app.state, "worker_pool", None)
        return {
            "status": "ok",
            "ready": worker_pool_state is not None and len(worker_pool_state.workers) > 0,
            "num_gpus": worker_pool_state.num_gpus if worker_pool_state else 0,
            "available_workers": worker_pool_state.available_workers.qsize() if worker_pool_state else 0,
        }

    @app.get("/stats")
    async def stats():
        worker_pool_state = app.state.worker_pool
        return {
            "total_workers": len(worker_pool_state.workers),
            "available_workers": worker_pool_state.available_workers.qsize(),
            "busy_workers": len(worker_pool_state.workers) - worker_pool_state.available_workers.qsize(),
            "workers": [
                {
                    "gpu_id": w.gpu_id,
                    "device": str(w.device),
                }
                for w in worker_pool_state.workers
            ],
        }

    return app


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    config = ChatWebConfig(
        num_gpus=args.num_gpus,
        source=args.source,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        model_tag=args.model_tag,
        step=args.step,
        dtype=args.dtype,
        device_type=args.device_type,
        host=args.host,
        port=args.port,
    )

    from uvicorn import run as uvicorn_run

    print("Starting NanoChat Web Server")
    print(f"Temperature: {config.temperature}, Top-k: {config.top_k}, Max tokens: {config.max_tokens}")
    uvicorn_run(build_chat_web_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
