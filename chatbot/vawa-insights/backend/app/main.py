"""
FastAPI entrypoint for the VAWA Insights Bot.

Design goals:
- Keep dependencies minimal (no vector DB, no heavy LLM framework)
- Never hallucinate numeric values: numbers come only from structured tools
- Always return citations
- Keep code modular so we can upgrade retrieval later
"""

import asyncio
import json
import os
from functools import partial

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, ChatResponse
from app.logic import answer_chat


app = FastAPI(title="VAWA Insights Bot API", version="0.1.0")

# CORS: local dev origins + optional production origins from CORS_ORIGINS (comma-separated).
# Example: CORS_ORIGINS=https://your-frontend.vercel.app
_cors_extra = [o.strip() for o in (os.environ.get("CORS_ORIGINS") or "").split(",") if o.strip()]
_cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    *_cors_extra,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return answer_chat(req)


def _sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Server-Sent Events: `start` (immediate), then `delta` chunks for direct_answer,
    then `done` with full answer + debug. Computation runs once; deltas improve UX for long replies.
    """

    async def events():
        yield _sse_data({"type": "start"})
        loop = asyncio.get_event_loop()
        try:
            result: ChatResponse = await loop.run_in_executor(None, partial(answer_chat, req))
            text = result.answer.direct_answer or ""
            step = 24
            for i in range(0, len(text), step):
                yield _sse_data({"type": "delta", "text": text[i : i + step]})
                await asyncio.sleep(0)
            yield _sse_data(
                {
                    "type": "done",
                    "answer": result.answer.model_dump(),
                    "debug": result.debug.model_dump(),
                }
            )
        except Exception as e:
            yield _sse_data({"type": "error", "message": str(e)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

