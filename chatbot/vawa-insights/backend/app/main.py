"""
FastAPI entrypoint for the VAWA Insights Bot (V1 prototype).

Design goals for V1:
- Keep dependencies minimal (no vector DB, no heavy LLM framework)
- Never hallucinate numeric values: numbers come only from structured tools
- Always return citations
- Keep code modular so we can swap in stronger RAG later
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.logic import answer_chat


app = FastAPI(title="VAWA Insights Bot API", version="0.1.0")

# Local dev convenience: allow the React dev server to call the API.
# For production, lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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

