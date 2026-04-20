"""
FastAPI entrypoint for the Victim Tool Bot (V1 prototype).

Design goals for V1:
- Keep dependencies minimal
- Use deterministic, transparent logic
- Never fabricate local resources/addresses/phone numbers
- Provide trauma-informed wording and clear caveats
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.logic import answer_chat


app = FastAPI(title="Victim Tool Bot API", version="0.1.0")

# Local dev convenience: allow the React dev server to call the API.
# For production, lock this down.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Victim Tool frontend dev server (Vite)
        "http://localhost:5174",
        "http://127.0.0.1:5174",

        # Extra local dev ports (optional)
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

