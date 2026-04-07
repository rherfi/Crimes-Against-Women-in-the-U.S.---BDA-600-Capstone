# VAWA Insights Bot (Backend) — FastAPI

This is a **local prototype** backend for the first “VAWA Insights Bot”.

## What it does (V1)
- Accepts chat questions at `POST /api/chat`
- Routes intent to:
  - **structured data tools** (for exact numbers), and/or
  - **document retrieval** (for policy + methodology context)
- Returns a structured JSON answer with **citations** and **debug info**

## Run locally

From the repo root:

```bash
cd "chatbot/vawa-insights/backend"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then test:

```bash
curl -s http://127.0.0.1:8000/health
```

