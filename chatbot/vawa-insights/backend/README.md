# VAWA Insights Bot (Backend) — FastAPI

This folder contains the backend API for the VAWA Insights Bot.

## What it does
- Accepts chat questions at `POST /api/chat`
- Routes intent to:
  - **structured data tools** (for exact numbers), and/or
  - **document retrieval** (for policy + methodology context)
- Provides victim resources search at `POST /api/resources/search` and via chat phrases like “find a shelter near …”
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

## Victim resources data
- The victim resources dataset lives at `backend/app/data/resources.csv`.
- Add rows for shelters, clinics, legal aid, etc. Include `latitude`/`longitude` when possible for best “near me” accuracy.
- If coordinates are missing, the API can still return national resources; location-based ranking requires coordinates.

