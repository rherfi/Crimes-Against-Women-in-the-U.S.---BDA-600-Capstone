# VAWA Insights Bot (V1 prototype)

This folder contains the **first chatbot only**: a local prototype that demonstrates:
- a simple React chat UI
- a FastAPI backend API
- structured tool functions for **exact numbers** from a sample dataset
- lightweight document retrieval (RAG-style) from a sample knowledge base
- answers returned in a structured format with **citations** and **debug intent**

## Folder layout
- `backend/` — FastAPI app + sample CSV + sample knowledge base docs
- `frontend/` — React UI (Vite)

## Run it locally (2 terminals)

### 1) Backend

```bash
cd "chatbot/vawa-insights/backend"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Quick check:

```bash
curl -s http://127.0.0.1:8000/health
```

### 2) Frontend

```bash
cd "chatbot/vawa-insights/frontend"
npm install
npm run dev
```

Open:
- `http://127.0.0.1:5173`

## Example questions to try
- Compare California and Texas in firearm involvement from 2021 to 2024.
- Which states had the highest dv_rate in 2024?
- What does VAWA 2022 change about dating partners?
- Give me a risk profile for New Mexico in 2024.
- Did reporting increase after 2022, and what are the caveats?

## Notes (V1)
- The structured tools read from `backend/app/data/metrics_sample.csv` and `backend/app/data/risk_components_sample.csv`.
- The knowledge base docs live in `backend/app/kb/` and are retrieved using a simple keyword-overlap retriever (easy to upgrade later).
- This prototype prioritizes **no hallucinated numbers**: if the data tool can’t find rows, the bot will say so.

