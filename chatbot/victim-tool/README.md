# Victim Tool Bot (V1 prototype)

This is a **victim-facing support + resource + plain-language VAWA info** chatbot prototype.

## Safety notice (read first)

- This bot is **not** a therapist, lawyer, doctor, or emergency responder.
- If you are in immediate danger, call **911 (U.S.)** or your local emergency number.
- The backend **will not invent local resources**. It only returns resources present in its dataset.
- In this prototype, some “local” resources are clearly labeled **DEMO DATA** and are **not verified**.

## Folder layout

- `backend/` — FastAPI app + sample resource dataset + small markdown knowledge base
- `frontend/` — React UI (Vite)

## Run it locally (2 terminals)

### 1) Backend (FastAPI)

Important: the `cd` paths below assume you run them from the **repo root**.
If your terminal prompt already ends in `.../chatbot %`, see the troubleshooting section at the bottom.

```bash
cd "chatbot/victim-tool/backend"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Quick check:

```bash
curl -s http://127.0.0.1:8001/health
```

### 2) Frontend (React)

```bash
cd "chatbot/victim-tool/frontend"
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5174`

## Example questions to try

- “I need help near Albuquerque.”
- “I live in Arizona and need domestic violence shelter options.”
- “Are there tribal services near me? I’m in Apache County, AZ.”
- “What protections did VAWA add for dating partners?”
- “I am scared and I do not know what to do.”

## Data & knowledge base (V1)

- **Resources dataset**: `backend/app/data/resources_sample.json`
  - Includes a few **national hotlines** and a few **DEMO-only** local placeholder entries.
  - Replace this file later with verified resources.
- **Knowledge base docs**: `backend/app/kb/*.md`
  - Small plain-language documents with frontmatter metadata.

## Troubleshooting (common issues)

### If `cd "chatbot/victim-tool/backend"` says “no such file or directory”

You are probably already inside the `chatbot/` folder.

From a prompt like:

- `.../Crimes-Against-Women.../chatbot %`

Use these instead:

```bash
cd "victim-tool/backend"
```

And for the frontend:

```bash
cd "victim-tool/frontend"
```

### If `pip` or `uvicorn` says it’s trying to run from `victim-tool-bot/...`

That means the Python virtual environment (`.venv`) was created before you renamed/moved folders, and its scripts still point to the old path.

Fix by recreating the venv:

```bash
cd "chatbot/victim-tool/backend"
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### If `uvicorn` says “Address already in use”

Another process is already using that port.

Option A (quick): use a different port:

```bash
uvicorn app.main:app --reload --port 8010
```

Then tell the frontend which port to use by creating `frontend/.env`:

```bash
cd "chatbot/victim-tool/frontend"
echo 'VITE_API_BASE=http://127.0.0.1:8010' > .env
npm run dev
```

Option B (clean): find and stop the process using the port:

```bash
lsof -nP -iTCP:8001 -sTCP:LISTEN
kill <PID>
```

