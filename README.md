# VLM Person Alert WebApp

MVP webapp for local vision alerts using a USB webcam + Ollama VLM.

## Current Features
- Captures frames from local webcam (`/dev/video0` by default)
- Sends frame to local Ollama model (`gemma3:4b` by default)
- Detects whether a person is present
- Streams detections to browser via SSE
- Browser notification on person detection

## Stack
- Backend: FastAPI + OpenCV + Ollama HTTP API
- Frontend: single-page HTML/JS dashboard

## Run
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

## Config
Environment variables:
- `OLLAMA_URL` (default `http://127.0.0.1:11434/api/generate`)
- `OLLAMA_MODEL` (default `gemma3:4b`)
- `CAMERA_INDEX` (default `0`)
- `CHECK_INTERVAL_SECONDS` (default `5`)

## Next Steps
- Add dedupe/cooldown for repeated alerts
- Add activity recognition labels
- Add webhook/Slack/Discord outbound notifications
- Add frame preview and model response debugging panel
