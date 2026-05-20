import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, List

import cv2
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CHECK_INTERVAL_SECONDS = float(os.getenv("CHECK_INTERVAL_SECONDS", "5"))

app = FastAPI(title="VLM Person Alert")

status: Dict[str, object] = {
    "running": True,
    "last_check": None,
    "last_result": None,
    "error": None,
}

event_queue: Deque[Dict[str, object]] = deque(maxlen=200)


def push_event(event_type: str, payload: Dict[str, object]) -> None:
    event_queue.append(
        {
            "id": int(time.time() * 1000),
            "type": event_type,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "payload": payload,
        }
    )


def capture_frame(camera_index: int) -> bytes:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from camera")

    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buffer.tobytes()


def ask_vlm_person_present(jpeg_bytes: bytes) -> Dict[str, object]:
    prompt = (
        "You are a strict vision classifier. "
        "Answer ONLY valid JSON with fields: person_detected (boolean), confidence (0-1 number), summary (string). "
        "Detect if at least one person is visible in this image."
    )
    image_b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    raw_response = data.get("response", "{}")
    parsed = json.loads(raw_response)

    return {
        "person_detected": bool(parsed.get("person_detected", False)),
        "confidence": float(parsed.get("confidence", 0.0)),
        "summary": str(parsed.get("summary", "")),
    }


def detection_loop() -> None:
    while status.get("running", True):
        try:
            jpeg = capture_frame(CAMERA_INDEX)
            result = ask_vlm_person_present(jpeg)
            status["last_check"] = time.strftime("%Y-%m-%d %H:%M:%S")
            status["last_result"] = result
            status["error"] = None
            push_event("detection", result)

            if result.get("person_detected"):
                push_event("alert", result)
        except Exception as exc:  # pragma: no cover
            status["error"] = str(exc)
            push_event("error", {"message": str(exc)})

        time.sleep(CHECK_INTERVAL_SECONDS)


@app.on_event("startup")
async def startup_event() -> None:
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/status")
def get_status() -> JSONResponse:
    return JSONResponse(status)


@app.get("/events")
async def events() -> StreamingResponse:
    async def event_generator() -> List[str]:
        last_seen = 0
        while True:
            while event_queue and event_queue[-1]["id"] > last_seen:
                # Send only unseen events in order.
                unseen = [e for e in event_queue if e["id"] > last_seen]
                for evt in unseen:
                    last_seen = evt["id"]
                    yield f"event: {evt['type']}\n"
                    yield f"data: {json.dumps(evt)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/shutdown")
def shutdown() -> JSONResponse:
    status["running"] = False
    return JSONResponse({"ok": True})
