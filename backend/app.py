import asyncio
import base64
import json
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import cv2
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FALLBACK_CAMERA_INDEXES = [int(x) for x in os.getenv("FALLBACK_CAMERA_INDEXES", "1").split(",") if x.strip()]
CHECK_INTERVAL_SECONDS = float(os.getenv("CHECK_INTERVAL_SECONDS", "5"))

app = FastAPI(title="VLM Person Alert")

status: Dict[str, object] = {
    "running": True,
    "last_check": None,
    "last_result": None,
    "error": None,
}

event_queue: Deque[Dict[str, object]] = deque(maxlen=200)
latest_jpeg: Optional[bytes] = None
frame_lock = threading.Lock()
prompt_lock = threading.Lock()

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict vision classifier. "
    "Answer ONLY valid JSON with fields: person_detected (boolean), confidence (0-1 number), summary (string). "
    "Detect if at least one person is visible in this image."
)
current_system_prompt = DEFAULT_SYSTEM_PROMPT


class PromptConfig(BaseModel):
    system_prompt: str


def push_event(event_type: str, payload: Dict[str, object]) -> None:
    event_queue.append(
        {
            "id": int(time.time() * 1000),
            "type": event_type,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "payload": payload,
        }
    )


def camera_capture_loop() -> None:
    global latest_jpeg
    indexes = [CAMERA_INDEX] + [idx for idx in FALLBACK_CAMERA_INDEXES if idx != CAMERA_INDEX]
    while status.get("running", True):
        cap = None
        chosen_index = None
        for idx in indexes:
            test_cap = cv2.VideoCapture(idx)
            if test_cap.isOpened():
                cap = test_cap
                chosen_index = idx
                break
            test_cap.release()

        if cap is None:
            status["error"] = f"Could not open camera indexes: {indexes}"
            push_event("error", {"message": status["error"]})
            time.sleep(2)
            continue

        status["camera_index"] = chosen_index
        status["error"] = None

        try:
            while status.get("running", True):
                ok, frame = cap.read()
                if not ok or frame is None:
                    status["error"] = f"Camera read failed on index {chosen_index}, retrying"
                    push_event("error", {"message": status["error"]})
                    break
                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                with frame_lock:
                    latest_jpeg = buffer.tobytes()
        finally:
            cap.release()
            time.sleep(0.2)


def mjpeg_frame_generator():
    while status.get("running", True):
        frame = None
        with frame_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.06)


def ask_vlm_person_present(jpeg_bytes: bytes) -> Dict[str, object]:
    with prompt_lock:
        prompt = current_system_prompt
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
            with frame_lock:
                jpeg = latest_jpeg
            if jpeg is None:
                status["error"] = "Waiting for first camera frame"
                time.sleep(0.2)
                continue
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
    camera_thread = threading.Thread(target=camera_capture_loop, daemon=True)
    camera_thread.start()
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


@app.get("/config/prompt")
def get_prompt_config() -> JSONResponse:
    with prompt_lock:
        return JSONResponse({"system_prompt": current_system_prompt})


@app.post("/config/prompt")
def set_prompt_config(config: PromptConfig) -> JSONResponse:
    global current_system_prompt
    new_prompt = (config.system_prompt or "").strip()
    if not new_prompt:
        return JSONResponse({"ok": False, "error": "system_prompt cannot be empty"}, status_code=400)
    with prompt_lock:
        current_system_prompt = new_prompt
    push_event("config", {"message": "System prompt updated"})
    return JSONResponse({"ok": True, "system_prompt": current_system_prompt})


@app.get("/camera")
def camera() -> StreamingResponse:
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/shutdown")
def shutdown() -> JSONResponse:
    status["running"] = False
    return JSONResponse({"ok": True})
