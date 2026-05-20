import asyncio
import base64
import glob
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
last_camera_error: Optional[str] = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a strict vision classifier. "
    "Answer ONLY valid JSON with fields: person_detected (boolean), confidence (0-1 number), summary (string). "
    "Detect if at least one person is visible in this image."
)
current_system_prompt = DEFAULT_SYSTEM_PROMPT


class PromptConfig(BaseModel):
    system_prompt: str


def normalize_model_response(raw_response: str) -> Dict[str, object]:
    raw = (raw_response or "").strip()
    if not raw:
        return {
            "person_detected": False,
            "confidence": 0.0,
            "summary": "",
            "count": 0,
            "raw_response": raw_response,
        }

    parsed_obj = None
    try:
        parsed_obj = json.loads(raw)
    except Exception:
        # Try to recover a JSON object/array from mixed text output.
        start_obj = raw.find("{")
        end_obj = raw.rfind("}")
        start_arr = raw.find("[")
        end_arr = raw.rfind("]")
        candidate = None
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            candidate = raw[start_obj : end_obj + 1]
        elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            candidate = raw[start_arr : end_arr + 1]

        if candidate:
            try:
                parsed_obj = json.loads(candidate)
            except Exception:
                parsed_obj = raw
        else:
            parsed_obj = raw

    # If the model returns a bare numeric JSON value or plain numeric text.
    if isinstance(parsed_obj, int) or (isinstance(parsed_obj, str) and parsed_obj.strip().isdigit()):
        count = int(parsed_obj)
        return {
            "person_detected": count > 0,
            "confidence": 1.0 if count > 0 else 0.0,
            "summary": f"Detected {count} people",
            "count": count,
            "raw_response": raw_response,
        }

    if isinstance(parsed_obj, dict):
        # Preferred explicit schema.
        if "person_detected" in parsed_obj:
            detected = bool(parsed_obj.get("person_detected", False))
            confidence = float(parsed_obj.get("confidence", 1.0 if detected else 0.0))
            summary = str(parsed_obj.get("summary", "") or "")
            count_val = parsed_obj.get("count")
            count = int(count_val) if isinstance(count_val, (int, float)) else (1 if detected else 0)
            return {
                "person_detected": detected,
                "confidence": confidence,
                "summary": summary,
                "count": count,
                "raw_response": raw_response,
            }

        # Count-based schema.
        if "count" in parsed_obj:
            try:
                count = int(parsed_obj.get("count", 0))
            except Exception:
                count = 0
            return {
                "person_detected": count > 0,
                "confidence": 1.0 if count > 0 else 0.0,
                "summary": str(parsed_obj.get("summary", f"Detected {count} people")),
                "count": count,
                "raw_response": raw_response,
            }

        # Fallback for any unknown JSON object.
        return {
            "person_detected": False,
            "confidence": 0.0,
            "summary": str(parsed_obj.get("summary", "")),
            "count": 0,
            "raw_response": raw_response,
        }

    return {
        "person_detected": False,
        "confidence": 0.0,
        "summary": str(parsed_obj),
        "count": 0,
        "raw_response": raw_response,
    }


def push_event(event_type: str, payload: Dict[str, object]) -> None:
    event_queue.append(
        {
            "id": int(time.time() * 1000),
            "type": event_type,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "payload": payload,
        }
    )


def discover_camera_indexes() -> List[int]:
    discovered = []
    for path in sorted(glob.glob("/dev/video*")):
        suffix = path.replace("/dev/video", "")
        if suffix.isdigit():
            discovered.append(int(suffix))
    preferred = [CAMERA_INDEX] + [idx for idx in FALLBACK_CAMERA_INDEXES if idx != CAMERA_INDEX]
    ordered = preferred + [idx for idx in discovered if idx not in preferred]
    return ordered if ordered else preferred


def camera_capture_loop() -> None:
    global latest_jpeg, last_camera_error
    while status.get("running", True):
        indexes = discover_camera_indexes()
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
            msg = f"Could not open camera indexes: {indexes}"
            status["error"] = msg
            if last_camera_error != msg:
                push_event("error", {"message": msg})
                last_camera_error = msg
            time.sleep(2)
            continue

        status["camera_index"] = chosen_index
        status["error"] = None
        last_camera_error = None

        try:
            while status.get("running", True):
                ok, frame = cap.read()
                if not ok or frame is None:
                    msg = f"Camera read failed on index {chosen_index}, retrying"
                    status["error"] = msg
                    if last_camera_error != msg:
                        push_event("error", {"message": msg})
                        last_camera_error = msg
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
    return normalize_model_response(raw_response)


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
