"""
SentinelAI Backend — FastAPI + Ultralytics YOLO
================================================
Serves your trained YOLO model for unattended object detection.

Usage:
  1. Place your trained weights (best.pt) in the same folder, or set MODEL_PATH env var.
  2. pip install fastapi uvicorn ultralytics opencv-python
  3. python server.py

Endpoints:
  GET  /api/v1/health              → service health + model classes
  GET  /api/v1/stream              → MJPEG video stream (webcam or RTSP)
  GET  /api/v1/detections          → latest detections JSON
  GET  /api/v1/events              → event log for alert cards
  GET  /api/v1/analytics/timeline  → detection counts per 10s bucket (last 5 min)
  GET  /api/v1/analytics/heatmap   → accumulated heatmap points
  GET  /api/v1/evidence            → evidence snapshots with metadata
  POST /api/v1/inference/unattended→ single-frame inference (base64 image)
  POST /api/v1/inference/toggle    → enable/disable inference
"""

import os, time, json, threading, base64, io
from datetime import datetime, timezone
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Config ──────────────────────────────────────────────
MODEL_PATH = r"C:\Users\Piyush\OneDrive\Desktop\Capcam Main Project\CapCam Backend\archive\runs\capcam_yolo11s\weights\best.pt"
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))  # baseline confidence
INFERENCE_SIZE = int(os.getenv("INFERENCE_SIZE", "1280"))
NMS_IOU_THRESHOLD = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Classes to detect — pose classes excluded (not relevant for weapon detection)
SKIP_CLASSES = {"DANGER_POSE", "NOT_DANGER_POSE"}

# Per-class confidence thresholds
CLASS_CONF_THRESHOLDS = {
    "AKM":    0.40,
    "KNIFE":  0.35,
    "pistol": 0.35,
    "M4":     0.40,
    "PKS":    0.40,
    "RPG":    0.40,
    "SNIPER": 0.40,
}
MULTI_SCALE_SIZES = [640, 1280]
LIVE_INFERENCE_SIZE = 640

try:
    CAMERA_SOURCE = int(CAMERA_SOURCE)
except ValueError:
    pass

# ── Load YOLO Model ────────────────────────────────────
from ultralytics import YOLO

print(f"[CapCam] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"[CapCam] Model loaded. Classes: {model.names}")
print(f"[CapCam] Per-class thresholds: {CLASS_CONF_THRESHOLDS}")
print(f"[CapCam] Multi-scale inference: {MULTI_SCALE_SIZES}")

# ── App Setup ───────────────────────────────────────────
app = FastAPI(title="CapCam Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Shared State ────────────────────────────────────────
latest_detections = []
latest_frame = None
frame_lock = threading.Lock()
events_log = deque(maxlen=100)
event_counter = 0
inference_enabled = True
fps_value = 0
start_time = time.time()

# Timeline: list of {timestamp, count} per 10s bucket
timeline_data = deque(maxlen=30)  # last 5 min in 10s buckets
timeline_bucket_start = time.time()
timeline_bucket_count = 0

# Heatmap: accumulated detection center points
heatmap_points = deque(maxlen=500)

# Evidence: auto-captured frames on detection
evidence_log = deque(maxlen=20)
evidence_counter = 0
last_evidence_time = 0

# Temporal smoothing: track recent detections to boost confidence
recent_detections_history = deque(maxlen=5)  # last 5 frames

# ── Preprocessing: CLAHE for contrast enhancement ──────
def enhance_frame(frame):
    """Apply CLAHE contrast enhancement to improve weapon visibility."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / max(a1 + a2 - inter, 1e-6)

def multi_scale_detect(frame, scales=None, use_augment=False):
    """Run inference at given scales and merge results."""
    if scales is None:
        scales = MULTI_SCALE_SIZES
    all_dets = []
    for sz in scales:
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=NMS_IOU_THRESHOLD,
                        imgsz=sz, verbose=False, augment=use_augment)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                # Skip excluded classes
                if cls_name in SKIP_CLASSES:
                    continue
                # Apply per-class threshold
                min_conf = CLASS_CONF_THRESHOLDS.get(cls_name, CONFIDENCE_THRESHOLD)
                if conf >= min_conf:
                    all_dets.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "confidence": conf, "class": cls_name, "class_id": cls_id
                    })
    # NMS merge across scales: keep highest confidence for overlapping boxes
    merged = []
    used = [False] * len(all_dets)
    all_dets.sort(key=lambda d: d["confidence"], reverse=True)
    for i, d in enumerate(all_dets):
        if used[i]: continue
        best = d
        for j in range(i+1, len(all_dets)):
            if used[j]: continue
            if all_dets[j]["class"] == d["class"]:
                iou = compute_iou(
                    [d["x1"],d["y1"],d["x2"],d["y2"]],
                    [all_dets[j]["x1"],all_dets[j]["y1"],all_dets[j]["x2"],all_dets[j]["y2"]]
                )
                if iou > 0.4:
                    used[j] = True
                    # Average the confidence from both scales
                    best["confidence"] = max(best["confidence"], all_dets[j]["confidence"])
        merged.append(best)
    return merged

def temporal_boost(detections):
    """Boost confidence if similar detection appeared in recent frames."""
    for det in detections:
        for prev_dets in recent_detections_history:
            for pd in prev_dets:
                if pd["class"] == det["class"]:
                    iou = compute_iou(
                        [det["x1"],det["y1"],det["x2"],det["y2"]],
                        [pd["x1"],pd["y1"],pd["x2"],pd["y2"]]
                    )
                    if iou > 0.3:
                        det["confidence"] = min(det["confidence"] * 1.1, 0.99)
                        break
    return detections

# ── Background Camera + Inference Loop ──────────────────
def camera_loop():
    global latest_detections, latest_frame, fps_value, event_counter
    global inference_enabled, timeline_bucket_start, timeline_bucket_count
    global evidence_counter, last_evidence_time

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[CapCam] ERROR: Cannot open camera source: {CAMERA_SOURCE}")
        return

    print(f"[CapCam] Camera opened: {CAMERA_SOURCE}")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        now = time.time()
        fps_value = round(1.0 / max(now - prev_time, 0.001))
        prev_time = now

        detections = []
        if inference_enabled:
            # Skip CLAHE on live stream to reduce lag (only used in deep scan API)
            # Single-scale inference for speed (multi-scale is too slow for live)
            raw_dets = multi_scale_detect(frame, scales=[LIVE_INFERENCE_SIZE], use_augment=False)
            # Temporal smoothing boost
            raw_dets = temporal_boost(raw_dets)
            recent_detections_history.append(raw_dets)
            for d in raw_dets:
                x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                conf = d["confidence"]
                cls_name = d["class"]
                det = {
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                    "confidence": round(conf, 3),
                    "class": cls_name, "class_id": d["class_id"]
                }
                detections.append(det)

                # Heatmap: add center point
                cx = round((x1 + x2) / 2)
                cy = round((y1 + y2) / 2)
                heatmap_points.append({"x": cx, "y": cy, "t": now})

                # Trigger event for ALL weapon classes
                WEAPON_CLASSES = {"AKM", "KNIFE", "pistol", "M4", "PKS", "RPG", "SNIPER"}
                if cls_name in WEAPON_CLASSES:
                    event_counter += 1
                    event = {
                        "id": f"EVT-{datetime.now(timezone.utc).strftime('%Y-%m')}-{event_counter:04d}",
                        "type": "Weapon Detected", "class": cls_name,
                        "confidence": conf, "camera": "CAM-01", "zone": "Zone C",
                        "status": "Active",
                        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                        "bbox": [round(x1), round(y1), round(x2), round(y2)]
                    }
                    events_log.appendleft(event)

                    # Evidence: auto-capture (max once per 10s)
                    if now - last_evidence_time > 10:
                        last_evidence_time = now
                        evidence_counter += 1
                        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        b64 = base64.b64encode(jpg.tobytes()).decode('utf-8')
                        evidence_log.appendleft({
                            "id": f"EV-{evidence_counter:04d}",
                            "event_id": event["id"],
                            "class": cls_name, "confidence": conf,
                            "timestamp": event["timestamp"],
                            "thumbnail": b64[:200] + "...",
                            "image_b64": b64
                        })

            # Timeline bucket
            timeline_bucket_count += len(detections)
            if now - timeline_bucket_start >= 10:
                timeline_data.append({
                    "t": datetime.now(timezone.utc).isoformat() + "Z",
                    "count": timeline_bucket_count
                })
                timeline_bucket_count = 0
                timeline_bucket_start = now

        with frame_lock:
            latest_detections = detections
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            latest_frame = buf.tobytes()

        time.sleep(0.01)  # reduced sleep for smoother camera feed

camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()

# ── Endpoints ───────────────────────────────────────────
@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy", "model": MODEL_PATH,
        "model_classes": model.names,
        "camera_source": str(CAMERA_SOURCE),
        "inference_enabled": inference_enabled,
        "uptime_seconds": round(time.time() - start_time),
        "fps": fps_value,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    }

@app.get("/api/v1/stream")
def video_stream():
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/v1/detections")
def get_detections():
    return {"detections": latest_detections, "inference_enabled": inference_enabled,
            "fps": fps_value, "timestamp": datetime.now(timezone.utc).isoformat() + "Z"}

@app.get("/api/v1/events")
def get_events():
    return {"events": list(events_log), "total": len(events_log)}

@app.get("/api/v1/analytics/timeline")
def get_timeline():
    return {"buckets": list(timeline_data), "bucket_seconds": 10}

@app.get("/api/v1/analytics/heatmap")
def get_heatmap():
    cutoff = time.time() - 300  # last 5 min
    points = [p for p in heatmap_points if p["t"] > cutoff]
    return {"points": [{"x": p["x"], "y": p["y"]} for p in points]}

@app.get("/api/v1/evidence")
def get_evidence():
    # Return list without full image data
    items = []
    for e in evidence_log:
        items.append({k: v for k, v in e.items() if k != "image_b64"})
    return {"evidence": items, "total": len(evidence_log)}

@app.get("/api/v1/evidence/{evidence_id}")
def get_evidence_image(evidence_id: str):
    for e in evidence_log:
        if e["id"] == evidence_id:
            img_bytes = base64.b64decode(e["image_b64"])
            return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/api/v1/inference/unattended")
async def infer_single_frame(request: Request):
    body = await request.json()
    img_b64 = body.get("image", "")
    if not img_b64:
        return JSONResponse({"error": "missing 'image' field"}, status_code=400)
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    enhanced = enhance_frame(frame)
    dets = multi_scale_detect(enhanced)
    result_dets = []
    for d in dets:
        result_dets.append({"x1": round(d["x1"]), "y1": round(d["y1"]),
                            "x2": round(d["x2"]), "y2": round(d["y2"]),
                            "confidence": round(d["confidence"], 3),
                            "class": d["class"], "class_id": d["class_id"]})
    return {"detections": result_dets, "timestamp": datetime.now(timezone.utc).isoformat() + "Z"}

@app.post("/api/v1/inference/toggle")
async def toggle_inference():
    global inference_enabled
    inference_enabled = not inference_enabled
    return {"inference_enabled": inference_enabled}

# ── Serve Frontend ──────────────────────────────────────
from fastapi.responses import RedirectResponse

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
def root():
    return RedirectResponse(url="/index.html")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    print("[SentinelAI] Starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
