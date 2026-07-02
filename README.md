<img align="center" width="100" height="100" alt="b8dbe6bb-5dbd-4f05-af62-604aba46c642-removebg-preview" src="https://github.com/user-attachments/assets/9341767a-cf43-4736-a5f2-80c7488682de" />

# CapCam — AI-Powered Security Surveillance System

CapCam is a real-time weapon detection system that combines a **YOLO-based computer vision backend** with a **web-based frontend dashboard** for live camera surveillance, threat alerting, and analytics.

---

## 🚀 Features

- 🔫 **Real-time weapon detection** (AKM, M4, PKS, RPG, Sniper, Pistol, Knife)
- 📹 **Live MJPEG video stream** with bounding box overlays
- 📊 **Analytics dashboard** — detection timeline & heatmap
- 📸 **Evidence capture** — auto-saves snapshots on weapon detection
- 🔔 **Event log** — timestamped alert cards with class & confidence
- ⚡ **Multi-scale inference** + temporal smoothing for accuracy
- 🔐 **Login-protected frontend**

---

## 🗂️ Project Structure

```
Capcam Main Project/
├── frontend/
│   ├── index.html       # Main dashboard UI
│   ├── login.html       # Login page
│   ├── app.js           # Frontend logic
│   ├── styles.css       # Styling
│   └── server.py        # FastAPI backend server
├── CapCam Backend/
│   └── archive/
│       ├── train_model.py    # YOLO training script
│       ├── data.yaml         # YOLO dataset config
│       ├── yolo11s.pt        # Base YOLO model weights
│       └── yolo26n.pt        # Additional model weights
├── requirements.txt
└── README.md
└── best.pt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/capcam.git
cd capcam
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the model path

Open `frontend/server.py` and update the `MODEL_PATH` variable to point to your trained YOLO model weights (e.g., `best.pt`):

```python
MODEL_PATH = r"path\to\your\best.pt"
```

> **Note:** The trained model weights (`best.pt`) are added recently in this repo due to file size. Train your own using the provided `train_model.py` or use provided (`best.pt`) file

### 5. Run the backend server
```bash
python frontend/server.py
```

The server starts at **http://localhost:8000**

---

## 🧠 Training Your Own Model

Use the provided training script and dataset config:

```bash
cd "CapCam Backend/archive"
python train_model.py
```

Ensure your dataset folders (`train/`, `valid/`, `test/`) and `data.yaml` are configured correctly.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Service health & model info |
| GET | `/api/v1/stream` | Live MJPEG video stream |
| GET | `/api/v1/detections` | Latest detection results |
| GET | `/api/v1/events` | Weapon alert event log |
| GET | `/api/v1/analytics/timeline` | Detection counts over time |
| GET | `/api/v1/analytics/heatmap` | Spatial heatmap data |
| GET | `/api/v1/evidence` | Evidence snapshot list |
| POST | `/api/v1/inference/unattended` | Single-frame inference (base64) |
| POST | `/api/v1/inference/toggle` | Enable/disable inference |

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **AI / CV:** Ultralytics YOLO, OpenCV, NumPy
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. Do not deploy in production without proper security hardening, authentication, and legal compliance review.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
