# REAR GUARD — Collision Avoidance System

A real-time obstacle detection and collision warning system that uses **YOLOv8** for object detection and distance estimation, with a **React Native** mobile app as the frontend and a **Flask** backend for processing.



## Architecture

```
[Android Phone (React Native)]  --WiFi/USB-->  [Laptop (Flask + YOLOv8)]
      Capture photo/video                          Detect objects & estimate distance
      Display zones & alerts                       Return results via REST API
```

### Backend (Python / Flask) — `backend/`

- **`app.py`** — Flask server exposing REST endpoints for image/video processing
- **`detection.py`** — YOLOv8-based object detection (car, truck, bus, person, etc.)
- **`distance.py`** — Distance estimation using bounding box size and focal length, with smoothing
- **`main.py`** — Local webcam test (no Flask needed)
- **`Calibrate.py`** — Camera focal length calibration tool

### Frontend (React Native) — `CollisionApp/`

- Android app with three modes:
  - **Photo** — Capture or upload an image for instant analysis
  - **Video** — Record or pick a video for frame-by-frame analysis
  - **Live** — Real-time camera feed with continuous detection (~2-3 fps)

### Results — `results/`

- Screenshots of detection zones (safe, warning, danger)

### Detection Zones

| Zone | Distance | Meaning |
|------|----------|---------|
| Danger | < 3m | Stop immediately |
| Warning | 3–8m | Slow down |
| Safe | > 8m | Clear road |

## Setup

### Prerequisites

- Python 3.10+
- Node.js 16+
- Android Studio + Android SDK (for running the mobile app)
- USB debugging enabled on your Android phone

### 1. Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv ../venv
source ../venv/bin/activate

# Install dependencies
pip install flask flask-cors opencv-python numpy ultralytics

# Start the server
python app.py
```

The backend runs on `http://0.0.0.0:5001`.

### 2. Frontend Setup

```bash
cd CollisionApp

# Install dependencies
npm install

# Start Metro bundler
npx react-native start --reset-cache
```

### 3. Connect Your Phone

Connect your Android phone via USB with USB debugging enabled.

```bash
# Set up port forwarding
adb reverse tcp:8081 tcp:8081
adb reverse tcp:5001 tcp:5001

# Build and install the app
npx react-native run-android
```

## When Your IP Changes

If your laptop reconnects to WiFi or changes network, follow these steps:

### Step 1 — Find your laptop's current IP

```bash
ipconfig getifaddr en0
```

### Step 2 — Update the server URL in `CollisionApp/App.tsx` (lines 24–25)

Replace the old IP with the new one in lines 24–25:

```typescript
const SERVER_URL        = 'http://<NEW_IP>:5001/process';
const VIDEO_SUMMARY_URL = 'http://<NEW_IP>:5001/process_video_summary';
```

### Step 3 — Start the backend

```bash
cd backend
source ../venv/bin/activate
python app.py
```

### Step 4 — Start Metro (if not already running)

```bash
cd CollisionApp
npx react-native start --reset-cache
```

### Step 5 — Set up adb reverse

```bash
adb reverse tcp:8081 tcp:8081
adb reverse tcp:5001 tcp:5001
```

### Step 6 — Update the dev server on your phone

1. Open the React Native dev menu (shake the phone, or run `adb shell input keyevent 82`)
2. Tap **Settings**
3. Tap **Debug server host & port for device**
4. Enter `<YOUR_NEW_IP>:8081`
5. Press **OK**
6. Go back and tap **Reload**

### Quick check — make sure both devices are on the same network

```bash
# Laptop IP
ipconfig getifaddr en0

# Phone IPs
adb shell ip addr | grep "inet "
```

Both should share the same subnet (e.g. both `192.168.100.x` or both `10.48.19.x`).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/process` | Detect obstacles in a single image (`multipart/form-data` with `image` field) |
| POST | `/process_video_summary` | Process a video and return frame-by-frame results (`multipart/form-data` with `video` field) |

## Project Structure

```
collision_avoidance/
├── backend/                # Python Flask backend
│   ├── app.py              # Flask server
│   ├── detection.py        # YOLOv8 object detection
│   ├── distance.py         # Distance estimation & smoothing
│   ├── main.py             # Local webcam test
│   ├── Calibrate.py        # Camera calibration tool
│   ├── requirements.txt    # Python dependencies
│   └── yolov8n.pt          # YOLOv8 nano model weights
├── CollisionApp/           # React Native frontend
│   ├── App.tsx             # Main app component
│   ├── package.json
│   └── android/            # Android native project
├── results/                # Detection result screenshots
│   ├── safe_zone.jpeg
│   ├── warning_zone.jpeg
│   └── danger_zone.jpeg
├── buildozer.spec          # Buildozer config (Android APK)
├── venv/                   # Python virtual environment (gitignored)
├── .gitignore
└── README.md
```
