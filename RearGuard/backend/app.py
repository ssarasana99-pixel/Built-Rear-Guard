# Roll Numbers: 23L-0549, 23L-0736, 23L-0848

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import tempfile, os, json

from detection import detect_obstacles
from distance  import estimate_distance, get_zone, SmoothingFilter, DEFAULT_FOCAL_LENGTH_PX

app = Flask(__name__)
CORS(app) 
FOCAL_LENGTH_PX = DEFAULT_FOCAL_LENGTH_PX


def _process_frame(frame, smoothers: dict):
    """Run detection + distance estimation on one BGR frame."""
    frame_h, frame_w = frame.shape[:2]
    obstacles        = detect_obstacles(frame)

    results          = []
    nearest_distance = float("inf")

    for i, det in enumerate(obstacles):
        x, y, w, h, label, conf = det

        # Hybrid distance: area-ratio heuristic (70%) + focal-length formula (30%)
        raw_dist = estimate_distance(h, w, h, frame_w, frame_h, label, FOCAL_LENGTH_PX)

        if i not in smoothers:
            smoothers[i] = SmoothingFilter(window=5)
        distance = smoothers[i].update(raw_dist)

        area_ratio = round((w * h) / (frame_w * frame_h), 3)

        results.append({
            "label":      label,
            "confidence": conf,
            "distance":   round(distance, 2),
            "box":        [x, y, w, h],
            "area_ratio": area_ratio,
        })

        if distance < nearest_distance:
            nearest_distance = distance

    # Purge stale smoothers for disappeared objects
    for key in list(smoothers.keys()):
        if key >= len(obstacles):
            del smoothers[key]

    if not results:
        return {"distance": None, "zone": "no object", "objects": []}

    zone = get_zone(nearest_distance)
    return {"distance": nearest_distance, "zone": zone, "objects": results}


# routes

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# single image
@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400

    data  = np.frombuffer(request.files["image"].read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    frame_h, frame_w = frame.shape[:2]
    print(f"\n[/process] Image {frame_w}x{frame_h}")

    payload = _process_frame(frame, {})
    print(f"  → {payload}")
    return jsonify(payload)


# video streaming
@app.route("/process_video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"error": "No video field"}), 400

    frame_step = int(request.args.get("frame_step", 5))
    max_frames = int(request.args.get("max_frames", 300))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        request.files["video"].save(tmp.name)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30
        total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"\n[/process_video] {vid_w}x{vid_h}  fps={fps:.1f}  frames={total_vid}")

        def generate():
            smoothers = {}
            frame_idx = 0
            sent      = 0

            while sent < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp = round(frame_idx / fps, 3)
                    payload   = _process_frame(frame, smoothers)
                    payload.update({"frame_index": frame_idx, "timestamp": timestamp})
                    yield json.dumps(payload) + "\n"
                    sent += 1

                frame_idx += 1

            cap.release()
            yield json.dumps({"done": True, "total_frames": frame_idx}) + "\n"

        return Response(generate(), mimetype="application/x-ndjson")

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# Video summary
@app.route("/process_video_summary", methods=["POST"])
def process_video_summary():
    if "video" not in request.files:
        return jsonify({"error": "No video field"}), 400

    frame_step = int(request.args.get("frame_step", 5))
    max_frames = int(request.args.get("max_frames", 200))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        request.files["video"].save(tmp.name)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps       = cap.get(cv2.CAP_PROP_FPS) or 30
        vid_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n[/process_video_summary] {vid_w}x{vid_h}  fps={fps:.1f}  frames={total_vid}")

        smoothers    = {}
        frames       = []
        frame_idx    = 0
        sent         = 0

        ZONE_RANK    = {"no object": 0, "safe": 1, "warning": 2, "danger": 3}
        worst_zone   = "no object"
        min_distance = float("inf")

        while sent < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                timestamp = round(frame_idx / fps, 3)
                payload   = _process_frame(frame, smoothers)
                payload.update({"frame_index": frame_idx, "timestamp": timestamp})
                frames.append(payload)
                sent += 1

                zone = payload.get("zone", "no object")
                if ZONE_RANK.get(zone, 0) > ZONE_RANK.get(worst_zone, 0):
                    worst_zone = zone

                d = payload.get("distance")
                if d is not None and d < min_distance:
                    min_distance = d

            frame_idx += 1

        cap.release()

        print(f"  → processed {sent} frames, worst_zone={worst_zone}, min_dist={min_distance}")

        return jsonify({
            "frames":                 frames,
            "worst_zone":             worst_zone,
            "min_distance":           None if min_distance == float("inf") else round(min_distance, 2),
            "total_frames_processed": sent,
            "video_duration":         round(frame_idx / fps, 2),
        })

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, threaded=False)