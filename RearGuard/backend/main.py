# 23L-0549, 23L-0736, 23L-0848

import cv2

from detection import detect_obstacles, draw_detections
from distance  import estimate_distance, get_zone, SmoothingFilter, DEFAULT_FOCAL_LENGTH_PX

# configuration
FOCAL_LENGTH_PX = DEFAULT_FOCAL_LENGTH_PX   # replace after calibration
CAMERA_INDEX    = 0                          # 0 = default webcam


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {CAMERA_INDEX}")
        return

    smoothers: dict[int, SmoothingFilter] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break

        # detetction
        obstacles = detect_obstacles(frame)

        frame_h, frame_w = frame.shape[:2]
        distances        = []
        nearest_distance = float("inf")

        for i, (x, y, w, h, label, _conf) in enumerate(obstacles):
            raw_dist = estimate_distance(
                pixel_height    = h,
                box_w           = w,
                box_h           = h,
                frame_w         = frame_w,
                frame_h         = frame_h,
                label           = label,
                focal_length_px = FOCAL_LENGTH_PX
            )

            # Smooth per-slot to reduce frame-to-frame jitter
            if i not in smoothers:
                smoothers[i] = SmoothingFilter(window=7)
            smoothed = smoothers[i].update(raw_dist)

            distances.append(smoothed)

            if smoothed < nearest_distance:
                nearest_distance = smoothed

        # Remove smoothers for slots no longer active
        for key in list(smoothers.keys()):
            if key >= len(obstacles):
                del smoothers[key]

        # zone + speed
        zone = get_zone(nearest_distance)

        speed_advice = {
            "danger":    "STOP  (0 km/h)",
            "warning":   "SLOW  (≤ 30 km/h)",
            "safe":      "OK    (≤ 60 km/h)",
            "no object": "No obstacle detected",
        }.get(zone, "")

        # draw
        output = draw_detections(frame, obstacles, distances, zone)

        cv2.putText(
            output, speed_advice,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

        if nearest_distance < 20.0:
            cv2.putText(
                output,
                f"Nearest: {nearest_distance:.2f} m",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2
            )

        cv2.imshow("Rear-End Collision Avoidance", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()