# 23L-0549, 23L-0736, 23L-0848

import cv2
import numpy as np

# parameters for calibration
KNOWN_DISTANCE_M  = 3.0    # how far the object is from camera (metres)
KNOWN_REAL_HEIGHT = 1.5    # real height of the object (metres)  e.g. 1.5 m car
CAMERA_INDEX      = 0

points = []


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        print(f"  Point {len(points)}: ({x}, {y})")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", click_event)

    print("Click the TOP of the object, then the BOTTOM.")
    print("Press R to reset, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        for pt in points:
            cv2.circle(display, pt, 6, (0, 255, 255), -1)

        if len(points) == 2:
            pixel_height = abs(points[1][1] - points[0][1])
            focal = (pixel_height * KNOWN_DISTANCE_M) / KNOWN_REAL_HEIGHT

            cv2.line(display, points[0], points[1], (0, 255, 0), 2)
            cv2.putText(
                display,
                f"pixel_h={pixel_height}px  focal={focal:.1f}px",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            print(f"\n✅  FOCAL_LENGTH_PX = {focal:.1f}")
            print("    Copy this value into app.py and main.py\n")

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            points.clear()
            print("Points reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()