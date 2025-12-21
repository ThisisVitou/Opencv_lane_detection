import cv2 as cv
import numpy as np
import os

def draw_grid(img, step=100, color=(100, 100, 100), thickness=1, font_scale=0.5, text_color=(180, 180, 180)):
    h, w = img.shape[:2]
    for x in range(0, w, step):
        cv.line(img, (x, 0), (x, h), color, thickness, lineType=cv.LINE_AA)
        cv.putText(img, f"x={x}", (x + 4, 16), cv.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv.LINE_AA)
    for y in range(0, h, step):
        cv.line(img, (0, y), (w, y), color, thickness, lineType=cv.LINE_AA)
        cv.putText(img, f"y={y}", (4, max(16, y - 4)), cv.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv.LINE_AA)

def main():

    # Use webcam instead of video file
    cap = cv.VideoCapture(1)  # 0 is usually the default webcam
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Failed to read frame from webcam.")
        return

    # Always resize to 720x480
    frame = cv.resize(frame, (720, 480))

    points = [
        (0, 450), #1
        (719, 450), #2
        (200, 250), #3
        (520, 250), #4

        # (520, 250), #4
        # (200, 250), #3
        # (719, 450), #2
        # (0, 450), #1

        (0, 479),
        (719, 479),
        (0, 0),
        (719, 0),

        # (719, 0),
        # (0, 0),
        # (719, 450),
        # (0, 450),
    ]

    # Draw points on frame
    vis = frame.copy()
    draw_grid(vis)
    for i, (x, y) in enumerate(points):
        color = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv.circle(vis, (x, y), 2, color, -1, cv.LINE_AA)
        cv.putText(vis, f"{i+1}", (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    print("Coordinate (x, y):")
    for r in points:
        print(f"{r[0]:.3f}, {r[1]:.3f}")

    np.savez("_point_.npz", points=np.array(points))

    cv.imshow("Frame + Points", vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()