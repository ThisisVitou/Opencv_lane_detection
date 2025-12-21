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

def convert_to_percentages(points, width, height):
    return [(x / width, y / height) for x, y in points]

def main():
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Failed to read frame from webcam.")
        return

    orig_h, orig_w = frame.shape[:2]
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(points) < 8:
                points.append((x, y))
                print(f"Point {len(points)}: ({x}, {y})")

    vis = frame.copy()
    draw_grid(vis)
    cv.namedWindow("Select 8 Points")
    cv.setMouseCallback("Select 8 Points", mouse_callback)

    while True:
        temp = vis.copy()
        for i, (x, y) in enumerate(points):
            color = (0, 0, 255) if i < 4 else (255, 0, 0)
            cv.circle(temp, (x, y), 4, color, -1, cv.LINE_AA)
            cv.putText(temp, f"{i+1}", (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
        cv.imshow("Select 8 Points", temp)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or len(points) == 8:  # ESC or 8 points
            break

    cv.destroyWindow("Select 8 Points")

    if len(points) != 8:
        print("You must select exactly 8 points.")
        return

    ratios = convert_to_percentages(points, orig_w, orig_h)
    np.savez("point_ratios.npz", ratios=np.array(ratios))
    print("Ratios (x%, y%):")
    for r in ratios:
        print(f"{r[0]:.3f}, {r[1]:.3f}")

    # Optionally, show the result
    for i, (x, y) in enumerate(points):
        color = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv.circle(frame, (x, y), 4, color, -1, cv.LINE_AA)
        cv.putText(frame, f"{i+1}", (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    cv.imshow("Selected Points", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()