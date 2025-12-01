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
    vid_path = 'test_video/test3.mp4'
    if not os.path.isfile(vid_path):
        print(f"File not found: {vid_path}")
        return

    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Failed to open video: {vid_path}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Failed to read first frame from video.")
        return

    orig_h, orig_w = frame.shape[:2]

    # points = [
    #     (200, 719), #1
    #     (1100, 719), #2
    #     (520, 580), #3
    #     (750, 580), #4

    #     (250, 719),
    #     (1030, 719),
    #     (250, 0),
    #     (1030, 0),
    # ]

    points = [
        (200, 719), #1
        (1100, 719), #2
        (400, 610), #3
        (830, 610), #4

        (250, 719),
        (1030, 719),
        (250, 0),
        (1030, 0),
    ]

    # Ratios based on original frame size
    ratios = convert_to_percentages(points, orig_w, orig_h)

    # Target resized dimensions
    ws = 480
    scale = ws / orig_w
    hs = int(orig_h * scale)

    # Resize frame
    resized = cv.resize(frame, (ws, hs), interpolation=cv.INTER_AREA)

    # Scaled points for resized frame
    scaled_points = [(int(rx * ws), int(ry * hs)) for rx, ry in ratios]

    # Draw original points on original frame
    orig_vis = frame.copy()
    draw_grid(orig_vis)
    for i, (x, y) in enumerate(points):
        color = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv.circle(orig_vis, (x, y), 2, color, -1, cv.LINE_AA)
        cv.putText(orig_vis, f"{i+1}", (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    # Draw scaled points on resized frame
    resized_vis = resized.copy()
    draw_grid(resized_vis)
    for i, (x, y) in enumerate(scaled_points):
        color = (0, 0, 255) if i < 4 else (255, 0, 0)
        cv.circle(resized_vis, (x, y), 2, color, -1, cv.LINE_AA)
        cv.putText(resized_vis, f"{i+1}", (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)

    print("Ratios (x%, y%):")
    for r in ratios:
        print(f"{r[0]:.3f}, {r[1]:.3f}")

    np.savez("point_ratios.npz", ratios=np.array(ratios))

    cv.imshow("Original Frame + Points", orig_vis)
    cv.imshow(f"Resized Frame ({ws}x{hs}) + Scaled Points", resized_vis)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()