import cv2 as cv
import numpy as np
import argparse

from edge import detect_edges
from inverse_perspective import inversePerspectiveTransform
from searchBox import SearchBox

def open_capture(source: str):
    # If source is a digit use webcam, else treat as file path
    if source.isdigit():
        cap = cv.VideoCapture(int(source))
        is_webcam = True
    else:
        cap = cv.VideoCapture(source)
        is_webcam = False
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap, is_webcam

def resize_frame(frame, width: int | None, height: int | None):
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width is not None and height is None:
        scale = width / w
        height = int(h * scale)
    elif height is not None and width is None:
        scale = height / h
        width = int(w * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def get_perspective_points(frame_width, frame_height):
    w, h = frame_width, frame_height
    # Example trapezoid (tune these ratios)
    top_y   = 0.556  
    bottom_y= 0.999

    left_top_x  = 0.312
    right_top_x = 0.625
    left_bottom_x  = 0.156
    right_bottom_x = 0.859

    # Order: TL, TR, BR, BL
    src_points = np.float32([
        [w * left_top_x,    h * top_y],      # Top-left
        [w * right_top_x,   h * top_y],      # Top-right
        [w * right_bottom_x,h * bottom_y],   # Bottom-right
        [w * left_bottom_x, h * bottom_y],   # Bottom-left
    ])

    # Output rectangle (can span full width)
    dst_points = np.float32([
        [w * 0.20, 0],       # Top-left
        [w * 0.80, 0],       # Top-right
        [w * 0.80, h],       # Bottom-right
        [w * 0.20, h],       # Bottom-left
    ])
    return src_points, dst_points

def create_box_visualization(birdseye, birdseye_edges, number_of_boxes=10):
    left_boxes = []
    right_boxes = []
    for i in range(number_of_boxes):
        y_pos = 245 - i * 21
        box = SearchBox(birdseye, birdseye_edges, x=80, y=y_pos, width=100, height=20)
        left_boxes.append(box)

    for i in range(number_of_boxes):
        y_pos = 245 - i * 21
        box = SearchBox(birdseye, birdseye_edges, x=280, y=y_pos, width=100, height=20)
        right_boxes.append(box)

    for box in left_boxes:
        box.detect()
    for box in right_boxes:
        box.detect()

    lvis = birdseye.copy()
    rvis = birdseye.copy()

    for box in left_boxes:
        lvis = box.visualize()
    for box in right_boxes:
        rvis = box.visualize()

    combined = cv.hconcat([lvis, rvis])
    
    return combined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or video file path")
    parser.add_argument("--width", type=int, default=None, help="Target width (file sources only)")
    parser.add_argument("--height", type=int, default=None, help="Target height (file sources only)")
    args = parser.parse_args()

    cap, is_webcam = open_capture(args.source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or read error.")
            break

        # Only resize for video file
        if not is_webcam:
            frame = resize_frame(frame, args.width, args.height)

        h, w = frame.shape[:2]
        src_points, dst_points = get_perspective_points(w, h)


        ## just test out the trapezoid area
        debug_frame = frame.copy()
        for i, p in enumerate(src_points):
            cv.circle(debug_frame, tuple(map(int, p)), 6, (0,0,255), -1)
            cv.putText(debug_frame, f"S{i}", tuple(map(int, p+5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv.polylines(debug_frame, [src_points.astype(int)], True, (0,255,0), 2)
        cv.imshow("src trapezoid", debug_frame)

        detector = detect_edges(frame) # create edge detector instance
        edges = detector.canny_edge(mask_height=150)

        ## apply inverse perspective transform
        ipt = inversePerspectiveTransform(frame) # create inverse perspective transform instance
        birdseye = ipt.inverse_perspective_transform(src_points, dst_points)

        ## Apply edge detection to birdseye view
        birdseye_detector = detect_edges(birdseye)
        birdseye_edges = birdseye_detector.canny_edge(mask_height=300)
        
        ## test search boxes on birdseye edges
        combined = create_box_visualization(birdseye, birdseye_edges, number_of_boxes=10)

        # cv.imshow("frame", frame)
        cv.imshow("edges", edges)
        cv.imshow("birdseye", birdseye)
        cv.imshow("birdseye_edges", birdseye_edges)
        cv.imshow("Search Boxes", combined)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()