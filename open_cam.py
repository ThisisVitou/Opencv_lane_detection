import cv2 as cv

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