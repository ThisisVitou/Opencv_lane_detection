import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break
    
    contrast = cv.convertScaleAbs(frame, alpha=0.2)
    gray = cv.cvtColor(contrast, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 100)
    canny = cv.Canny(blur, threshold1=18, threshold2=22)
    frame = cv.resize(canny, (720, 480))

    cv.imshow("Webcam", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()