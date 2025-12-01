import cv2 as cv
import numpy as np

class detect_edges():
    """Class for edge detection using Canny algorithm."""
    def __init__(self, frame, mask_height=400):
        self.frame = frame
        self.mask_height = mask_height

    def aoi_mask(self):
        """
        Create an Area of Interest (AOI) mask for the given frame.
        The AOI is defined as a rectangle centered horizontally at the bottom of the frame,
        with a height of 400 pixels and width equal to the frame width.
        """
        h, w = self.frame.shape[:2]
        img = np.zeros((h, w), dtype=np.uint8)  # assuming 720p frame size

        rect_width = w
        rect_height = self.mask_height

        x1 = w//2 - rect_width//2 # this just tell the first point of rectangle to x = 0
        y1 = h - rect_height 
        x2 = x1 + rect_width
        y2 = y1 + rect_height

        mask = cv.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=-1)
        # print(f"Video's width: {w}, height: {h}")

        return mask

    def canny_edge(self, low_threshold=25, high_threshold=80):
        """Apply Canny edge detection to the frame."""
        mask = self.aoi_mask()

        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (7, 7), 1)
        
        ## mask the area of interest
        aoi = cv.bitwise_and(blur, blur, mask=mask)
        
        canny = cv.Canny(aoi, low_threshold, high_threshold)
        return canny
