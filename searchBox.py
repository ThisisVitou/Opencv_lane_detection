import cv2 as cv
import numpy as np

class SearchBox():
    def __init__(self, frame, mask, x=80, y=245, width=100, height=20): ## assume the frame is 480pixel wide
        """
        Initialize the detector with a mask and ROI parameters.
        
        Parameters:
        - mask: Binary mask (numpy array)
        - x, y: Top-left corner coordinates of rectangle
        - width, height: Dimensions of rectangle
        """
        self.mask = mask
        self.frame = frame
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.roi_mask = None
        self.avg_x = None
    
    def set_roi(self, x, y, width, height):
        """
        Change the location and size of the ROI rectangle.
        
        Parameters:
        - x, y: New top-left corner coordinates
        - width, height: New dimensions
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def detect(self):
        """
        Detect and count pixels in the current ROI.

        """
        # Ensure coordinates are within bounds
        x = max(0, self.x) ## Using max(0, self.x) clamps it to >=0
        y = max(0, self.y)
        width = min(self.width, self.mask.shape[1] - x) ## preventing out-of-bounds
        height = min(self.height, self.mask.shape[0] - y)
        
        # Extract ROI from mask
        self.roi_mask = self.mask[y:y+height, x:x+width] ##I don't really know what this does

        # If ROI is 3-channel, convert to gray
        if self.roi_mask.ndim == 3:
            gray = cv.cvtColor(self.roi_mask, cv.COLOR_BGR2GRAY)
        else:
            gray = self.roi_mask

        # Find non-zero pixels and compute average x in global coords
        ys, xs = np.nonzero(gray)
        if xs.size > 0:
            self.avg_x = x + xs.mean()

            # Recenter: left edge so avg_x sits at box center
            new_x = int(self.avg_x - self.width // 2)

            # Clamp within image bounds
            new_x = max(0, min(new_x, self.mask.shape[1] - self.width))
            self.x = new_x

        return self.avg_x

    def visualize(self):
        """
        Create a visualization of the mask with ROI highlighted.
        
        """
        vis = self.frame.copy() if self.frame.ndim == 3 else cv.cvtColor(self.frame, cv.COLOR_GRAY2BGR)

        box_x = self.x
        box_y = self.y

        # Keep box_y in bounds
        box_y = max(0, min(box_y, self.mask.shape[0] - self.height))

        # Rectangle
        cv.rectangle(vis, (box_x, box_y), (box_x + self.width, box_y + self.height),
                        (0, 255, 0), 1)

        # Center marker
        center_x = box_x + self.width // 2
        center_y = box_y + self.height // 2
        cv.circle(vis, (center_x, center_y), 3, (0, 0, 255), -1)

        # Avg_x marker (if available)
        if self.avg_x is not None:
            cv.circle(vis, (int(self.avg_x), center_y), 4, (255, 0, 0), 1)

        return vis
    

        

        

    