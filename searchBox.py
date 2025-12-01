import cv2 as cv
import numpy as np

class SearchBox():
    def __init__(self, frame, mask, lx=80, rx=150, y=245, width=100, height=20): ## assume the frame is 480pixel wide
        """
        Initialize the detector with a mask and ROI parameters.
        
        Parameters:
        - mask: Binary mask (numpy array)
        - x, y: Top-left corner coordinates of rectangle
        - width, height: Dimensions of rectangle
        """
        self.mask = mask
        self.frame = frame
        self.lx = lx
        self.rx = rx
        self.y = y
        self.width = width
        self.height = height

        self.last_valid_lx = lx  # Store last valid positions
        self.last_valid_rx = rx

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

    def detect(self, y=None, side=None):
        """
        Detect and count pixels in the current ROI.

        """

        # Use provided y or fall back to self.y
        if y is None:
            y = self.y

        # Ensure coordinates are within bounds
        if side == "left":
            x = max(0, self.lx)
        elif side == "right":
            x = max(0, self.rx)
        else:
            x = max(0, self.lx)  # Default to left if side not specified


        y = max(0, y)
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
            if side == "left":
                self.lx = new_x
                self.last_valid_lx = new_x
            elif side == "right":
                self.rx = new_x
                self.last_valid_rx = new_x
        else:
            # No pixels found - use last valid position
            if side == "left":
                self.lx = self.last_valid_lx
            elif side == "right":
                self.rx = self.last_valid_rx

        return self.avg_x

    def visualize(self, num_boxes=10):
        """
        Create a visualization of the mask with ROI highlighted.
        
        """
        vis = self.frame.copy() if self.frame.ndim == 3 else cv.cvtColor(self.frame, cv.COLOR_GRAY2BGR)
        rlane = []
        llane = []

        for i in range(num_boxes):

            box_y = self.y - i * (self.height + 5)
            self.detect(1 + box_y, side="left")  # Update position based on detection
            
            box_x = self.lx 
            box_y = self.y - i * (self.height + 5)  # Stack boxes vertically with spacing
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
                llane.append(self.avg_x)

        for i in range(num_boxes):

            box_y = self.y - i * (self.height + 5)
            self.detect(1 + box_y, side="right")  # Update position based on detection
            
            box_x = self.rx 
            box_y = self.y - i * (self.height + 5)  # Stack boxes vertically with spacing
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
                rlane.append(self.avg_x)
            
            
        return vis, llane, rlane
    

        

        

    