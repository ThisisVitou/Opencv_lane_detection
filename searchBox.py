import cv2 as cv
import numpy as np

class SearchBox():
    def __init__(self, frame, mask, lx=80, rx=150, y=245, width=100, height=20, num_boxes=10):
        """
        Initialize the detector with a mask and ROI parameters.
        
        Parameters:
        - mask: Binary mask (numpy array)
        - lx, rx: Initial x positions for left and right lanes
        - y: Bottom starting y coordinate
        - width, height: Dimensions of rectangle
        - num_boxes: Number of boxes to stack
        """
        self.mask = mask
        self.frame = frame
        self.y = y
        self.width = width
        self.height = height
        self.num_boxes = num_boxes

        # Calculate center line
        self.center_x = mask.shape[1] // 2

        # Store independent positions for each box
        self.left_positions = [lx] * num_boxes
        self.right_positions = [rx] * num_boxes
        
        # Store last VALID detection for each box
        self.left_last_valid = [lx] * num_boxes
        self.right_last_valid = [rx] * num_boxes
        
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

    def detect(self, x, y):
        """
        Detect and count pixels in the current ROI.
        Returns the new x position or None if no detection.
        """
        # Ensure coordinates are within bounds
        x = max(0, x)
        y = max(0, y)
        width = min(self.width, self.mask.shape[1] - x)
        height = min(self.height, self.mask.shape[0] - y)
        
        # Extract ROI from mask
        self.roi_mask = self.mask[y:y+height, x:x+width]

        # If ROI is 3-channel, convert to gray
        if self.roi_mask.ndim == 3:
            gray = cv.cvtColor(self.roi_mask, cv.COLOR_BGR2GRAY)
        else:
            gray = self.roi_mask

        # Find non-zero pixels and compute average x in global coords
        ys, xs = np.nonzero(gray)
        if xs.size > 0:
            avg_x = x + xs.mean()
            # Recenter: left edge so avg_x sits at box center
            new_x = int(avg_x - self.width // 2)
            # Clamp within image bounds
            new_x = max(0, min(new_x, self.mask.shape[1] - self.width))
            return new_x
        else:
            # No pixels found - return None to signal no detection
            return None

    def visualize(self):
        """
        Create a visualization of the mask with ROI highlighted.
        Each box maintains its own position independently.
        """
        vis = self.frame.copy() if self.frame.ndim == 3 else cv.cvtColor(self.frame, cv.COLOR_GRAY2BGR)
        rlane = ([], [])
        llane = ([], [])

        # Process left lane boxes
        for i in range(self.num_boxes):
            box_y = self.y - i * (self.height + 5)
            box_y = max(0, min(box_y, self.mask.shape[0] - self.height))
            
            # Detect with current box position
            new_x = self.detect(self.left_positions[i], box_y)
            
            # Update position based on detection result
            if new_x is not None:
                # Check if box center would cross center line
                box_center = new_x + self.width // 2
                if box_center < self.center_x:
                    # Valid detection and doesn't cross - update both current and last valid
                    self.left_positions[i] = new_x
                    self.left_last_valid[i] = new_x
                else:
                    # Detection crosses center line - revert to last valid
                    self.left_positions[i] = self.left_last_valid[i]
            else:
                # No detection - revert to last valid position
                self.left_positions[i] = self.left_last_valid[i]
            
            box_x = self.left_positions[i]

            # Draw rectangle
            cv.rectangle(vis, (box_x, box_y), (box_x + self.width, box_y + self.height),
                        (0, 255, 0), 1)
            
            # Center marker
            center_x = box_x + self.width // 2
            center_y = box_y + self.height // 2
            cv.circle(vis, (center_x, center_y), 3, (0, 0, 255), -1)

            llane[0].append(center_x)
            llane[1].append(center_y)

        # Process right lane boxes
        for i in range(self.num_boxes):
            box_y = self.y - i * (self.height + 5)
            box_y = max(0, min(box_y, self.mask.shape[0] - self.height))
            
            # Detect with current box position
            new_x = self.detect(self.right_positions[i], box_y)
            
            # Update position based on detection result
            if new_x is not None:
                # Check if box center would cross center line
                box_center = new_x + self.width // 2
                if box_center > self.center_x:
                    # Valid detection and doesn't cross - update both current and last valid
                    self.right_positions[i] = new_x
                    self.right_last_valid[i] = new_x
                else:
                    # Detection crosses center line - revert to last valid
                    self.right_positions[i] = self.right_last_valid[i]
            else:
                # No detection - revert to last valid position
                self.right_positions[i] = self.right_last_valid[i]
            
            box_x = self.right_positions[i]

            # Draw rectangle
            cv.rectangle(vis, (box_x, box_y), (box_x + self.width, box_y + self.height),
                        (0, 255, 0), 1)
            
            # Center marker
            center_x = box_x + self.width // 2
            center_y = box_y + self.height // 2
            cv.circle(vis, (center_x, center_y), 3, (0, 0, 255), -1)

            rlane[0].append(center_x)
            rlane[1].append(center_y)
            
        return vis, llane, rlane