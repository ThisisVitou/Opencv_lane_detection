import cv2 as cv
import numpy as np

class SearchBox():
    def __init__(self, mask, x=0, y=0, width=100, height=100):
        """
        Initialize the detector with a mask and ROI parameters.
        
        Parameters:
        - mask: Binary mask (numpy array)
        - x, y: Top-left corner coordinates of rectangle
        - width, height: Dimensions of rectangle
        """
        self.mask = mask
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.roi_mask = None
        self.pixel_count = 0
        self.pixel_coords = []
    
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
        
        Returns:
        - dict: Contains roi_mask, pixel_count, and pixel_coords
        """
        # Ensure coordinates are within bounds
        x = max(0, self.x) ## Using max(0, self.x) clamps it to >=0
        y = max(0, self.y)
        width = min(self.width, self.mask.shape[1] - x) ## preventing out-of-bounds
        height = min(self.height, self.mask.shape[0] - y)
        
        # Extract ROI from mask
        self.roi_mask = self.mask[y:y+height, x:x+width] ##I don't really know what this does
        
        # Count non-zero pixels
        self.pixel_count = cv.countNonZero(self.roi_mask)
        
        # Get coordinates of non-zero pixels (relative to ROI)
        pixel_coords_rel = cv.findNonZero(self.roi_mask)
        
        # Convert to absolute coordinates
        self.pixel_coords = []
        if pixel_coords_rel is not None:
            for coord in pixel_coords_rel:
                px, py = coord[0]
                self.pixel_coords.append((px + x, py + y))
        
        return {
            'roi_mask': self.roi_mask,
            'pixel_count': self.pixel_count,
            'pixel_coords': self.pixel_coords
        }

    def visualize(self):
        """
        Create a visualization of the mask with ROI highlighted.
        
        """

        # Draw the ROI rectangle in green

        cv.rectangle(   img=    self.mask, 
                        pt1=    (self.x, self.y),
                        pt2=    (self.x + self.width, self.y + self.height),
                        color=  (0, 255, 0),
                        thickness= 2)
        
        return self.mask
    

        

        

    