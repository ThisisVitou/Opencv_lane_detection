import cv2 as cv
import numpy as np

def draw_lane_overlay(original_frame, llane, rlane, src_points, dst_points):
    """
    Draw lane lines on bird's eye view and transform back to original perspective.
    
    Parameters:
    - original_frame: Original frame to overlay on
    - llane: List of (x, y) tuples for left lane
    - rlane: List of (x, y) tuples for right lane
    - src_points: Source trapezoid points for perspective transform
    - dst_points: Destination rectangle points for perspective transform
    
    Returns:
    - Frame with lane overlay
    """
    h, w = original_frame.shape[:2]
    
    # Create blank canvas same size as bird's eye view
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw lane lines on overlay
    if len(llane) > 1:
        pts_left = np.array(llane, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(overlay, [pts_left], False, (255, 255, 0), 8)
    if len(rlane) > 1:
        pts_right = np.array(rlane, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(overlay, [pts_right], False, (0, 255, 255), 8)
    
    # Fill polygon between lanes if both exist
    if len(llane) > 1 and len(rlane) > 1:
        pts = np.array(llane + rlane[::-1], dtype=np.int32).reshape((-1, 1, 2))
        cv.fillPoly(overlay, [pts], (0, 255, 0))
    
    # Inverse perspective transform back to original view
    M_inv = cv.getPerspectiveTransform(dst_points, src_points)
    warped_overlay = cv.warpPerspective(overlay, M_inv, (w, h))
    
    # Blend with original frame
    result = cv.addWeighted(original_frame, 1, warped_overlay, 0.3, 0)
    
    return result