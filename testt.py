import cv2 as cv 
import numpy as np
from inverse_perspective import inversePerspectiveTransform
from searchBox import SearchBox
from edge import detect_edges

def open_camera(cap):
    _, frame_size = cap.read()
    h, w = frame_size.shape[:2]
    print(f"Frame size: {w}x{h}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (720, 480))  # Resize to 720x480

        cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

    return ret, frame, h, w

def get_perspective_points(frame_width, frame_height):
    w, h = frame_width, frame_height
    # Example trapezoid (tune these ratios)
    points = np.load("_point_.npz")["points"]

    # Order: TL, TR, BR, BL
    src_points = np.float32([
        points[2],  # Top-left
        points[3],  # Top-right
        points[1],  # Bottom-right
        points[0],  # Bottom-left
    ])

    # Output rectangle (can span full width)
    dst_points = np.float32([
        [w * 0.20, 0],       # Top-left
        [w * 0.80, 0],       # Top-right
        [w * 0.80, h],       # Bottom-right
        [w * 0.20, h],       # Bottom-left
    ])
    return src_points, dst_points

def debug_perspective_transform(frame, src_points):
    ## just test out the trapezoid area
    debug_frame = frame.copy()
    for i, p in enumerate(src_points):
        cv.circle(debug_frame, tuple(map(int, p)), 6, (0,0,255), -1)
        cv.putText(debug_frame, f"S{i}", tuple(map(int, p+5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv.polylines(debug_frame, [src_points.astype(int)], True, (0,255,0), 2)

    return debug_frame

def draw_lane_on_frame(frame, llane, rlane, src_points, dst_points, ipt):
    """
    Draw detected lanes as an overlay on the original frame.
    
    Parameters:
    - frame: Original frame
    - llane: Left lane points (x_coords, y_coords)
    - rlane: Right lane points (x_coords, y_coords)
    - src_points: Source perspective points
    - dst_points: Destination perspective points
    - ipt: inversePerspectiveTransform instance
    """
    # Create blank image for drawing lanes
    lane_img = np.zeros_like(frame)
    
    # Check if we have enough points
    if len(llane[0]) < 3 or len(rlane[0]) < 3:
        return frame
    
    # Fit polynomials (2nd degree)
    left_fit = np.polyfit(llane[1], llane[0], 2)
    right_fit = np.polyfit(rlane[1], rlane[0], 2)
    
    # Generate y coordinates for the lane
    ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0])
    
    # Calculate x coordinates using the fitted polynomials
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create points for polygon
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane on the blank image
    cv.fillPoly(lane_img, np.int32([pts]), (0, 255, 0))
    
    # Transform back to original perspective
    M_inv = cv.getPerspectiveTransform(dst_points, src_points)
    warped_lane = cv.warpPerspective(lane_img, M_inv, (frame.shape[1], frame.shape[0]))
    
    # Combine with original frame
    result = cv.addWeighted(frame, 1, warped_lane, 0.3, 0)
    
    return result




def main():
    cap = cv.VideoCapture(1)

    if not cap.isOpened():
        print("Failed to open webcam.")
        return

    _, frame_size = cap.read()
    h, w = frame_size.shape[:2]
    print(f"Frame size: {w}x{h}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (720, 480))  # Resize to 720x480

        ##detech each frame edges
        detector = detect_edges(frame) # create edge detector instance
        edges = detector.canny_edge()

        ## Birdeye view transformation
        src_points, dst_points = get_perspective_points(w, h)
        ipt = inversePerspectiveTransform(frame)
        birdeye_view = ipt.inverse_perspective_transform(src_points, dst_points)
        birdeye_edges = detect_edges(birdeye_view).canny_edge()

        ## debug draw trapezoid
        debug_frame = debug_perspective_transform(frame, src_points)

        ## box

        search_box = SearchBox(birdeye_view, birdeye_edges, lx=150, rx=420, y=450, width=100, height=20)
        vis, llane, rlane = search_box.visualize()


        # cv.imshow('Webcam', frame)
        # cv.imshow('Edges', edges)
        cv.imshow('Debug Frame', debug_frame)
        # cv.imshow('Birdseye View', birdeye_view)
        cv.imshow("search box visualization", vis)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

    pass
    

if __name__ == "__main__":
    main()
