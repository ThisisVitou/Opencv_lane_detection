# Lane Detection (OpenCV)

Simple lane detection with perspective transform, edge detection, sliding windows, and polygon overlay.

## How it works

Pipeline per frame:
1. Capture and optional resize ([main.py](main.py)).
2. Compute perspective ROI from ratios stored in `point_ratios.npz` via [`main.get_perspective_points`](main.py).
3. Edge detection in an Area-Of-Interest using Canny via [`edge.detect_edges.canny_edge`](edge.py).
4. Bird’s-eye transform via [`inverse_perspective.inversePerspectiveTransform.inverse_perspective_transform`](inverse_perspective.py).
5. Sliding windows to collect lane points via [`searchBox.SearchBox.visualize`](searchBox.py).
6. Fit second-order polynomials and fill lane polygon, inverse-warp, and blend via [`main.draw_lane_on_frame`](main.py).

Windows shown:
- src trapezoid (debug)
- birdseye_edges
- search box visualization
- frame with lane overlay

Press `q` to quit.

## Requirements

- Python 3.10+
- OpenCV and NumPy

```bash
pip install opencv-python numpy
```

## Running

Webcam (index 0):
```bash
python main.py --source 0
```

Video file:
```bash
python main.py --source test_video/test3.mp4 --width 480
```
- If only width or height is given, the other dimension auto-scales.

## Perspective points (ratios)

- Ratios for the perspective trapezoid are loaded from `point_ratios.npz` by [`main.get_perspective_points`](main.py).
- You can regenerate/tune them using [test.py](test.py) (it reads the first frame of a video, draws points, and saves normalized ratios to `point_ratios.npz`).

Generate/update ratios:
```bash
python test.py
```

## Tuning

- Edge AOI and thresholds: [`edge.detect_edges`](edge.py) and [`edge.detect_edges.canny_edge`](edge.py) (defaults: low=25, high=80; AOI height set via mask_height).
- Sliding windows start/size: edit the `SearchBox` call in [main.py](main.py), e.g.:
  ```python
  search_box = SearchBox(birdseye, birdseye_edges, lx=85, rx=280, y=230, width=100, height=20)
  ```
  See class [`searchBox.SearchBox`](searchBox.py).
- Perspective ROI placement: regenerate `point_ratios.npz` with [test.py](test.py) or replace the file.

## Files

- App entry: [main.py](main.py)
- Edge detection: [`edge.detect_edges`](edge.py)
- Perspective transform: [`inverse_perspective.inversePerspectiveTransform`](inverse_perspective.py)
- Sliding-window search: [`searchBox.SearchBox`](searchBox.py)
- Lane overlay (poly fit + inverse warp): [`main.draw_lane_on_frame`](main.py)
- Ratio helper scripts: [test.py](test.py), [testt.py](testt.py)
- Ratios data: [point_ratios.npz](point_ratios.npz)