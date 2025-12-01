import numpy as np

points = np.load("point_ratios.npz")["ratios"]

for r in points:
    print(f"{r[0]:.3f}, {r[1]:.3f}")