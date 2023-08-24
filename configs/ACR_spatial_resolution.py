#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import DRO

# FOV / [mm]
fov = np.array([50, 50])

# Number of pixels for simulation. Grid points are the centers of the 
# pixels. The origin is the center of the pixel with indices [matrix[0] // 2, 
# matrix[1] // 2].
matrix = np.array([512, 512])

# Gaussian noise standard deviation (choose either None or 0.0 for no noise).
# Noise is added individually to the real and imaginary part of the DRO in k-space.
noise = 0.2 * np.sqrt(np.prod(matrix))

pixel_area = (fov[0] / matrix[0]) * (fov[1] / matrix[1])

radii = [0.45, 0.4, 0.35]
subsection_disctance = 15.0

# Definition of DRO (list of objects from DRO.py)
obj = DRO.new()
for i in range(3):

    # Adjust sizes of holes (each next pair consists of smaller holes)
    r = radii[i]
    center_x = (i - 1) * subsection_disctance

    for j in range(4):
        for k in range(4):

            # Top square
            top_y_coord = - j * 4 * r
            top_x_coord = center_x - k * 4 * r - j * 0.5 * r

            obj.add(DRO.Disk(density=(1.0 / pixel_area), radius=r, center=[top_y_coord, top_x_coord]))

            # Bottom square
            # First hole of the bottom square is dropped!
            if j == 0 and k == 0:
                continue

            bottom_y_coord = j * 4 * r + k * 0.5 * r
            bottom_x_coord = center_x + k * 4 * r

            obj.add(DRO.Disk(density=(1.0 / pixel_area), radius=r, center=[bottom_y_coord, bottom_x_coord]))