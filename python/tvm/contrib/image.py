"""Common system utilities"""
from __future__ import absolute_import as _abs
import math
import numpy as np

def bilinear_weights(height, width, new_h, new_w, align_corners=False):
    """ Helper function to generate weights for bilinear scaling """

    if align_corners:
        x_ratio = width/new_w
        y_ratio = height/new_h
    else:
        x_ratio = (width-1)/(new_w-1)
        y_ratio = (height-1)/(new_h-1)

    def _bilinear_interpolation(y, x):
        x_coord = math.floor(x_ratio * x)
        y_coord = math.floor(y_ratio * y)
        x_diff = (x_ratio * x) - x_coord
        y_diff = (y_ratio * y) - y_coord

        return [y_coord, x_coord, y_diff, x_diff]

    weights = np.empty([new_h, new_w, 4], dtype='float32')

    for i in range(new_h):
        for j in range(new_w):
            weights[i][j] = _bilinear_interpolation(i, j)
    return weights
