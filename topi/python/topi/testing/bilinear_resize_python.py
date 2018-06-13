# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Bilinear Scale in python"""
import math
import numpy as np

def bilinear_weights(height, width, new_h, new_w, align_corners=False):
    """ Helper function to generate weights for bilinear scaling """

    if align_corners:
        x_ratio = np.float32(np.float32(width)/np.float32(new_w))
        y_ratio = np.float32(np.float32(height)/np.float32(new_h))
    else:
        x_ratio = np.float32(np.float32(width-1)/np.float32(new_w-1))
        y_ratio = np.float32(np.float32(height-1)/np.float32(new_h-1))

    def _bilinear_interpolation(y, x):
        x_coord = math.floor(x_ratio * x)
        y_coord = math.floor(y_ratio * y)
        x_diff = np.float32((x_ratio * x) - x_coord)
        y_diff = np.float32((y_ratio * y) - y_coord)

        return [y_coord, x_coord, y_diff, x_diff]

    # weights to hold (srcx, srcy, x_diff, y_diff) for each out value.
    weights = np.empty([new_h, new_w, 4], dtype='float32')

    for i in range(new_h):
        for j in range(new_w):
            weights[i][j] = _bilinear_interpolation(i, j)
    return weights

def bilinear_resize_python(image, out_size, layout, align_corners=False):
    """ Bilinear scaling using python"""
    (new_h, new_w) = out_size

    if layout == 'NHWC':
        (batch, h, w, channel) = image.shape
        scaled_image = np.ones((batch, new_h, new_w, channel))
    else:
        (batch, channel, h, w) = image.shape
        scaled_image = np.ones((batch, channel, new_h, new_w))

    weights = bilinear_weights(h, w, new_h, new_w, align_corners)

    for b in range(batch):
        for i in range(channel):
            for j in range(new_h):
                for k in range(new_w):
                    y0 = int(weights[j][k][0])
                    x0 = int(weights[j][k][1])

                    x1 = min((x0+1), (w-1))
                    y1 = min((y0+1), (h-1))

                    y_diff = weights[j][k][2]
                    x_diff = weights[j][k][3]

                    if layout == 'NHWC':
                        A = image[b][y0][x0][i]
                        B = image[b][y0][x1][i]
                        C = image[b][y1][x0][i]
                        D = image[b][y1][x1][i]
                    else:
                        A = image[b][i][y0][x0]
                        B = image[b][i][y0][x1]
                        C = image[b][i][y1][x0]
                        D = image[b][i][y1][x1]

                    pixel = np.float32((A*(1-x_diff)*(1-y_diff) +
                                        B*(x_diff)*(1-y_diff) +
                                        C*(y_diff)*(1-x_diff) +
                                        D*(x_diff*y_diff)))

                    if layout == 'NHWC':
                        scaled_image[b][j][k][i] = pixel
                    else:
                        scaled_image[b][i][j][k] = pixel

    return scaled_image
