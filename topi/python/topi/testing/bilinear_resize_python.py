# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Bilinear Scale in python"""
import numpy as np

def bilinear_resize_python(image, weights, out_size, layout):
    """ Bilinear scaling using python"""
    (new_h, new_w) = out_size

    if layout == 'NHWC':
        (batch, h, w, channel) = image.shape
        scaled_image = np.ones((batch, new_h, new_w, channel))
    else:
        (batch, channel, h, w) = image.shape
        scaled_image = np.ones((batch, channel, new_h, new_w))

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

                    pixel = (A*(1-x_diff)*(1-y_diff) +
                             B*(x_diff)*(1-y_diff) +
                             C*(y_diff)*(1-x_diff) +
                             D*(x_diff*y_diff))

                    if layout == 'NHWC':
                        scaled_image[b][j][k][i] = pixel
                    else:
                        scaled_image[b][i][j][k] = pixel

    return scaled_image
