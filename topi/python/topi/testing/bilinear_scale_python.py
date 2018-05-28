# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Bilinear Scale in python"""
import numpy as np

def bilinear_scale_python(image, weights, out_size, layout):
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
                    x1 = int(weights[j][k][0])
                    y1 = int(weights[j][k][1])

                    x_diff = weights[j][k][2]
                    y_diff = weights[j][k][3]

                    if layout == 'NHWC':
                        A = image[b][y1][x1][i]
                        B = image[b][y1][x1+1][i]
                        C = image[b][y1+1][x1][i]
                        D = image[b][y1+1][x1+1][i]
                    else:
                        A = image[b][i][y1][x1]
                        B = image[b][i][y1][x1+1]
                        C = image[b][i][y1+1][x1]
                        D = image[b][i][y1+1][x1+1]

                    pixel = (A*(1-x_diff)*(1-y_diff) +
                             B*(x_diff)*(1-y_diff) +
                             C*(y_diff)*(1-x_diff) +
                             D*(x_diff*y_diff))

                    if layout == 'NHWC':
                        scaled_image[b][j][k][i] = pixel
                    else:
                        scaled_image[b][i][j][k] = pixel

    return scaled_image
