# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Upsampling in python"""
import numpy as np

def upsample_nearest(arr, scale):
    return arr.repeat(scale, axis=0).repeat(scale, axis=1)

def upsampling_python(data, scale):
    ishape = data.shape
    oshape = (ishape[0], ishape[1], ishape[2]*scale, ishape[3]*scale)
    output_np = np.zeros(oshape, dtype=data.dtype)
    for b in range(oshape[0]):
        for c in range(oshape[1]):
            output_np[b, c, :, :] = upsample_nearest(data[b, c, :, :], scale)
    return output_np
