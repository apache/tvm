# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Upsampling in python"""
import numpy as np

def upsample_nearest(arr, scale):
    """ Populate the array by scale factor"""
    return arr.repeat(scale, axis=0).repeat(scale, axis=1)

def upsampling_python(data, scale, layout='NCHW'):
    """ Python version of scaling using nearest neighbour """

    ishape = data.shape
    if layout == 'NCHW':
        oshape = (ishape[0], ishape[1], ishape[2]*scale, ishape[3]*scale)
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[1]):
                output_np[b, c, :, :] = upsample_nearest(data[b, c, :, :], scale)
        return output_np
    elif layout == 'NHWC':
        oshape = (ishape[0], ishape[1]*scale, ishape[1]*scale, ishape[3])
        output_np = np.zeros(oshape, dtype=data.dtype)
        for b in range(oshape[0]):
            for c in range(oshape[3]):
                output_np[b, :, :, c] = upsample_nearest(data[b, :, :, c], scale)
        return output_np
    else:
        raise ValueError("not support this layout {} yet".format(layout))
