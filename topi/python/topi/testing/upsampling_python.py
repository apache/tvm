# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Upsampling in python"""
import numpy as np
import scipy.signal
from skimage.transform import resize


def upsampling_python(data, scale):
    ishape = data.shape
    oshape = (ishape[0], ishape[1], ishape[2]*scale, ishape[3]*scale)
    output_np = np.zeros(oshape, dtype=data.dtype)
    for b in range(oshape[0]):
        for c in range(oshape[1]):
            output_np[b,c,:,:] = resize(data[b,c,:,:] , (oshape[2], oshape[3]), order=0, mode="reflect")
    return output_np
