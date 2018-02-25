# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Reorg in python"""
import numpy as np

def reorg_python(a_np, stride):
    """Reorg operator

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    stride : int
        Stride size

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    batch, in_channel, in_height, in_width = a_np.shape
    a_np = np.reshape(a_np, batch*in_channel*in_height*in_width)
    out_c = int(in_channel/(stride*stride))
    out_channel = in_channel*stride*stride
    out_height = int(in_height/stride)
    out_width = int(in_width/stride)
    b_np = np.zeros(batch*out_channel*out_height*out_width)
    cnt = 0
    for b in range(batch):
        for k in range(in_channel):
            for j in range(in_height):
                for i in range(in_width):
                    c2 = k % out_c
                    offset = int(k / out_c)
                    w2 = int(i*stride + offset % stride)
                    h2 = int(j*stride + offset / stride)
                    out_index = int(w2 + in_width*stride*(h2 + in_height*stride*(c2 + out_c*b)))
                    b_np[cnt] = a_np[int(out_index)]
                    cnt = cnt+1
    b_np = np.reshape(b_np, (batch, out_channel, out_height, out_width))
    return b_np
