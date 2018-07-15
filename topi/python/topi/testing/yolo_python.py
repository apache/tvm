# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Yolo operator in python"""
import numpy as np

def entry_index(batch, w, h, outputs, classes, coords, location, entry):
    n = int(location/(w*h))
    loc = location%(w*h)
    return batch*outputs + n*w*h*(coords+classes+1) + entry*w*h + loc

def yolo_python(a_np, N, classes):
    """Yolo operator
    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    N : int
        Darknet layer parameter n

    classes : int
        Darknet layer parameter classes

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    batch, in_channel, in_height, in_width = a_np.shape
    a_np_temp = np.reshape(a_np, batch*in_channel*in_height*in_width)
    outputs = batch*in_channel*in_height*in_width
    b_np = np.zeros(batch*in_channel*in_height*in_width)
    for i in range(batch*in_channel*in_height*in_width):
        b_np[i] = a_np_temp[i]
    for b in range(batch):
        for n in range(N):
            index = entry_index(b, in_width, in_height, outputs, classes, 4, n*in_width*in_height, 0)
            b_np[index: index+2*in_width*in_height] = 1/(1+np.exp(-1*b_np[index: index+2*in_width*in_height]))
            index = entry_index(b, in_width, in_height, outputs, classes, 4, n*in_width*in_height, 4)
            b_np[index: index+(1+classes)*in_width*in_height] = 1/(1+np.exp(-1*b_np[index: index+(1+classes)*in_width*in_height]))

    b_np = np.reshape(b_np, (batch, in_channel, in_height, in_width))
    return b_np
