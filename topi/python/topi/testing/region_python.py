# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Region in python"""
import numpy as np

def entry_index(batch, w, h, outputs, classes, coords, location, entry):
    n = int(location/(w*h))
    loc = location%(w*h)
    return batch*outputs + n*w*h*(coords+classes+1) + entry*w*h + loc

def region_python(a_np, N, classes, coords, background, softmax):
    """Region operator
    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_channel, in_height, in_width]

    N : int
        Darknet layer parameter n

    classes : int
        Darknet layer parameter classes

    coords : int
        Darknet layer parameter coords

    background : int
        Darknet layer parameter background

    softmax : int
        Darknet layer parameter softmax

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
            index = entry_index(b, in_width, in_height, outputs, classes, coords, n*in_width*in_height, 0)
            b_np[index: index+2*in_width*in_height] = 1/(1+np.exp(-1*b_np[index: index+2*in_width*in_height]))
            index = entry_index(b, in_width, in_height, outputs, classes, coords, n*in_width*in_height, coords)
            if not background:
                b_np[index: index+in_width*in_height] = 1/(1+np.exp(-1*b_np[index: index+in_width*in_height]))

    b_np = np.reshape(b_np, (batch, in_channel, in_height, in_width))
    def local_softmax(data_in):
        data_c, data_h, data_w = data_in.shape
        largest = np.max(data_in, axis=1)
        data_out = np.zeros((data_c, data_h, data_w))
        for i in range(data_h):
            for j in range(data_w):
                data_out[:, i, j] = np.exp(data_in[:, i, j] - largest[i, j])
        return data_out/data_out.sum(axis=0)

    if softmax:
        index = coords + int(not background)
        for b in range(batch):
            for i in range(N):
                b_np_index = int(i*(in_channel/N) + index)
                b_np[b, b_np_index: b_np_index + classes+background, :, :] = local_softmax(b_np[b, b_np_index:b_np_index + classes+background, :, :])

    return b_np
