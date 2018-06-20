# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""Shortcut in python"""
import numpy as np

def shortcut_python(a_np1, a_np2):
    """Reorg operator

    Parameters
    ----------
    a_np1 : numpy.ndarray
        4-D with shape [batch1, in_channel1, in_height1, in_width1]

    a_np2 : numpy.ndarray
        4-D with shape [batch2, in_channel2, in_height2, in_width2]

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch1, out_channel1, out_height1, out_width1]
    """

    batch1, in_channel1, in_height1, in_width1 = a_np1.shape
    batch2, in_channel2, in_height2, in_width2 = a_np2.shape
    a_np1_temp = np.reshape(a_np1, batch1*in_channel1*in_height1*in_width1)
    a_np2_temp = np.reshape(a_np2, batch2*in_channel2*in_height2*in_width2)
    b_np = np.zeros(batch1*in_channel1*in_height1*in_width1)
    stride = int(in_width1/in_width2)
    sample = int(in_width2/in_width1)
    if stride < 1:
        stride = 1
    if sample < 1:
        sample = 1
    minw = min(in_width1, in_width2)
    minh = min(in_height1, in_height2)
    minc = min(in_channel1, in_channel2)

    for i in range((batch1*in_channel1*in_height1*in_width1)):
        b_np[i] = a_np1_temp[i]
    for b in range(batch1):
        for k in range(minc):
            for j in range(minh):
                for i in range(minw):
                    out_index = i*sample + in_width2*(j*sample + in_height2*(k + in_channel2*b))
                    add_index = i*stride + in_width1*(j*stride + in_height1*(k + in_channel1*b))
                    b_np[out_index] = a_np1_temp[out_index] + a_np2_temp[add_index]
    b_np = np.reshape(b_np, (batch1, in_channel1, in_height1, in_width1))
    return b_np
