# pylint: disable=invalid-name, unused-variable
"""NN operator common utilities"""
from __future__ import absolute_import

import tvm
from ..util import get_const_int
import numpy as np
from topi.transform import concatenate


def infer_pad(data, data_pad):
    """Infer the padding from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    data_pad : Tensor
        pad stage.

    Returns
    -------
    hpad : int
        padding size on height
    wpad : int
        padding size on width
    """
    if data_pad is None:
        return 0, 0
    _, _, IH, IW = data.shape
    _, _, TH, TW = data_pad.shape
    hpad = (TH - IH) // 2
    wpad = (TW - IW) // 2
    return get_const_int(hpad), get_const_int(wpad)

def infer_stride(data, kernel, out):
    """Infer the stride from stages in reverse.

    Parameters
    ----------
    data : Tensor
        data stage.

    kernel : Tensor
        kernel stage.

    out : Tensor
        output stage.

    Returns
    -------
    hstride : int
        stride size on height
    wstride : int
        stride size on width
    """
    _, _, IH, IW = data.shape
    _, _, KH, KW = kernel.shape
    _, _, OH, OW = out.shape
    hstride = (IH - KH) // tvm.make.Max(OH - 1, 1) + tvm.select(OH == 1, 1, 0)
    wstride = (IW - KW) // tvm.make.Max(OW - 1, 1) + tvm.select(OW == 1, 1, 0)
    return get_const_int(hstride), get_const_int(wstride)


def get_pad_tuple(padding, kernel):
    """Common code to get the pad option

    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']

    kernel : tuple of int
        Conv kernel size

    Returns
    -------
    pad_top : int
        Padding size on top

    pad_left : int
        Padding size on left

    pad_down : int
        Padding size on down.

    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        pad_h = padding[0] * 2
        pad_w = padding[1] * 2
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


# Packs quantized data into packed bitplanes
# pack_axis = Axis to compress of original tensor
# bit_axis = Axis to place bitplanes in the resulting tensor
# pack_type = Datatype to pack elements into 
def bitpack(data, bits, pack_axis, bit_axis, pack_type, name="QuantizeInput"):
    ishape = data.shape
    n = len(ishape)
    if pack_type == 'uint8':
        data_width = 8
    elif pack_type == 'uint16':
        data_width = 16
    elif pack_type == 'uint32':
        data_width = 32
    elif pack_type == 'uint64':
        data_width = 64
  
    # Data must be in multiples of the data_width
    assert get_const_int(ishape[pack_axis]) % data_width == 0, "Not a multiple of word size"

    shape_vec = list(ishape)
    shape_vec[pack_axis] = (shape_vec[pack_axis] // data_width)
    shape_vec.insert(bit_axis, 1)
    bitserial_oshape = tuple(shape_vec)
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])

    # pack axis shifts if bit axis comes before
    if bit_axis <= pack_axis:
        pack_axis += 1 

    def _bitpack(*indices):
        packed_data = [tvm.const(0, pack_type)] * bits
        for k in range(data_width):
            # Translate indices for packed data back to original
            idx = [0] * n
            j = 0
            for i in range(n+1):
                if i == bit_axis:
                    continue
                elif i == pack_axis:
                    idx[j] = indices[i] * data_width + k
                else:
                    idx[j] = indices[i]
                j += 1       
            
            element = data(*idx)
            for b in range(bits):
                extracted_bit = ((element & tvm.const(masks[b])) >> b).astype(pack_type)
                packed_data[b] = (packed_data[b] | extracted_bit)
                if k < data_width - 1 :
                    packed_data[b] = packed_data[b] << 1

            if k == data_width - 1:
                return tuple(packed_data)

    output_tuple = tvm.compute(bitserial_oshape, _bitpack, name=name, tag='bitpack')

    if bits > 1:
        return concatenate(output_tuple, axis=bit_axis)
    else:
        return output_tuple  

