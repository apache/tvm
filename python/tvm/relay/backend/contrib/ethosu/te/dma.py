# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unnecessary-lambda
"""Tensor Expressions for operations supported by the NPU DMA engine"""
from typing import Callable, Tuple, Optional, List

import tvm  # type: ignore
from tvm import te
from tvm.topi.utils import equal_const_int  # type: ignore


def _pad_tensor(
    tensor: te.Tensor, pad_before: List[int], pad_after: Optional[List[int]] = None
) -> Callable:
    """Generate a padded tensor.

    Parameters
    ----------
    tensor : te.Tensor
        The tensor to pad.
    pad_before : tuple of int
        The 'before' padding on each axis.
    pad_after : tuple of int
        The 'after' padding on each axis.
    Returns
    -------
    _pad : callable
        The padded tensor.

    """
    pad_after = pad_after or pad_before
    dims = len(tensor.shape)
    assert len(pad_before) == dims
    assert len(pad_after) == dims

    def _pad(*indices):
        not_zero = []  # A list of padding conditions that aren't trivial (zero padding)
        index_tuple = []  # The indices with which to access the padded tensor
        for i in range(dims):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < tensor.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tvm.tir.all(*not_zero)
            return tvm.tir.if_then_else(
                not_zero, tensor(*index_tuple), tvm.tir.const(0, tensor.dtype)
            )
        return tensor(*index_tuple)

    return _pad


def read_compute(tensor: te.Tensor, layout: str, zero_point: int, scale: float) -> te.Tensor:
    """A tensor expression which represents a read.

    Parameters
    ----------
    tensor : te.Tensor
        The tensor to read.
    layout : str
        The layout of the tensor, either NHWC or NHCWB16.
    zero_point : int
        The zero point of the tensor.
    scale : float
        The scale of the tensor.

    Returns
    -------
    te.Tensor
        The tensor having been read.

    """
    assert layout in {"NHWC", "NHCWB16"}
    read_attrs = {
        "op": "ethosu_read",
        "layout": layout,
        "zero_point": zero_point,
        "scale": scale,
    }
    return te.compute(tensor.shape, lambda *i: tensor(*i), name="ethosu_read", attrs=read_attrs)


def write_compute(tensor: te.Tensor, layout: str, zero_point: int, scale: float) -> te.Tensor:
    """A tensor expression which represents a write.

    Parameters
    ----------
    tensor : te.Tensor
        The tensor to write.
    layout : str
        The layout of the tensor, either NHWC or NHCWB16.
    zero_point : int
        The zero point of the tensor.
    scale : float
        The scale of the tensor.

    Returns
    -------
    te.Tensor
        The tensor having been written.

    """
    assert layout in {"NHWC", "NHCWB16"}
    write_attrs = {
        "op": "ethosu_write",
        "layout": layout,
        "zero_point": zero_point,
        "scale": scale,
    }
    return te.compute(
        tensor.shape,
        lambda *i: tensor(*i),
        name="ethosu_write",
        attrs=write_attrs,
    )


def convert_to_nhwc_compute(tensor: te.Tensor, layout: str, channels: int) -> te.Tensor:
    """Converts a tensor into NHWC layout if it's in NHWCB16 layout.

    Parameters
    ----------
    tensor : te.Tensor
        The tensor to convert.
    layout : str
        The layout of the tensor, either NHWC or NHCWB16.
    channels : int
        The number of valid channels for the tensor.

    Returns
    -------
    te.Tensor
        The converted tensor in NHWC layout.

    """
    assert layout in {"NHWC", "NHCWB16"}
    convert_to_nhwc_attrs = {
        "op": "ethosu_convert_to_nhwc",
        "layout": layout,
    }
    if layout == "NHCWB16":
        return te.compute(
            (tensor.shape[0], tensor.shape[1], tensor.shape[3], channels),
            lambda nn, hh, ww, cc: tensor(nn, hh, te.indexdiv(cc, 16), ww, te.indexmod(cc, 16)),
            name="ethosu_convert_to_nhwc",
            attrs=convert_to_nhwc_attrs,
        )

    return te.compute(
        tensor.shape,
        lambda *i: tensor(*i),
        name="ethosu_convert_to_nhwc",
        attrs=convert_to_nhwc_attrs,
    )


def convert_to_nhcwb16_compute(tensor: te.Tensor, layout: str, channels: int) -> te.Tensor:
    """Converts a tensor into NHCWB16 layout if it's in NHWC layout.

    Parameters
    ----------
    tensor : te.Tensor
        The tensor to convert.
    layout : str
        The layout of the tensor, either NHWC or NHCWB16.
    channels : int
        The number of valid channels for the tensor.

    Returns
    -------
    te.Tensor
        The converted tensor in NHCWB16 layout.

    """
    assert layout in {"NHWC", "NHCWB16"}
    convert_to_nhcwb16_attrs = {
        "op": "ethosu_convert_to_nhcwb16",
        "layout": layout,
    }
    if layout == "NHCWB16":
        out_channel_bricks = te.indexdiv(channels - 1, 16) + 1
        output_shape = (1, tensor.shape[1], out_channel_bricks, tensor.shape[2], 16)
        return te.compute(
            output_shape,
            lambda nn, hh, cc, ww, cb: tvm.tir.if_then_else(
                cc * 16 + cb < channels,
                tensor(nn, hh, ww, cc * 16 + cb),
                tvm.tir.IntImm(tensor.dtype, 0),
            ),
            name="ethosu_convert_to_nhcwb16",
            attrs=convert_to_nhcwb16_attrs,
        )

    return te.compute(
        tensor.shape,
        lambda *i: tensor(*i),
        name="ethosu_convert_to_nhcwb16",
        attrs=convert_to_nhcwb16_attrs,
    )


def pad_compute(tensor: te.Tensor, padding: tuple) -> te.Tensor:
    """Pad an NHWC tensor in the height and width axes.

    Parameters
    ----------
    tensor : te.Tensor
        The tensor to pad.
    padding : tuple
        The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).

    Returns
    -------
    te.Tensor
        The padded tensor.

    """
    pad_top, pad_left, pad_down, pad_right = padding
    pad_before = [0, int(pad_top), int(pad_left), 0]
    pad_after = [0, int(pad_down), int(pad_right), 0]
    pad_attrs = {
        "op": "ethosu_pad",
    }
    shape = tensor.shape
    return te.compute(
        (shape[0], shape[1] + pad_top + pad_down, shape[2] + pad_left + pad_right, shape[3]),
        lambda nn, hh, ww, cc: _pad_tensor(tensor, pad_before, pad_after)(nn, hh, ww, cc),
        name="ethosu_pad",
        attrs=pad_attrs,
    )


def dma_ifm_compute(
    ifm: te.Tensor,
    layout: str,
    zero_point: int,
    scale: float,
    channels: int,
    padding: Tuple[int, int, int, int],
) -> te.Tensor:
    """A sequence of compute operators representing the DMA capabilities for an IFM.

    Parameters
    ----------
    ifm : te.Tensor
        The Input Feature Map (IFM) tensor.
    layout : str
        The layout of the data, either NHWC or NHCWB16.
    zero_point : int
        The zero point of the data.
    scale : float
        The scale of the data.
    channels : int
        The number of valid channels for the data.
    padding : tuple
        The 4 dimensional padding as (pad_top, pad_left, pad_bottom, pad_right).

    Returns
    -------
    te.Tensor
        The dma-ed IFM tensor.

    """
    read_ifm = read_compute(ifm, layout, zero_point, scale)
    convert_to_nhwc_ifm = convert_to_nhwc_compute(read_ifm, layout, channels)
    return pad_compute(convert_to_nhwc_ifm, padding)


def dma_ofm_compute(
    ofm: te.Tensor, layout: str, zero_point: int, scale: float, channels: int
) -> te.Tensor:
    """A sequence of compute operators representing the DMA capabilities for an OFM.

    Parameters
    ----------
    ofm : te.Tensor
        The Output Feature Map (OFM) tensor.
    layout : str
        The layout of the data, either NHWC or NHCWB16.
    zero_point : int
        The zero point of the data.
    scale : float
        The scale of the data.
    channels : int
        The number of valid channels for the data.

    Returns
    -------
    te.Tensor
        The dma-ed OFM tensor.

    """
    convert_to_nhcwb16_ofm = convert_to_nhcwb16_compute(ofm, layout, channels)
    return write_compute(convert_to_nhcwb16_ofm, layout, zero_point, scale)
