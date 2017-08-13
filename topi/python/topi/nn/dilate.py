"""Dilation operators"""
from __future__ import absolute_import as _abs
from .util import get_const_tuple
import tvm

@tvm.tag_scope(tag="dilation")
def dilate_nchw(Input, stride_h, stride_w):
    """Dilate Input with zeros.

    Parameters
    ----------
    Input : tvm.Tensor
        Input tensor, layout is NCHW.

    stride_h : int
        Dilation extent on H dimension, 1 means no dilation.

    stride_w : int
        Dilation extent on W dimension, 1 means no dilation.

    Returns
    -------
    Output : tvm.Tensor
        Output tensor, layout is NCHW.
    """
    N, C, H, W = get_const_tuple(Input.shape)
    Output = tvm.compute(
        (N, C, (H-1)*stride_h+1, (W-1)*stride_w+1),
        lambda n, c, h, w: tvm.select(
            tvm.all(tvm.make.EQ(h%stride_h, 0), tvm.make.EQ(w%stride_w, 0)),
            Input(n, c, h/stride_h, w/stride_w), tvm.const(0.0)),
        name='DilatedInput')
    return Output


@tvm.tag_scope(tag="dilation")
def dilate_nhwc(Input, stride_h, stride_w):
    """Dilate Input with zeros.

    Parameters
    ----------
    Input : tvm.Tensor
        Input tensor, layout is NHWC.

    stride_h : int
        Dilation extent on H dimension, 1 means no dilation.

    stride_w : int
        Dilation extent on W dimension, 1 means no dilation.

    Returns
    -------
    Output : tvm.Tensor
        Output tensor, layout is NHWC.
    """
    N, H, W, C = get_const_tuple(Input.shape)
    Output = tvm.compute(
        (N, (H-1)*stride_h+1, (W-1)*stride_w+1, C),
        lambda n, h, w, c: tvm.select(
            tvm.all(tvm.make.EQ(h%stride_h, 0), tvm.make.EQ(w%stride_w, 0)),
            Input(n, h/stride_h, w/stride_w, c), tvm.const(0.0)),
        name='DilatedInput')
    return Output
