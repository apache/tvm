"""External function interface to NNPlhsCK libraroes."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin

def fully_connected_inference(lhs, rhs):
    """Create an extern op that compute fully connected of 1D tensor lhs and
    2D tensor rhs with nnpack.

    Parameters
    ----------
    lhs : Tensor
        lhs 1D array input[input_channels] of FP32 elements
    rhs : Tensor
        lhs 2D matrix kernel[output_channels][input_channels] of FP32 elements

    Returns
    -------
    C : Tensor
        lhs 1D array out[output_channels] of FP32 elements.
    """
    m = rhs.shape[0]
    return _api.extern(
        (m, ), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.fully_connected_inference",
            ins[0], ins[1], outs[0]), name="C")

def fully_connected_output(lhs, rhs):
    """Create an extern op that compute fully connected of 2D tensor lhs and
    2D tensor rhs with nnpack.

    Parameters
    ----------
    lhs : Tensor
        lhs 2D matrix input[batch_size][input_channels] of FP32 elements
    rhs : Tensor
        lhs 2D matrix kernel[output_channels][input_channels] of FP32 elements

    Returns
    -------
    C : Tensor
        lhs 2D array out[batch_size][output_channels] of FP32 elements.
    """
    n = lhs.shape[0]
    m = rhs.shape[0]
    return _api.extern(
        (n, m), [lhs, rhs],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.fully_connected_output",
            ins[0], ins[1], outs[0]), name="C")
