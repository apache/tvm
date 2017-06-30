"""External function interface to NNPACK libraroes."""
from __future__ import absolute_import as _abs

from .. import api as _api
from .. import intrin as _intrin

def fully_connected_inference(A, B):
    """Create an extern op that compute fully connected of 1D tensor A and
    2D tensor B with nnpack.

    Parameters
    ----------
    A : Tensor
        A 1D array input[input_channels] of FP32 elements
    B : Tensor
        A 2D matrix kernel[output_channels][input_channels] of FP32 elements

    Returns
    -------
    C : Tensor
        A 1D array out[output_channels] of FP32 elements.
    """
    n = A.shape[0]
    m = B.shape[0]
    l = B.shape[1]
    return _api.extern(
        (m, ), [A, B],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.fully_connected_inference",
            ins[0], ins[1], outs[0]), name="C")

def fully_connected_output(A, B):
    """Create an extern op that compute fully connected of 2D tensor A and
    2D tensor B with nnpack.

    Parameters
    ----------
    A : Tensor
        A 2D matrix input[batch_size][input_channels] of FP32 elements
    B : Tensor
        A 2D matrix kernel[output_channels][input_channels] of FP32 elements

    Returns
    -------
    C : Tensor
        A 2D array out[batch_size][output_channels] of FP32 elements.
    """
    n = A.shape[0]
    m = B.shape[0]
    return _api.extern(
        (n, m), [A, B],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.nnpack.fully_connected_output",
            ins[0], ins[1], outs[0]), name="C")


