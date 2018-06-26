# pylint: disable=invalid-name
"""TVM operator for local response norm compute."""
from __future__ import absolute_import
import tvm
from .. import cpp

@tvm.target.generic_func
def lrn(data, size, axis=1, alpha=0.0001, beta=0.75, bias=2):
    """Perform the across channels local response normalisation
    on the input data.

    sum_sqr_up^i{x, y} = (bias+((alpha/size)* \
                                {sum_{j=max(0, i-size/2)}^{min(N-1,i+size/2)} \
                                     (data^j{x,y})^2}))^beta
    output^i{x, y} = data^i{x, y}/sum_sqr_up^i{x, y}
    N is the number for input channels

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    size : int
        normalisation window size

    axis : int
        input data layout channel axis
        default value is 1 for NCHW format

    bias : float
        offset to avoid dividing by 0

    alpha : float
        to be divided

    beta : float
        exponent

    Returns
    -------
    output : tvm.Tensor
        4-D output with same shape
    """
    return cpp.nn.lrn(data, size, axis, alpha, beta, bias)
