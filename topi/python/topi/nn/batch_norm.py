"""TVM operator batch normalization compute."""
from __future__ import absolute_import
import tvm
from .. import tag

@tvm.tag_scope(tag=tag.BROADCAST)
def batch_norm_inference(data, gamma, beta, moving_mean, moving_var, eps, fix_gamma):
    """Batch normalization inference operator in NCHW layout.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    gamma : tvm.Tensor
        1-D with shape [channel]

    beta : tvm.Tensor
        1-D with shape [channel]

    moving_mean : tvm.Tensor
        1-D with shape [channel]

    moving_var : tvm.Tensor
        1-D with shape [channel]

    eps : float
        Epsilon to prevent div 0.

    fix_gamma : boolean
        Fix gamma while training

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    mean : tvm.Tensor
        1-D with shape [channel]

    var : tvm.Tensor
        1-D with shape [channel]
    """
    assert len(data.shape) == 4, "only support 4-dim batch norm"
    batch, channel, height, width = data.shape
    if fix_gamma:
        out = tvm.compute((batch, channel, height, width), \
            lambda b, c, h, w: (data[b, c, h, w] - moving_mean[c]) / \
            tvm.intrin.sqrt(moving_var[c] + eps) + beta[c])
    else:
        out = tvm.compute((batch, channel, height, width), \
            lambda b, c, h, w: (data[b, c, h, w] - moving_mean[c]) / \
            tvm.intrin.sqrt(moving_var[c] + eps) * gamma[c] + beta[c])
    mean = tvm.compute((C, ), lambda c: moving_mean[c])
    var = tvm.compute((C, ), lambda c: moving_var[c])
    return [out, mean, var]
