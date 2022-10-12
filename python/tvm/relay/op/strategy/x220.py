"""Definition of x220 operator strategy."""
# Referenced: tvm/relay/op/strategy/generic.py

import logging

from tvm import topi
from .generic import *
from .. import op as _op

logger = logging.getLogger("strategy")

@schedule_pool.register("x220")
def schedule_pool_x220(attrs, outs, target):
    with target:
        return topi.x220.schedule_pool(outs, attrs.layout)

@conv2d_strategy.register("x220")
def conv2d_strategy_x220(attrs, inputs, out_type, target):
    """conv2d x220 strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups): # ic == oc == groups
        assert data.shape[0] == 1 # Only supports when batch=1
        logger.warning("** depthwise_conv2d is optimized for x220 platform. **")
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.x220.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.x220.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.x220",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.x220.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.x220.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.x220",
            )
        else:
            raise RuntimeError("Currently NOT Supported layout: %s"%(layout))
    else:
        raise RuntimeError("x220 only supports depthwise_conv2d")
    return strategy
