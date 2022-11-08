"""Definition of x220 operator strategy."""
# Referenced: tvm/relay/op/strategy/generic.py

import logging

from tvm import topi
from .generic import *
from .. import op as _op

logger = logging.getLogger("strategy")


@max_pool2d_strategy.register("x220")
def max_pool2d_strategy_x220(attrs, inputs, out_type, target):
    """max_pool2d x220 strategy"""
    logger.warning("max_pool2d strategy optimized for x220")
    strategy = _op.OpStrategy()
    data = inputs
    dilation = get_const_tuple(attrs.dilation)
    layout = attrs.layout # data_layout
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_pool2d(topi.x220.max_pool2d_nchw, 'max', need_out_dtype=True),
            wrap_topi_schedule(topi.x220.schedule_pool_nchw),
            name="pool2d_nchw.x220",
        )
    else:
        raise RuntimeError("Unsupported max_pool2d layout {}".format(layout))
        # relay/op/nn/_nn.py::convert_max_pool2d(attrs, inputs, tinfos, desired_layouts)
    return strategy


@avg_pool2d_strategy.register("x220")
def avg_pool2d_strategy_x220(attrs, inputs, out_type, target):
    """avg_pool2d x220 strategy"""
    logger.warning("avg_pool2d strategy optimized for x220")
    strategy = _op.OpStrategy()
    data = inputs
    dilation = get_const_tuple(attrs.dilation)
    layout = attrs.layout # data_layout
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_pool2d(topi.x220.avg_pool2d_nchw, 'avg', need_out_dtype=True),
            wrap_topi_schedule(topi.x220.schedule_pool_nchw),
            name="pool2d_nchw.x220",
        )
    else:
        raise RuntimeError("Unsupported max_pool2d layout {}".format(layout))
        # relay/op/nn/_nn.py::convert_max_pool2d(attrs, inputs, tinfos, desired_layouts)
    return strategy


@schedule_injective.register("x220")
def schedule_injective_x220(attrs, outs, target):
    with target:
        return topi.x220.schedule_injective(outs)


@conv2d_strategy.register("x220")
def conv2d_strategy_x220(attrs, inputs, out_type, target):
    """conv2d x220 strategy"""
    logger.warning("conv2d optimized for x220")
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
