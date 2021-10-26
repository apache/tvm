
from .generic import *
from .. import op as _op

@conv1d_strategy.register("pulp")
def conv1d_strategy(attrs, inputs, out_type, target):
    """conv1d pulp strategy"""
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if layout == "NCW":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.pulp.conv1d_ncw_pulp),
            wrap_topi_schedule(topi.pulp.schedule_conv1d_ncw),
            name="conv1d_ncw.pulp"
        )
    elif layout == "NWC":
        if kernel_layout == "WIO":
            strategy.add_implementation(
                wrap_compute_conv1d(topi.pulp.conv1d_nwc_pulp),
                topi.pulp.schedule_conv1d_nwc,
                name="conv1d_nwc.pulp"
            )
        elif kernel_layout == "OWI":
            strategy.add_implementation(
                wrap_compute_conv1d(topi.pulp.conv1d_nwc_owi),
                topi.pulp.schedule_conv1d_nwc_owi,
                name="conv1d_nwc_owi.pulp"
            )
    else:
        raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy




@conv2d_strategy.register("pulp")
def conv2d_strategy(attrs, inputs, out_type, target):
    """conv2d pulp strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    (dilation_h, dilation_w) = dilation
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nchw),
                wrap_topi_schedule(topi.pulp.schedule_conv2d_nchw),
                name="conv2d_nchw.pulp",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO" or kernel_layout == "OHWI"
            if kernel_layout == "HWIO":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                    topi.pulp.schedule_conv2d_nhwc,
                    name="conv2d_nhwc.pulp",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_nhwc_ohwi),
                    topi.pulp.schedule_conv2d_nhwc_ohwi,
                    name="conv2d_nhwc_ohwi.pulp"
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_hwcn),
                wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
                name="conv2d_hwcn.generic",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {}".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.generic",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.generic",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                name="group_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy
