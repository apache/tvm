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
"""Definition of generic operator strategy."""
# pylint: disable=invalid-name,unused-argument
import logging
import re

from tvm import _ffi, ir, te, topi
from tvm.target import generic_func, override_native_generic_func
from tvm.topi.utils import get_const_float, get_const_int, get_const_tuple, get_float_tuple

from .. import op as _op

logger = logging.getLogger("strategy")


def naive_schedule(_, outs, target):
    """Return the naive default schedule.
    This function acts as a placeholder for op implementations that uses auto-scheduler.
    Implemenations using this function should only be used along with auto-scheduler.
    """
    if "gpu" in target.keys:
        # For GPU, we at least need thread binding to make a valid schedule.
        # So the naive schedule cannot be compiled.
        logger.debug(
            "Cannot compile for GPU targets if no tuned schedule is found. "
            "Please see the warning messages above for more information about the failed workloads."
        )
    return te.create_schedule(outs[-1].op)


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper


def wrap_topi_compute(topi_compute):
    """Wrap TOPI compute which doesn't use attrs"""

    def wrapper(attrs, inputs, out_type):
        return [topi_compute(*inputs)]

    return wrapper


def get_conv2d_in_channels(data_shape, data_layout):
    """Get conv2d input channels"""
    data_shape = get_const_tuple(data_shape)
    if len(data_shape) == 4:
        idx = data_layout.find("C")
        assert idx >= 0, "Invalid conv2d data layout {}".format(data_layout)
        return data_shape[idx]
    if re.match(r"NCHW\d*c", data_layout):
        # NCHW[8]c
        return data_shape[1] * data_shape[4]
    raise ValueError("Unknown conv2d data layout {}".format(data_layout))


def get_conv2d_out_channels(kernel_shape, kernel_layout):
    """Get conv2d output channels"""
    kernel_shape = get_const_tuple(kernel_shape)
    if len(kernel_shape) == 4:
        idx = kernel_layout.find("O")
        assert idx >= 0, "Invalid conv2d kernel layout {}".format(kernel_layout)
        return kernel_shape[idx]
    if re.match(r"OIHW\d*i\d*o", kernel_layout):
        return kernel_shape[0] * kernel_shape[5]
    if re.match(r"OIHW\d*o", kernel_layout):
        return kernel_shape[0] * kernel_shape[4]
    raise ValueError("Unknown conv2d kernel layout {}".format(kernel_layout))


def is_depthwise_conv2d(data_shape, data_layout, kernel_shape, kernel_layout, groups):
    ic = get_conv2d_in_channels(data_shape, data_layout)
    oc = get_conv2d_out_channels(kernel_shape, kernel_layout)
    return ic == oc == groups


@generic_func
def schedule_injective(attrs, outs, target):
    """Schedule injective ops"""
    with target:
        return topi.generic.schedule_injective(outs)


@generic_func
def schedule_reduce(attrs, outs, target):
    """Schedule reduction ops"""
    with target:
        return topi.generic.schedule_reduce(outs)


_op._schedule_injective = schedule_injective
_op._schedule_reduce = schedule_reduce

# concatenate
@generic_func
def schedule_concatenate(attrs, outs, target):
    """Schedule concatenate op"""
    with target:
        return topi.generic.schedule_injective(outs)


# pool
@generic_func
def schedule_pool(attrs, outs, target):
    """Schedule pooling ops"""
    with target:
        return topi.generic.schedule_pool(outs, attrs.layout)


# pool_grad
@generic_func
def schedule_pool_grad(attrs, outs, target):
    """Schedule pooling gradient ops"""
    with target:
        return topi.generic.schedule_pool_grad(outs)


# adaptive pool
@generic_func
def schedule_adaptive_pool(attrs, outs, target):
    """Schedule adaptive pooling ops"""
    with target:
        return topi.generic.schedule_adaptive_pool(outs)


# softmax
def wrap_compute_softmax(topi_compute):
    """Wrap softmax topi compute"""

    def _compute_softmax(attrs, inputs, out_type):
        axis = attrs.get_int("axis")
        return [topi_compute(inputs[0], axis)]

    return _compute_softmax


@override_native_generic_func("softmax_strategy")
def softmax_strategy(attrs, inputs, out_type, target):
    """softmax generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.generic.schedule_softmax),
        name="softmax.generic",
    )
    return strategy


@override_native_generic_func("fast_softmax_strategy")
def fast_softmax_strategy(attrs, inputs, out_type, target):
    """fast softmax generic strategy"""
    # NOTE: This op does not have an optimized manual schedule,
    # so it should only be used together with auto-scheduler.
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.fast_softmax),
        wrap_topi_schedule(topi.generic.schedule_fast_softmax),
        name="fast_softmax.generic",
    )
    return strategy


@override_native_generic_func("log_softmax_strategy")
def log_softmax_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.log_softmax),
        wrap_topi_schedule(topi.generic.schedule_softmax),
        name="log_softmax.generic",
    )
    return strategy


# lrn
@generic_func
def schedule_lrn(attrs, outs, target):
    """Schedule LRN op"""
    with target:
        return topi.generic.schedule_lrn(outs)


# bitpack
@generic_func
def schedule_bitpack(attrs, outs, target):
    """Schedule bitpack"""
    with target:
        return topi.generic.schedule_bitpack(outs)


get_auto_scheduler_rewritten_layout = _ffi.get_global_func(
    "relay.attrs.get_auto_scheduler_rewritten_layout"
)

# conv2d
def wrap_compute_conv2d(
    topi_compute,
    need_data_layout=False,
    need_out_layout=False,
    has_groups=False,
    need_auto_scheduler_layout=False,
):
    """Wrap conv2d topi compute"""

    def _compute_conv2d(attrs, inputs, out_type):
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        dilation = get_const_tuple(attrs.dilation)
        data_layout = attrs.get_str("data_layout")
        out_layout = attrs.get_str("out_layout")
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        args = [inputs[0], inputs[1], strides, padding, dilation]
        if has_groups:
            args.append(attrs.groups)
        if need_data_layout:
            args.append(data_layout)
        if need_out_layout:
            args.append(out_layout)
        args.append(out_dtype)
        if need_auto_scheduler_layout:
            args.append(get_auto_scheduler_rewritten_layout(attrs))
        return [topi_compute(*args)]

    return _compute_conv2d


@override_native_generic_func("conv2d_strategy")
def conv2d_strategy(attrs, inputs, out_type, target):
    """conv2d generic strategy"""
    logger.warning("conv2d is not optimized for this platform.")
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
                wrap_topi_schedule(topi.generic.schedule_conv2d_nchw),
                name="conv2d_nchw.generic",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                wrap_topi_schedule(topi.generic.schedule_conv2d_nhwc),
                name="conv2d_nhwc.generic",
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


# conv2d_NCHWc
@override_native_generic_func("conv2d_NCHWc_strategy")
def conv2d_NCHWc_strategy(attrs, inputs, out_type, target):
    """conv2d_NCHWc generic strategy"""
    logger.warning("conv2d_NCHWc is not optimized for this platform.")
    strategy = _op.OpStrategy()
    if inputs[0].dtype == "int8" or inputs[0].dtype == "uint8":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_NCHWc_int8, True, True),
            wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc_int8),
            name="conv2d_NCHWc_int8.generic",
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_NCHWc, True, True),
            wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.generic",
        )
    return strategy


# depthwise_conv2d_NCHWc
@override_native_generic_func("depthwise_conv2d_NCHWc_strategy")
def depthwise_conv2d_NCHWc_strategy(attrs, inputs, out_type, target):
    """depthwise_conv2d generic strategy"""
    logger.warning("depthwise_conv2d_NCHWc is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.nn.depthwise_conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_NCHWc),
        name="depthwise_conv2d_NCHWc.generic",
    )
    return strategy


# conv2d_winograd_without_weight_transform
@override_native_generic_func("conv2d_winograd_without_weight_transform_strategy")
def conv2d_winograd_without_weight_transfrom_strategy(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom generic strategy"""
    raise ValueError("No generic implemenation for conv2d_winograd_without_weight_transform")


# conv2d_gemm_without_weight_transform
@override_native_generic_func("conv2d_gemm_without_weight_transform_strategy")
def conv2d_gemm_without_weight_transform_strategy(attrs, inputs, out_type, target):
    """conv2d_gemm_without_weight_transfrom generic strategy"""
    raise ValueError("No generic implemenation for conv2d_gemm_without_weight_transform")


# conv2d_winograd_weight_transform
@generic_func
def schedule_conv2d_winograd_weight_transform(attrs, outs, target):
    """Schedule conv2d_winograd_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_winograd_weight_transform(outs)


# conv2d_winograd_nnpack_weight_transform
@generic_func
def schedule_conv2d_winograd_nnpack_weight_transform(attrs, outs, target):
    """Schedule conv2d_winograd_nnpack_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_winograd_nnpack_weight_transform(outs)


# conv2d_gemm_weight_transform
@generic_func
def schedule_conv2d_gemm_weight_transform(attrs, outs, target):
    """Schedule conv2d_gemm_weight_transform"""
    with target:
        return topi.generic.schedule_conv2d_gemm_weight_transform(outs)


# deformable_conv2d
def wrap_compute_deformable_conv2d(topi_compute):
    """wrap deformable_conv2d topi compute"""

    def _compute_deformable_conv2d(attrs, inputs, out_dtype):
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        dilation = get_const_tuple(attrs.dilation)
        deformable_groups = attrs.deformable_groups
        groups = attrs.groups
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        out = topi_compute(
            inputs[0],
            inputs[1],
            inputs[2],
            strides,
            padding,
            dilation,
            deformable_groups,
            groups,
            out_dtype,
        )
        return [out]

    return _compute_deformable_conv2d


@override_native_generic_func("deformable_conv2d_strategy")
def deformable_conv2d_strategy(attrs, inputs, out_type, target):
    """deformable_conv2d generic strategy"""
    layout = attrs.data_layout
    strategy = _op.OpStrategy()

    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_deformable_conv2d(topi.nn.deformable_conv2d_nchw),
            wrap_topi_schedule(topi.generic.schedule_deformable_conv2d_nchw),
            name="deformable_conv2d_nchw.generic",
        )
    elif layout == "NHWC":
        # This implementation should never be picked by autotvm
        strategy.add_implementation(
            wrap_compute_deformable_conv2d(topi.nn.deformable_conv2d_nhwc),
            naive_schedule,
            name="deformable_conv2d_nhwc.generic",
        )
    else:
        raise RuntimeError("Layout %s is not supported in deformable conv2d" % layout)
    return strategy


# conv2d_transpose
def wrap_compute_conv2d_transpose(topi_compute, has_groups=False):
    """wrap conv2d_transpose topi compute"""

    def compute_conv2d_transpose(attrs, inputs, out_dtype):
        """Compute definition of conv2d_transpose"""
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        output_padding = get_const_tuple(attrs.output_padding)
        # out = topi_compute(inputs[0], inputs[1], strides, padding, out_dtype, output_padding)
        args = [inputs[0], inputs[1], strides, padding, out_dtype, output_padding]
        if has_groups:
            args.append(attrs.groups)
        out = topi_compute(*args)
        return [out]

    return compute_conv2d_transpose


@override_native_generic_func("conv2d_transpose_strategy")
def conv2d_transpose_strategy(attrs, inputs, out_type, target):
    """conv2d_transpose generic strategy"""
    logger.warning("conv2d_transpose is not optimized for this platform.")
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    strategy = _op.OpStrategy()
    if groups == 1:
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.nn.conv2d_transpose_nchw),
            wrap_topi_schedule(topi.generic.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.generic",
        )
    else:  # group_transpose_conv2d
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.nn.group_conv2d_transpose_nchw, has_groups=True),
            wrap_topi_schedule(topi.generic.schedule_group_conv2d_transpose_nchw),
            name="group_conv2d_transpose_nchw.generic",
        )
    return strategy


# conv3d_transpose
def wrap_compute_conv3d_transpose(topi_compute):
    """wrap conv3d_transpose topi compute"""

    def compute_conv3d_transpose(attrs, inputs, out_dtype):
        """Compute definition of conv3d_transpose"""
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        output_padding = get_const_tuple(attrs.output_padding)
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        out = topi_compute(inputs[0], inputs[1], strides, padding, out_dtype, output_padding)
        return [out]

    return compute_conv3d_transpose


@override_native_generic_func("conv3d_transpose_strategy")
def conv3d_transpose_strategy(attrs, inputs, out_type, target):
    """conv3d_transpose generic strategy"""
    logger.warning("conv3d_transpose is not optimized for this platform.")
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCDHW", "only support ncdhw for now"
    assert dilation == (1, 1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv3d_transpose(topi.nn.conv3d_transpose_ncdhw),
        wrap_topi_schedule(topi.generic.schedule_conv3d_transpose_ncdhw),
        name="conv3d_transpose_ncdhw.generic",
    )
    return strategy


# conv3d
def wrap_compute_conv3d(topi_compute, need_layout=False, need_auto_scheduler_layout=False):
    """wrap conv3d topi compute"""

    def _compute_conv3d(attrs, inputs, out_type):
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        dilation = get_const_tuple(attrs.dilation)
        groups = attrs.groups
        layout = attrs.data_layout
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype

        (dilation_d, dilation_h, dilation_w) = dilation
        if dilation_d < 1 or dilation_h < 1 or dilation_w < 1:
            raise ValueError("Dilation should be positive value")
        if groups != 1:
            raise ValueError("Not support arbitrary group number for conv3d")

        args = [inputs[0], inputs[1], strides, padding, dilation]
        if need_layout:
            args.append(layout)
        args.append(out_dtype)
        if need_auto_scheduler_layout:
            args.append(get_auto_scheduler_rewritten_layout(attrs))
        return [topi_compute(*args)]

    return _compute_conv3d


@override_native_generic_func("conv3d_strategy")
def conv3d_strategy(attrs, inputs, out_type, target):
    """conv3d generic strategy"""
    logger.warning("conv3d is not optimized for this platform.")
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCDHW":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.nn.conv3d_ncdhw),
            wrap_topi_schedule(topi.generic.schedule_conv3d_ncdhw),
            name="conv3d_ncdhw.generic",
        )
    elif layout == "NDHWC":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.nn.conv3d_ndhwc),
            wrap_topi_schedule(topi.generic.schedule_conv3d_ndhwc),
            name="conv3d_ndhwc.generic",
        )
    else:
        raise ValueError("Not support this layout {} yet".format(layout))
    return strategy


# conv3d_winograd_without_weight_transform
@override_native_generic_func("conv3d_winograd_without_weight_transform_strategy")
def conv3d_winograd_without_weight_transfrom_strategy(attrs, inputs, out_type, target):
    """conv3d_winograd_without_weight_transfrom generic strategy"""
    raise ValueError("No generic implemenation for conv3d_winograd_without_weight_transform")


# conv3d_winograd_weight_transform
@generic_func
def schedule_conv3d_winograd_weight_transform(attrs, outs, target):
    """Schedule conv3d_winograd_weight_transform"""
    with target:
        return topi.generic.schedule_conv3d_winograd_weight_transform(outs)


# conv1d
def wrap_compute_conv1d(topi_compute):
    """wrap conv1d topi compute"""

    def _compute_conv1d(attrs, inputs, out_type):
        """Compute definition of conv1d"""
        strides = get_const_tuple(attrs.strides)
        padding = get_const_tuple(attrs.padding)
        dilation = get_const_tuple(attrs.dilation)
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        return [topi_compute(inputs[0], inputs[1], strides, padding, dilation, out_dtype)]

    return _compute_conv1d


@override_native_generic_func("conv1d_strategy")
def conv1d_strategy(attrs, inputs, out_type, target):
    """conv1d generic strategy"""
    logger.warning("conv1d is not optimized for this platform.")
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if layout == "NCW":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.nn.conv1d_ncw),
            wrap_topi_schedule(topi.generic.schedule_conv1d_ncw),
            name="conv1d_ncw.generic",
        )
    elif layout == "NWC":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.nn.conv1d_nwc),
            wrap_topi_schedule(topi.generic.schedule_conv1d_nwc),
            name="conv1d_nwc.generic",
        )
    else:
        raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy


# conv1d_transpose
def wrap_compute_conv1d_transpose(topi_compute):
    """wrap conv1d_transpose topi compute"""

    def _compute_conv1d_tranpsoe(attrs, inputs, out_type):
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        output_padding = get_const_tuple(attrs.output_padding)
        out = topi_compute(inputs[0], inputs[1], strides, padding, out_dtype, output_padding)
        return [out]

    return _compute_conv1d_tranpsoe


@override_native_generic_func("conv1d_transpose_strategy")
def conv1d_transpose_strategy(attrs, inputs, out_type, target):
    """conv1d_transpose generic strategy"""
    logger.warning("conv1d_transpose is not optimized for this platform.")
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCW", "conv1d_transpose ncw only supported"
    assert dilation == (1,), "conv1d_transpose dilation is not supported"
    assert groups == 1, "conv1d_transpose groups == 1 only supported"
    strategy.add_implementation(
        wrap_compute_conv1d_transpose(topi.nn.conv1d_transpose_ncw),
        wrap_topi_schedule(topi.generic.schedule_conv1d_transpose_ncw),
        name="conv1d_transpose_ncw.generic",
    )
    return strategy


# dilation2d
def wrap_compute_dilation2d(topi_compute, need_data_layout=False):
    """Wrap dilation2d topi compute"""

    def _compute_dilation2d(attrs, inputs, out_type):
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        dilations = get_const_tuple(attrs.dilations)
        data_layout = attrs.get_str("data_layout")
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype in ("same", "") else out_dtype
        args = [inputs[0], inputs[1], strides, padding, dilations]
        if need_data_layout:
            args.append(data_layout)
        args.append(out_dtype)
        return [topi_compute(*args)]

    return _compute_dilation2d


@override_native_generic_func("dilation2d_strategy")
def dilation2d_strategy(attrs, inputs, out_type, target):
    """dilation2d_strategy generic strategy"""
    logger.warning("dilation2d_strategy is not optimized for this platform.")
    strategy = _op.OpStrategy()
    dilations = get_const_tuple(attrs.dilations)
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout

    assert layout in ["NCHW", "NHWC"]
    (dilation_h, dilation_w) = dilations
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if layout == "NCHW":
        assert kernel_layout == "IHW"
        strategy.add_implementation(
            wrap_compute_dilation2d(topi.image.dilation2d_nchw),
            wrap_topi_schedule(topi.generic.schedule_dilation2d_nchw),
            name="dilation2d_nchw.generic",
        )
    elif layout == "NHWC":
        assert kernel_layout == "HWI"
        strategy.add_implementation(
            wrap_compute_dilation2d(topi.image.dilation2d_nhwc),
            wrap_topi_schedule(topi.generic.schedule_dilation2d_nhwc),
            name="dilation2d_nhwc.generic",
        )
    else:
        raise RuntimeError("Unsupported dilation2d layout {}".format(layout))
    return strategy


def copy_if_identical(tensor_a, tensor_b):
    """
    When two inputs to batch_matul or dense are the same tensor, e.g. batch_matmul(x, x),
    compilation fails because TE thinks there is only one input tensor x, and doing
    cache_read(x) on the same tensor twice results in an error.
    To prevent such errors, we make the second tensor be the copy of the first one
    when two input tensors are identical.
    """
    if tensor_a == tensor_b:
        return te.compute(tensor_a.shape, lambda *ind: tensor_a[ind])
    return tensor_b


# matmul
def wrap_compute_matmul(topi_compute, need_auto_scheduler_layout=False):
    """wrap matmul topi compute"""

    def _compute_matmul(attrs, inputs, out_type):
        """Compute definition of matmul"""
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype
        args = [
            inputs[0],
            inputs[1],
            None,
            out_dtype,
            attrs.transpose_a,
            attrs.transpose_b,
        ]
        if need_auto_scheduler_layout:
            args.append(get_auto_scheduler_rewritten_layout(attrs))
        args[1] = copy_if_identical(inputs[0], inputs[1])
        return [topi_compute(*args)]

    return _compute_matmul


@override_native_generic_func("matmul_strategy")
def matmul_strategy(attrs, inputs, out_type, target):
    """matmul generic strategy"""
    logger.warning("matmul is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_matmul(topi.nn.matmul),
        wrap_topi_schedule(topi.generic.schedule_matmul),
        name="matmul.generic",
    )
    return strategy


# dense
def wrap_compute_dense(topi_compute, need_auto_scheduler_layout=False):
    """wrap dense topi compute"""

    def _compute_dense(attrs, inputs, out_type):
        """Compute definition of dense"""
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype
        args = [inputs[0], inputs[1], None, out_dtype]
        if need_auto_scheduler_layout:
            args.append(get_auto_scheduler_rewritten_layout(attrs))
        args[1] = copy_if_identical(inputs[0], inputs[1])
        return [topi_compute(*args)]

    return _compute_dense


@override_native_generic_func("dense_strategy")
def dense_strategy(attrs, inputs, out_type, target):
    """dense generic strategy"""
    logger.warning("dense is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.nn.dense),
        wrap_topi_schedule(topi.generic.schedule_dense),
        name="dense.generic",
    )
    return strategy


@override_native_generic_func("dense_pack_strategy")
def dense_pack_strategy(attrs, inputs, out_type, target):
    """dense_pack generic strategy"""
    logger.warning("dense_pack is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.nn.dense_pack),
        wrap_topi_schedule(topi.generic.schedule_dense),
        name="dense_pack.generic",
    )
    return strategy


# batch_matmul
def wrap_compute_batch_matmul(topi_compute, need_auto_scheduler_layout=False, need_out_dtype=False):
    """wrap batch_matmul topi compute"""

    def _compute_batch_matmul(attrs, inputs, out_type):
        args = [inputs[0], inputs[1], out_type.shape]
        args.append(out_type.dtype if need_out_dtype else None)
        args.append(attrs.transpose_a)
        args.append(attrs.transpose_b)
        if need_auto_scheduler_layout:
            args.append(get_auto_scheduler_rewritten_layout(attrs))
        args[1] = copy_if_identical(inputs[0], inputs[1])
        return [topi_compute(*args)]

    return _compute_batch_matmul


@override_native_generic_func("batch_matmul_strategy")
def batch_matmul_strategy(attrs, inputs, out_type, target):
    """batch_matmul generic strategy"""
    logger.warning("batch_matmul is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.nn.batch_matmul),
        wrap_topi_schedule(topi.generic.schedule_batch_matmul),
        name="batch_matmul.generic",
    )
    return strategy


# sparse dense
def wrap_compute_sparse_dense(topi_compute):
    """wrap sparse dense topi compute"""

    def _compute_sparse_dense(attrs, inputs, out_type):
        return [topi_compute(inputs[0], inputs[1], inputs[2], inputs[3], attrs["sparse_lhs"])]

    return _compute_sparse_dense


@override_native_generic_func("sparse_dense_strategy")
def sparse_dense_strategy(attrs, inputs, out_type, target):
    """sparse dense generic strategy"""
    logger.warning("sparse dense is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_dense(topi.nn.sparse_dense),
        wrap_topi_schedule(topi.generic.schedule_sparse_dense),
        name="sparse_dense.generic",
    )
    return strategy


@override_native_generic_func("sparse_dense_padded_strategy")
def sparse_dense_padded_strategy(attrs, inputs, out_type, target):
    """sparse dense padded generic strategy"""
    raise NotImplementedError("sparse_dense_padded is only implemented for cuda")


# sparse_add
def wrap_compute_sparse_add(topi_compute):
    """wrap sparse add topi compute"""

    def _compute_sparse_add(attrs, inputs, out_type):
        return [topi_compute(inputs[0], inputs[1], inputs[2], inputs[3])]

    return _compute_sparse_add


@override_native_generic_func("sparse_add_strategy")
def sparse_add_strategy(attrs, inputs, out_type, target):
    """sparse add generic strategy"""
    logger.warning("sparse add is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_add(topi.nn.sparse_add),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="sparse_add.generic",
    )
    return strategy


# sparse_transpose
@generic_func
def schedule_sparse_transpose(attrs, outs, target):
    """schedule sparse_transpose"""
    with target:
        return topi.generic.schedule_sparse_transpose(outs)


# sparse conv2d
def wrap_compute_sparse_conv2d(topi_compute):
    """wrap sparse conv2d topi compute"""

    def _compute_sparse_conv2d(attrs, inputs, out_type):
        return [topi_compute(inputs[0], inputs[1], inputs[2], inputs[3], attrs["layout"])]

    return _compute_sparse_conv2d


@override_native_generic_func("sparse_conv2d_strategy")
def sparse_conv2d_strategy(attrs, inputs, out_type, target):
    """sparse conv2d generic strategy"""
    logger.warning("sparse conv2d is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_conv2d(topi.nn.sparse_conv2d),
        wrap_topi_schedule(topi.generic.schedule_sparse_conv2d),
        name="sparse_conv2d.generic",
    )
    return strategy


# sort
def wrap_compute_sort(topi_compute):
    """Wrap sort topi compute"""

    def _compute_sort(attrs, inputs, _):
        axis = get_const_int(attrs.axis)
        is_ascend = bool(get_const_int(attrs.is_ascend))
        return [topi_compute(inputs[0], axis=axis, is_ascend=is_ascend)]

    return _compute_sort


@override_native_generic_func("sort_strategy")
def sort_strategy(attrs, inputs, out_type, target):
    """sort generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sort(topi.sort),
        wrap_topi_schedule(topi.generic.schedule_sort),
        name="sort.generic",
    )
    return strategy


# argsort
def wrap_compute_argsort(topi_compute):
    """Wrap argsort topi compute"""

    def _compute_argsort(attrs, inputs, _):
        axis = get_const_int(attrs.axis)
        is_ascend = bool(get_const_int(attrs.is_ascend))
        dtype = attrs.dtype
        return [topi_compute(inputs[0], axis=axis, is_ascend=is_ascend, dtype=dtype)]

    return _compute_argsort


@override_native_generic_func("argsort_strategy")
def argsort_strategy(attrs, inputs, out_type, target):
    """argsort generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argsort(topi.argsort),
        wrap_topi_schedule(topi.generic.schedule_argsort),
        name="argsort.generic",
    )
    return strategy


# topk
def wrap_compute_topk(topi_compute):
    """Wrap topk compute"""

    def _compute_topk(attrs, inputs, out_type):
        if attrs.k is not None:
            k = attrs.k
        else:
            k = inputs[1]
        axis = get_const_int(attrs.axis)
        ret_type = attrs.ret_type
        is_ascend = bool(get_const_int(attrs.is_ascend))
        dtype = attrs.dtype
        out = topi_compute(inputs[0], k, axis, ret_type, is_ascend, dtype)
        out = out if isinstance(out, list) else [out]
        return out

    return _compute_topk


@override_native_generic_func("topk_strategy")
def topk_strategy(attrs, inputs, out_type, target):
    """topk generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_topk(topi.topk),
        wrap_topi_schedule(topi.generic.schedule_topk),
        name="topk.generic",
    )
    return strategy


# searchsorted
def wrap_compute_searchsorted(topi_compute):
    """Wrap searchsorted compute"""

    def _compute_searchsorted(attrs, inputs, out_type):
        right = attrs.right
        dtype = attrs.dtype
        return [topi_compute(inputs[0], inputs[1], right, dtype)]

    return _compute_searchsorted


# searchsorted_strategy
@override_native_generic_func("searchsorted_strategy")
def searchsorted_strategy(attrs, inputs, out_type, target):
    """searchsorted generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_searchsorted(topi.searchsorted),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="searchsorted.generic",
    )
    return strategy


# multibox_prior
def wrap_compute_multibox_prior(topi_compute):
    """Wrap multibox_prior compute"""

    def _compute_multibox_prior(attrs, inputs, _):
        """Compute definition of multibox_prior"""
        sizes = get_float_tuple(attrs.sizes)
        ratios = get_float_tuple(attrs.ratios)
        steps = get_float_tuple(attrs.steps)
        offsets = get_float_tuple(attrs.offsets)
        clip = bool(get_const_int(attrs.clip))
        return [topi_compute(inputs[0], sizes, ratios, steps, offsets, clip)]

    return _compute_multibox_prior


@override_native_generic_func("multibox_prior_strategy")
def multibox_prior_strategy(attrs, inputs, out_type, target):
    """multibox_prior generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_multibox_prior(topi.vision.ssd.multibox_prior),
        wrap_topi_schedule(topi.generic.schedule_multibox_prior),
        name="multibox_prior.generic",
    )
    return strategy


# multibox_transform_loc
def wrap_compute_multibox_transform_loc(topi_compute):
    """Wrap multibox_transform_loc compute"""

    def _compute_multibox_transform_loc(attrs, inputs, _):
        """Compute definition of multibox_detection"""
        clip = bool(get_const_int(attrs.clip))
        threshold = get_const_float(attrs.threshold)
        variances = get_float_tuple(attrs.variances)
        return topi_compute(inputs[0], inputs[1], inputs[2], clip, threshold, variances)

    return _compute_multibox_transform_loc


@override_native_generic_func("multibox_transform_loc_strategy")
def multibox_transform_loc_strategy(attrs, inputs, out_type, target):
    """schedule multibox_transform_loc"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_multibox_transform_loc(topi.vision.ssd.multibox_transform_loc),
        wrap_topi_schedule(topi.generic.schedule_multibox_transform_loc),
        name="multibox_transform_loc.generic",
    )
    return strategy


# get_valid_counts
def wrap_compute_get_valid_counts(topi_compute):
    """wrap get_valid_counts topi compute"""

    def _compute_get_valid_counts(attrs, inputs, out_type):
        score_threshold = inputs[1]
        id_index = get_const_int(attrs.id_index)
        score_index = get_const_int(attrs.score_index)
        if attrs.score_threshold is not None:
            score_threshold = get_const_float(attrs.score_threshold)
        return topi_compute(inputs[0], score_threshold, id_index, score_index)

    return _compute_get_valid_counts


@override_native_generic_func("get_valid_counts_strategy")
def get_valid_counts_strategy(attrs, inputs, out_type, target):
    """get_valid_counts generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_get_valid_counts(topi.vision.get_valid_counts),
        wrap_topi_schedule(topi.generic.schedule_get_valid_counts),
        name="get_valid_counts.generic",
    )
    return strategy


# non-maximum suppression
def wrap_compute_nms(topi_compute):
    """wrap nms topi compute"""

    def _compute_nms(attrs, inputs, out_type):
        max_output_size = inputs[3]
        iou_threshold = inputs[4]
        return_indices = bool(get_const_int(attrs.return_indices))
        force_suppress = bool(get_const_int(attrs.force_suppress))
        top_k = get_const_int(attrs.top_k)
        coord_start = get_const_int(attrs.coord_start)
        score_index = get_const_int(attrs.score_index)
        id_index = get_const_int(attrs.id_index)
        invalid_to_bottom = bool(get_const_int(attrs.invalid_to_bottom))
        if return_indices:
            return topi_compute(
                inputs[0],
                inputs[1],
                inputs[2],
                max_output_size,
                iou_threshold,
                force_suppress,
                top_k,
                coord_start,
                score_index,
                id_index,
                return_indices,
                invalid_to_bottom,
            )
        return [
            topi_compute(
                inputs[0],
                inputs[1],
                inputs[2],
                max_output_size,
                iou_threshold,
                force_suppress,
                top_k,
                coord_start,
                score_index,
                id_index,
                return_indices,
                invalid_to_bottom,
            )
        ]

    return _compute_nms


@override_native_generic_func("non_max_suppression_strategy")
def nms_strategy(attrs, inputs, out_type, target):
    """nms generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_nms(topi.vision.non_max_suppression),
        wrap_topi_schedule(topi.generic.schedule_nms),
        name="nms.generic",
    )
    return strategy


def wrap_compute_all_class_nms(topi_compute):
    """wrap all class nms topi compute"""

    def _compute_nms(attrs, inputs, out_type):
        max_output_size = inputs[2]
        iou_threshold = inputs[3]
        score_threshold = inputs[4]
        output_format = attrs.output_format
        return topi_compute(
            inputs[0],
            inputs[1],
            max_output_size,
            iou_threshold,
            score_threshold,
            output_format,
        )

    return _compute_nms


@override_native_generic_func("all_class_non_max_suppression_strategy")
def all_class_nms_strategy(attrs, inputs, out_type, target):
    """all class nms generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_all_class_nms(topi.vision.all_class_non_max_suppression),
        wrap_topi_schedule(topi.generic.schedule_nms),
        name="all_class_nms.generic",
    )
    return strategy


# roi_align
def wrap_compute_roi_align(topi_compute):
    """wrap roi_align topi compute"""

    def _compute_roi_align(attrs, inputs, out_type):
        pooled_size = get_const_tuple(attrs.pooled_size)
        mode = bytes(attrs.mode, "utf-8")
        return [
            topi_compute(
                inputs[0],
                inputs[1],
                pooled_size=pooled_size,
                spatial_scale=attrs.spatial_scale,
                sample_ratio=attrs.sample_ratio,
                mode=mode,
            )
        ]

    return _compute_roi_align


@override_native_generic_func("roi_align_strategy")
def roi_align_strategy(attrs, inputs, out_type, target):
    """roi_align generic strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_roi_align(topi.vision.rcnn.roi_align_nchw),
            wrap_topi_schedule(topi.generic.schedule_roi_align),
            name="roi_align.generic",
        )
    else:
        assert layout == "NHWC", "layout must be NCHW or NHWC."
        strategy.add_implementation(
            wrap_compute_roi_align(topi.vision.rcnn.roi_align_nhwc),
            wrap_topi_schedule(topi.generic.schedule_roi_align),
            name="roi_align.generic",
        )
    return strategy


# sparse_fill_empty_rows
@override_native_generic_func("sparse_fill_empty_rows_strategy")
def sparse_fill_empty_rows_strategy(attrs, outs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_fill_empty_rows(topi.sparse_fill_empty_rows),
        wrap_topi_schedule(topi.generic.schedule_sparse_fill_empty_rows),
        name="sparse_fill_empty_rows.generic",
    )
    return strategy


def wrap_compute_sparse_fill_empty_rows(topi_compute):
    """Wrap sparse_fill_empty_rows compute"""

    def _compute_sparse_fill_empty_rows(attrs, inputs, output_type):
        return topi_compute(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            output_type.fields[0].shape,
            output_type.fields[1].shape,
            output_type.fields[2].shape,
        )

    return _compute_sparse_fill_empty_rows


# sparse_reshape
@override_native_generic_func("sparse_reshape_strategy")
def sparse_reshape_strategy(attrs, outs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_reshape(topi.sparse_reshape),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="sparse_reshape.generic",
    )
    return strategy


def wrap_compute_sparse_reshape(topi_compute):
    """Wrap sparse_reshape compute"""

    def _compute_sparse_reshape(attrs, inputs, output_type):
        return topi_compute(
            inputs[0],
            inputs[1],
            inputs[2],
            output_type.fields[0].shape,
            output_type.fields[1].shape,
        )

    return _compute_sparse_reshape


# roi_pool
@generic_func
def schedule_roi_pool(attrs, outs, target):
    """schedule roi_pool"""
    with target:
        return topi.generic.schedule_roi_pool(outs)


# proposal
def wrap_compute_proposal(topi_compute):
    """wrap proposal topi compute"""

    def _compute_proposal(attrs, inputs, out_type):
        scales = get_float_tuple(attrs.scales)
        ratios = get_float_tuple(attrs.ratios)
        feature_stride = attrs.feature_stride
        threshold = attrs.threshold
        rpn_pre_nms_top_n = attrs.rpn_pre_nms_top_n
        rpn_post_nms_top_n = attrs.rpn_post_nms_top_n
        rpn_min_size = attrs.rpn_min_size
        iou_loss = bool(get_const_int(attrs.iou_loss))
        return [
            topi_compute(
                inputs[0],
                inputs[1],
                inputs[2],
                scales,
                ratios,
                feature_stride,
                threshold,
                rpn_pre_nms_top_n,
                rpn_post_nms_top_n,
                rpn_min_size,
                iou_loss,
            )
        ]

    return _compute_proposal


@override_native_generic_func("proposal_strategy")
def proposal_strategy(attrs, inputs, out_type, target):
    """proposal generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_proposal(topi.vision.rcnn.proposal),
        wrap_topi_schedule(topi.generic.schedule_proposal),
        name="proposal.generic",
    )
    return strategy


# scatter
@override_native_generic_func("scatter_strategy")
def scatter_strategy(attrs, outs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter(topi.scatter),
        wrap_topi_schedule(topi.generic.schedule_scatter),
        name="scatter.generic",
    )
    return strategy


def wrap_compute_scatter(topi_compute):
    """Wrap scatter topi compute"""

    def _compute_scatter(attrs, inputs, _):
        return [topi_compute(inputs[0], inputs[1], inputs[2], attrs.axis)]

    return _compute_scatter


@override_native_generic_func("scatter_add_strategy")
def scatter_add_strategy(attrs, outs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter(topi.scatter_add),
        wrap_topi_schedule(topi.generic.schedule_scatter),
        name="scatter_add.generic",
    )
    return strategy


# scatter_nd
@override_native_generic_func("scatter_nd_strategy")
def scatter_nd_strategy(attrs, inputs, out_type, target):
    """scatter_nd generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scatter_nd(topi.scatter_nd),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="scatter_nd.generic",
    )
    return strategy


def wrap_compute_scatter_nd(topi_compute):
    """Wrap scatter_nd topi compute"""

    def _compute_scatter_nd(attrs, inputs, _):
        return [topi_compute(inputs[0], inputs[1], inputs[2], attrs.mode)]

    return _compute_scatter_nd


# bitserial_conv2d
def wrap_compute_bitserial_conv2d(topi_compute):
    """wrap bitserial_conv2d topi compute"""

    def compute_bitserial_conv2d(attrs, inputs, out_dtype):
        """Compute definition for bitserial conv2d."""
        padding = get_const_tuple(attrs.padding)
        strides = get_const_tuple(attrs.strides)
        activation_bits = attrs.activation_bits
        weight_bits = attrs.weight_bits
        pack_dtype = attrs.pack_dtype
        out_dtype = attrs.out_dtype
        unipolar = attrs.unipolar
        return [
            topi_compute(
                inputs[0],
                inputs[1],
                strides,
                padding,
                activation_bits,
                weight_bits,
                pack_dtype,
                out_dtype,
                unipolar,
            )
        ]

    return compute_bitserial_conv2d


@override_native_generic_func("bitserial_conv2d_strategy")
def bitserial_conv2d_strategy(attrs, inputs, out_type, target):
    """bitserial_conv2d generic strategy"""
    logger.warning("bitserial_conv2d is not optimized for this platform.")
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.nn.bitserial_conv2d_nchw),
            wrap_topi_schedule(topi.generic.schedule_bitserial_conv2d_nchw),
            name="bitserial_conv2d_nchw.generic",
        )
    elif layout == "NHWC":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.nn.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.generic.schedule_bitserial_conv2d_nhwc),
            name="bitserial_conv2d_nhwc.generic",
        )
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy


# bitserial_dense
def wrap_compute_bitserial_dense(topi_compute):
    """wrap bitserial_dense topi compute"""

    def compute_bitserial_dense(attrs, inputs, out_type):
        """Compute definition of bitserial dense"""
        data_bits = attrs.data_bits
        weight_bits = attrs.weight_bits
        pack_dtype = attrs.pack_dtype
        out_dtype = attrs.out_dtype
        out_dtype = inputs[0].dtype if out_dtype == "" else out_dtype
        unipolar = attrs.unipolar
        return [
            topi_compute(
                inputs[0], inputs[1], data_bits, weight_bits, pack_dtype, out_dtype, unipolar
            )
        ]

    return compute_bitserial_dense


@override_native_generic_func("bitserial_dense_strategy")
def bitserial_dense_strategy(attrs, inputs, out_type, target):
    """bitserial_dense generic strategy"""
    logger.warning("bitserial_dense is not optimized for this platform.")
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_bitserial_dense(topi.nn.bitserial_dense),
        wrap_topi_schedule(topi.generic.schedule_bitserial_dense),
        name="bitserial_dense.generic",
    )
    return strategy


# correlation
def wrap_compute_correlation(topi_compute):
    """wrap correlation topi compute"""

    def _compute_correlation(attrs, inputs, out_type):
        kernel_size = attrs.kernel_size
        max_displacement = attrs.max_displacement
        stride1 = attrs.stride1
        stride2 = attrs.stride2
        padding = get_const_tuple(attrs.padding)
        is_multiply = attrs.is_multiply
        return [
            topi_compute(
                inputs[0],
                inputs[1],
                kernel_size,
                max_displacement,
                stride1,
                stride2,
                padding,
                is_multiply,
            )
        ]

    return _compute_correlation


@override_native_generic_func("correlation_strategy")
def correlation_strategy(attrs, inputs, out_type, target):
    """correlation generic strategy"""
    logger.warning("correlation is not optimized for this platform.")
    layout = attrs.layout
    assert layout == "NCHW", "Only support NCHW layout"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_correlation(topi.nn.correlation_nchw),
        wrap_topi_schedule(topi.generic.schedule_correlation_nchw),
        name="correlation.generic",
    )
    return strategy


# argwhere
def wrap_compute_argwhere(topi_compute):
    """wrap argwhere topi compute"""

    def _compute_argwhere(attrs, inputs, out_type):
        output_shape = []
        for s in out_type.shape:
            if hasattr(s, "value"):
                output_shape.append(s)
            else:
                output_shape.append(te.var("any_dim", "int32"))
        new_output_type = ir.TensorType(output_shape, "int32")
        return [topi_compute(new_output_type, inputs[0])]

    return _compute_argwhere


@override_native_generic_func("argwhere_strategy")
def argwhere_strategy(attrs, inputs, out_type, target):
    """argwhere generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_argwhere(topi.argwhere),
        wrap_topi_schedule(topi.generic.schedule_argwhere),
        name="argwhere.generic",
    )
    return strategy


# threefry_generate
def wrap_compute_threefry_generate(topi_compute):
    """Wrap threefry_generate topi compute"""

    def _compute_threefry_generate(attrs, inputs, _):
        return topi_compute(inputs[0], attrs.out_shape)

    return _compute_threefry_generate


@override_native_generic_func("threefry_generate_strategy")
def threefry_generate_strategy(attrs, inputs, out_type, target):
    """threefry_generate generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_threefry_generate(topi.random.threefry_generate),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="threefry_generate.generic",
    )
    return strategy


# threefry_split
def wrap_compute_threefry_split(topi_compute):
    """Wrap threefry_split topi compute"""

    def _compute_threefry_split(attrs, inputs, _):
        return topi_compute(inputs[0])

    return _compute_threefry_split


@override_native_generic_func("threefry_split_strategy")
def threefry_split_strategy(attrs, inputs, out_type, target):
    """threefry_split generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_threefry_split(topi.random.threefry_split),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="threefry_split.generic",
    )
    return strategy


# uniform
def wrap_compute_uniform(topi_compute):
    """Wrap uniform topi compute"""

    def _compute_uniform(attrs, inputs, _):
        return list(topi_compute(inputs[0], inputs[1], inputs[2], attrs.out_shape, attrs.out_dtype))

    return _compute_uniform


@override_native_generic_func("uniform_strategy")
def uniform_strategy(attrs, inputs, out_type, target):
    """uniform generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_uniform(topi.random.uniform),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="uniform.generic",
    )
    return strategy


def wrap_compute_scanop(topi_compute):
    """Wrap scanop style topi compute"""

    def _compute_scanop(attrs, inputs, _):
        return [topi_compute(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]

    return _compute_scanop


@override_native_generic_func("cumsum_strategy")
def cumsum_strategy(attrs, inputs, out_type, target):
    """cumsum generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cumsum),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="cumsum.generic",
    )
    return strategy


@override_native_generic_func("cumprod_strategy")
def cumprod_strategy(attrs, inputs, out_type, target):
    """cumprod generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_scanop(topi.cumprod),
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="cumprod.generic",
    )
    return strategy


def wrap_compute_unique(topi_compute):
    """Wrap unique topi compute"""

    def _compute_unique(attrs, inputs, _):
        return topi_compute(inputs[0], attrs.sorted, attrs.return_counts)

    return _compute_unique


@override_native_generic_func("unique_strategy")
def unique_strategy(attrs, inputs, out_type, target):
    """unique generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_unique(topi.unique),
        wrap_topi_schedule(topi.generic.schedule_unique),
        name="unique.generic",
    )
    return strategy


@generic_func
def schedule_transpose(attrs, outs, target):
    """schedule transpose"""
    with target:
        return schedule_injective(attrs, outs, target)


# invert_permutation
def wrap_compute_invert_permutation(topi_compute):
    """wrap invert_permutation topi compute"""

    def _compute_invert_permutation(attrs, inputs, out_type):
        return [topi_compute(inputs[0])]

    return _compute_invert_permutation


@override_native_generic_func("invert_permutation_strategy")
def invert_permutation_strategy(attrs, inputs, out_type, target):
    """invert_permutation generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_invert_permutation(topi.invert_permutation),
        wrap_topi_schedule(topi.generic.schedule_injective),
        name="invert_permutation.generic",
    )
    return strategy


def wrap_compute_einsum(topi_compute):
    """Wrap einsum topi compute"""

    def _compute_einsum(attrs, inputs, _):
        return [topi_compute(attrs.equation, *inputs)]

    return _compute_einsum


@override_native_generic_func("einsum_strategy")
def einsum_strategy(attrs, inputs, out_type, target):
    """einsum generic strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_einsum(topi.einsum),
        wrap_topi_schedule(topi.generic.schedule_einsum),
        name="einsum.generic",
    )
    return strategy
