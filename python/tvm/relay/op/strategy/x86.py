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
"""Definition of x86 operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import logging

import re
from tvm import topi
from tvm.te import SpecializedCondition
from .generic import *
from .. import op as _op

logger = logging.getLogger("strategy")

_NCHWc_matcher = re.compile("^NCHW[0-9]+c$")
_OIHWio_matcher = re.compile("^OIHW[0-9]+i[0-9]+o$")


@schedule_injective.register("cpu")
def schedule_injective_cpu(attrs, outs, target):
    """schedule injective ops for x86"""
    with target:
        return topi.x86.schedule_injective(outs)


@schedule_reduce.register("cpu")
def schedule_reduce_cpu(attrs, outs, target):
    """schedule reduction ops for x86"""
    with target:
        return topi.x86.schedule_reduce(outs)


@schedule_concatenate.register("cpu")
def schedule_concatenate_cpu(attrs, outs, target):
    """schedule concatenate op for x86"""
    with target:
        return topi.x86.schedule_concatenate(outs)


@schedule_pool.register("cpu")
def schedule_pool_cpu(attrs, outs, target):
    """schedule pooling ops for x86"""
    with target:
        return topi.x86.schedule_pool(outs, attrs.layout)


@schedule_adaptive_pool.register("cpu")
def schedule_adaptive_pool_cpu(attrs, outs, target):
    """schedule adaptive pooling ops for x86"""
    with target:
        return topi.x86.schedule_adaptive_pool(outs)


@softmax_strategy.register("cpu")
def softmax_strategy_cpu(attrs, inputs, out_type, target):
    """softmax x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.x86.schedule_softmax),
        name="softmax.x86",
    )
    return strategy


@schedule_log_softmax.register("cpu")
def schedule_log_softmax_cpu(attrs, outs, target):
    """schedule log_softmax op for x86"""
    with target:
        return topi.x86.schedule_softmax(outs)


@conv2d_strategy.register("cpu")
def conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if topi.x86.is_int8_hw_support(data.dtype, kernel.dtype):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.x86.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.x86.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.x86",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.x86.conv2d_nchw),
                    wrap_topi_schedule(topi.x86.schedule_conv2d_nchw),
                    name="conv2d_nchw.x86",
                )
        elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
            assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
            return conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target)
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            logger.warning("For x86 target, NCHW layout is recommended for conv2d.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                wrap_topi_schedule(topi.x86.schedule_conv2d_nhwc),
                name="conv2d_nhwc.x86",
            )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            logger.warning("conv2d HWCN layout is not optimized for x86.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_hwcn),
                wrap_topi_schedule(topi.generic.schedule_conv2d_hwcn),
                name="conv2d_hwcn.generic",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {} for x86".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            channel_multiplier = get_const_tuple(inputs[1].shape)[1]
            if channel_multiplier == 1 and dilation_h == 1 and dilation_w == 1:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.x86.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.x86",
                )
            else:
                logger.warning(
                    "For x86 target, depthwise_conv2d with channel "
                    "multiplier greater than 1 is not optimized"
                )
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.generic.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.generic",
                )
        elif _NCHWc_matcher.match(layout):  # check if layout is NCHWxc
            assert _OIHWio_matcher.match(kernel_layout)  # check if kernel is OIHWio
            return depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target)
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            logger.warning("depthwise_conv2d NHWC layout is not optimized for x86.")
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
            logger.warning("group_conv2d is not optimized for x86.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.generic",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            logger.warning("group_conv2d is not optimized for x86.")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.group_conv2d_nhwc, has_groups=True),
                wrap_topi_schedule(topi.generic.schedule_group_conv2d_nhwc),
                name="group_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy


@conv2d_NCHWc_strategy.register("cpu")
def conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc x86 strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    if topi.x86.is_int8_hw_support(data.dtype, kernel.dtype):
        strategy.add_implementation(
            wrap_compute_conv2d(topi.x86.conv2d_NCHWc_int8, True, True),
            wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc_int8),
            name="conv2d_NCHWc_int8.x86",
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.x86.conv2d_NCHWc, True, True),
            wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.x86",
        )
    return strategy


@depthwise_conv2d_NCHWc_strategy.register("cpu")
def depthwise_conv2d_NCHWc_strategy_cpu(attrs, inputs, out_type, target):
    """depthwise_conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.x86.depthwise_conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.x86.schedule_depthwise_conv2d_NCHWc),
        name="depthwise_conv2d_NCHWc.x86",
    )
    return strategy


@conv2d_transpose_strategy.register("cpu")
def conv2d_transpose_strategy_cpu(attrs, inputs, out_type, target):
    """conv2d_transpose x86 strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d_transpose(topi.x86.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.x86.schedule_conv2d_transpose_nchw),
        name="conv2d_transpose_nchw.x86",
    )
    return strategy


@conv3d_transpose_strategy.register("cpu")
def conv3d_transpose_strategy_cpu(attrs, inputs, out_type, target):
    """conv3d_transpose x86 strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCDHW", "only support ncdhw for now"
    assert dilation == (1, 1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv3d_transpose(topi.x86.conv3d_transpose_ncdhw),
        wrap_topi_schedule(topi.x86.schedule_conv3d_transpose_ncdhw),
        name="conv3d_transpose_ncdhw.x86",
    )
    return strategy


@conv3d_strategy.register("cpu")
def conv3d_strategy_cpu(attrs, inputs, out_type, target):
    """conv3d generic strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCDHW":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.x86.conv3d_ncdhw),
            wrap_topi_schedule(topi.x86.schedule_conv3d_ncdhw),
            name="conv3d_ncdhw.x86",
        )
    elif layout == "NDHWC":
        strategy.add_implementation(
            wrap_compute_conv3d(topi.x86.conv3d_ndhwc),
            wrap_topi_schedule(topi.x86.schedule_conv3d_ndhwc),
            name="conv3d_ndhwc.x86",
        )
    else:
        raise ValueError("Not support this layout {} yet".format(layout))
    return strategy


@conv1d_strategy.register("cpu")
def conv1d_strategy_cpu(attrs, inputs, out_type, target):
    """conv1d x86 strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if layout == "NCW":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.nn.conv1d_ncw),
            wrap_topi_schedule(topi.x86.schedule_conv1d_ncw),
            name="conv1d_ncw.x86",
        )
    elif layout == "NWC":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.nn.conv1d_nwc),
            wrap_topi_schedule(topi.x86.schedule_conv1d_nwc),
            name="conv1d_nwc.x86",
        )
    else:
        raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy


@dense_strategy.register("cpu")
def dense_strategy_cpu(attrs, inputs, out_type, target):
    """dense x86 strategy"""
    strategy = _op.OpStrategy()
    m, _ = inputs[0].shape
    same_type = inputs[0].dtype == inputs[1].dtype == out_type.dtype
    dtype = inputs[0].dtype
    u8s8s32 = dtype == "uint8" and inputs[1].dtype == "int8" and out_type.dtype == "int32"
    strategy.add_implementation(
        wrap_compute_dense(topi.x86.dense_nopack),
        wrap_topi_schedule(topi.x86.schedule_dense_nopack),
        name="dense_nopack.x86",
        plevel=10,
    )
    if "cblas" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"]):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_cblas),
                wrap_topi_schedule(topi.x86.schedule_dense_cblas),
                name="dense_cblas.x86",
                plevel=13,
            )
    if "mkl" in target.libs:
        with SpecializedCondition(same_type and dtype in ["float32", "float64"] or u8s8s32):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_mkl),
                wrap_topi_schedule(topi.x86.schedule_dense_mkl),
                name="dense_mkl.x86",
                plevel=14,
            )
    if "mkldnn" in target.libs:
        with SpecializedCondition(same_type and dtype == "float32"):
            strategy.add_implementation(
                wrap_compute_dense(topi.x86.dense_mkldnn),
                wrap_topi_schedule(topi.x86.schedule_dense_mkldnn),
                name="dense_mkldnn.x86",
                plevel=15,
            )
    with SpecializedCondition(m >= 16):
        # this implementation may not be well-optimized, so use plevel=8 for now.
        strategy.add_implementation(
            wrap_compute_dense(topi.x86.dense_pack),
            wrap_topi_schedule(topi.x86.schedule_dense_pack),
            name="dense_pack.x86",
            plevel=5,
        )
    return strategy


@batch_matmul_strategy.register("cpu")
def batch_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """batch_matmul x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.x86.batch_matmul),
        wrap_topi_schedule(topi.x86.schedule_batch_matmul),
        name="batch_matmul.x86",
        plevel=10,
    )
    if "cblas" in target.libs:
        strategy.add_implementation(
            wrap_compute_batch_matmul(topi.x86.batch_matmul_cblas),
            wrap_topi_schedule(topi.x86.schedule_batch_matmul_cblas),
            name="batch_matmul_cblas.x86",
            plevel=15,
        )
    return strategy


@sparse_dense_strategy.register("cpu")
def sparse_dense_strategy_cpu(attrs, inputs, out_type, target):
    """sparse dense x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_sparse_dense(topi.nn.sparse_dense),
        wrap_topi_schedule(topi.x86.schedule_sparse_dense),
        name="sparse_dense.x86",
        plevel=10,
    )
    return strategy


@roi_align_strategy.register("cpu")
def roi_align_strategy_cpu(attrs, inputs, out_type, target):
    """roi_align x86 strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.layout
    assert layout == "NCHW", "only support nchw for now"
    strategy.add_implementation(
        wrap_compute_roi_align(topi.x86.roi_align_nchw),
        wrap_topi_schedule(topi.generic.schedule_roi_align),
        name="roi_align.x86",
    )
    return strategy


@bitserial_conv2d_strategy.register("cpu")
def bitserial_conv2d_strategy_cpu(attrs, inputs, out_type, target):
    """bitserial_conv2d x86 strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.x86.bitserial_conv2d_nchw),
            wrap_topi_schedule(topi.x86.schedule_bitserial_conv2d_nchw),
            name="bitserial_conv2d_nchw.x86",
        )
    elif layout == "NHWC":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.x86.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.x86.schedule_bitserial_conv2d_nhwc),
            name="bitserial_conv2d_nhwc.x86",
        )
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy


@bitserial_dense_strategy.register("cpu")
def bitserial_dense_strategy_cpu(attrs, inputs, out_type, target):
    """bitserial_dense x86 strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_bitserial_dense(topi.x86.bitserial_dense),
        wrap_topi_schedule(topi.x86.schedule_bitserial_dense),
        name="bitserial_dense.x86",
    )
    return strategy
