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
"""Definition of HLS operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
from tvm import topi
from .generic import *
from .. import op as _op


@schedule_injective.register("hls")
def schedule_injective_hls(attrs, outs, target):
    """schedule injective ops for hls"""
    with target:
        return topi.hls.schedule_injective(outs)


@schedule_reduce.register("hls")
def schedule_reduce_hls(attrs, outs, target):
    """schedule reduction ops for hls"""
    with target:
        return topi.hls.schedule_reduce(outs)


@schedule_concatenate.register("hls")
def schedule_concatenate_hls(attrs, outs, target):
    """schedule concatenate for hls"""
    with target:
        return topi.hls.schedule_injective(outs)


@schedule_pool.register("hls")
def schedule_pool_hls(attrs, outs, target):
    """schedule pooling ops for hls"""
    with target:
        return topi.hls.schedule_pool(outs, attrs.layout)


@schedule_adaptive_pool.register("hls")
def schedule_adaptive_pool_hls(attrs, outs, target):
    """schedule adaptive pooling ops for hls"""
    with target:
        return topi.hls.schedule_adaptive_pool(outs)


@softmax_strategy.register("hls")
def softmax_strategy_hls(attrs, inputs, out_type, target):
    """softmax hls strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.hls.schedule_softmax),
        name="softmax.hls",
    )
    return strategy


@log_softmax_strategy.register("hls")
def log_softmax_strategy_hls(attrs, inputs, out_type, target):
    """log_softmax hls strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.log_softmax),
        wrap_topi_schedule(topi.hls.schedule_softmax),
        name="log_softmax.hls",
    )
    return strategy


@conv2d_strategy.register("hls")
def conv2d_strategy_hls(attrs, inputs, out_type, target):
    """conv2d hls strategy"""
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
                wrap_topi_schedule(topi.hls.schedule_conv2d_nchw),
                name="conv2d_nchw.hls",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                wrap_topi_schedule(topi.hls.schedule_conv2d_nhwc),
                name="conv2d_nhwc.hls",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {}".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.hls.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.hls",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.hls.schedule_depthwise_conv2d_nhwc),
                name="depthwise_nhwc.hls",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        raise RuntimeError("group_conv2d is not supported for hls")
    return strategy


@conv2d_NCHWc_strategy.register("hls")
def conv2d_NCHWc_strategy_hls(attrs, inputs, out_type, target):
    """conv2d_NCHWc hls strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.nn.conv2d_NCHWc, need_data_layout=True, need_out_layout=True),
        wrap_topi_schedule(topi.hls.schedule_conv2d_NCHWc),
        name="conv2d_NCHWc.hls",
    )
    return strategy


@conv2d_transpose_strategy.register("hls")
def conv2d_transpose_strategy_hls(attrs, inputs, out_type, target):
    """conv2d_transpose hls strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d_transpose(topi.nn.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.hls.schedule_conv2d_transpose_nchw),
        name="conv2d_transpose_nchw.hls",
    )
    return strategy


@dense_strategy.register("hls")
def dense_strategy_hls(attrs, inputs, out_type, target):
    """dense hls strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.nn.dense),
        wrap_topi_schedule(topi.hls.schedule_dense),
        name="dense.hls",
    )
    return strategy


@bitserial_conv2d_strategy.register("hls")
def bitserial_conv2d_strategy_hls(attrs, inputs, out_type, target):
    """bitserial_conv2d hls strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    if layout == "NCHW":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.nn.bitserial_conv2d_nchw),
            wrap_topi_schedule(topi.hls.schedule_bitserial_conv2d_nchw),
            name="bitserial_conv2d_nchw.hls",
        )
    elif layout == "NHWC":
        strategy.add_implementation(
            wrap_compute_bitserial_conv2d(topi.nn.bitserial_conv2d_nhwc),
            wrap_topi_schedule(topi.hls.schedule_bitserial_conv2d_nhwc),
            name="bitserial_conv2d_nhwc.hls",
        )
    else:
        raise ValueError("Data layout {} not supported.".format(layout))
    return strategy
