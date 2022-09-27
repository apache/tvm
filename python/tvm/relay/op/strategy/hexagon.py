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
"""Definition of Hexagon operator strategy."""

# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import

from tvm import topi
from .generic import *
from .. import op as _op

# --- Op strategy registration


@batch_matmul_strategy.register("hexagon")
def batch_matmul_strategy_hexagon(attrs, inputs, out_type, target):
    """batch_matmul strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.nn.batch_matmul),
        wrap_topi_schedule(topi.hexagon.schedule_batch_matmul),
        name="batch_matmul.hexagon",
    )
    return strategy


@concatenate_strategy.register("hexagon")
def concatenate_strategy_hexagon(attrs, inputs, out_type, target):
    """concatenate strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_concat(topi.concatenate),
        wrap_topi_schedule(topi.hexagon.schedule_injective),
        name="concatenate.hexagon",
    )
    return strategy


@conv2d_strategy.register("hexagon")
def conv2d_strategy_hexagon(attrs, inputs, out_type, target):
    """Conv2d strategy for Hexagon"""
    strategy = _op.OpStrategy()
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    groups = attrs.groups
    data, kernel = inputs
    layout = attrs.data_layout

    if groups == 1:
        if data_layout == "NHWC" and kernel_layout == "HWIO":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc),
                wrap_topi_schedule(topi.hexagon.schedule_conv2d_nhwc),
                name="conv2d_nhwc.hexagon",
            )
        elif data_layout == "NCHW" and kernel_layout == "OIHW":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nchw),
                wrap_topi_schedule(topi.hexagon.schedule_conv2d_nchw),
                name="conv2d_nchw.hexagon",
            )
        else:
            raise RuntimeError(
                f"Unsupported layouts: data_layout:{data_layout}, kernel_layout:{kernel_layout}, "
                f"groups:{attrs.groups}"
            )
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.hexagon.schedule_depthwise_conv2d_nchw),
                name="depthwise_conv2d_nchw.generic",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.hexagon.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.generic",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        raise RuntimeError(f"Unsupported group_conv2d layout {layout}")

    return strategy


@dense_strategy.register("hexagon")
def dense_strategy_hexagon(attrs, inputs, out_type, target):
    """Dense strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.nn.dense),
        wrap_topi_schedule(topi.hexagon.schedule_dense),
        name="dense.hexagon",
    )
    return strategy


@softmax_strategy.register("hexagon")
def softmax_strategy_hexagon(attrs, inputs, out_type, target):
    """Softmax strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.hexagon.schedule_softmax),
        name="softmax.hexagon",
    )
    return strategy


@conv2d_transpose_strategy.register("hexagon")
def conv2d_transpose_strategy_hexagon(attrs, inputs, out_type, target):
    """conv2d_transpose hexagon strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    strategy = _op.OpStrategy()
    if groups == 1:
        strategy.add_implementation(
            wrap_compute_conv2d_transpose(topi.nn.conv2d_transpose_nchw),
            wrap_topi_schedule(topi.hexagon.schedule_conv2d_transpose_nchw),
            name="conv2d_transpose_nchw.generic",
        )
    else:
        raise RuntimeError("Unsupported conv2d_transpose layout {}".format(layout))
    return strategy


# --- Op schedule registration


@schedule_adaptive_pool.register("hexagon")
def schedule_adaptive_pool_hexagon(attrs, outs, target):
    """Schedule adaptive pool ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_adaptive_pool(outs)


@schedule_injective.register("hexagon")
def schedule_injective_hexagon(attrs, outs, target):
    """Schedule injective ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_injective(outs)


@schedule_concatenate.register("hexagon")
def schedule_concatenate_hexagon(attrs, outs, target):
    """Schedule concatenate ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_injective(outs)


@schedule_pad.register("hexagon")
def schedule_pad_hexagon(attrs, outs, target):
    """Schedule pad ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_pad(outs)


@schedule_pool.register("hexagon")
def schedule_pool_hexagon(attrs, outs, target):
    """Schedule pool ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_pool(outs)


@schedule_reduce.register("hexagon")
def schedule_reduce_hexagon(attrs, outs, target):
    """Schedule reduction ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_reduce(outs)
