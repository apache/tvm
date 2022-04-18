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
def batch_matmul_strategy_cpu(attrs, inputs, out_type, target):
    """batch_matmul strategy for Hexagon"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_batch_matmul(topi.nn.batch_matmul),
        wrap_topi_schedule(topi.hexagon.schedule_batch_matmul),
        name="batch_matmul.hexagon",
    )
    return strategy


@conv2d_strategy.register("hexagon")
def conv2d_strategy_hexagon(attrs, inputs, out_type, target):
    """Conv2d strategy for Hexagon"""
    strategy = _op.OpStrategy()
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout

    if data_layout == "NHWC" and kernel_layout == "HWIO":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_nhwc),
            wrap_topi_schedule(topi.hexagon.schedule_conv2d_nhwc),
            name="conv2d_nhwc.hexagon",
        )
        return strategy

    if data_layout == "NCHW" and kernel_layout == "OIHW":
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_nchw),
            wrap_topi_schedule(topi.hexagon.schedule_conv2d_nchw),
            name="conv2d_nchw.hexagon",
        )
        return strategy

    raise RuntimeError(
        f"Unsupported layouts: data_layout:{data_layout}, kernel_layout:{kernel_layout}, "
        f"groups:{attrs.groups}"
    )


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


# --- Op schedule registration


@schedule_adaptive_pool.register("hexagon")
def schedule_adaptive_pool_hexagon(attrs, outs, target):
    """Schedule adaptive pool ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_adaptive_pool(outs)


@schedule_concatenate.register("hexagon")
def schedule_concatenate_hexagon(attrs, outs, target):
    """Schedule concatenate ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_injective(outs)


@schedule_injective.register("hexagon")
def schedule_injective_hexagon(attrs, outs, target):
    """Schedule injective ops for Hexagon"""
    with target:
        return topi.hexagon.schedule_injective(outs)


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
