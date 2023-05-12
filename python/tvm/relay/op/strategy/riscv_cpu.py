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

# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
"""Definition of RISCV CPU operator strategy."""

from functools import reduce
import logging

from tvm import topi
from .. import op as _op
from .generic import *
from .x86 import conv2d_strategy_cpu

logger = logging.getLogger("strategy")


@schedule_injective.register("riscv_cpu")
def schedule_injective_riscv_cpu(_, outs, target):
    """schedule injective ops for riscv_cpu"""
    with target:
        return topi.riscv_cpu.schedule_injective(outs)


@schedule_reduce.register("riscv_cpu")
def schedule_reduce_riscv_cpu(_, outs, target):
    """schedule reduction ops for riscv_cpu"""
    with target:
        return topi.x86.schedule_reduce(outs)


@conv2d_strategy.register("riscv_cpu")
def conv2d_strategy_riscv_cpu(attrs, inputs, out_type, target):
    """conv2d riscv_cpu strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            is_int8 = topi.riscv_cpu.is_int8_hw_support(data.dtype, kernel.dtype)
            # Vector instructions with int8 show more performance at a larger size.
            if is_int8 and kernel.shape[1] >= 128:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.riscv_cpu.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.riscv",
                    plevel=15,
                )
                return strategy

    return conv2d_strategy_cpu(attrs, inputs, out_type, target)


@conv2d_NCHWc_strategy.register("riscv_cpu")
def conv2d_NCHWc_strategy_riscv_cpu(attrs, inputs, out_type, target):
    """conv2d_NCHWc adopted from x86"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    is_int8 = topi.riscv_cpu.is_int8_hw_support(data.dtype, kernel.dtype)
    # Vector instructions with int8 show more performance at a larger size.
    if is_int8 and kernel.shape[1] >= 128:
        strategy.add_implementation(
            wrap_compute_conv2d(
                topi.riscv_cpu.conv2d_NCHWc_int8, need_data_layout=True, need_out_layout=True
            ),
            wrap_topi_schedule(topi.riscv_cpu.schedule_conv2d_NCHWc_int8),
            name="conv2d_NCHWc_int8.riscv_cpu",
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(topi.x86.conv2d_NCHWc, need_data_layout=True, need_out_layout=True),
            wrap_topi_schedule(topi.x86.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.x86",
        )
    return strategy
