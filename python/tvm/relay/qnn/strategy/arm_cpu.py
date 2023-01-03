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
"""Quantized operator strategy for Arm CPU.

As quantized op schedules, these are only used if the qnn.Legalize pass is disabled. The current
schedules only work for fused operators with bias, as this is the most common use case. Only
regular/depthwise conv2d is supported, but qnn_dense will be added eventually."""

from tvm import topi, TVMError
from .generic import qnn_conv2d_strategy
from ... import op as _op
from ...op.strategy.generic import is_depthwise_conv2d


@qnn_conv2d_strategy.register("arm_cpu")
def qnn_conv2d_strategy_arm_cpu(attrs, inputs, _out_type, target):
    """qnn.conv2d strategy for Arm Cortex-M CPUs with DSP.

    When computing convolutions, we want data that will be used to compute the same output values to
    be adjacent in memory, as this lets us reuse memory loads and use more SIMD instructions.

    For depthwise convolutions, channels do not interact with each other, so the NCHW and IOHW
    layouts to the best job of keeping "related" data close. In contrast, computing one output of a
    regular convolution requires reading all input channels, so NHWC and OHWI are best. Hence, these
    are the layouts we support.
    """

    if not (target.features.has_dsp and "cortex-m" in target.mcpu):
        raise TVMError(
            "Quantized Arm schedules only exist for Cortex-M with DSP! "
            "The qnn.Legalize pass should be run for other Arm processors."
        )

    data = inputs[0]
    kernel = inputs[1]
    data_layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    groups = attrs.groups
    strategy = _op.OpStrategy()

    if groups == 1:
        if data_layout == "NHWC" and kernel_layout == "OHWI":
            strategy.add_implementation(
                topi.arm_cpu.qnn_conv2d,
                topi.arm_cpu.schedule_qnn_conv2d,
                name="qnn_conv2d.arm_cpu",
            )
    elif is_depthwise_conv2d(data.shape, data_layout, kernel.shape, kernel_layout, groups):
        if data_layout == "NCHW" and kernel_layout == "IOHW":
            strategy.add_implementation(
                topi.arm_cpu.qnn_depthwise_conv2d,
                topi.arm_cpu.schedule_qnn_depthwise_conv2d,
                name="qnn_depthwise_conv2d.arm_cpu",
            )
    else:
        raise TVMError("No Arm Cortex-M DSP strategy exists for generic group qnn.conv2d")

    return strategy
