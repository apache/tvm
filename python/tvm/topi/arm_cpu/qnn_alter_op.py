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
"""Arm Cortex-M specific optimizations for quantized operators."""

import numpy as np

from tvm import nd, relay, target
from ..nn import qnn_requantize_alter_layout, qnn_add_alter_layout


@qnn_requantize_alter_layout.register(["arm_cpu"])
def alter_requantize_layout(attrs, inputs, _tinfos, _out_type):
    """Changes a floating point requantize op to use int64 multiply + shift for microTVM.

    Usually, this is done by QNN legalization. However, microTVM wants to manually choose the
    integer rounding constants in order to:
        (a) Have int32, not int64 constants
        (b) Use a constant rounding shift to skip a memory load.

    Ideally, we would pick these constants in the requantize (or fused) schedule. Unfortunately that
    is not currently possible, so we pick them with `alter_layout` as a hack. This will only work if
    the requantize schedule "plays along" with this hack.
    """

    # Only microTVM Cortex-M boards with DSP use the relevant schedules
    current_target = target.Target.current(allow_none=False)
    if not (current_target.features.has_dsp and "cortex-m" in current_target.mcpu):
        return None

    _, in_scale, _, out_scale, _ = inputs
    in_scale_numpy = in_scale.data.numpy().astype("float64")
    out_scale_scalar = out_scale.data.numpy().item()

    # Shifting by 33 and rounding means shifting by 32, adding 1, and shifting by 1 again. This is
    # useful, because shifting a multiplication product by 32 can be done for "free" with SMMUL
    scales = ((in_scale_numpy / out_scale_scalar) * 2**33).astype("int32")

    # Requantize ops in Relay do not support int32 scales - if we try to use one, requantize.cc will
    # raise an error. As a hacky work-around, we change the scale dtype to float32, without changing
    # underlying data. This works, as our compute function knows to interpret the scale as an int32.

    # This is only a work-around - a better long-term solution would be adding a new integer
    # requantize op, which takes integer scales, shifts, and rounding behavior.
    fake_float_scales = scales.view("float32")

    scale_constant = relay.Constant(nd.array(fake_float_scales))
    return relay.qnn.op.requantize(inputs[0], scale_constant, *inputs[2:], **attrs)


def _is_qnn_op_depthwise_conv2d(qnn_conv2d_op):
    return relay.op.strategy.generic.is_depthwise_conv2d(
        qnn_conv2d_op.args[0].type_annotation.shape,
        qnn_conv2d_op.attrs.data_layout,
        qnn_conv2d_op.args[1].data.shape,
        qnn_conv2d_op.attrs.kernel_layout,
        qnn_conv2d_op.attrs.groups,
    )


@qnn_add_alter_layout.register(["arm_cpu"])
def alter_add_layout(_attrs, inputs, _tinfos, _out_type):
    """Fuses the zero point for a previous quantized operator with this add operation.

    Currently only supports qnn.conv2d, but qnn.dense support should be added. Note that this
    optimization means we must pad tensors with the input zero point, and NOT with zero.
    """

    prev_op, biases = inputs
    if not hasattr(prev_op, "op"):
        return None
    if prev_op.op.name != "qnn.conv2d":
        return None

    # We should not perform this alteration if the target has a uint * int SIMD MAC operation (since
    # these do (x - (-128)) * y efficiently, and conv_input_zp is usually -128). For now, we
    # restrict this optimization to just Cortex-M devices, but it might be helpful on others too.
    current_target = target.Target.current(allow_none=False)
    if not "cortex-m" in current_target.mcpu:
        return None

    conv_input_zp = prev_op.args[2].data.numpy().item()
    kernel = prev_op.args[1].data.numpy()

    if _is_qnn_op_depthwise_conv2d(prev_op):
        axes_to_sum = "HW"
    elif prev_op.attrs.groups == 1:
        axes_to_sum = "HWI"
    else:
        # This alteration does not currently support grouped conv2d
        return None
    axes_to_sum = tuple(map(prev_op.attrs.kernel_layout.index, axes_to_sum))
    element_sums = np.sum(kernel, axis=axes_to_sum).flatten()

    # The zero point is subtracted from the input elements, so we need a "-" sign here
    zp_shifted_sums = element_sums * (-conv_input_zp)

    # We want to make sure new_biases is representable as an int32. It's tempting to just check
    # whether arr.dtype == "int32" (since Numpy will automatically increase dtype in some cases)
    # but this leads to weird wrapping behavior and doesn't work. We must do it manually.
    new_biases = biases.data.numpy().astype("int64") + zp_shifted_sums
    if new_biases.min() < -(2**31) or new_biases.max() > 2**31 - 1:
        return None

    new_input_zp = relay.Constant(nd.array(np.int32(0)))
    new_conv_args = (*prev_op.args[:2], new_input_zp, *prev_op.args[3:])
    new_conv_op = relay.qnn.op.conv2d(*new_conv_args, **prev_op.attrs)
    bias_constant = relay.Constant(nd.array(new_biases.astype("int32")))
    return relay.add(new_conv_op, bias_constant)
