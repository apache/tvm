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

import numpy as np

from tvm import nd, relay
from ..nn import qnn_requantize_alter_layout, qnn_add_alter_layout

@qnn_requantize_alter_layout.register(["arm_cpu"])
def alter_requantize_layout(attrs, inputs, tinfos, out_type):
    bias_op, in_scale, _in_zp, out_scale, _out_zp = inputs
    in_scale_numpy = in_scale.data.numpy().astype("float64")
    out_scale_scalar = out_scale.data.numpy().item()
    scales = (in_scale_numpy / out_scale_scalar) * 2**33
    scale_constant = relay.Constant(nd.array(scales.astype("int32")))
    return relay.qnn.op.requantize(inputs[0], scale_constant, *inputs[2:], **attrs)


def _is_qnn_op_depthwise_conv2d(qnn_conv2d_op):
    return relay.op.strategy.generic.is_depthwise_conv2d(
        qnn_conv2d_op.args[0].type_annotation.shape,
        qnn_conv2d_op.attrs.data_layout,
        qnn_conv2d_op.args[1].data.shape,
        qnn_conv2d_op.attrs.kernel_layout,
        qnn_conv2d_op.attrs.groups
    )


@qnn_add_alter_layout.register(["arm_cpu"])
def alter_add_layout(attrs, inputs, tinfos, out_type):
    prev_op, biases = inputs
    if prev_op.op.name != "qnn.conv2d":
        return None

    conv_input_zp = prev_op.args[2].data.numpy().item()
    assert conv_input_zp == -128
    kernel = prev_op.args[1].data.numpy()

    kernel_layout = prev_op.attrs.kernel_layout
    axes_to_sum = [kernel_layout.index("H"), kernel_layout.index("W")]
    if not _is_qnn_op_depthwise_conv2d(prev_op):
        axes_to_sum.append(kernel_layout.index("I"))
    element_sums = np.sum(kernel, axis=tuple(axes_to_sum)).flatten()


    # The zero point is subtracted from the input elements, so we need a "-" sign here
    zp_shifted_sums = element_sums * (-conv_input_zp)

    # We want to make sure new_biases is representable as an int32. It's tempting to just check
    # whether arr.dtype == "int32" (since Numpy will automatically increase dtype in some cases)
    # but this leads to weird wrapping behavior and doesn't work. We must do it manually.
    new_biases = biases.data.numpy().astype("int64") + zp_shifted_sums
    if new_biases.min() < -2**31 or new_biases.max() > 2**31 - 1:
        return None

    new_input_zp = relay.Constant(nd.array(np.int32(0)))
    new_conv_args = (*prev_op.args[:2], new_input_zp, *prev_op.args[3:])
    new_conv_op = relay.qnn.op.conv2d(*new_conv_args, **prev_op.attrs)
    bias_constant = relay.Constant(nd.array(new_biases.astype("int32")))
    return relay.add(new_conv_op, bias_constant)