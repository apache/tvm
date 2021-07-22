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
"""Arm(R) Ethos(TM)-U NPU supported operators."""
import numpy as np

from tvm.relay.expr import Constant
from tvm.relay.op.contrib.register import register_pattern_table
from tvm.relay.dataflow_pattern import wildcard, is_op, is_constant
from tvm.relay.backend.contrib.ethosu.util import QConv2DArgs
from tvm.relay.backend.contrib.ethosu.util import BiasAddArgs
from tvm.relay.backend.contrib.ethosu.util import RequantArgs
from tvm.relay.backend.contrib.ethosu.util import get_dim_value
from ethosu.vela import api as vapi


def check_strides(strides):
    """Checks whether strides are within the limits supported by the hardware"""
    stride_range = (1, 3)
    smin, smax = stride_range
    if not smax >= strides[0] >= smin:
        return False
    if not smax >= strides[1] >= smin:
        return False
    return True


def check_valid_dtypes(tensor_params):
    """Check whether dtypes are supported by the hardware"""
    supported_dtypes = (np.uint8, np.int8)
    for tep in tensor_params:
        # Check for dtypes
        if np.dtype(tep.dtype) not in supported_dtypes:
            return False
        # Check for shape sizes
        if any(dimlen > 65536 for dimlen in tep.shape):
            return False
    return True


def check_weights(weights, dilation):
    """Checks whether weight tensor is compatible with HW"""
    dilated_height_range = (1, 64)
    dilated_hxw_range = (1, 64 * 64)
    weights_limit = 127 * 65536
    dilated_width = (weights.shape[get_dim_value(weights.layout, "W")] - 1) * dilation[0] + 1
    dilated_height = (weights.shape[get_dim_value(weights.layout, "H")] - 1) * dilation[1] + 1
    dh_min, dh_max = dilated_height_range
    if not dh_min <= dilated_height <= dh_max:
        return False
    dilated_hxw = dilated_height * dilated_width
    dhxw_min, dhxw_max = dilated_hxw_range
    if not dhxw_min <= dilated_hxw <= dhxw_max:
        return False
    # A saturation upper bound check for accumulators
    weights.values = weights.values - weights.q_params.zero_point
    axis = (
        get_dim_value(weights.layout, "H"),
        get_dim_value(weights.layout, "W"),
        get_dim_value(weights.layout, "I"),
    )
    sum_weights = np.amax(np.sum(np.absolute(weights.values), axis=axis))
    if not sum_weights <= weights_limit:
        return False
    return True


def check_bias(bias):
    """Check whether the bias values fit in 40 bits"""
    if bias and bias.dtype == np.dtype("int64"):
        valid = all(len(bin(bias_value)[2:]) <= 40 for bias_value in bias.values)
        return valid
    return True


def check_batch_size(ifm):
    """Checks for the number of batches vela currently supports"""
    if ifm.shape[0] != 1:
        return False
    return True


def check_dilation(dilation):
    """Checks whether dilation is within the limits supported by the hardware"""
    dilation_range = (1, 2)
    dmin, dmax = dilation_range
    if not dmin <= dilation[0] <= dmax:
        return False
    if not dmin <= dilation[1] <= dmax:
        return False
    return True


def check_padding(padding, bounds):
    """Checks whether padding is within the limits supported by the hardware"""
    if len(padding) != 4 or len(bounds) != 4:
        return False
    top, left, bottom, right = padding
    topb, leftb, bottomb, rightb = bounds
    if top > topb or left > leftb or bottom > bottomb or right > rightb:
        return False
    return True


class TensorParams:
    """
    This class will parse a tvm Expr along with quantization scale
    and zero point to populate parameters that are required
    for the creation of tensors in Vela.
    """

    def __init__(self, tensor, layout=None, scale=None, zero_point=None):
        self.tensor = tensor
        if isinstance(tensor, Constant):
            self.values = tensor.data.asnumpy()
        else:
            self.values = None
        self.dtype = tensor.checked_type.dtype
        self.shape = [int(i) for i in tensor.checked_type.shape]
        self.layout = layout

        if scale is not None and zero_point is not None:
            self.q_params = vapi.NpuQuantization(
                scale.data.asnumpy().astype("float32"), zero_point.data.asnumpy().astype(self.dtype)
            )
        else:
            # put default values
            self.q_params = vapi.NpuQuantization(1.0, 0)


class QnnConv2DParams:
    """
    This class will parse a Call to a ethosu.qnn_conv2d_clip composite function
    and extract quantization information of all the associated tensors.
    """

    composite_name = "ethosu.qnn_conv2d"
    # The hardware only supports padding upto the numbers as follows
    padding_bounds = [31, 31, 32, 32]
    activation_map = {"clip": "CLIP"}

    def __init__(self, func_body):
        activation = None
        if str(func_body.op) in self.activation_map.keys():
            activation = func_body
            requantize_op = activation.args[0]
        else:
            requantize_op = func_body
        bias_add = requantize_op.args[0]
        qnn_conv2d = bias_add.args[0]
        data_layout = qnn_conv2d.attrs.data_layout
        kernel_layout = qnn_conv2d.attrs.kernel_layout
        # We consider the weights & biases as params as it should be a Constant
        self.weights = TensorParams(
            qnn_conv2d.args[QConv2DArgs.weights.value],
            kernel_layout,
            qnn_conv2d.args[QConv2DArgs.weights_scale.value],
            qnn_conv2d.args[QConv2DArgs.weights_zero_point.value],
        )

        self.biases = TensorParams(
            bias_add.args[BiasAddArgs.biases.value],
            data_layout,
            requantize_op.args[RequantArgs.ifm_scale.value],
            requantize_op.args[RequantArgs.ifm_zero_point.value],
        )
        self.ifm = TensorParams(
            qnn_conv2d.args[QConv2DArgs.ifm.value],
            data_layout,
            qnn_conv2d.args[QConv2DArgs.ifm_scale.value],
            qnn_conv2d.args[QConv2DArgs.ifm_zero_point.value],
        )
        self.ofm = TensorParams(
            func_body,
            data_layout,
            requantize_op.args[RequantArgs.ofm_scale.value],
            requantize_op.args[RequantArgs.ofm_zero_point.value],
        )
        self.padding = qnn_conv2d.attrs.padding
        self.strides = qnn_conv2d.attrs.strides
        self.dilation = qnn_conv2d.attrs.dilation
        self.activation = activation

        # If groups are equal to channel, its a depthwise_conv2d
        self.groups = qnn_conv2d.attrs.groups
        self.is_depthwise = False
        channels_axis = {"HWIO": 3, "HWOI": 2}
        if qnn_conv2d.attrs.groups == self.weights.shape[channels_axis[kernel_layout]]:
            self.is_depthwise = True

    def is_valid(self):
        """
        Checks whether QnnConv2D with Clip has compatible attributes with HW
        """
        tensor_params = [self.weights, self.ifm, self.ofm]
        if not check_valid_dtypes(tensor_params):
            return False
        if not check_weights(self.weights, self.dilation):
            return False
        if not check_bias(self.biases):
            return False
        if not check_strides(self.strides):
            return False
        if not check_batch_size(self.ifm):
            return False
        if not check_dilation(self.dilation):
            return False
        if not check_padding(self.padding, self.padding_bounds):
            return False
        legal_groups = [1, self.ofm.shape[3]]
        if self.groups not in legal_groups:
            return False
        # This should be a valid QnnDepthwise2DParams, not QnnConv2DParams
        if self.is_depthwise:
            return False
        return True


def qnn_conv2d_pattern():
    """
    Create pattern for qnn.conv2D with optional fused relu
    """
    qnn_conv2d = is_op("qnn.conv2d")(
        wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
    ).has_attr({"kernel_layout": "HWIO"})
    bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
    req = is_op("qnn.requantize")(
        qnn_conv2d | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
    )
    clip_or_req = req.optional(is_op("clip"))
    return clip_or_req


@register_pattern_table("ethosu")
def pattern_table():
    return [
        ("ethosu.qnn_conv2d", qnn_conv2d_pattern(), lambda pat: QnnConv2DParams(pat).is_valid())
    ]
