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

"""CMSIS-NN functions for testing networks"""

import math
from typing import List, Union, Tuple
import numpy as np

import tvm
from tvm import relay


def skip_if_no_reference_system(func):
    return tvm.testing.skip_if_32bit(reason="Reference system unavailable in i386 container")(func)


def count_num_calls(mod):
    """Counts number of CallNode(s) in the IRModule"""

    class CallCounter(relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    counter = CallCounter()
    for var in mod.get_global_vars():
        counter.visit(mod[var.name_hint])
    return counter.count


def assert_partitioned_function(orig_mod, cmsisnn_mod):
    """If kCompiler attribute is missing, this function raises assertion"""
    attrs = [
        cmsisnn_mod[var.name_hint].attrs
        for var in cmsisnn_mod.get_global_vars()
        if cmsisnn_mod[var.name_hint].attrs
    ]
    assert any(attrs), "At least one function with external attributes was expected."

    compilers = [
        key == "Compiler" and value == "cmsis-nn" for attr in attrs for key, value in attr.items()
    ]
    assert any(compilers), "Module does not contain function for cmsisnn target."

    assert count_num_calls(orig_mod) == count_num_calls(
        cmsisnn_mod
    ), "Number of calls changed during partitioning"


def assert_no_external_function(mod):
    attrs = [mod[var.name_hint].attrs for var in mod.get_global_vars() if mod[var.name_hint].attrs]
    assert not any(attrs), "No function should have an external attribute."


def get_range_for_dtype_str(dtype):
    """
    Produces the min,max for a give data type.

    Parameters
    ----------
    dtype : str
        a type string (e.g., int8)

    Returns
    -------
    type_info.min : int
        the minimum of the range
    type_info.max : int
        the maximum of the range
    """

    try:
        type_info = np.iinfo(dtype)
    except ValueError:
        type_info = np.finfo(dtype)
    return type_info.min, type_info.max


def make_module(func):
    """Creates IRModule from Function"""
    func = relay.Function(relay.analysis.free_vars(func), func)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod


def get_same_padding(in_shape, kernel, dilation, stride):
    """
    Provides CMSIS-NN padding when output dim == input dim.
    This is TFLu's "SAME" padding case.
    """
    dilated_kernel_h = dilation[0] * (kernel[0] - 1) + 1
    out = int(math.ceil(float(in_shape[0]) / float(stride[0])))
    pad = max(0, (out - 1) * stride[0] + dilated_kernel_h - in_shape[0])
    pad_top = pad // 2
    pad_bottom = pad - pad_top

    dilated_kernel_w = dilation[1] * (kernel[1] - 1) + 1
    out = int(math.ceil(float(in_shape[1]) / float(stride[1])))
    pad = max(0, (out - 1) * stride[1] + dilated_kernel_w - in_shape[1])
    pad_left = pad // 2
    pad_right = pad - pad_left
    return [pad_top, pad_left, pad_bottom, pad_right]


def get_conv2d_qnn_params(
    weight_shape: List[int],
    input_scale: float,
    input_zp: int,
    weights_scale: Union[float, List[float]],
    weights_zp: int,
    input_dtype: str = "int8",
    weights_dtype: str = "int8",
    output_dtype: str = "int8",
    is_depthwise: bool = False,
) -> Tuple[float, int]:
    """
    Calculate the output quantization parameters for convolution based on the input and
    weights quantization paramters and the data types.

    Parameters
    ----------
    weight_shape : List[int]
        shape of the weights
    input_scale : float
        scale of the input tensor
    input_zp : int
        zero point of the input tensor
    weights_scale : Union[float, List[float]]
        scale(s) of the weights tensor
    weights_zp : int
        zero point of the weights tensor
    is_depthwise : bool
        whether it is a depthwise convolution
    input_dtype : str
        data type of the input tensor
    weights_dtype : str
        data type of the weights tensor
    output_dtype : str
        data type of the output tensor

    Returns
    -------
    output_scale : float
        scale of the output tensor
    output_zp : int
        zero point of the output tensor
    """
    input_dtype_min, input_dtype_max = get_range_for_dtype_str(input_dtype)
    input_max = input_scale * (input_dtype_max - input_zp)
    input_min = input_scale * (input_dtype_min - input_zp)

    weights_dtype_min, weights_dtype_max = get_range_for_dtype_str(weights_dtype)
    weights_sc_max = np.max(weights_scale)
    weights_max = weights_sc_max * (weights_dtype_max - weights_zp)

    weights_sc_min = np.min(weights_scale)
    weights_min = weights_sc_min * (weights_dtype_min - weights_zp)

    weights_h = weight_shape[1]
    weights_w = weight_shape[2]
    channels = weight_shape[3]
    num_elements = weights_h * weights_w * channels
    # Adjust the result if it is a depthwise convolution
    if is_depthwise:
        num_elements = num_elements / channels

    # The smallest and largest possible values in the unquantized output tensor
    output_limits = [
        weights_max * input_max * num_elements,
        weights_min * input_max * num_elements,
        weights_min * input_min * num_elements,
        weights_max * input_min * num_elements,
    ]

    output_max = max(output_limits)
    output_min = min(output_limits)
    output_dtype_min, output_dtype_max = get_range_for_dtype_str(output_dtype)

    output_scale = (output_max - output_min) / (output_dtype_max - output_dtype_min)
    output_zp = int(output_dtype_min - (output_min / output_scale))

    return output_scale, output_zp


def make_qnn_relu(expr, fused_activation_fn, scale, zero_point, dtype):
    """Mimics convert_qnn_fused_activation_function from TFLite frontend"""
    quantize = lambda x: float(int(round(x / scale)) + zero_point)

    # Get min/max of the output dtype. This will be used to ensure that clip a_min/a_max are not
    # beyond the dtype range.
    qmin, qmax = get_range_for_dtype_str(dtype)

    # The input expr is a quantized tensor with its scale and zero point. We calculate the
    # suitable clip off points based on these scale and zero point.
    if fused_activation_fn == "NONE":
        return expr
    if fused_activation_fn == "RELU6":
        return tvm.relay.op.clip(expr, a_min=max(qmin, quantize(0)), a_max=min(qmax, quantize(6.0)))
    if fused_activation_fn == "RELU_N1_TO_1":
        return tvm.relay.op.clip(
            expr, a_min=max(qmin, quantize(-1.0)), a_max=min(qmax, quantize(1.0))
        )
    if fused_activation_fn == "RELU":
        return tvm.relay.op.clip(expr, a_min=max(qmin, quantize(0.0)), a_max=qmax)

    raise ValueError("Invalid argument provided with fused_activation_fn")
