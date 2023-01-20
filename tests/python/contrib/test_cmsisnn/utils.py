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
from tvm.testing.aot import AOTTestRunner, get_dtype_range


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


def assert_partitioned_function(orig_mod, cmsisnn_mod, expected_ops_unchanged=True):
    """
    if KCompiler attribute is missing, this function raises an assertion.

    Parameters
    ----------
    orig_mod : IRModule
        Pre-partitioning module
    cmsisnn_mod : IRModule
        Post-partitioning module
    is_num_calls_same: bool
        Are number of CallNode(s) before and after partitioning expected to be the same
    """
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

    if expected_ops_unchanged:
        assert count_num_calls(orig_mod) == count_num_calls(
            cmsisnn_mod
        ), "Number of calls changed during partitioning"


def assert_no_external_function(mod):
    attrs = [mod[var.name_hint].attrs for var in mod.get_global_vars() if mod[var.name_hint].attrs]
    assert not any(attrs), "No function should have an external attribute."


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


def get_kernel_bias_dtype(input_dtype):
    """
    Returns (kernel_dtype, bias_dtype) based on input's dtype.
    """
    # uint8 corresponds to an invalid case, so returning int types
    # does not cause tests to break
    if input_dtype in ("int8", "uint8"):
        return ("int8", "int32")
    elif input_dtype == "int16":
        return ("int8", "int64")
    raise ValueError("Invalid dtype provided to get_kernel_bias_dtype()")


def get_conv2d_qnn_params(
    kernel_shape: List[int],
    input_scale: float,
    input_zp: int,
    kernel_scale: Union[float, List[float]],
    kernel_zp: int,
    input_dtype: str = "int8",
    kernel_dtype: str = "int8",
    output_dtype: str = "int8",
    is_depthwise: bool = False,
) -> Tuple[float, int]:
    """
    Calculate the output quantization parameters for convolution based on the input and
    kernel quantization paramters and the data types.

    Parameters
    ----------
    kernel_shape : List[int]
        shape of the kernel
    input_scale : float
        scale of the input tensor
    input_zp : int
        zero point of the input tensor
    kernel_scale : Union[float, List[float]]
        scale(s) of the kernel tensor
    kernel_zp : int
        zero point of the kernel tensor
    is_depthwise : bool
        whether it is a depthwise convolution
    input_dtype : str
        data type of the input tensor
    kernel_dtype : str
        data type of the kernel tensor
    output_dtype : str
        data type of the output tensor

    Returns
    -------
    output_scale : float
        scale of the output tensor
    output_zp : int
        zero point of the output tensor
    """
    input_dtype_min, input_dtype_max = get_dtype_range(input_dtype)
    input_max = input_scale * (input_dtype_max - input_zp)
    input_min = input_scale * (input_dtype_min - input_zp)

    kernel_dtype_min, kernel_dtype_max = get_dtype_range(kernel_dtype)
    kernel_sc_max = np.max(kernel_scale)
    kernel_max = kernel_sc_max * (kernel_dtype_max - kernel_zp)

    kernel_sc_min = np.min(kernel_scale)
    kernel_min = kernel_sc_min * (kernel_dtype_min - kernel_zp)

    kernel_h = kernel_shape[1]
    kernel_w = kernel_shape[2]
    channels = kernel_shape[3]
    num_elements = kernel_h * kernel_w * channels
    # Adjust the result if it is a depthwise convolution
    if is_depthwise:
        num_elements = num_elements / channels

    # The smallest and largest possible values in the unquantized output tensor
    output_limits = [
        kernel_max * input_max * num_elements,
        kernel_min * input_max * num_elements,
        kernel_min * input_min * num_elements,
        kernel_max * input_min * num_elements,
    ]

    output_max = max(output_limits)
    output_min = min(output_limits)
    output_dtype_min, output_dtype_max = get_dtype_range(output_dtype)

    output_scale = (output_max - output_min) / (output_dtype_max - output_dtype_min)
    output_zp = int(output_dtype_min - (output_min / output_scale))

    return output_scale, output_zp


def make_qnn_relu(expr, fused_activation_fn, scale, zero_point, dtype):
    """Mimics convert_qnn_fused_activation_function from TFLite frontend"""
    quantize = lambda x: float(int(round(x / scale)) + zero_point)

    # Get min/max of the output dtype. This will be used to ensure that clip a_min/a_max are not
    # beyond the dtype range.
    qmin, qmax = get_dtype_range(dtype)

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


class CheckForPadsWithinCompositeFunc(tvm.relay.ExprVisitor):
    """Provides method to test number of pads present inside the function being visited."""

    def __init__(self):
        super().__init__()
        self.num_pads_ = 0

    def visit_call(self, call):
        super().visit_call(call)
        if (
            isinstance(call, tvm.relay.Call)
            and isinstance(call.op, tvm.ir.op.Op)
            and call.op.name == "nn.pad"
        ):
            self.num_pads_ += 1

    def assert_no_pads_within_func(self):
        assert self.num_pads_ == 0, "CMSIS-NN composite function should not have pads."

    def assert_pads_within_func(self):
        assert self.num_pads_ > 0, "Composite function should have pads within it."


def create_test_runner(compiler_cpu="cortex-m55", cpu_flags=""):
    """
    Creates AOT test runner for CMSIS-NN tests.

    Parameters
    ----------
    compiler_cpu : str
       Equivalent of gcc option mcpu
       Options:  cortex-m55, cortex-m7
    cpu_flags: str
        Disable Arm(R) Cortex(R)-M profile vector extension (mve)
        Options:
        Arm(R) Cortex(R)-M55: when null +mve is set by default.
            +nomve disables vector extensions.
        Arm(R) Cortex(R)-M7 does not support mve.
    """
    # cmsis_cpu is used to find out start up code inside CMSIS package
    cmsis_cpu = "ARMCM7" if compiler_cpu == "cortex-m7" else "ARMCM55"
    mfloat_abi = "soft" if compiler_cpu == "cortex-m7" else "hard"
    return AOTTestRunner(
        makefile="corstone300",
        prologue="""
        UartStdOutInit();
        """,
        includes=["uart_stdout.h"],
        pass_config={
            "relay.ext.cmsisnn.options": {
                "mcpu": compiler_cpu + cpu_flags,
            },
            "tir.usmp.enable": True,
            "tir.disable_storage_rewrite": True,
        },
        parameters={
            "ARM_CPU": cmsis_cpu,
            "MCPU": compiler_cpu,
            "MCPU_FLAGS": cpu_flags,
            "MFLOAT_ABI": mfloat_abi,
        },
    )
