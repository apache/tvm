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
# pylint: disable=invalid-name, unused-argument
"""Arm(R) CMSIS-NN supported operators for Cortex-M."""
import tvm.ir
from tvm.target import Target
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import is_constant, is_op, wildcard
from .register import register_pattern_table

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


def enabled():
    return "cmsis-nn" in Target.list_kinds()


def partition_for_cmsisnn(mod, params=None, **opts):
    """Partition the graph greedily offloading supported
    operators on Cortex-M using CMSIS-NN

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : Module
        annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("cmsis-nn"),
            transform.PartitionGraph(),
            GenerateCMSISNNConstants(),
            ExtractConstantsFromPartitionedFunction(),
            transform.InferType(),
        ]
    )
    return seq(mod)


@register_pattern_table("cmsis-nn")
def pattern_table():
    """Get the CMSIS-NN compiler pattern table."""

    def qnn_softmax_pattern():
        """Create pattern for quantized softmax"""
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.softmax")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def check_qnn_softmax(pattern):
        """Check if softmax is supported by CMSIS-NN."""
        dequantize_call = pattern.args[0].args[0]
        scale = pattern.args[1].data.numpy().item(0)
        zero_point = pattern.args[2].data.numpy().item(0)

        # check for dtypes of quantize and dequantize
        return (
            (scale == 1.0 / 256 and zero_point == -128)
            and pattern.attrs.out_dtype == "int8"
            and dequantize_call.args[0].checked_type.dtype == "int8"
        )

    def qnn_conv2d_pattern():
        """Create pattern for qnn.conv2D with optional fused relu."""
        qnn_conv2d = is_op("qnn.conv2d")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
        req = is_op("qnn.requantize")(
            qnn_conv2d | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
        )
        clip_or_req = req.optional(is_op("clip"))
        return clip_or_req

    def check_qnn_conv2d(pattern):
        """Check if the Conv2D is supported by CMSIS-NN."""
        if str(pattern.op.name) == "clip":
            relu = pattern
            requantize = relu.args[0]
        else:
            requantize = pattern
        requantize_input = requantize.args[0]
        bias_add = None
        bias_dtype = "int32"
        if str(requantize_input.op.name) == "nn.bias_add":
            bias_add = requantize_input
            conv2d = bias_add.args[0]
            bias_dtype = bias_add.args[1].checked_type.dtype
        else:
            conv2d = requantize_input
        conv2d_input = conv2d.args[0]
        conv2d_weight = conv2d.args[1]

        # kernel zero_point should be 0
        kernel_zp = conv2d.args[3].data.numpy()
        kernel_zp = [kernel_zp] if kernel_zp.ndim == 0 else kernel_zp

        return (
            conv2d.attrs.out_dtype == "int32"
            and conv2d.attrs.padding[2] == 0
            and conv2d.attrs.padding[3] == 0
            and conv2d_input.checked_type.dtype == "int8"
            and conv2d_weight.checked_type.dtype == "int8"
            and pattern.checked_type.dtype == "int8"
            and bias_dtype == "int32"
            and all([zp == 0 for zp in kernel_zp])
        )

    def binary_op_pattern(op):
        """Matches QNN binary operation"""
        return is_op(f"qnn.{op}")(
            wildcard(),
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )

    def check_qnn_binary_op(extract):
        """Check if multiply is supported by CMSIS-NN."""
        return (
            extract.args[0].checked_type.dtype == "int8"
            and extract.args[1].checked_type.dtype == "int8"
        )

    return [
        ("cmsis-nn.qnn_softmax", qnn_softmax_pattern(), check_qnn_softmax),
        ("cmsis-nn.qnn_conv2d", qnn_conv2d_pattern(), check_qnn_conv2d),
        ("cmsis-nn.qnn_mul", binary_op_pattern("mul"), check_qnn_binary_op),
        ("cmsis-nn.qnn_add", binary_op_pattern("add"), check_qnn_binary_op),
    ]
