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


def _find_last(pattern):
    if hasattr(pattern, "args"):
        return _find_last(pattern.args[0])
    return pattern


def partition_for_cmsisnn(mod, params=None, mod_name="default", **opts):
    """Partition the graph greedily offloading supported
    operators on Cortex-M using CMSIS-NN

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    mod_name: str, optional
        The module name

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
            transform.PartitionGraph(mod_name=mod_name),
            GenerateCMSISNNConstants(),
            ScalarToTensorConstants(),
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

        # check if depthwise Conv2D
        kernel_layout = conv2d.attrs.kernel_layout
        pos_o = kernel_layout.index("O")
        groups = conv2d.attrs.groups
        is_depthwise = False
        if groups == int(conv2d_input.checked_type.shape[3]) and groups == int(
            conv2d_weight.checked_type.shape[pos_o]
        ):
            is_depthwise = True

        return (
            conv2d.attrs.out_dtype == "int32"
            and conv2d_input.checked_type.dtype == "int8"
            and conv2d_weight.checked_type.dtype == "int8"
            and pattern.checked_type.dtype == "int8"
            and bias_dtype == "int32"
            and all([zp == 0 for zp in kernel_zp])
            and (not is_depthwise or bias_add is not None)
        )

    def qnn_fully_connected_pattern():
        """Create pattern for qnn.dense with optional Relu."""
        qnn_fc = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        bias_add = is_op("nn.bias_add")(qnn_fc, is_constant())
        req = is_op("qnn.requantize")(
            qnn_fc | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
        )
        clip_or_req = req.optional(is_op("clip"))
        return clip_or_req

    def check_qnn_fully_connected(pattern):
        """Check if the fully connected is supported by CMSIS-NN."""
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
            fc = bias_add.args[0]
            bias_dtype = bias_add.args[1].checked_type.dtype
        else:
            fc = requantize_input
        fc_input = fc.args[0]
        fc_weight = fc.args[1]

        # kernel zero_point should be 0
        kernel_zp = fc.args[3].data.numpy().item(0)

        return (
            fc.attrs.out_dtype == "int32"
            and fc_input.checked_type.dtype == "int8"
            and fc_weight.checked_type.dtype == "int8"
            and pattern.checked_type.dtype == "int8"
            and bias_dtype == "int32"
            and kernel_zp == 0
        )

    def qnn_avg_pool2d_pattern():
        """Matches average pooling with optional Relu"""
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def check_qnn_avg_pool2d(pattern):
        """Check if avg pool2d is supported by CMSIS-NN."""
        output = pattern
        input_var = _find_last(pattern)

        if str(pattern.op.name) == "clip":
            pooling = pattern.args[0].args[0]
        else:
            pooling = pattern.args[0]

        return (
            pooling.attrs.layout == "NHWC"
            and bool(input_var.checked_type.shape[0] == 1)
            and input_var.checked_type.dtype == "int8"
            and output.checked_type.dtype == "int8"
        )

    def qnn_max_pool2d_pattern():
        """Matches max pool2d with optional Relu"""
        pattern = is_op("nn.max_pool2d")(wildcard())
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def check_qnn_max_pool2d(pattern):
        """Check if max pool2d is supported by CMSIS-NN."""
        output = pattern
        input_var = _find_last(pattern)

        if str(pattern.op.name) == "clip":
            pooling = pattern.args[0]
        else:
            pooling = pattern

        return (
            pooling.attrs.layout == "NHWC"
            and bool(input_var.checked_type.shape[0] == 1)
            and input_var.checked_type.dtype == "int8"
            and output.checked_type.dtype == "int8"
        )

    def binary_op_pattern(op):
        """Matches QNN binary operation"""
        pattern = is_op(f"qnn.{op}")(
            wildcard(),
            wildcard(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        return pattern.optional(is_op("clip"))

    def check_qnn_binary_op(pattern):
        """Check if binary op is supported by CMSIS-NN."""
        binary_op = pattern
        if str(pattern.op.name) == "clip":
            binary_op = pattern.args[0]

        arg0 = binary_op.args[0]
        arg1 = binary_op.args[1]
        both_args_scalar = False
        if (
            isinstance(arg0, tvm.relay.expr.Constant)
            and len(arg0.checked_type.shape) == 0
            and isinstance(arg1, tvm.relay.expr.Constant)
            and len(arg1.checked_type.shape) == 0
        ):
            both_args_scalar = True

        return (
            arg0.checked_type.dtype == "int8"
            and arg1.checked_type.dtype == "int8"
            and not both_args_scalar
        )

    return [
        ("cmsis-nn.qnn_conv2d", qnn_conv2d_pattern(), check_qnn_conv2d),
        ("cmsis-nn.qnn_fully_connected", qnn_fully_connected_pattern(), check_qnn_fully_connected),
        ("cmsis-nn.qnn_avg_pool2d", qnn_avg_pool2d_pattern(), check_qnn_avg_pool2d),
        ("cmsis-nn.qnn_max_pool2d", qnn_max_pool2d_pattern(), check_qnn_max_pool2d),
        ("cmsis-nn.qnn_mul", binary_op_pattern("mul"), check_qnn_binary_op),
        ("cmsis-nn.qnn_add", binary_op_pattern("add"), check_qnn_binary_op),
        ("cmsis-nn.qnn_softmax", qnn_softmax_pattern(), check_qnn_softmax),
    ]
