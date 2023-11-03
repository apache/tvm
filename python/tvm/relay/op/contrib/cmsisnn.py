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
            CMSISNNFusePads(),
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
        if (
            (scale == 1.0 / 256 and zero_point == -128)
            and pattern.attrs.out_dtype == "int8"
            and dequantize_call.args[0].checked_type.dtype == "int8"
        ):
            return True

        if (
            (scale == 1.0 / 32768 and zero_point == 0)
            and pattern.attrs.out_dtype == "int16"
            and dequantize_call.args[0].checked_type.dtype == "int16"
        ):
            return True

        return False

    def qnn_conv2d_pattern(with_pad):
        """Create pattern for qnn.conv2D with optional pad and/or optional fused relu."""
        conv2d_input = wildcard()
        if with_pad:
            conv2d_input = is_op("nn.pad")(wildcard(), is_constant())
        qnn_conv2d = is_op("qnn.conv2d")(
            conv2d_input,
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
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
        if str(requantize_input.op.name) == "nn.bias_add":
            bias_add = requantize_input
            conv2d = bias_add.args[0]
        else:
            conv2d = requantize_input
        conv2d_input = conv2d.args[0]
        conv2d_weight = conv2d.args[1]

        # check if depthwise Conv2D
        kernel_layout = conv2d.attrs.kernel_layout
        pos_o = kernel_layout.index("O")
        groups = conv2d.attrs.groups
        is_depthwise = False
        if groups == int(conv2d_input.checked_type.shape[3]) and groups == int(
            conv2d_weight.checked_type.shape[pos_o]
        ):
            is_depthwise = True

        # check if dtypes are supported for the following entities
        # (input_dtype, weight_dtype, bias_dtype, out_dtype, pattern_dtype)
        are_dtypes_valid = False
        conv2d_input_dtype = conv2d_input.checked_type.dtype
        if bias_add:
            bias_dtype = bias_add.args[1].checked_type.dtype
        else:
            # this is only to enable to following check that validates all sorts of dtypes
            bias_dtype = "int32" if conv2d_input_dtype == "int8" else "int64"
        valid_dtypes = None
        if conv2d_input_dtype == "int8":
            valid_dtypes = ("int8", "int8", "int32", "int32", "int8")
        elif conv2d_input_dtype == "int16":
            valid_dtypes = ("int16", "int8", "int64", "int64", "int16")

        if (
            conv2d_input_dtype,
            conv2d_weight.checked_type.dtype,
            bias_dtype,
            conv2d.attrs.out_dtype,
            pattern.checked_type.dtype,
        ) == valid_dtypes:
            are_dtypes_valid = True

        # input_zero_point should be 0 when int16
        valid_input_zp = True
        if conv2d_input_dtype == "int16" and conv2d.args[2].data.numpy().item(0) != 0:
            valid_input_zp = False

        # kernel zero_point should be 0
        kernel_zp = conv2d.args[3].data.numpy()
        kernel_zp = [kernel_zp] if kernel_zp.ndim == 0 else kernel_zp

        # combination of all checks to decide if pattern is eligible for partitioning
        ret = (
            are_dtypes_valid
            and valid_input_zp
            and all([zp == 0 for zp in kernel_zp])
            and (not is_depthwise or bias_add is not None)
        )
        return ret

    def check_qnn_conv2d_pad(pattern):
        """Check if the Pad followed by Conv2D is supported by CMSIS-NN."""
        if str(pattern.op.name) == "clip":
            relu = pattern
            requantize = relu.args[0]
        else:
            requantize = pattern
        requantize_input = requantize.args[0]
        if str(requantize_input.op.name) == "nn.bias_add":
            bias_add = requantize_input
            conv2d = bias_add.args[0]
        else:
            conv2d = requantize_input
        conv2d_input = conv2d.args[0]

        # check if sum of paddings from pad() and conv2d() satisfies CMSIS-NN constraints
        can_pad_be_fused = True
        if isinstance(conv2d_input, tvm.relay.expr.Call) and str(conv2d_input.op.name) == "nn.pad":
            pad_top, pad_left, pad_bottom, pad_right = GetEffectiveConv2DPadding(
                conv2d, conv2d_input
            )
            # check if difference in the side paddings is 1 along each dimension
            pad_w_diff = int(pad_right - pad_left)
            pad_h_diff = int(pad_bottom - pad_top)
            can_pad_be_fused = pad_w_diff in [0, 1] and pad_h_diff in [0, 1]

        ret = check_qnn_conv2d(pattern) and can_pad_be_fused
        return ret

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
        if str(requantize_input.op.name) == "nn.bias_add":
            bias_add = requantize_input
            fc = bias_add.args[0]
        else:
            fc = requantize_input
        fc_input = fc.args[0]
        fc_weight = fc.args[1]

        are_dtypes_valid = False
        fc_input_dtype = fc_input.checked_type.dtype
        if bias_add:
            bias_dtype = bias_add.args[1].checked_type.dtype
        else:
            bias_dtype = "int32" if fc_input_dtype == "int8" else "int64"

        valid_dtypes = None
        if fc_input_dtype == "int8":
            valid_dtypes = ("int8", "int8", "int32", "int32", "int8")
        elif fc_input_dtype == "int16":
            valid_dtypes = ("int16", "int8", "int64", "int64", "int16")

        if (
            fc_input_dtype,
            fc_weight.checked_type.dtype,
            bias_dtype,
            fc.attrs.out_dtype,
            pattern.checked_type.dtype,
        ) == valid_dtypes:
            are_dtypes_valid = True

        # kernel zero_point should be 0
        kernel_zp = fc.args[3].data.numpy().item(0)

        return are_dtypes_valid and kernel_zp == 0

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

        if str(pattern.op.name) == "clip":
            pooling = pattern.args[0].args[0]
        else:
            pooling = pattern.args[0]
        input_op = pooling.args[0].args[0]

        return (
            pooling.attrs.layout == "NHWC"
            and int(input_op.checked_type.shape[0]) == 1
            and (
                (input_op.checked_type.dtype == "int8" and output.checked_type.dtype == "int8")
                or (input_op.checked_type.dtype == "int16" and output.checked_type.dtype == "int16")
            )
        )

    def qnn_max_pool2d_pattern():
        """Matches max pool2d with optional Relu"""
        pattern = is_op("nn.max_pool2d")(wildcard())
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def check_qnn_max_pool2d(pattern):
        """Check if max pool2d is supported by CMSIS-NN."""
        output = pattern

        if str(pattern.op.name) == "clip":
            pooling = pattern.args[0]
        else:
            pooling = pattern
        input_op = pooling.args[0]

        return (
            pooling.attrs.layout == "NHWC"
            and int(input_op.checked_type.shape[0]) == 1
            and (
                (input_op.checked_type.dtype == "int8" and output.checked_type.dtype == "int8")
                or (input_op.checked_type.dtype == "int16" and output.checked_type.dtype == "int16")
            )
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

        # Check arguments are not scalar.
        if (
            isinstance(arg0, tvm.relay.expr.Constant)
            and len(arg0.checked_type.shape) == 0
            and isinstance(arg1, tvm.relay.expr.Constant)
            and len(arg1.checked_type.shape) == 0
        ):
            return False

        arg0_type = arg0.checked_type.dtype
        arg1_type = arg1.checked_type.dtype

        # Check arguments are of valid type.
        if arg0_type not in ["int8", "int16"]:
            return False

        # Check arguments are the same type.
        if arg0_type != arg1_type:
            return False

        # Check zero points are non-zero (arm_elementwise_(add|mul)_s16 does not
        # handle non-zero zero points).
        if arg0_type == "int16" and str(binary_op.op.name) in ["qnn.add", "qnn.mul"]:
            arg_0_zero_point = binary_op.args[3].data.numpy()
            arg_1_zero_point = binary_op.args[5].data.numpy()
            output_zero_point = binary_op.args[7].data.numpy()
            if any([arg_0_zero_point, arg_1_zero_point, output_zero_point]):
                return False

        return True

    return [
        ("cmsis-nn.qnn_conv2d", qnn_conv2d_pattern(with_pad=True), check_qnn_conv2d_pad),
        ("cmsis-nn.qnn_conv2d", qnn_conv2d_pattern(with_pad=False), check_qnn_conv2d),
        ("cmsis-nn.qnn_fully_connected", qnn_fully_connected_pattern(), check_qnn_fully_connected),
        ("cmsis-nn.qnn_avg_pool2d", qnn_avg_pool2d_pattern(), check_qnn_avg_pool2d),
        ("cmsis-nn.qnn_max_pool2d", qnn_max_pool2d_pattern(), check_qnn_max_pool2d),
        ("cmsis-nn.qnn_mul", binary_op_pattern("mul"), check_qnn_binary_op),
        ("cmsis-nn.qnn_add", binary_op_pattern("add"), check_qnn_binary_op),
        ("cmsis-nn.qnn_softmax", qnn_softmax_pattern(), check_qnn_softmax),
    ]
