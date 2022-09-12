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
"""Arm(R) Ethos(TM)-N NPU supported operators."""
from enum import Enum
import warnings
from distutils.version import LooseVersion

import tvm.ir
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import is_constant, is_op, wildcard
from . import _ethosn
from .register import register_pattern_table


class Available(Enum):
    UNAVAILABLE = 0
    SW_ONLY = 1
    SW_AND_HW = 2

    def __bool__(self):
        return self != Available.UNAVAILABLE


def ethosn_available():
    """Return whether Ethos-N software and hardware support is available"""
    if not tvm.get_global_func("relay.ethos-n.query", True):
        print("skip because Ethos-N module is not available")
        return Available.UNAVAILABLE
    hw = tvm.get_global_func("relay.ethos-n.query")()
    return Available.SW_AND_HW if hw else Available.SW_ONLY


def ethosn_api_version() -> str:
    """
    Returns the semantic version of the driver stack api that is
    being used.

    Returns
    -------
    str
        Semantic version string (e.g. 3.0.1).
    """
    return tvm.get_global_func("relay.ethos-n.api.version")()


def ConvertEquivalents() -> tvm.ir.IRModule:  # pylint: disable=invalid-name
    """Converts operations into a numerically equivalent form
    that can be understood by the NPU codegen.

    Return
    ------
    Pass
        The module pass.
    """
    return _ethosn.ConvertEquivalents()


def partition_for_ethosn(mod, params=None, **opts):
    """Partition the graph greedily offloading supported
    operators to Arm Ethos-N NPU.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    opts = opts or {}
    if "variant" not in opts:
        raise ValueError("Please specify a variant in the target string, e.g. -variant=n78.")

    # -variant=ethos-n78 deprecated in favour of -variant=n78
    if opts["variant"].lower() == "ethos-n78":
        warnings.warn(
            "Please use '-variant=n78' instead of the deprecated "
            "'-variant=ethos-n78', which will be removed in TVM v0.9.",
            DeprecationWarning,
        )
    elif opts["variant"] != "n78":
        raise ValueError("When targeting Ethos(TM)-N78, -variant=n78 should be set.")

    api_version = ethosn_api_version()
    supported_api_versions = ["3.0.1", "3.1.0"]
    if all(api_version != LooseVersion(exp_ver) for exp_ver in supported_api_versions):
        raise ValueError(
            f"Driver stack version {api_version} is unsupported. "
            f"Please use version in {supported_api_versions}."
        )

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("ethos-n"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            ConvertEquivalents(),
        ]
    )
    return seq(mod)


@register_pattern_table("ethos-n")
def pattern_table():
    """Get the Ethos-N compiler pattern table."""

    def qnn_conv_pattern():
        pattern = is_op("nn.pad")(wildcard(), wildcard()) | wildcard()
        pattern = is_op("qnn.conv2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("nn.bias_add")(pattern, is_constant())
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_fc_pattern():
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_op("nn.bias_add")(pattern, is_constant())
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_avg_pool2d_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def qnn_sigmoid_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("sigmoid")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_mean_pattern():
        pattern = is_op("cast")(wildcard())
        pattern = is_op("mean")(pattern)
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_tanh_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("tanh")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_leaky_relu_pattern():
        pattern = is_op("qnn.dequantize")(wildcard(), is_constant(), is_constant())
        pattern = is_op("nn.leaky_relu")(pattern)
        pattern = is_op("qnn.quantize")(pattern, is_constant(), is_constant())
        return pattern

    def qnn_requantize_pattern():
        pattern = is_op("qnn.requantize")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_resize_pattern():
        pattern = is_op("image.resize2d")(wildcard()).has_attr({"method": "nearest_neighbor"})
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def qnn_mul_pattern():
        """
        Multiply is supported when one input is a constant of shape [1, ..., C],
        where C matches the number of channels of the other input.
        """
        mul_op = is_op("qnn.mul")
        gen_mul_inputs = lambda x, y: mul_op(
            x,
            y,
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        input_is_left = gen_mul_inputs(wildcard(), is_constant())
        input_is_right = gen_mul_inputs(is_constant(), wildcard())
        return input_is_left | input_is_right

    def qnn_add_pattern():
        add_op = is_op("qnn.add")
        gen_add_inputs = lambda x, y: add_op(
            x,
            y,
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        two_inputs = gen_add_inputs(wildcard(), wildcard())
        input_is_left = gen_add_inputs(wildcard(), is_constant())
        input_is_right = gen_add_inputs(is_constant(), wildcard())

        return input_is_left | input_is_right | two_inputs

    def qnn_conv2d_transpose_pattern():
        pattern = is_op("qnn.conv2d_transpose")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        ).has_attr({"data_layout": "NHWC"})
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = is_op("qnn.requantize")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant()
        )
        return pattern

    def check_conv2d(extract):
        """Check if a conv2d is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.conv2d(extract)

    def check_fc(extract):
        """Check if a fully connected is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.fc(extract)

    def check_avg_pool2d(extract):
        """Check if a avg pool2d is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.avg_pool2d(extract)

    def check_mean(extract):
        """Check if mean is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.mean(extract)

    def check_conv2d_transpose(extract):
        """Check if conv2d_transpose is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.conv2d_transpose(extract)

    def check_sigmoid(extract):
        """Check if a sigmoid is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.sigmoid(extract)

    def check_tanh(extract):
        """Check if tanh is supported by Ethos-N."""
        if not ethosn_available():
            return False

        return _ethosn.tanh(extract)

    def check_leaky_relu(extract):
        """Check if Leaky ReLU is supported."""
        if not ethosn_available():
            return False

        return _ethosn.leaky_relu(extract)

    def check_mul(extract):
        """Check if Mul is supported."""
        if not ethosn_available():
            return False
        # Do not support scalar constants for now
        check_scalar = lambda i: isinstance(i, tvm.relay.Constant) and len(i.data.shape) == 0
        if check_scalar(extract.args[0]) or check_scalar(extract.args[1]):
            return False
        extract = _ethosn.ConvertQnnMultiply(extract)
        return _ethosn.conv2d(extract)

    def check_requantize(extract):
        """Check if requantize is supported."""
        if not ethosn_available():
            return False

        return _ethosn.requantize(extract)

    def check_resize(extract):
        """Check if resize (nearest neighbor) is supported."""
        if not ethosn_available():
            return False

        return _ethosn.resize(extract)

    def check_add(extract):
        """Check if an addition is supported by Ethos-N."""
        if not ethosn_available():
            return False
        # Do not support scalar constants for now
        check_scalar = lambda i: isinstance(i, tvm.relay.Constant) and len(i.data.shape) == 0
        if check_scalar(extract.args[0]) or check_scalar(extract.args[1]):
            return False

        inputs = extract.args[0:2]
        if any([isinstance(i, tvm.relay.Constant) for i in inputs]):
            extract = _ethosn.ConvertQnnAdd(extract)
            return _ethosn.conv2d(extract)
        return _ethosn.addition(extract)

    return [
        ("ethos-n.qnn_mul", qnn_mul_pattern(), check_mul),
        ("ethos-n.qnn_add", qnn_add_pattern(), check_add),
        ("ethos-n.qnn_conv2d", qnn_conv_pattern(), check_conv2d),
        ("ethos-n.qnn_conv2d_transpose", qnn_conv2d_transpose_pattern(), check_conv2d_transpose),
        ("ethos-n.qnn_avg_pool2d", qnn_avg_pool2d_pattern(), check_avg_pool2d),
        ("ethos-n.qnn_sigmoid", qnn_sigmoid_pattern(), check_sigmoid),
        ("ethos-n.qnn_fc", qnn_fc_pattern(), check_fc),
        ("ethos-n.qnn_mean", qnn_mean_pattern(), check_mean),
        ("ethos-n.qnn_tanh", qnn_tanh_pattern(), check_tanh),
        ("ethos-n.qnn_leaky_relu", qnn_leaky_relu_pattern(), check_leaky_relu),
        ("ethos-n.qnn_resize", qnn_resize_pattern(), check_resize),
        ("ethos-n.qnn_requantize", qnn_requantize_pattern(), check_requantize),
    ]


def _is_ethosn_composite(node):
    if isinstance(node, tvm.relay.expr.Call) and isinstance(node.op, tvm.relay.Function):
        if "Composite" in node.op.attrs:
            comp_name = node.op.attrs["Composite"]
            return comp_name.split(".")[0] == "ethos-n"

    return False


@tvm.ir.register_op_attr("nn.max_pool2d", "target.ethos-n")
def max_pool2d(expr):
    """Check if a max pool2d is supported by Ethos-N."""
    if not ethosn_available():
        return False

    return _ethosn.max_pool2d(expr)


@tvm.ir.register_op_attr("reshape", "target.ethos-n")
def reshape(expr):
    """Check if a reshape is supported by Ethos-N."""
    if not ethosn_available():
        return False

    return _ethosn.reshape(expr)


@tvm.ir.register_op_attr("qnn.concatenate", "target.ethos-n")
def qnn_concatenate(expr):
    """Check if a concatenate is supported by Ethos-N."""
    if not ethosn_available():
        return False
    if not _ethosn.concatenate(expr):
        return False

    # Support library has some unenforced restrictions on qnn params
    args = expr.args
    min_range = 1e9
    max_range = -1e9
    qnn_params = []
    for i in range(len(args[1].fields)):
        scale = args[1].fields[i].data.numpy()
        zero_point = args[2].fields[i].data.numpy()
        min_range = min(-1 * zero_point * scale, min_range)
        max_range = max((255 - zero_point) * scale, max_range)
        qnn_params.append((scale, zero_point))

    scale = (max_range - min_range) / 255
    zero_point = int(-min_range / scale)
    if (scale, zero_point) in qnn_params:
        return True

    return False


@tvm.ir.register_op_attr("split", "target.ethos-n")
def split(expr):
    """Check if a split is supported by Ethos-N."""
    if not ethosn_available():
        return False
    if ethosn_api_version() == LooseVersion("3.0.1"):
        return False
    if not _ethosn.split(expr):
        return False

    return True


@tvm.ir.register_op_attr("nn.depth_to_space", "target.ethos-n")
def depth_to_space(expr):
    """Check if a depth_to_space is supported by Ethos-N."""
    if not ethosn_available():
        return False
    if not _ethosn.depth_to_space(expr):
        return False

    return True


@tvm.ir.register_op_attr("clip", "target.ethos-n")
def clip(expr):
    """Check if a clip is supported by Ethos-N."""
    if not ethosn_available():
        return False
    if not _ethosn.relu(expr):
        return False

    return True
