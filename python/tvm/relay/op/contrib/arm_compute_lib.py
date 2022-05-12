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
# pylint: disable=invalid-name, unused-argument, dangerous-default-value
"""Arm Compute Library supported operators."""
import tvm
from tvm import relay
from tvm._ffi import register_func
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr import const

from ...dataflow_pattern import is_constant, is_expr, is_op, is_tuple, wildcard
from ..strategy.generic import is_depthwise_conv2d
from .register import register_pattern_table


def is_arm_compute_runtime_enabled():
    """Check if the ACL graph executor is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_arm_compute_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def partition_for_arm_compute_lib(mod, params=None, disabled_ops=["concatenate"], **opts):
    """Partition the graph greedily offloading supported
    operators to Arm Compute Library.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    disabled_ops : Optional[list]
        Ops do not want to offload to ACL.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(arm_compute_lib_pattern_table(disabled_ops)),
            transform.AnnotateTarget("arm_compute_lib", False),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)


@register_func("relay.ext.arm_compute_lib.optimize")
def preprocess_module(mod):
    """
    Pre-process a module containing functions ready for ACL codegen. For now we enforce OHWI
    kernel layout and fold the transforms away.

    Parameters
    ----------
    mod : Module
        The module to run passes on.

    Returns
    -------
    preprocessed_mod : The processed module.
    """

    def convert_layout_conv2d(conv2d_function):
        def convert_conv(attrs, inputs, tinfos, desired_layouts):
            new_attrs = dict(attrs)
            data_info = tinfos[0]
            weight_info = tinfos[1]
            desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
            new_attrs["data_layout"] = desired_data_layout
            new_attrs["kernel_layout"] = desired_kernel_layout

            if is_depthwise_conv2d(
                data_info.shape,
                attrs["data_layout"],
                weight_info.shape,
                attrs["kernel_layout"],
                attrs["groups"],
            ):
                dkl = desired_kernel_layout
                new_attrs["kernel_layout"] = dkl[3] + dkl[1:3] + dkl[0]
            return conv2d_function(*inputs, **new_attrs)

        return convert_conv

    with OpAttrContext(
        "nn.conv2d", "FTVMConvertOpLayout", convert_layout_conv2d(tvm.relay.nn.conv2d)
    ), OpAttrContext(
        "qnn.conv2d", "FTVMConvertOpLayout", convert_layout_conv2d(tvm.relay.qnn.op.conv2d)
    ):
        seq = tvm.transform.Sequential(
            [
                transform.ConvertLayout(
                    {"nn.conv2d": ["NHWC", "OHWI"], "qnn.conv2d": ["NHWC", "OHWI"]}
                ),
                transform.FoldConstant(),
            ]
        )
        preprocessed_mod = seq(mod)
    return preprocessed_mod


@register_pattern_table("arm_compute_lib")
def arm_compute_lib_pattern_table(disabled_ops=["concatenate"]):
    """Get the ACL pattern table."""

    def conv_pattern():
        """Create a convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.pad")(wildcard(), wildcard()) | wildcard()
        pattern = is_op("nn.conv2d")(pattern, is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(is_op("nn.relu"))
        return pattern

    def qnn_conv_pattern():
        """Create a quantized convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.pad")(wildcard(), wildcard()) | wildcard()
        pattern = is_op("qnn.conv2d")(
            pattern, is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = is_op("qnn.requantize")(
            pattern, wildcard(), wildcard(), is_constant(), is_constant()
        )
        return pattern

    def dense_pattern():
        """Create a dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern

    def qnn_dense_pattern():
        """Create a quantized dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = is_op("qnn.requantize")(
            pattern, wildcard(), wildcard(), is_constant(), is_constant()
        )
        return pattern

    def avg_pool2d_pattern():
        """Creates a pattern that matches either quantized
        avg_pool2d or quantized global_avg_pool2d.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("cast")(wildcard())
        pattern = is_op("nn.avg_pool2d")(pattern) | is_op("nn.global_avg_pool2d")(pattern)
        pattern = is_op("cast")(pattern)
        return pattern

    def l2_pool2d_pattern():
        """Create an l2 pooling pattern from equivalent relay operators.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("power")(wildcard(), is_expr(const(2.0)))
        pattern = is_op("nn.avg_pool2d")(pattern)
        pattern = is_op("sqrt")(pattern)
        return pattern

    def concatenate_pattern():
        """Create an concatenate pattern from equivalent relay operators.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the concatenate pattern.
        """
        pattern = is_op("concatenate")(is_tuple(None))
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by ACL."""
        call = extract
        while call.op.name != "nn.conv2d":
            call = call.args[0]
        return conv2d(call)

    def check_qnn_conv(extract):
        """Check qnn conv pattern is supported by ACL."""
        if extract.attrs.out_dtype != "uint8":
            return False
        call = extract
        while call.op.name != "qnn.conv2d":
            call = call.args[0]
        return qnn_conv2d(call)

    def check_dense(extract):
        """Check conv pattern is supported by ACL."""
        call = extract
        while call.op.name != "nn.dense":
            call = call.args[0]
        return dense(call)

    def check_qnn_dense(extract):
        """Check qnn conv pattern is supported by ACL."""
        if extract.attrs.out_dtype != "uint8":
            return False
        call = extract
        while call.op.name != "qnn.dense":
            call = call.args[0]
        return qnn_dense(call)

    def check_avg_pool2d(extract):
        """Check average pool2d pattern is supported by ACL."""
        if extract.attrs.dtype != "uint8":
            return False
        pool = extract.args[0]
        if pool.args[0].attrs.dtype != "int32":
            return False
        return avg_pool2d(pool, from_quantized_composite=True)

    def check_l2_pool2d(extract):
        """Check l2 pool2d pattern is supported by ACL."""
        pool = extract.args[0]
        return avg_pool2d(pool)

    def check_concatenate(expr):
        """Check concatenate pattern is supported by ACL."""
        if "concatenate" in disabled_ops:
            return False
        attrs, type_args = expr.attrs, expr.type_args
        for idx in range(len(type_args[0].fields)):
            if type_args[0].fields[idx].dtype not in ["float32", "uint8"]:
                return False
        # ACL concatenate only supports maximum 4 dimensions input tensor
        if attrs.axis not in [-4, -3, -2, -1, 0, 1, 2, 3]:
            return False
        return True

    return [
        ("arm_compute_lib.conv2d", conv_pattern(), check_conv),
        ("arm_compute_lib.qnn_conv2d", qnn_conv_pattern(), check_qnn_conv),
        ("arm_compute_lib.dense", dense_pattern(), check_dense),
        ("arm_compute_lib.qnn_dense", qnn_dense_pattern(), check_qnn_dense),
        ("arm_compute_lib.qnn_conv2d", qnn_conv_pattern(), check_qnn_conv),
        ("arm_compute_lib.avg_pool2d", avg_pool2d_pattern(), check_avg_pool2d),
        ("arm_compute_lib.l2_pool2d", l2_pool2d_pattern(), check_l2_pool2d),
        ("arm_compute_lib.concatenate", concatenate_pattern(), check_concatenate),
    ]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.arm_compute_lib")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("reshape")


@tvm.ir.register_op_attr("nn.conv2d", "target.arm_compute_lib")
def conv2d(expr):
    """Check if the external ACL codegen for conv2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "float32":
        return False
    is_depthwise = is_depthwise_conv2d(
        data_typ.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        return depthwise_conv2d(attrs, args)
    # ACL doesn't support grouped convolution
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


def qnn_conv2d(expr):
    """Check if the external ACL codegen for qnn.conv2d should be used."""
    attrs, args = expr.attrs, expr.args

    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "int32" and attrs.out_dtype != "":
        return False
    data_typ = args[0].checked_type
    if len(data_typ.shape) != 4 or data_typ.shape[0] != 1 or data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 4 or kernel_typ.dtype != "uint8":
        return False
    is_depthwise = is_depthwise_conv2d(
        data_typ.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        return depthwise_conv2d(attrs, args)
    # ACL doesn't support grouped convolution
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


def depthwise_conv2d(attrs, args):
    """Check if the external ACL codegen for depthwise convolution should be used.

    Note
    ----
    Relay does not have a depthwise conv2d operator whilst ACL does. We simply
    separate the checks for depthwise for clarity.
    """
    kernel_typ = args[1].checked_type
    # Only supports 3x3, 5x5 depthwise
    if (
        kernel_typ.shape[0] not in [3, 5]
        or kernel_typ.shape[1] not in [3, 5]
        or kernel_typ.shape[0] != kernel_typ.shape[1]
    ):
        return False
    # Stride must be (1, 1) or (2, 2)
    if (attrs.strides[0], attrs.strides[1]) not in [(1, 1), (2, 2)]:
        return False
    return True


@tvm.ir.register_op_attr("nn.dense", "target.arm_compute_lib")
def dense(expr):
    """Check if the external ACL codegen for dense should be used."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "float32":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "float32":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


def qnn_dense(expr):
    """Check if the external ACL codegen for qnn.dense should be used."""
    attrs, args = expr.attrs, expr.args
    data_typ = args[0].checked_type
    if data_typ.dtype != "uint8":
        return False
    kernel_typ = args[1].checked_type
    if len(kernel_typ.shape) != 2 or kernel_typ.dtype != "uint8":
        return False
    if attrs.out_dtype != "int32":
        return False
    return True


def check_dilation(attrs):
    """Prevents offloading if dilation other than (1, 1)"""
    if not isinstance(attrs, relay.op.op_attrs.GlobalPool2DAttrs):
        if not (len(attrs.dilation) == 2 and attrs.dilation[0] == 1 and attrs.dilation[1] == 1):
            return False
    return True


@tvm.ir.register_op_attr("nn.max_pool2d", "target.arm_compute_lib")
def max_pool2d(expr):
    """Check if the external ACL codegen for maxpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    typ = args[0].checked_type
    if typ.dtype not in ["float32", "uint8"]:
        return False
    return check_dilation(attrs)


@tvm.ir.register_op_attr("nn.avg_pool2d", "target.arm_compute_lib")
def avg_pool2d(expr, from_quantized_composite=False):
    """Check if the external ACL codegen for avgpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type

    if from_quantized_composite:
        if typ.dtype != "int32":
            return False
    else:
        if typ.dtype not in ["float32"]:
            return False
    if attrs.layout != "NHWC":
        return False

    return check_dilation(attrs)


@tvm.ir.register_op_attr("nn.global_max_pool2d", "target.arm_compute_lib")
def global_max_pool2d(expr):
    """Check if the external ACL codegen for gloval_maxpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type
    if typ.dtype not in ["float32", "uint8"]:
        return False
    if attrs.layout != "NHWC":
        return False
    return True


@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.arm_compute_lib")
def global_avg_pool2d(expr):
    """Check if the external ACL codegen for global_avgpool2d should be used."""
    attrs, args = expr.attrs, expr.args
    typ = args[0].checked_type
    if typ.dtype not in ["float32"]:
        return False
    if attrs.layout != "NHWC":
        return False
    return True


@tvm.ir.register_op_attr("maximum", "target.arm_compute_lib")
def maximum(expr):
    """Check if the external ACL codegen for maximum should be used."""
    args = expr.args
    type_a = args[0].checked_type
    type_b = args[0].checked_type
    return (type_a.dtype == "float32") and (type_b.dtype == "float32")


@tvm.ir.register_op_attr("add", "target.arm_compute_lib")
def add(expr):
    """Check if the external ACL codegen for add should be used."""
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "float32":
            return False

    return True


@tvm.ir.register_op_attr("qnn.add", "target.arm_compute_lib")
def qnn_add(expr):
    """Check if the external ACL codegen for add should be used."""
    args = expr.args
    for typ in [args[0].checked_type, args[1].checked_type]:
        if typ.dtype != "uint8":
            return False

    return True


class OpAttrContext(object):
    """Temporarily changes the attr of an op."""

    def __init__(self, op_name, attr_key, attr_value):
        """Saves the required info for RAII pattern usage.

        Parameters
        ----------
        op_name : str
            The op name.

        attr_key : str
            The attribute name.

        attr_value : object
            The attribute value.
        """
        self.op = relay.op.get(op_name)
        self.attr_key = attr_key
        self.attr_value = attr_value

    def __enter__(self):
        self.older_attr = self.op.get_attr(self.attr_key)
        self.op.reset_attr(self.attr_key)
        self.op.set_attr(self.attr_key, self.attr_value)
        return self

    def __exit__(self, ptype, value, trace):
        self.op.reset_attr(self.attr_key)
        if self.older_attr:
            self.op.set_attr(self.attr_key, self.older_attr)
