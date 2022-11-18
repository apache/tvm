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
"""CLML Library supported operators."""
import tvm

from tvm import relay
from tvm._ffi import register_func
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import wildcard, is_op, is_constant, is_tuple_get_item, is_tuple
from .register import register_pattern_table
from ..strategy.generic import is_depthwise_conv2d


def is_clml_runtime_enabled():
    """Check if the CLML graph runtime is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_clml_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def partition_for_clml(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to CLML Library.

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

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.MergeComposite(clml_pattern_table()),
            transform.AnnotateTarget("clml", False),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    result_mod = seq(mod)
    return result_mod


@register_func("relay.ext.clml.optimize")
def preprocess_module(mod):
    """
    Pre-process a module containing functions ready for CLML codegen. For now we enforce OIHW
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
                new_attrs["kernel_layout"] = dkl[1] + dkl[0] + dkl[2] + dkl[3]
            return conv2d_function(*inputs, **new_attrs)

        return convert_conv

    with OpAttrContext(
        "nn.conv2d", "FTVMConvertOpLayout", convert_layout_conv2d(tvm.relay.nn.conv2d)
    ):
        seq = tvm.transform.Sequential(
            [
                transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"]}),
                transform.FoldConstant(),
            ]
        )
        preprocessed_mod = seq(mod)
    return preprocessed_mod


@register_pattern_table("clml")
def clml_pattern_table():
    """Get the CLML pattern table."""

    def conv_pattern():
        """Create a convolution pattern."""
        pattern = is_op("nn.conv2d")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(
            lambda x: is_tuple_get_item(
                is_op("nn.batch_norm")(
                    x, is_constant(), is_constant(), is_constant(), is_constant()
                )
            )
        )
        pattern = pattern.optional(is_op("nn.relu"))
        pattern = pattern.optional(is_op("clip"))
        return pattern

    def batch_norm_pattern():
        """Create a batch norm pattern."""
        pattern = is_op("nn.batch_norm")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        pattern = is_tuple_get_item(pattern)
        return pattern

    def concat_pattern():
        """Create a concat pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the concat pattern.
        """
        pattern = is_tuple(None)
        pattern = is_op("concatenate")(pattern)

        return pattern

    def dense_pattern():
        """Create a dense pattern."""
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("add")(x, is_constant()))
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern

    def pad_pattern():
        """Create a pad pattern."""
        pattern = is_op("nn.pad")(wildcard(), is_constant())
        return pattern

    def check_conv(extract):
        """Check conv pattern is supported by CLML."""
        call = extract
        clip_found = False
        if isinstance(call, tvm.relay.expr.TupleGetItem):
            call = call.tuple_value
        elif call.op.name == "nn.relu":
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value
        elif call.op.name == "clip":
            clip_found = True
            if call.attrs["a_min"] != 0.0 or call.attrs["a_max"] != 6.0:
                return False
            call = call.args[0]
            if isinstance(call, tvm.relay.expr.TupleGetItem):
                call = call.tuple_value

        while call.op.name != "nn.conv2d":
            call = call.args[0]
        attrs, args = call.attrs, call.args
        if attrs.data_layout != "NCHW":
            return False
        if (
            (not clip_found)
            and (attrs.kernel_size[0] == 3)
            and (attrs.dilation[0] != 1)
            and (attrs.groups != 1)
            and (attrs.channels == attrs.groups)
        ):
            return False
        data_typ = args[0].checked_type
        kernel_typ = args[1].checked_type
        is_depthwise = is_depthwise_conv2d(
            data_typ.shape,
            attrs["data_layout"],
            kernel_typ.shape,
            attrs["kernel_layout"],
            attrs["groups"],
        )
        if attrs.groups != 1 and not is_depthwise:
            return False
        return True

    def check_binary_op(extract):
        call = extract
        if len(call.args[1].checked_type.shape) > 0:
            return True
        return False

    def check_pad_op(extract):
        call = extract
        if len(call.attrs["pad_width"]) != 4:
            return False
        return True

    def check_softmax_op(extract):
        call = extract
        if len(call.args[0].checked_type.shape) > 2:
            return False
        return True

    def check_default_op(extract):
        return True

    return [
        ("clml.conv2d", conv_pattern(), check_conv),
        ("clml.dense", dense_pattern(), check_default_op),
        ("clml.pad", pad_pattern(), check_pad_op),
        ("clml.concat", concat_pattern(), check_default_op),
        ("clml.batch_norm", batch_norm_pattern(), check_default_op),
        ("clml.add", is_op("add")(wildcard(), wildcard()), check_binary_op),
        ("clml.subtract", is_op("subtract")(wildcard(), wildcard()), check_binary_op),
        ("clml.multiply", is_op("multiply")(wildcard(), wildcard()), check_binary_op),
        ("clml.softmax", is_op("nn.softmax")(wildcard()), check_softmax_op),
        ("clml.reshape", is_op("reshape")(wildcard()), check_default_op),
        ("clml.avg_pool2d", is_op("nn.avg_pool2d")(wildcard()), check_default_op),
        ("clml.max_pool2d", is_op("nn.max_pool2d")(wildcard()), check_default_op),
        ("clml.global_avg_pool2d", is_op("nn.global_avg_pool2d")(wildcard()), check_default_op),
        ("clml.global_max_pool2d", is_op("nn.global_max_pool2d")(wildcard()), check_default_op),
        ("clml.relu", is_op("nn.relu")(wildcard()), check_default_op),
        ("clml.clip", is_op("clip")(wildcard()), check_default_op),
    ]


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.clml")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("minimum")
_register_external_op_helper("maximum")


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
