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
# pylint: disable=invalid-name, unused-argument, broad-except
"""Marvell Library supported operators."""

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import Call, TupleGetItem
from tvm.contrib import mrvl as mrvl_contrib

from ...dataflow_pattern import (
    wildcard,
    is_op,
    is_constant,
    is_tuple,
    is_tuple_get_item,
    is_var,
)
from .register import register_pattern_table
from ..strategy.generic import is_depthwise_conv2d


def partition_for_mrvl(
    mod,
    params=None,
    **kwargs,
):
    """Partition the graph greedily into Marvell graph region(s) and a LLVM region(s). The LLVM
    region will contain ops not supported by the Marvell backend.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    mod_mrvl_llvm_regions : annotated & partitioned module (of Mrvl region(s) & LLVM region(s))
    """

    # setup & register convert layout options
    convert_layout_dict = {
        "nn.conv2d": ["NHWC", "OHWI"],
        "nn.max_pool2d": ["NHWC"],
        "nn.avg_pool2d": ["NHWC"],
        "nn.global_avg_pool2d": ["NHWC"],
    }

    mrvl_register_conv2d_attr_funcs_for_convert_layout()
    mrvl_register_max_pool2d_attr_funcs_for_convert_layout()
    mrvl_register_avg_pool2d_attr_funcs_for_convert_layout()
    mrvl_register_global_avg_pool2d_attr_funcs_for_convert_layout()

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    opt_level = 3
    disabled_pass_list = ["AlterOpLayout"]
    annotate_target_str = "mrvl"
    annotate_target_include_non_call_ops = True

    seq_tvmc_pre_repartition = tvm.transform.Sequential(
        passes=[
            relay.transform.InferType(),
            MrvlRemoveDropoutPass(),
            MrvlRemoveCopyPass(),
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
            relay.transform.InferType(),
            relay.transform.ConvertLayout(convert_layout_dict),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
            relay.transform.InferType(),
            relay.transform.MergeComposite(mrvl_pattern_table()),
            relay.transform.AnnotateTarget(
                annotate_target_str,
                annotate_target_include_non_call_ops,
            ),
            relay.transform.MergeCompilerRegions(),
            relay.transform.PartitionGraph(""),
            relay.transform.InferType(),
        ]
    )

    # convert layout back to NCHW for ops in main
    desired_layouts_in_main = {
        "nn.conv2d": ["NCHW", "OIHW"],
        "nn.max_pool2d": ["NCHW"],
        "nn.avg_pool2d": ["NCHW"],
        "nn.global_avg_pool2d": ["NCHW"],
    }

    seq_tvmc_post_repartition = tvm.transform.Sequential(
        passes=[
            # Convert Layout of conv ops in main to NCHW (as expected by LLVM).
            # This pass does not change layout of ops already partitioned into
            # Marvell regions.
            relay.transform.ConvertLayout(desired_layouts_in_main),
            relay.transform.FoldConstant(),
            relay.transform.SimplifyExpr(),
            relay.transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass_list):
        tmp_mod1 = seq_tvmc_pre_repartition(mod)
        tmp_mod1 = repartition_mrvl_subgraphs(tmp_mod1)
        tmp_mod1 = seq_tvmc_post_repartition(tmp_mod1)
        mod_mrvl_llvm_regions = add_attributes(tmp_mod1, annotate_target_str, **kwargs)

    return mod_mrvl_llvm_regions


def is_activation(pattern):
    """
    Check if pattern in Marvell supported activations list
    """
    mrvl_activations = [
        "nn.relu",
    ]
    activation_pattern = None
    for ptrn in mrvl_activations:
        activ = is_op(ptrn)
        if activation_pattern is None:
            activation_pattern = activ
        else:
            activation_pattern |= activ
    pattern = pattern.optional(activation_pattern)
    return pattern


class IsComputeIntensiveGraph(ExprVisitor):
    """
    Visits the graph recursively and checks if it contains compute heavy ops like
    convolutions and dense.
    """

    def __init__(self):
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call):
        compute_intensive_ops = {
            "nn.conv2d",
            "nn.dense",
        }
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op.name) in compute_intensive_ops:
                self.is_compute_intensive = True

        return super().visit_call(call)

    def is_graph_compute_intensive(self, subgraph):
        """
        This function recursively visits the graph and checks if it's compute intensive"
        """
        self.visit(subgraph)
        return self.is_compute_intensive


class IsSupportedGraph(ExprVisitor):
    """
    Visits the graph recursively and checks if function inputs feed into
    any unsupported ops.
    """

    def __init__(self, function):
        ExprVisitor.__init__(self)
        self.is_supported = True
        self.function = function
        self.input_op_list = []

    def _check_legal(self, node, parent_call):
        unsupported_ops = {
            "mrvl.sum2d",
            "mrvl.concat",
        }

        input_ops = {
            "mrvl.reshape",
        }

        if isinstance(node, relay.Function):
            if node.attrs["Composite"] in unsupported_ops:
                self.is_supported = False
            if node.attrs["Composite"] in input_ops:
                self.input_op_list.append(parent_call)

    def visit_call(self, call):
        for args in call.args:
            if args in self.function.params or args in self.input_op_list:
                relay.analysis.post_order_visit(
                    call, lambda expr, parent_call=call: self._check_legal(expr, parent_call)
                )

        return super().visit_call(call)

    def is_supported_subgraph(self):
        """
        This function recursively visits the graph and checks if graph is legal"
        """
        self.visit(self.function.body)
        return self.is_supported


def first_op_unsupported(function):
    return not IsSupportedGraph(function).is_supported_subgraph()


def repartition_subgraph(function):
    """
    Revert back to LLVM if the subgraph is not compute intensive or marked as
    force_llvm.
    """
    if not IsComputeIntensiveGraph().is_graph_compute_intensive(function.body):
        return True

    if first_op_unsupported(function):
        return True

    return False


def repartition_mrvl_subgraphs(mod):
    """
    Un-partition those partitions which:
     - are not computationally intensive subgraph
     - cannot be supported by the backend currently
    """
    global_vars_to_inline = [
        gv
        for gv in mod.get_global_vars()
        if mod[gv].attrs and mod[gv].attrs["Compiler"] == "mrvl" and repartition_subgraph(mod[gv])
    ]
    return relay.transform.InlineCompilerFunctionsBoundTo(global_vars_to_inline)(mod)


def add_attributes(mod, annotate_target_str, **kwargs):
    """This method iterates across all Marvell partitioned functions in the
    module and attaches attributes which are supplied by the user from the CLI.
    Use good defaults in case a particular option is not specified. These options
    are later accessed by codegen and are embedded into the runtime.

    Parameters
    ----------
    mod : Module
        The module to attach attributes to
    kwargs : Dict[str, str]
        Dictionary with command line options

    Returns
    -------
    mod : module with attributes
    """
    working_dir = mrvl_contrib.get_working_dir()
    sim_attr_found = False
    hw_attr_found = False

    if "mattr" in kwargs:
        base_opts_str = kwargs.get("mattr")

        # Set defaults to options if explicit command line option is not given
        if "arch" not in base_opts_str:
            base_opts_str = f"{base_opts_str} -arch=mlip"

        if "quantize" not in base_opts_str:
            base_opts_str = f"{base_opts_str} -quantize=fp16"

        if "wb_pin_ocm" not in base_opts_str:
            base_opts_str = f"{base_opts_str} -wb_pin_ocm=0"

        if "sim" in base_opts_str:
            sim_attr_found = True
            base_opts_str = base_opts_str.replace("sim", "")

        if "hw" in base_opts_str:
            hw_attr_found = True
            base_opts_str = base_opts_str.replace("hw", "")

    else:
        base_opts_str = "-arch=mlip -quantize=fp16 -wb_pin_ocm=0"

    if "num_tiles" in kwargs:
        base_opts_str = f"{base_opts_str} -num_tiles={kwargs.get('num_tiles')}"
    elif "num_tiles" not in base_opts_str:
        base_opts_str = f"{base_opts_str} -num_tiles=8"

    mode_string = "sim"
    if sim_attr_found:
        mode_string = "sim"
    elif hw_attr_found:
        mode_string = "hw"

    for var in mod.get_global_vars():
        func_name = var.name_hint
        func = mod[func_name]

        if annotate_target_str in func_name:
            func = func.with_attr("working_dir", working_dir)
            func = func.with_attr("compiler_opts_string", base_opts_str)
            func = func.with_attr("mode", mode_string)
            mod.update_func(var, func)

    return mod


def is_valid_batch_size(batch_size):
    if isinstance(batch_size, type(relay.Any())):
        return False
    elif batch_size > 8:
        return False
    else:
        return True


def mrvl_register_conv2d_attr_funcs_for_convert_layout():
    """register the conv2d attr func(s) to convert op layout"""
    # reset first in order to register & use a new nn.conv2d convert layout function
    relay.op.get("nn.conv2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.conv2d", "FTVMConvertOpLayout")
    def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
        if not is_valid_batch_size(tinfos[0].shape[0]):
            return relay.nn.conv2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        weight_info_const = tinfos[1]
        new_attrs["channels"] = weight_info_const.shape[0]
        desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
        new_attrs["data_layout"] = desired_data_layout
        new_attrs["kernel_layout"] = desired_kernel_layout
        new_attrs["out_layout"] = desired_data_layout
        return relay.nn.conv2d(*inputs, **new_attrs)

    return convert_conv2d


def mrvl_register_max_pool2d_attr_funcs_for_convert_layout():
    """register the max_pool2d attr func(s) to convert op layout"""
    # reset first in order to register & use a new nn.max_pool2d convert layout function
    relay.op.get("nn.max_pool2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.max_pool2d", "FTVMConvertOpLayout")
    def convert_max_pool2d(attrs, inputs, tinfos, desired_layouts):
        if not is_valid_batch_size(tinfos[0].shape[0]):
            return relay.nn.max_pool2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        new_attrs["out_layout"] = str(desired_layouts[0])
        return relay.nn.max_pool2d(*inputs, **new_attrs)

    return convert_max_pool2d


def mrvl_register_avg_pool2d_attr_funcs_for_convert_layout():
    """register the avg_pool2d attr func(s) to convert op layout"""
    # reset first in order to register& use a new nn.avg_pool2d convert layout function
    relay.op.get("nn.avg_pool2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.avg_pool2d", "FTVMConvertOpLayout")
    def convert_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
        if (tinfos[0].shape[0] != 1) and not isinstance(tinfos[0].shape[0], type(relay.Any())):
            return relay.nn.avg_pool2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        new_attrs["out_layout"] = str(desired_layouts[0])
        return relay.nn.avg_pool2d(*inputs, **new_attrs)

    return convert_avg_pool2d


def mrvl_register_global_avg_pool2d_attr_funcs_for_convert_layout():
    """register the global_avg_pool2d attr func(s) to convert op layout"""
    # reset first in order to register& use a new nn.global_avg_pool2d convert layout function
    relay.op.get("nn.global_avg_pool2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.global_avg_pool2d", "FTVMConvertOpLayout")
    def convert_global_avg_pool2d(attrs, inputs, tinfos, desired_layouts):
        if (tinfos[0].shape[0] != 1) and not isinstance(tinfos[0].shape[0], type(relay.Any())):
            return relay.nn.global_avg_pool2d(*inputs, **attrs)
        new_attrs = dict(attrs)
        new_attrs["layout"] = str(desired_layouts[0])
        new_attrs["out_layout"] = str(desired_layouts[0])
        return relay.nn.global_avg_pool2d(*inputs, **new_attrs)

    return convert_global_avg_pool2d


@register_pattern_table("mrvl")
def mrvl_pattern_table():
    """Get the Mrvl pattern table."""

    def conv2d_nhwc2nhwc_pattern():
        """Create a convolution-2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution-2d pattern.
        """

        def conv2d_base_pattern(pattern):
            pattern = is_op("nn.conv2d")(pattern, is_constant())
            pattern = pattern.optional(
                lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
            )

            def conv2d_no_batchnorm(pattern):
                # conv + [add] + [relu]
                pattern1 = is_activation(pattern)
                return pattern1

            def conv2d_batchnorm(pattern):
                pattern2 = is_op("nn.batch_norm")(
                    pattern, is_constant(), is_constant(), is_constant(), is_constant()
                )
                pattern2 = is_tuple_get_item(pattern2, 0)
                pattern2 = is_activation(pattern2)
                return pattern2

            pattern1 = conv2d_no_batchnorm(pattern)
            pattern2 = conv2d_batchnorm(pattern)

            return pattern1 | pattern2

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = conv2d_base_pattern(pad)
        no_pad = wildcard()
        no_pad = conv2d_base_pattern(no_pad)

        return pad | no_pad

    def sum_pattern():
        """Create a sum pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the sum pattern.
        """
        pattern = is_op("add")(wildcard(), wildcard())
        pattern = is_activation(pattern)
        return pattern

    def concat_pattern():
        """Create a concat pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the concat pattern.
        """
        pattern = is_op("concatenate")(is_tuple(None))
        return pattern

    def fc_pattern():
        """Create a fc (fully-connected) pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the fc pattern.
        """

        def fc_base_pattern(pattern):
            pattern = is_op("nn.dense")(pattern, is_constant())
            pattern = pattern.optional(
                lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
            )
            pattern = is_activation(pattern)

            return pattern

        transform1 = is_op("layout_transform")(wildcard()).has_attr(
            {"src_layout": "NHWC", "dst_layout": "NCHW"}
        )
        reshape = is_op("reshape")(transform1)
        flatten = is_op("nn.batch_flatten")(transform1)
        flatten = reshape | flatten
        flatten = fc_base_pattern(flatten)

        no_flatten = wildcard()
        no_flatten = fc_base_pattern(no_flatten)

        return flatten | no_flatten

    def maxpool2d_pattern():
        """Create a maxpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the maxpool2d pattern.
        """

        def maxpool2d_base_pattern(pattern):
            pattern = is_op("nn.max_pool2d")(pattern)
            return pattern

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = maxpool2d_base_pattern(pad)

        no_pad = wildcard()
        no_pad = maxpool2d_base_pattern(no_pad)

        return pad | no_pad

    def avgpool2d_pattern():
        """Create a avgpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the avgpool2d pattern.
        """

        def avgpool2d_base_pattern(pattern):
            pattern = is_op("nn.avg_pool2d")(pattern)

            return pattern

        pad = is_op("nn.pad")(wildcard(), wildcard())
        pad = avgpool2d_base_pattern(pad)

        no_pad = wildcard()
        no_pad = avgpool2d_base_pattern(no_pad)

        return pad | no_pad

    def globalavgpool2d_pattern():
        """Create a globalavgpool2d pattern.
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the globalavgpool2d pattern.
        """
        pattern = is_op("nn.global_avg_pool2d")(wildcard())
        return pattern

    def globalmaxpool2d_pattern():
        """Create a globalmaxpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples
        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the globalmaxpool2d pattern.
        """
        pattern = is_op("nn.global_max_pool2d")(wildcard())
        return pattern

    def reshape_pattern():
        pattern = is_op("reshape")(wildcard())
        return pattern

    def batch_flatten_pattern():
        pattern = is_op("nn.batch_flatten")(wildcard())
        return pattern

    def squeeze_pattern():
        pattern = is_op("squeeze")(wildcard())
        return pattern

    def layout_transform_nchw2nhwc_pattern():
        pattern = is_op("layout_transform")(is_var(), wildcard(), wildcard()).has_attr(
            {"src_layout": "NCHW", "dst_layout": "NHWC"}
        )
        return pattern

    def check_conv2d(extract):
        """Check conv pattern is supported by Mrvl."""
        call = extract
        while isinstance(call, TupleGetItem) or (call.op.name != "nn.conv2d"):
            if isinstance(call, TupleGetItem):
                call = call.tuple_value
            else:
                call = call.args[0]
        return conv2d_nhwc2nhwc(call)

    def check_fc(extract):
        """Check fc pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.dense":
            call = call.args[0]
        return fc_ni2no(call)

    def check_maxpool2d(extract):
        """Check maxpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.max_pool2d":
            call = call.args[0]
        return maxpool2d_nhwc2nhwc(call)

    def check_avgpool2d(extract):
        """Check avgpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.avg_pool2d":
            call = call.args[0]
        return avgpool2d_nhwc2nhwc(call)

    def check_globalavgpool2d(extract):
        """Check globalavgpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.global_avg_pool2d":
            call = call.args[0]
        return globalavgpool2d_nhwc2nhwc(call)

    def check_globalmaxpool2d(extract):
        """Check globalmaxpool2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "nn.global_max_pool2d":
            call = call.args[0]
        return globalmaxpool2d_nhwc2nhwc(call)

    def check_reshape(extract):
        call = extract
        while call.op.name != "reshape":
            call = call.args[0]
        return reshape_mrvl(call)

    def check_batch_flatten(extract):
        call = extract
        while call.op.name != "nn.batch_flatten":
            call = call.args[0]
        return batch_flatten_mrvl(call)

    def check_squeeze(extract):
        call = extract
        while call.op.name != "squeeze":
            call = call.args[0]
        return squeeze_mrvl(call)

    def check_layout_transform_nchw2nhwc(extract):
        call = extract
        while call.op.name != "layout_transform":
            call = call.args[0]
        return layout_transform_nchw2nhwc(call)

    def check_sum(extract):
        """Check sum2d pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "add":
            call = call.args[0]
        return summation(call)

    def check_concat(extract):
        """Check concat pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "concatenate":
            call = call.args[0]
        return concat(call)

    return [
        ("mrvl.conv2d_nhwc2nhwc", conv2d_nhwc2nhwc_pattern(), check_conv2d),
        ("mrvl.fc_ni2no", fc_pattern(), check_fc),
        ("mrvl.maxpool2d_nhwc2nhwc", maxpool2d_pattern(), check_maxpool2d),
        ("mrvl.avgpool2d_nhwc2nhwc", avgpool2d_pattern(), check_avgpool2d),
        ("mrvl.globalavgpool2d_nhwc2nhwc", globalavgpool2d_pattern(), check_globalavgpool2d),
        ("mrvl.globalmaxpool2d_nhwc2nhwc", globalmaxpool2d_pattern(), check_globalmaxpool2d),
        ("mrvl.sum", sum_pattern(), check_sum),
        ("mrvl.concat", concat_pattern(), check_concat),
        (
            "mrvl.layout_transform_nchw2nhwc",
            layout_transform_nchw2nhwc_pattern(),
            check_layout_transform_nchw2nhwc,
        ),
        ("mrvl.reshape", reshape_pattern(), check_reshape),
        ("mrvl.batch_flatten", batch_flatten_pattern(), check_batch_flatten),
        ("mrvl.squeeze", squeeze_pattern(), check_squeeze),
    ]


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.conv2d", "target.mrvl")
def conv2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for conv2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.data_layout != "NHWC":
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    data_type = args[0].checked_type
    if (
        (len(data_type.shape) != 4)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32"])
    ):
        return False
    kernel_typ = args[1].checked_type
    if (len(kernel_typ.shape) != 4) or (kernel_typ.dtype not in ["float32"]):
        return False

    is_depthwise = is_depthwise_conv2d(
        data_type.shape,
        attrs["data_layout"],
        kernel_typ.shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    if is_depthwise:
        # Mrvl support grouped conv only for groups == ch
        return bool(attrs.groups == kernel_typ.shape[0])
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("add", "target.mrvl")
def summation(expr):
    """Check if the external Mrvl codegen for sum should be used."""
    arg0 = expr.args[0]

    # - need to further checking if the call_func of arg0 is not nn.conv2d nor nn.dense
    if (
        isinstance(arg0, Call)
        and isinstance(arg0.op, tvm.ir.Op)
        and arg0.op.name in ["nn.conv2d", "nn.dense"]
    ):
        return False

    # - need to further checking if dimension of input or output tensor is 4
    data_type = arg0.checked_type
    if (
        (len(data_type.shape) != 4 and len(data_type.shape) != 3)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32"])
    ):
        return False

    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("concatenate", "target.mrvl")
def concat(expr):
    """Check if the external Mrvl codegen for concat should be used."""
    attrs, args = expr.attrs, expr.args
    arg0 = args[0]
    assert not isinstance(arg0, Call)

    # check data types for both inputs
    # - only support 4-dimension input tensors in NHWC
    # - only support batch size is 1
    data_type_a = arg0.checked_type.fields[0]
    data_type_b = arg0.checked_type.fields[1]
    if (
        (len(data_type_a.shape) != 4)
        or (len(data_type_b.shape) != 4)
        or (data_type_a.shape[0] != 1)
        or (data_type_b.shape[0] != 1)
        or (data_type_a.dtype not in ["float32"])
        or (data_type_b.dtype not in ["float32"])
    ):
        return False

    for data_type in arg0.checked_type.fields:
        if (
            (len(data_type.shape) != 4)
            or (data_type.shape[0] != 1)
            or (data_type.dtype not in ["float32"])
        ):
            return False

    if attrs["axis"] != 3:
        return False

    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.dense", "target.mrvl")
def fc_ni2no(expr):
    """Check if the external Mrvl codegen for fc_ni2no should be used."""
    attrs, args = expr.attrs, expr.args
    data_type = args[0].checked_type
    if data_type.dtype not in ["float32"]:
        return False
    kernel_typ = args[1].checked_type
    if (len(kernel_typ.shape) != 2) or (kernel_typ.dtype not in ["float32"]):
        return False
    if attrs.out_dtype != "float32" and attrs.out_dtype != "":
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.max_pool2d", "target.mrvl")
def maxpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for maxpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if (
        (len(data_type.shape) != 4)
        or not is_valid_batch_size(data_type.shape[0])
        or (data_type.dtype not in ["float32"])
    ):
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.avg_pool2d", "target.mrvl")
def avgpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for avgpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if (
        (len(data_type.shape) != 4)
        or ((data_type.shape[0] != 1) and not isinstance(data_type.shape[0], type(relay.Any())))
        or (data_type.dtype not in ["float32"])
    ):
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.global_avg_pool2d", "target.mrvl")
def globalavgpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for globalavgpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if not (len(data_type.shape) == 4 or len(data_type.shape) == 2):
        return False
    if (
        (len(data_type.shape) != 4)
        or ((data_type.shape[0] != 1) and not isinstance(data_type.shape[0], type(relay.Any())))
        or (data_type.dtype not in ["float32"])
    ):
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("nn.global_max_pool2d", "target.mrvl")
def globalmaxpool2d_nhwc2nhwc(expr):
    """Check if the external Mrvl codegen for globalmaxpool2d_nhwc2nhwc should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if not (len(data_type.shape) == 4 or len(data_type.shape) == 2):
        return False
    if (len(data_type.shape) != 4) or (data_type.dtype not in ["float32"]):
        return False
    return True


@tvm.ir.register_op_attr("reshape", "target.mrvl")
def reshape_mrvl(expr):
    """Check if the external Mrvl codegen for reshape should be used."""
    if expr.op.name != "reshape":
        return False
    data_type = expr.checked_type
    if not (len(data_type.shape) == 4 or len(data_type.shape) == 2):
        return False

    args = expr.args
    data_type = args[0].checked_type
    return True


@tvm.ir.register_op_attr("nn.batch_flatten", "target.mrvl")
def batch_flatten_mrvl(expr):
    """Check if the external Mrvl codegen for batch_flatten should be used."""
    if expr.op.name != "nn.batch_flatten":
        return False
    else:
        data_type = expr.checked_type
        if len(data_type.shape) != 2:
            return False

        args = expr.args
        data_type = args[0].checked_type

        if not (len(data_type.shape) == 4 or len(data_type.shape) == 2):
            return False

        return True


@tvm.ir.register_op_attr("squeeze", "target.mrvl")
def squeeze_mrvl(expr):
    """Check if the external Mrvl codegen for squeeze should be used."""
    if expr.op.name != "squeeze":
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("layout_transform", "target.mrvl")
def layout_transform_nchw2nhwc(expr):
    """Check if the external Mrvl codegen for Layout Transform should be used."""
    attrs, args = expr.attrs, expr.args
    if attrs.src_layout != "NCHW":
        return False
    if attrs.dst_layout != "NHWC":
        return False
    data_type = args[0].checked_type
    if data_type.dtype not in ["float32"]:
        return False
    return True


class RemoveDropout(ExprMutator):
    """Removes all nn.dropout from an expr."""

    def visit_tuple_getitem(self, op):
        visit = super().visit_tuple_getitem(op)
        if visit.index != 0:
            return visit
        if (
            isinstance(visit.tuple_value, Call)
            and visit.tuple_value.op.name == "nn.dropout"
            and visit.index == 0
        ):
            # skip nn.dropout call and return arg0 instead
            return visit.tuple_value.args[0]
        return visit


@relay.transform.function_pass(opt_level=0)
class MrvlRemoveDropoutPass:
    """Removes Dropouts."""

    def transform_function(self, func, mod, _):
        """call RemoveDropout func."""
        return RemoveDropout().visit(func)


class RemoveCopy(ExprMutator):
    """
    Delete Copy expression
    """

    def visit_call(self, call):
        visit = super().visit_call(call)
        if visit.op.name in ["copy"]:
            return visit.args[0]
        return visit


@relay.transform.function_pass(opt_level=0)
class MrvlRemoveCopyPass:
    """Removes Copy."""

    def transform_function(self, func, mod, _):
        """call RemoveCopy func."""
        return RemoveCopy().visit(func)
