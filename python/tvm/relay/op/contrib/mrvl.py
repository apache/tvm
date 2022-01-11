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
"""
file mrvl.py
Marvell MLIP specific API
"""


import re
import base64
import json
import yaml

import tvm
from tvm import relay
from tvm.relay.transform import _ffi_api

from tvm.relay.build_module import bind_params_by_name
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import (
    Call,
    Let,
    Var,
    GlobalVar,
    If,
    Tuple,
    TupleGetItem,
    RefCreate,
    RefWrite,
    RefRead,
)
from tvm.relay.function import Function

from ...dataflow_pattern import (
    wildcard,
    is_op,
    is_constant,
    is_tuple_get_item,
    is_var,
)
from .register import register_pattern_table
from ..strategy.generic import is_depthwise_conv2d


def clear_ext_json_flag():
    """clear_ext_json_flag

    Returns
    -------
    ret: none
    """
    ext_json = tvm.get_global_func("relay.mrvl.clear_ext_json_flag")
    ext_json()


def is_mrvl_runtime_enabled():
    """Check if the Mrvl graph executor is present.

    Returns
    -------
    ret: bool
        True if present, False if not.
    """
    check_enabled = tvm.get_global_func("relay.op.is_mrvl_runtime_enabled", True)
    if check_enabled:
        return check_enabled()
    return False


def mrvl_register_op_attr_funcs_for_convert_layout():
    """ FIXME """
    # NOTE: for max_pool2d, global_max_pool2d, avg_pool2d, and global_avg_pool2d,
    #       we can rely on registered convert layout functions defined in
    #       the tvm/python/tvm/relay/op/nn/_nn.py file

    # reset first in order to register & use a new nn.conv2d convert layout function
    relay.op.get("nn.conv2d").reset_attr("FTVMConvertOpLayout")

    @tvm.ir.register_op_attr("nn.conv2d", "FTVMConvertOpLayout")
    def convert_conv2d(attrs, inputs, tinfos, desired_layouts):
        new_attrs = dict(attrs)
        # original input data shape is in NCHW format
        # data_info_const = tinfos[0]
        # original kernel shape is in OIHW format
        weight_info_const = tinfos[1]
        # output channels
        new_attrs["channels"] = weight_info_const.shape[0]

        # convert shapes for input data, kernel, and output to use NHWC, OHWI,
        #   and NHWC, respectively
        desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
        # allow us to set input tensor's data_layout == output tensor's out_layout
        new_attrs["data_layout"] = desired_data_layout
        new_attrs["kernel_layout"] = desired_kernel_layout
        new_attrs["out_layout"] = desired_data_layout
        return relay.nn.conv2d(*inputs, **new_attrs)

    return convert_conv2d


def partition_for_mrvl(
    mod,
    params=None,
    tvm_custom_dict=None,
    gen_non_mrvl_subgraph=True,
    flow_pass=1,
    **opts,
):
    """Partition the graph greedily offloading supported
    operators to Mrvl

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    mod_mrvl : annotated and partitioned module - part 1, the mrvl sub graph
    mod_other : annotated and partitioned module - part 2, if any, the rest sub graph
    params : TBA
    opt_level : TBA
    disabled_pass_list : TBA
    mod : TBA
    mrvl_layers_in_mrvl_subgraph : TBA
    """
    clear_ext_json_flag()

    # permanently use Mrvl defined convert layout functions
    mrvl_register_op_attr_funcs_for_convert_layout()

    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    # tvm.transform.Sequential()'s default opt_level is 2
    opt_level = 3
    disabled_pass_list = ["AlterOpLayout"]
    seq = tvm.transform.Sequential(
        passes=[
            # available but not used tvm passes are:
            # - 0, " FoldExplicitPadding", {"InferType"}  // extra leading space?
            # - 0, "SimplifyInference", {"InferType"}
            # - 1, "FuseOps", {"InferType"}
            # - 1, "Legalize", {"InferType"}
            # - 1, "RewriteAnnotatedOps", {"InferType"}
            # - 3, "AlterOpLayout", {"InferType"}
            # - 3, "AutoSchedulerLayoutRewrite", {"InferType"}
            # - 3, "BackwardFoldScaleAxis", {"InferType"}
            # - 3, "CanonicalizeCast", {"InferType"}
            # - 3, "CanonicalizeOps", {"InferType"}
            # - 3, "DefuseOps", {"InferType"}
            # - 3, "EliminateCommonSubexpr", {"InferType"}
            # - 3, "ForwardFoldScaleAxis", {"InferType"}
            # - 4, "CombineParallelBatchMatmul", {"InferType"}
            # - 4, "CombineParallelConv2d", {"InferType"}
            # - 4, "CombineParallelDense", {"InferType"}
            # - 4, "CombineParallelOpBatch", {"InferType"}
            # - 4, "FastMath", {"InferType"}
            # trigger tvm existing relay pass, which contains sub-passes: type_infer.cc
            # - (0, "InferType", {});
            relay.transform.InferType(),
            # tvm.transform.PrintIR("after InferType"), # ~/a
            # implement mrvl own pass (opt_level=0) for nn.dropout
            MrvlRemoveDropoutPass(),
            # tvm.transform.PrintIR("after MrvlRemoveDropout"), # ~/b
            # trigger tvm existing relay pass, which contains sub-passes:
            #   relay/backend/vm/removed_unused_funcs.cc
            # - (1, "RemoveUnusedFunctions", {});
            relay.transform.RemoveUnusedFunctions(),
            # tvm.transform.PrintIR("after RemoveUnusedFunctions"), # ~/c
            # trigger tvm existing relay ConvertLayout pass: convert_layout.cc
            # - (3, "CanonicalizeOps", {"InferType"})
            # - (3, "ConvertLayout", {"InferType", "CanonicalizeOps"})
            # - we can describe mrvl-specific format
            # - we can also implement mrvl per-relay-op conversion functions
            # - we can hook them to relay-op framework using Python @ decorator
            relay.transform.ConvertLayout(
                {"nn.conv2d": ["NHWC", "OHWI"], "nn.max_pool2d": ["NHWC"]}
            ),
            # tvm.transform.PrintIR("after ConvertLayout"), # ~/d
            # trigger tvm existing relay pass, which contains sub-passes: fold_constant.cc
            # - (2, "FoldConstant", {})
            relay.transform.FoldConstant(),
            # tvm.transform.PrintIR("after FoldConstant"), # ~/e
            # trigger tvm existing relay pass, which contains sub-passes: simplify_expr.cc
            # - (0, "SimplifyExpr", {"InferType"})
            # - ConcretizeZerosLikeRewrite, ConcretizeOnesLikeRewrite, ConcretizeFullLikeRewrite,
            # - ConcretizeReshapeLikeRewrite, ConcretizeCollapseSumLikeRewrite,
            #   ConcretizeBroadcastToLikeRewrite,
            # - EliminateIdentityRewrite, SimplifyReshape, SimplifyTranspose,
            # - SimplifyCast, # - FullElementwise,
            relay.transform.SimplifyExpr(),
            # tvm.transform.PrintIR("after SimplifyExpr"), # ~/e
            # implement mrvl-specific drop-noop-transpose pass: drop_noop_transpose.cc
            # - (0, "DropNoopTranspose", {"InferType"})
            # - we can implement mrvl C++ pass
            # - we can hook it to relay-pass framework:
            #   + first using C++ TVM_REGISTER_GLOBAL("relay._transform.DropNoopTranspose").
            #     set_body_typed(DropNoopTranspose);
            #   + then using Python @ decorator below
            _ffi_api.DropNoopTranspose(),
            relay.transform.InferType(),
            # tvm.transform.PrintIR("after DropNoopTranspose"), # ~/e
            # trigger tvm existing relay pass, which contains sub-passes: merge_composite.cc
            # - (0, "MergeComposite", {})
            # - we can also implement mrvl specific composite patterns
            # - we can hook them to relay-merge-composite framework using Python @ decorator
            relay.transform.MergeComposite(mrvl_pattern_table()),
            # tvm.transform.PrintIR("after MergeComposite"), # ~/f
            # trigger tvm existing relay pass, which contains sub-passes: annotate_target.cc
            # - 0, "AnnotateTargetFunc", {"InferType"}
            relay.transform.AnnotateTarget("mrvl", False),
            # tvm.transform.PrintIR("after AnnotateTarget mrvl"), # ~/g
            # this call (partition_graph.cc) can trigger @register_func("relay.ext.mrvl.optimize"),
            #   if defined
            # - (0, "FlattenNestedTuples", {}), (0, "RemoveDefaultAnnotations", {}),
            #   and (0, "PartitionGraph", {})
            # - mangle module name: "tvmgen_" + "mrvl_main_" with a post-fix <#>
            relay.transform.PartitionGraph(""),
            # tvm.transform.PrintIR("after PartitionGraph"), # ~/h
            # trigger tvm existing relay pass, which contains sub-passes: type_infer.cc
            # - (0, "InferType", {});
            relay.transform.InferType(),
            # tvm.transform.PrintIR("final IR"), # ~/h
        ]
    )
    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=disabled_pass_list):
        # triggers tvm/ir/transform.py: transform.pass => __call__() =>
        #   _ffi_transform_api.RunPass(self, mod)
        # - src/tvm/ir/transform.cc: IRMod <= transform.RunPass()
        mod = seq(mod)
        mutator = MrvlIRGraphUtils()
        # print("3.a mutator (inst of MrvlIRGraphUtils): {}".format(mutator), flush=True)
        mod_mrvl, mod_other, mrvl_layers_in_mrvl_subgraph = mutator.compute_two_subgraphs(
            mod, gen_non_mrvl_subgraph=gen_non_mrvl_subgraph, flow_pass=flow_pass
        )

    # annotated and partitioned mod_mrvl
    return (
        mod_mrvl,
        mod_other,
        params,
        opt_level,
        disabled_pass_list,
        mod,
        mrvl_layers_in_mrvl_subgraph,
    )


def defuse_mrvl_layers_in_mrvl_subgraph(mod, defuse_mrvl_layers_list):
    """given a Mrvl subgraph, user can decide to use only a subset of the Mrvl subgraph; and
    this can be done by: (a) use a graph viewer to see structure of the Mrvl subgraph
    including names of consecutive Mrvl layers; and (b) to identify what set of Mrvl
    layer names to be cut (e.g., by treating them as defuse nodes)
    """
    mutator = MrvlIRGraphUtils()
    # print("3.b mutator (inst of MrvlIRGraphUtils): {}".format(mutator), flush=True)
    mod_mrvl, mod_other, mrvl_layers_in_mrvl_subgraph = mutator.compute_two_subgraphs(
        mod,
        defuse_mrvl_layers_list=defuse_mrvl_layers_list,
        gen_non_mrvl_subgraph=True,
        flow_pass=2,
    )
    return mod_mrvl, mod_other, mrvl_layers_in_mrvl_subgraph


def dump_json_meta_data_files(external_graph_json, const_params, filename_prefix="metadata"):
    """Generate two meta data json file and return their filenames

    Parameters
    ----------
    external_graph_json : str
        The json string that can be accepted by graph executor.
        It is generated from the GetExternalJSON() function
    const_params: constant params
    filename_prefix : Optional json filename prefix

    Returns
    -------
    node_json_filename : json filename for nodes and etc.
    const_json_filename : meta data json filename for parameters
    """
    relay_json_obj = yaml.load(
        """\n%(json_str)s
        """
        % {"json_str": external_graph_json}
    )
    node_json_filename = "{}-byoc.json".format(filename_prefix)
    with open(node_json_filename, "w+") as json_f:
        json.dump(relay_json_obj, json_f, indent=2)
    # with open(node_json_filename, "r") as inp_f:
    #     node_json_obj = json.load(inp_f)

    # const params have been erased from graph_json and moved to
    #   metadata module
    const_json_filename = "{}-byoc-const.json".format(filename_prefix)
    with open("{}".format(const_json_filename), "w+") as json_c:
        json_c.write("{\n")
        first_const = True
        for const_key, const_value in const_params.items():
            if ("mrvl" not in const_key) or ("const" not in const_key):
                continue
            if first_const:
                json_c.write('  "{}": {}\n'.format(const_key, "{"))
            else:
                json_c.write('  {},\n  "{}": {}\n'.format("}", const_key, "{"))
            shape_str = str(const_value.shape)
            shape_str = shape_str.replace("(", "[")
            shape_str = shape_str.replace(")", "]")
            # need to take care of special case: composite FC with batch 1 and a scalar add() bias
            # - e.g.: its shape: (32,) needs to be converted to [32,] and then to [1,32]
            shape_re = "[[](?P<scalar_dim_val>[1-9][0-9]+),[]]"
            match_obj = re.match(shape_re, shape_str)
            if match_obj:
                shape_str = "[1, {}]".format(match_obj.group("scalar_dim_val"))
            json_c.write('    "shape": {},\n'.format(shape_str))
            json_c.write('    "dtype": "{}",\n'.format(const_value.dtype))
            json_c.write(
                '    "data_base64": "{}"\n'.format(
                    base64.b64encode(const_value.asnumpy()).decode("utf-8")
                )
            )
            first_const = False
        json_c.write("  }\n}\n")

    # with open(const_json_filename, "r") as inp_f:
    #     const_json_obj = json.load(inp_f)

    return node_json_filename, const_json_filename


def convert_consts_json_meta_data_to_string(
    const_params,
):
    """Generate two meta data json file and return their filenames

    Parameters
    ----------
    const_params: constant params

    Returns
    -------
    const_json_string : meta data of params in json string
    """
    # const params have been erased from graph_json and moved to
    #   metadata module
    json_str = "{\n"
    first_const = True
    for const_key, const_value in const_params.items():
        if ("mrvl" not in const_key) or ("const" not in const_key):
            continue
        if first_const:
            json_str = json_str + '  "{}": {}\n'.format(const_key, "{")
        else:
            json_str = json_str + '  {},\n  "{}": {}\n'.format("}", const_key, "{")
        shape_str = str(const_value.shape)
        shape_str = shape_str.replace("(", "[")
        shape_str = shape_str.replace(")", "]")
        # need to take care of special case: composite FC with batch 1 and a scalar add() bias
        # - e.g.: its shape: (32,) needs to be converted to [32,] and then to [1,32]
        shape_re = "[[](?P<scalar_dim_val>[1-9][0-9]+),[]]"
        match_obj = re.match(shape_re, shape_str)
        if match_obj:
            shape_str = "[1, {}]".format(match_obj.group("scalar_dim_val"))
        json_str = json_str + '    "shape": {},\n'.format(shape_str)
        json_str = json_str + '    "dtype": "{}",\n'.format(const_value.dtype)
        json_str = json_str + '    "data_base64": "{}"\n'.format(
            base64.b64encode(const_value.asnumpy()).decode("utf-8")
        )
        first_const = False
    json_str = json_str + "  }\n}\n"

    return json_str


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
        pattern = is_op("nn.pad")(wildcard()) | wildcard()
        pattern = is_op("nn.conv2d")(pattern, is_constant())
        pattern = pattern.optional(
            lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
        )

        # conv + [add] + [relu]
        pattern1 = pattern.optional(is_op("nn.relu"))

        # conv + [add] + batch_norm + %.0 + [relu]
        pattern2 = is_op("nn.batch_norm")(pattern, wildcard(), wildcard(), wildcard(), wildcard())
        pattern2 = is_tuple_get_item(pattern2, 0)
        pattern2 = pattern2.optional(is_op("nn.relu"))

        return pattern1 | pattern2

    def sum2d_pattern():
        """Create a sum2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the sum2d pattern.
        """
        # do these in check_sum2d
        # - need to further checking if the call_func of args[0] is not nn.conv2d nor nn.dense
        # - need to further checking if dimension of input or output tensor is 4
        pattern = is_op("add")(wildcard(), wildcard())
        pattern = pattern.optional(is_op("nn.relu"))
        return pattern

    def fc_pattern():
        """Create a fc (fully-connected) pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the fc pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(
            lambda x: (is_op("nn.bias_add")(x, is_constant()) | is_op("add")(x, is_constant()))
        )
        pattern = pattern.optional(is_op("nn.relu"))
        return pattern

    def maxpool2d_pattern():
        """Create a maxpool2d pattern.
           review tvm/tests/python/relay/test_dataflow_pattern.py for examples

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the maxpool2d pattern.
        """
        pattern = is_op("nn.max_pool2d")(wildcard())
        return pattern

    def layout_transform_pattern():
        # pattern = is_op("layout_transform")(wildcard().match(GlobalVar), wildcard(),
        #                 wildcard()).has_attr(
        #                 {"src_layout": "NCHW", "dst_layout": "NHWC"})
        pattern = is_op("layout_transform")(is_var(), wildcard(), wildcard()).has_attr(
            {"src_layout": "NCHW", "dst_layout": "NHWC"}
        )
        return pattern

    def check_conv2d(extract):
        """Check conv pattern is supported by Mrvl."""
        call = extract
        # loop over fused Mrvl conv2d sub graph to find the conv2d op
        # - it is okay if we also fused nn.pad because, in conv2d_nhwc2nhwc(),
        #   we do checks starting from conv2d op
        # - in case of nn.batch_norm, a tuple-get-item node exists inside
        #   the fused conv2d sub graph
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

    def check_layout_transform(extract):
        call = extract
        while call.op.name != "layout_transform":
            call = call.args[0]
        return layout_transform_nchw2nhwc(call)

    def check_sum2d(extract):
        """Check maxpool pattern is supported by Mrvl."""
        call = extract
        while call.op.name != "add":
            call = call.args[0]
        return sum2d(call)

    return [
        ("mrvl.conv2d_nhwc2nhwc", conv2d_nhwc2nhwc_pattern(), check_conv2d),
        ("mrvl.fc_ni2no", fc_pattern(), check_fc),
        ("mrvl.maxpool2d_nhwc2nhwc", maxpool2d_pattern(), check_maxpool2d),
        ("mrvl.sum2d", sum2d_pattern(), check_sum2d),
        ("mrvl.layout_transform_nchw2nhwc", layout_transform_pattern(), check_layout_transform),
    ]


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported by Mrvl.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.mrvl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_flatten")
_register_external_op_helper("reshape")
_register_external_op_helper("transpose")


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
        or (data_type.shape[0] != 1)
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
        return depthwise_conv2d_nhwc2nhwc(attrs, args)
    # Mrvl doesn't support grouped convolution
    if attrs.groups != 1 and not is_depthwise:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("add", "target.mrvl")
def sum2d(expr):
    """Check if the external Mrvl codegen for sum2d should be used."""
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
        (len(data_type.shape) != 4)
        or (data_type.shape[0] != 1)
        or (data_type.dtype not in ["float32"])
    ):
        return False
    return True


# TODO(ccjoechou): register a helper function to indicate that the given operator
#   can be supported by Mrvl.
def depthwise_conv2d_nhwc2nhwc(attrs, args):
    """Check if the external Mrvl codegen for depthwise convolution should be used.

    Note
    ----
    Relay does not have a depthwise conv2d_nhwc2nhwc operator whilst Mrvl does. We simply
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
    if data_type.dtype not in ["float32"]:
        return False
    return True


# register a helper function to indicate that the given operator can be supported by Mrvl.
@tvm.ir.register_op_attr("layout_transform", "target.mrvl")
def layout_transform_nchw2nhwc(expr):
    """ FIXME """
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
    def transform_function(self, func, mod, _):
        return RemoveDropout().visit(func)


class MrvlLayers(ExprMutator):
    """experimental class:
    do post-order DFS traverse analysis based on the value of the !mrvl_color attribute
    to decide whether a Mrvl layer/node has an output for the IR sub graph of consecutive
    Mrvl layers
    """

    def __init__(
        self,
        mutate_style="compute-mrvl-color",
        mrvl_layer_names=None,
        defuse_mrvl_layers_list=None,
        mrvl_layers_consecutive=None,
        mrvl_layers_outputs=None,
        debug=False,
    ):
        ExprMutator.__init__(self)
        self._debug = debug
        self._compute_mrvl_color = False
        self._get_mrvl_subgraph = False
        if mutate_style in ["compute-mrvl-color"]:
            self._compute_mrvl_color = True
            # dictionary for consecutive Mrvl layers
            self._mrvl_layers_consecutive = {}
            # dictionary for non-consecutive Mrvl layers, which need to be defused
            self._mrvl_layers_to_defuse = {}
            if defuse_mrvl_layers_list is not None:
                # user has provided initial names of Mrvl layers
                #   to be de-fused based on previous run
                for name in defuse_mrvl_layers_list:
                    if name not in mrvl_layer_names:
                        raise RuntimeError(
                            "TVM-Mrvl-BYOC: defuse name ({}) isn't in Mrvl subgraph ({})".format(
                                name, mrvl_layer_names
                            )
                        )
                    self._mrvl_layers_to_defuse[name] = True
        elif mutate_style in ["get-mrvl-subgraph"]:
            self._get_mrvl_subgraph = True
            assert defuse_mrvl_layers_list is None
            self._mrvl_layers_consecutive = mrvl_layers_consecutive
            self._outputs_mrvl_name = mrvl_layers_outputs
            self._outputs_call = []
            self._inputs = []
        else:
            raise RuntimeError("TVM-Mrvl-BYOC: unsupported mutate style: {}".format(visit_style))

    def dump_debug_text_info(self, n, label):
        astext_list = n.astext(False).splitlines()
        if astext_list[-1:][0] in [" */"]:
            str_list = astext_list[-5:-4][0].split(") /*")
        else:
            str_list = astext_list[-1:][0].split(") /*")
        print("{}: {})".format(label, str_list[0]), flush=True)

    def post_order_analysis(self, call, name, layer_type):
        """do post-order DFS traverse analysis: using the mrvl_color attribute where:
        if mrvl_color == True: this Mrvl layer call is in the group (or inside the subgraph)
        of consecutive Mrvl layers
        """
        if self._debug:
            call_astext_list = call.astext(False).splitlines()
            if call_astext_list[-1:][0] in [" */"]:
                call_func_str_list = call_astext_list[-5:-4][0].split(") /*")
            else:
                call_func_str_list = call_astext_list[-1:][0].split(") /*")
            print("Debug: post-order {} {})".format(layer_type, call_func_str_list[0]), flush=True)

        assert hasattr(call, "mrvl_color")
        if call.mrvl_color:
            if name in self._mrvl_layers_to_defuse:
                # allow use-provided names of to be defused Mrvl layers
                call.mrvl_color = False
            else:
                self._mrvl_layers_consecutive[name] = True
        elif (not call.mrvl_color) and isinstance(call.op, GlobalVar):
            self._mrvl_layers_to_defuse[name] = True

    def visit_function(self, fn):
        """override base class ExprMutator's visit_function() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the Function obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        if self._compute_mrvl_color:
            params_mrvl_color = True
        if self._get_mrvl_subgraph:
            params_has_none = False
        new_params = []
        for x in fn.params:
            new_param = self.visit(x)
            new_params.append(new_param)
            if self._compute_mrvl_color:
                assert hasattr(new_param, "mrvl_color")
                if not new_param.mrvl_color:
                    params_mrvl_color = False
            if self._get_mrvl_subgraph:
                if new_param is None:
                    params_has_none = True

        new_body = self.visit(fn.body)
        if self._get_mrvl_subgraph:
            if (new_body is None) or params_has_none:
                if self._debug:
                    self.dump_debug_text_info(fn, "drop fn")
                return None
        new_fn = Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)
        if self._compute_mrvl_color:
            assert hasattr(new_body, "mrvl_color")
            new_fn.mrvl_color = params_mrvl_color and new_body.mrvl_color
        return new_fn

    def visit_let(self, let):
        """override base class ExprMutator's visit_let() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the Let obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        new_var = self.visit(let.var)
        new_value = self.visit(let.value)
        new_body = self.visit(let.body)
        if self._get_mrvl_subgraph:
            if (new_var is None) or (new_value is None) or (new_body is None):
                if self._debug:
                    self.dump_debug_text_info(let, "drop let")
                return None
        new_let = Let(new_var, new_value, new_body)
        if self._compute_mrvl_color:
            assert hasattr(new_var, "mrvl_color")
            assert hasattr(new_value, "mrvl_color")
            assert hasattr(new_body, "mrvl_color")
            new_let.mrvl_color = new_var.mrvl_color and new_value.mrvl_color and new_body.mrvl_color
        return new_let

    def visit_call(self, call):
        """override base class ExprMutator's visit_call() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the Call obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        name = None
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
        else:
            name = call.op.name

        if self._compute_mrvl_color:
            if isinstance(call.op, GlobalVar):
                layer_type = "mrvl-layer:    "
            else:
                assert isinstance(call.op, tvm.ir.Op)
                layer_type = "non-mrvl-layer:"
            args_mrvl_color = True

        new_fn = self.visit(call.op)
        if self._get_mrvl_subgraph:
            args_has_none = False
        new_args = []
        for idx, arg in enumerate(call.args):
            if self._compute_mrvl_color and self._debug and (idx > 0):
                print("Debug: post-order: visit call-arg{} @{}".format(idx, name), flush=True)
            new_arg = self.visit(arg)
            new_args.append(new_arg)
            if self._compute_mrvl_color:
                assert hasattr(new_arg, "mrvl_color")
                if not new_arg.mrvl_color:
                    args_mrvl_color = False
            if self._get_mrvl_subgraph:
                if new_arg is None:
                    args_has_none = True

        if self._get_mrvl_subgraph:
            if name not in self._mrvl_layers_consecutive:
                if self._debug:
                    self.dump_debug_text_info(call, "drop call")
                return None

        new_call = Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        if self._compute_mrvl_color:
            assert hasattr(new_fn, "mrvl_color")
            new_call.mrvl_color = args_mrvl_color and new_fn.mrvl_color
            self.post_order_analysis(new_call, name, layer_type)
        if self._get_mrvl_subgraph:
            assert not args_has_none
            if name in self._outputs_mrvl_name:
                if self._debug:
                    print("add outputs: {}".format(name), flush=True)
                self._outputs_call.append(new_call)
        return new_call

    def visit_var(self, var):
        """override base class ExprMutator's visit_var() so that
        (1) we can add & use the mrvl_color attribute, or
        (2) return only Mrvl subgraph
        """
        if self._compute_mrvl_color:
            var.mrvl_color = True
        if self._get_mrvl_subgraph:
            if self._debug:
                self.dump_debug_text_info(var, "add inputs: var")
            self._inputs.append(var)
        return var

    def visit_global_id(self, global_var):
        """override base class ExprMutator's visit_global_id() so that
        we can add & use the mrvl_color attribute
        """
        if self._compute_mrvl_color:
            global_var.mrvl_color = True
        return global_var

    def visit_if(self, ite):
        """override base class ExprMutator's visit_if() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the If obj is inside the group (or the sub graph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        new_cond = self.visit(ite.cond)
        new_true_branch = self.visit(ite.true_branch)
        new_false_branch = self.visit(ite.false_branch)
        if self._get_mrvl_subgraph:
            if (new_cond is None) or (new_true_branch is None) or (new_false_branch is None):
                if self._debug:
                    self.dump_debug_text_info(ite, "drop ite")
                return None
        new_if = If(new_cond, new_true_branch, new_false_branch)
        if self._compute_mrvl_color:
            assert hasattr(new_cond, "mrvl_color")
            assert hasattr(new_true_branch, "mrvl_color")
            assert hasattr(new_false_branch, "mrvl_color")
            new_if.mrvl_color = (
                new_cond.mrvl_color and new_true_branch.mrvl_color and new_false_branch.mrvl_color
            )
        return new_if

    def visit_tuple(self, tup):
        """override base class ExprMutator's visit_tuple() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the Tuple obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        if self._compute_mrvl_color:
            fields_mrvl_color = True
        if self._get_mrvl_subgraph:
            fields_has_none = False
        new_fields = []
        for field in tup.fields:
            new_field = self.visit(field)
            new_fields.append(new_field)
            if self._compute_mrvl_color:
                assert hasattr(new_field, "mrvl_color")
                if not new_field.mrvl_color:
                    fields_mrvl_color = False
            if self._get_mrvl_subgraph:
                if new_field is None:
                    fields_has_none = True

        if self._get_mrvl_subgraph:
            if fields_has_none:
                if self._debug:
                    self.dump_debug_text_info(tup, "drop tup")
                return None
        new_tup = Tuple(new_fields, tup.span)
        if self._compute_mrvl_color:
            new_tup.mrvl_color = fields_mrvl_color
        return new_tup

    def visit_tuple_getitem(self, op):
        """override base class ExprMutator's visit_tuple_getitem() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the op or TupleGetItem obj is inside the group (or subgraph)
        of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        tuple_value = self.visit(op.tuple_value)
        if self._get_mrvl_subgraph:
            if tuple_value is None:
                if self._debug:
                    self.dump_debug_text_info(op, "drop op")
                return None
        if not tuple_value.same_as(op.tuple_value):
            new_tup_get_item = TupleGetItem(tuple_value, op.index)
            if self._compute_mrvl_color:
                assert hasattr(tuple_value, "mrvl_color")
                new_tup_get_item.mrvl_color = tuple_value.mrvl_color
            return new_tup_get_item

        # usually we do not get here, but, if we do, we can only
        #   add the mrvl_color attribute to the original IR graph
        if self._compute_mrvl_color:
            if not hasattr(op, "mrvl_color"):
                op.mrvl_color = True
        return op

    def visit_global_var(self, gvar):
        """override base class ExprMutator's visit_global_var() so that
        we can add & use the mrvl_color attribute
        """
        if self._compute_mrvl_color:
            gvar.mrvl_color = True
        return gvar

    def visit_op(self, op):
        """override base class ExprMutator's visit_op() so that
        we can add & use the mrvl_color attribute
        """
        if self._compute_mrvl_color:
            # - all Mrvl layers are GlobalVar objs
            # - all ops are non Mrvl layers
            op.mrvl_color = False
        return op

    def visit_constant(self, const):
        """override base class ExprMutator's visit_constant() so that
        we can add & use the mrvl_color attribute
        """
        if self._compute_mrvl_color:
            const.mrvl_color = True
        return const

    def visit_ref_create(self, r):
        """override base class ExprMutator's visit_ref_create() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the RefCreate obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        new_value = self.visit(r.value)
        if self._get_mrvl_subgraph:
            if new_value is None:
                if self._debug:
                    self.dump_debug_text_info(r, "drop ref_create")
                return None
        new_refcreate = RefCreate(new_value)
        if self._compute_mrvl_color:
            assert hasattr(new_value, "mrvl_color")
            new_refcreate.mrvl_color = new_value.mrvl_color
        return new_refcreate

    def visit_ref_write(self, r):
        """override base class ExprMutator's visit_ref_create() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the RefWrite obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        new_ref = self.visit(r.ref)
        new_value = self.visit(r.value)
        if self._get_mrvl_subgraph:
            if (new_ref is None) or (new_value is None):
                if self._debug:
                    self.dump_debug_text_info(r, "drop ref_create")
                return None
        new_refwrite = RefWrite(new_ref, new_value)
        if self._compute_mrvl_color:
            assert hasattr(new_ref, "mrvl_color")
            assert hasattr(new_value, "mrvl_color")
            new_refwrite.mrvl_color = new_ref.mrvl_color and new_value.mrvl_color
        return new_refwrite

    def visit_ref_read(self, r):
        """override base class ExprMutator's visit_ref_create() so that
        (1) we can add & use the mrvl_color attribute to determine whether
        the RefRead obj is inside the group (or subgraph) of consecutive Mrvl layers, or
        (2) return only Mrvl subgraph
        """
        new_ref = self.visit(r.ref)
        if self._get_mrvl_subgraph:
            if new_ref is None:
                if self._debug:
                    self.dump_debug_text_info(r, "drop ref_create")
                return None
        new_refread = RefRead(new_ref)
        if self._compute_mrvl_color:
            assert hasattr(new_ref, "mrvl_color")
            new_refread.mrvl_color = new_ref.mrvl_color
        return new_refread

    def compute_main_func_mrvl_color(self, main_func):
        """initiate post-order DFS traverse from each output tensore of
        the main_func argument, i.e., mod["main"],
        in order to find and return the group (or the Mrvl sub graph)
        of consecutive Mrvl layers, as well as, Mrvl layers, which
        need to be defused back to their original operators
        """
        assert main_func
        assert self._compute_mrvl_color
        if self._debug:
            print("mod[main] => {}".format(main_func.astext(False)), flush=True)
        return self.visit(main_func)

    def get_consecutive_layers(self):
        """return names of Mrvl layers inside the Mrvl subgraph
        and names of Mrvl layers outside the Mrvl subgraph
        """
        assert self._compute_mrvl_color
        return self._mrvl_layers_consecutive, self._mrvl_layers_to_defuse

    def get_main_func_mrvl_subgraph(self, main_func):
        """return only Mrvl subgraph"""
        assert self._get_mrvl_subgraph
        new_main_func = self.visit(main_func)
        if new_main_func is not None:
            if self._debug:
                print("return new_main_func: {})".format(new_main_func.astext(False)), flush=True)
            return new_main_func

        # we need to instantiate a new output or output tuple
        if self._debug:
            print(
                "got new_main_func==None and need to construct a tuple outputs (size={})".format(
                    len(self._outputs_call)
                ),
                flush=True,
            )
        assert len(self._outputs_call) > 0
        if len(self._outputs_call) == 1:
            if self._debug:
                print("take the only output call")
            new_main_func = Function(list(self._inputs), self._outputs_call[0])
        else:
            if self._debug:
                print("tuple generated")
            new_out_tup = Tuple(self._outputs_call)
            new_main_func = Function(list(self._inputs), new_out_tup)
        if self._debug:
            print("new main func generated")
        return new_main_func


# TODO(ccjoechou): Need to find all the possible cut points so that many corner
# cases can be identifed and fixed.
class RestOfMrvlLayers(ExprMutator):
    """experimental class:
    Figures out restof subgraph based on the input nodes id
    and returns restof subgraph of a given model.
    """

    def __init__(
        self, mrvl_layers_consecutive=None, rest_of_subgraph_inputs_en_id=None, debug=False
    ):
        ExprMutator.__init__(self)
        self._debug = debug
        self._first_function_visit = True
        self._mrvl_layers_consecutive = mrvl_layers_consecutive
        self._inputs_restof_subgraph_en_id = rest_of_subgraph_inputs_en_id
        self._outputs_call = None
        self._inputs = []
        self._inputs_call_names = {}

    def dump_debug_text_info(self, n, label):
        astext_list = n.astext(False).splitlines()
        if astext_list[-1:][0] in [" */"]:
            str_list = astext_list[-5:-4][0].split(") /*")
        else:
            str_list = astext_list[-1:][0].split(") /*")
        print("{}: {})".format(label, str_list[0]), flush=True)

    def visit_function(self, fn):
        """override base class ExprMutator's visit function"""
        # if self._compute_mrvl_color: params_mrvl_color = True
        new_params = []
        save_first_func = None
        if self._first_function_visit and len(self._mrvl_layers_consecutive) > 0:
            self._first_function_visit = False
            save_first_func = fn
        else:
            for x in fn.params:
                new_param = self.visit(x)
                new_params.append(new_param)
        new_body = self.visit(fn.body)
        if save_first_func:
            assert new_body
            self._outputs_call = new_body
            return None
        new_fn = Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)
        return new_fn

    def visit_let(self, let):
        """override base class ExprMutator's visit function"""
        new_var = self.visit(let.var)
        new_value = self.visit(let.value)
        new_body = self.visit(let.body)
        if (new_var is None) or (new_value is None) or (new_body is None):
            if self._debug:
                self.dump_debug_text_info(let, "drop let")
            return None
        new_let = Let(new_var, new_value, new_body)
        return new_let

    def visit_call(self, call):
        """override base class ExprMutator's visit function"""
        old_call_en_id = tvm.relay._ffi_api.get_en_id(call)
        name = None
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
        else:
            name = call.op.name
        new_fn = self.visit(call.op)
        new_args = []
        for idx, arg in enumerate(call.args):
            new_arg = self.visit(arg)
            new_args.append(new_arg)
            if new_arg:
                if (
                    isinstance(new_arg, Var)
                    and old_call_en_id in self._inputs_restof_subgraph_en_id
                ):
                    assert name not in self._mrvl_layers_consecutive
                    if new_arg.name_hint not in self._inputs_call_names:
                        self._inputs.append(new_arg)
                        self._inputs_call_names[new_arg.name_hint] = new_arg
            else:
                if old_call_en_id in self._inputs_restof_subgraph_en_id:
                    assert name not in self._mrvl_layers_consecutive
                    if arg.op.name_hint not in self._inputs_call_names:
                        var = Var(arg.op.name_hint, arg.op.checked_type.ret_type)
                        self._inputs.append(var)
                        self._inputs_call_names[var.name_hint] = var
                        new_args[idx] = var
                    else:
                        new_args[idx] = self._inputs_call_names[arg.op.name_hint]
        if name in self._mrvl_layers_consecutive:
            if self._debug:
                self.dump_debug_text_info(call, "drop call")
            return None
        return Call(new_fn, new_args, call.attrs, call.type_args, call.span)

    def visit_if(self, ite):
        """override base class ExprMutator's visit function"""
        new_cond = self.visit(ite.cond)
        new_true_branch = self.visit(ite.true_branch)
        new_false_branch = self.visit(ite.false_branch)
        if (new_cond is None) or (new_true_branch is None) or (new_false_branch is None):
            if self._debug:
                self.dump_debug_text_info(ite, "drop ite")
            return None
        new_if = If(new_cond, new_true_branch, new_false_branch)
        return new_if

    def visit_tuple(self, tup):
        """override base class ExprMutator's visit function"""
        fields_has_none = False
        new_fields = []
        old_tup_en_id = tvm.relay._ffi_api.get_en_id(tup)
        for idx, field in enumerate(tup.fields):
            new_field = self.visit(field)
            new_fields.append(new_field)
            if new_field:
                continue
            if old_tup_en_id in self._inputs_restof_subgraph_en_id:
                if field.op.name_hint not in self._inputs_call_names:
                    var = Var(field.op.name_hint, field.op.checked_type.ret_type)
                    self._inputs.append(var)
                    self._inputs_call_names[var.name_hint] = var
                    new_fields[idx] = var
                else:
                    new_fields[idx] = self._inputs_call_names[field.op.name_hint]
            else:
                fields_has_none = True
        if fields_has_none:
            if self._debug:
                self.dump_debug_text_info(tup, "drop tup")
            return None
        new_tup = Tuple(new_fields, tup.span)
        return new_tup

    def get_restof_subgraph(self, main_func):
        """return rest of subgraph"""
        new_main_func = self.visit(main_func)
        if new_main_func is not None:
            if self._debug:
                print("return new_main_func: {})".format(new_main_func.astext(False)), flush=True)
            return new_main_func
        # we need to instantiate a new output or output tuple
        if self._debug:
            print(
                "got new_main_func==None and need to construct a tuple outputs (size={})".format(
                    len(self._outputs_call)
                ),
                flush=True,
            )
        assert self._outputs_call
        new_main_func = Function(self._inputs, self._outputs_call)
        if self._debug:
            print("new main func generated")
        return new_main_func


class MrvlLayersGetOutputs(ExprVisitor):
    """ FIXME """

    def __init__(self, mrvl_consecutive_layers, mrvl_layers_to_defuse, debug=False):
        """ FIXME """
        ExprVisitor.__init__(self)
        self._debug = debug
        self._mrvl_consecutive_layers = mrvl_consecutive_layers
        self._mrvl_layers_to_defuse = mrvl_layers_to_defuse
        self._outputs = {}

    def dump_debug_text_info(self, n, label):
        """ FIXME """
        astext_list = n.astext(False).splitlines()
        if astext_list[-1:][0] in [" */"]:
            str_list = astext_list[-5:-4][0].split(") /*")
        else:
            str_list = astext_list[-1:][0].split(") /*")
        print("{}: {})".format(label, str_list[0]), flush=True)

    def add_to_outputs_if_consecutive_mrvl_layer(self, n):
        """ FIXME """
        # tuple is not a consecutive Mrvl layer
        if (
            isinstance(n, Call)
            and isinstance(n.op, GlobalVar)
            and (n.op.name_hint in self._mrvl_consecutive_layers)
        ):
            self._outputs[n.op.name_hint] = True
            if self._debug:
                print("add outputs: {}".format(n.op.name_hint), flush=True)

    def visit_call(self, call):
        """ FIXME """
        if self._debug:
            self.dump_debug_text_info(call, "call")
        call_mrvl_color_false = True
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
            assert (name in self._mrvl_consecutive_layers) or (name in self._mrvl_layers_to_defuse)
            if self._debug:
                print("mrvl_layer: {}".format(name), flush=True)
            if name in self._mrvl_consecutive_layers:
                call_mrvl_color_false = False
        else:
            assert isinstance(call.op, tvm.ir.Op)
            name = call.op.name
            if self._debug:
                print("non-mrvl-layer: {}".format(name), flush=True)

        if call_mrvl_color_false:
            for arg in call.args:
                if self._debug:
                    self.dump_debug_text_info(arg, "arg")
                # add all consecutive Mrvl layers to outputs
                self.add_to_outputs_if_consecutive_mrvl_layer(arg)

        super().visit_call(call)

    def visit_tuple(self, tup):
        """ FIXME """
        # tuple is not a consecutive Mrvl layer
        for field in tup.fields:
            if self._debug:
                self.dump_debug_text_info(field, "field")
            # add all consecutive Mrvl layers to outputs
            self.add_to_outputs_if_consecutive_mrvl_layer(field)

        super().visit_tuple(tup)

    def visit_tuple_getitem(self, t):
        """ FIXME """
        # tuple_getitem is not a consecutive Mrvl layer
        self.add_to_outputs_if_consecutive_mrvl_layer(t.tuple_value)
        super().visit_tuple_getitem(t)

    def visit_let(self, let):
        """ FIXME """
        # let is not a consecutive Mrvl layer
        self.add_to_outputs_if_consecutive_mrvl_layer(let.var)
        self.add_to_outputs_if_consecutive_mrvl_layer(let.value)
        self.add_to_outputs_if_consecutive_mrvl_layer(let.body)
        super().visit_let(let)

    def visit_if(self, i):
        """ FIXME """
        # if is not a consecutive Mrvl layer
        self.add_to_outputs_if_consecutive_mrvl_layer(i.cond)
        self.add_to_outputs_if_consecutive_mrvl_layer(i.true_branch)
        self.add_to_outputs_if_consecutive_mrvl_layer(i.false_branch)
        super().visit_if(i)

    def visit_ref_create(self, r):
        """ FIXME """
        # ref_create is not a consecutive Mrvl layer
        self.add_to_outputs_if_consecutive_mrvl_layer(r.value)
        super().visit_ref_create(r)

    def visit_ref_read(self, r):
        """ FIXME """
        # ref_read is not a consecutive Mrvl layer
        self.add_to_outputs_if_consecutive_mrvl_layer(r.ref)
        super().visit_ref_read(r)

    def visit_ref_write(self, r):
        """ FIXME """
        # ref_ref_write is not a consecutive Mrvl layer
        self.add_to_outputs_if_consecutive_mrvl_layer(t.tuple_value)
        super().visit_ref_write(r)

    def run(self, main_func):
        """ FIXME """
        self.visit(main_func)
        # at least one output
        outputs_keys = self._outputs.keys()
        # in a model containing all Mrvl layers, this can be []
        assert len(outputs_keys) >= 0
        return outputs_keys


# TODO(ccjoechou): Need to find all the possible cut points so that many corner
# cases can be identifed and fixed.
class RestMrvlLayersGetInputs(ExprVisitor):
    """ FIXME """

    def __init__(self, mrvl_consecutive_layers, mrvl_layers_to_defuse, debug=False):
        """ FIXME """
        ExprVisitor.__init__(self)
        self._debug = debug
        self._mrvl_consecutive_layers = mrvl_consecutive_layers
        self._mrvl_layers_to_defuse = mrvl_layers_to_defuse
        self._inputs = {}

    def dump_debug_text_info(self, n, label):
        """ FIXME """
        astext_list = n.astext(False).splitlines()
        if astext_list[-1:][0] in [" */"]:
            str_list = astext_list[-5:-4][0].split(") /*")
        else:
            str_list = astext_list[-1:][0].split(") /*")
        print("{}: {})".format(label, str_list[0]), flush=True)

    def add_to_inputs_if_not_consecutive_mrvl_layer(self, n):
        """ FIXME """
        # tuple is not a consecutive Mrvl layer
        callnode_name = self.get_callnode_name(n)
        if callnode_name is None:
            return
        en_id = tvm.relay._ffi_api.get_en_id(n)
        if en_id not in self._inputs:
            self._inputs[en_id] = callnode_name
            if self._debug:
                print("add inputs: {}".format(callnode_name), flush=True)

    def get_callnode_name(self, call):
        """ FIXME """
        if isinstance(call, Call):
            if isinstance(call.op, GlobalVar):
                name = call.op.name_hint
                if self._debug:
                    print("layer: {}".format(name), flush=True)
            else:
                assert isinstance(call.op, tvm.ir.Op)
                name = call.op.name
                if self._debug:
                    print("non-mrvl-layer: {}".format(name), flush=True)
        elif isinstance(call, Tuple):
            name = "Tup_node"
        else:
            name = None
        return name

    def visit_call(self, call):
        """ FIXME """
        call_mrvl_color_false = False
        callnode_name = self.get_callnode_name(call)
        if callnode_name in self._mrvl_consecutive_layers:
            return
        for arg in call.args:
            if isinstance(arg, Var):
                call_mrvl_color_false = True
                # This callnode has a direct var input
                break
            arg_name = self.get_callnode_name(arg)
            if arg_name in self._mrvl_consecutive_layers:
                call_mrvl_color_false = True
                break
        if call_mrvl_color_false:
            self.add_to_inputs_if_not_consecutive_mrvl_layer(call)
        super().visit_call(call)

    def visit_tuple(self, tup):
        """ FIXME """
        # tuple is not a consecutive Mrvl layer
        call_mrvl_color_false = False
        for field in tup.fields:
            # add all consecutive Mrvl layers to outputs
            if isinstance(field, Var):
                # This callnode has a direct var input
                call_mrvl_color_false = True
                break
            arg_name = self.get_callnode_name(field)
            if arg_name in self._mrvl_consecutive_layers:
                call_mrvl_color_false = True
                break
        if call_mrvl_color_false:
            self.add_to_inputs_if_not_consecutive_mrvl_layer(tup)
        super().visit_tuple(tup)

    def visit_tuple_getitem(self, t):
        """ FIXME """
        # tuple_getitem is not a consecutive Mrvl layer
        self.add_to_inputs_if_not_consecutive_mrvl_layer(t)
        super().visit_tuple_getitem(t)

    def visit_let(self, let):
        """ FIXME """
        # let is not a consecutive Mrvl layer
        self.add_to_inputs_if_not_consecutive_mrvl_layer(let.var)
        self.add_to_inputs_if_not_consecutive_mrvl_layer(let.value)
        self.add_to_inputs_if_not_consecutive_mrvl_layer(let.body)
        super().visit_let(let)

    def visit_if(self, i):
        """ FIXME """
        # if is not a consecutive Mrvl layer
        self.add_to_inputs_if_not_consecutive_mrvl_layer(i.cond)
        self.add_to_inputs_if_not_consecutive_mrvl_layer(i.true_branch)
        self.add_to_inputs_if_not_consecutive_mrvl_layer(i.false_branch)
        super().visit_if(i)

    def visit_ref_create(self, r):
        """ FIXME """
        # ref_create is not a consecutive Mrvl layer
        self.add_to_inputs_if_not_consecutive_mrvl_layer(r.value)
        super().visit_ref_create(r)

    def visit_ref_read(self, r):
        """ FIXME """
        # ref_read is not a consecutive Mrvl layer
        self.add_to_inputs_if_not_consecutive_mrvl_layer(r.ref)
        super().visit_ref_read(r)

    def visit_ref_write(self, r):
        """ FIXME """
        # ref_ref_write is not a consecutive Mrvl layer
        self.add_to_inputs_if_not_consecutive_mrvl_layer(r.tuple_value)
        super().visit_ref_write(r)

    def run(self, main_func):
        """ FIXME """
        self.visit(main_func)
        # at least one output
        inputs_en_id = self._inputs.keys()
        if self._debug:
            for key, value in self._inputs.items():
                print("{}.{}".format(value, key))
        # in a model containing all Mrvl layers, this can be []
        return inputs_en_id


class MrvlSubgraphToRevert(ExprMutator):
    """Reverts subgraphs, which are listed in the subgraphs_to_revert list,
    back to TVM operators instead of using an external codegen (in Mrvl layers).
    """

    def __init__(self, subgraphs_to_revert, mod):
        ExprMutator.__init__(self)
        self._subgraphs_to_revert = subgraphs_to_revert
        self._mod = mod

    def visit_call(self, call):
        if isinstance(call.op, GlobalVar):
            name = call.op.name_hint
            if name in self._subgraphs_to_revert:
                # "Inline" the subgraph back into new main function.
                func = self._mod[name]
                var_map = {}
                for arg, param in zip(call.args, func.params):
                    var_map[param] = super().visit(arg)
                new_body = relay.bind(func.body, var_map)
                # return the original TVM function body, instead of Mrvl Layer
                return new_body
            # if call is not "def @main(...) { ... }"
            if name != "main":
                args = []
                for arg in call.args:
                    args.append(super().visit(arg))
                return call.op(*args)
        return super().visit_call(call)


def revert_mrvl_mod_to_orig(mod_mrvl, mrvl_layers_in_mrvl_subgraph, debug=False):
    """revert Mrvl subgraph mod and return its (original) TVM IR and params
    mod_mrvl: Mrvl subgraph IR with parameters - in fused Mrvl layers
    mrvl_layers_in_mrvl_subgraph: list of Mrvl layer composite function names, which
                                  are going to be reverted
    """

    def run_opt_pass(mod, passes):
        passes = passes if isinstance(passes, list) else [passes]
        seq = tvm.transform.Sequential(passes)
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
        return mod

    if debug:
        print("Debug: mod_mrvl:\n{}\n\n)".format(mod_mrvl.astext(False)), flush=True)
    mod_new = tvm.IRModule(mod_mrvl.functions, mod_mrvl.type_definitions)
    mod_new["main"] = MrvlSubgraphToRevert(mrvl_layers_in_mrvl_subgraph, mod_mrvl).visit(
        mod_mrvl["main"]
    )
    mod_new = relay.transform.RemoveUnusedFunctions()(mod_new)
    mod_new = relay.transform.InferType()(mod_new)
    if debug:
        print("Debug: mod_new (defused level1):\n{}\n\n)".format(mod_new.astext(False)), flush=True)

    mod_new = run_opt_pass(mod_new, relay.transform.DefuseOps())
    if debug:
        print("Debug: mod_new (defused level2):\n{}\n\n)".format(mod_new.astext(False)), flush=True)

    # need to reset back to use the default FTVConvertOpLayout function
    mod_new = run_opt_pass(
        mod_new,
        relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"], "nn.max_pool2d": ["NCHW"]}),
    )
    if debug:
        print(
            "Debug: mod_new (convert layout):\n{}\n\n)".format(mod_new.astext(False)),
            flush=True,
        )

    mod_new = run_opt_pass(mod_new, relay.transform.SimplifyExpr())
    if debug:
        print("Debug: mod_new (simplified):\n{}\n\n)".format(mod_new.astext(False)), flush=True)
    mod_new = run_opt_pass(mod_new, relay.transform._ffi_api.DropNoopTranspose())
    if debug:
        print(
            "Debug: mod_new (drop noop transpose):\n{}\n\n)".format(mod_new.astext(False)),
            flush=True,
        )
    mod_new = run_opt_pass(mod_new, relay.transform.InferType())
    if debug:
        print("Debug: mod_new (infertype):\n{}\n\n)".format(mod_new.astext(False)), flush=True)
    return mod_new


class MrvlIRGraphUtils:
    """Mrvl IR graph analysis utilities"""

    def __init__(self, debug=False):
        self._debug = debug

    def get_mrvl_layers_and_main_func(self, mod):
        """get all mrvl layers (which are annotated mrvl target sub graphs)"""
        main_func = None
        mrvl_layers = []
        for annotated_subgraph in mod.get_global_vars():
            name = annotated_subgraph.name_hint
            if name in ["main"]:
                main_func = mod[name]
            if (not mod[name].attrs) or (mod[name].attrs["Compiler"] != "mrvl"):
                continue
            mrvl_layers.append(name)
        return mrvl_layers, main_func

    def dump_main_func(self, mod, prefix_str="mod[main]"):
        """dump def @main of mod"""
        main_func = None
        for func_var in mod.get_global_vars():
            name = func_var.name_hint
            if name in ["main"]:
                main_func = mod[name]
        assert main_func
        print("{} => {}".format(prefix_str, main_func.astext(False)), flush=True)

    def compute_two_subgraphs(
        self, mod, defuse_mrvl_layers_list=None, gen_non_mrvl_subgraph=False, flow_pass=1
    ):
        """produce a Mrvl-layer sub graph and a graph where non-consecutive Mrvl layers
        are de-fused back to TVM operators
        """

        # find call.op names for Mrvl layers
        mrvl_layers, main_func = self.get_mrvl_layers_and_main_func(mod)
        assert main_func

        # find consecutive Mrvl layers and Mrvl layers, which need to be defused
        mutator = MrvlLayers(
            mutate_style="compute-mrvl-color",
            mrvl_layer_names=mrvl_layers,
            defuse_mrvl_layers_list=defuse_mrvl_layers_list,
            debug=self._debug,
        )
        mod["main"] = mutator.compute_main_func_mrvl_color(main_func)
        mrvl_layers_consecutive, mrvl_layers_to_defuse = mutator.get_consecutive_layers()
        mrvl_layers_consecutive_keys = mrvl_layers_consecutive.keys()
        mrvl_layers_to_defuse_keys = mrvl_layers_to_defuse.keys()
        if self._debug:
            print("\nDebug: flow_pass: {}".format(flow_pass), flush=True)
            print(
                "\nDebug: to {} mrvl layers - {})".format(len(mrvl_layers), mrvl_layers), flush=True
            )
            print(
                "\nDebug: to {} mrvl consecutive layers - {})".format(
                    len(mrvl_layers_consecutive_keys), mrvl_layers_consecutive_keys
                ),
                flush=True,
            )
            print(
                "\nDebug: to {} mrvl to defuse layers - {})".format(
                    len(mrvl_layers_to_defuse_keys), mrvl_layers_to_defuse_keys
                ),
                flush=True,
            )
            print(
                "\nDebug: given defuse_mrvl_layers_list - {})".format(defuse_mrvl_layers_list),
                flush=True,
            )
        assert len(mrvl_layers) == (
            len(mrvl_layers_consecutive_keys) + len(mrvl_layers_to_defuse_keys)
        )

        # FIXME: do post-order DFS traverse analysis based on the value of the !mrvl_color attribute
        #   to decide whether to exclude non-Mrvl-layer operators from IR graph
        # - ran into TVM error: Check failed: (n.defined()) is false: Found null
        #   pointer node while traversing AST. The previous pass may have generated invalid data
        # figure out outputs, which is a list of Mrvl layers
        mrvl_layers_outputs = MrvlLayersGetOutputs(
            mrvl_layers_consecutive, mrvl_layers_to_defuse, debug=self._debug
        ).run(mod["main"])
        # generate a subgraph for consecutive Mrvl layers
        mod_mrvl = tvm.IRModule(mod.functions, mod.type_definitions)
        mutator2 = MrvlLayers(
            mutate_style="get-mrvl-subgraph",
            mrvl_layers_consecutive=mrvl_layers_consecutive,
            mrvl_layers_outputs=mrvl_layers_outputs,
            debug=self._debug,
        )
        # print("type(mutator2): {}".format(type(mutator2).__name__), flush=True)
        mod_mrvl["main"] = mutator2.get_main_func_mrvl_subgraph(mod["main"])
        mod_mrvl = relay.transform.InferType()(mod_mrvl)
        if self._debug:
            print("Debug: mod_mrvl: {})".format(mod_mrvl.astext(False)), flush=True)
        rest_of_subgraph_inputs_en_id = RestMrvlLayersGetInputs(
            mrvl_layers_consecutive, mrvl_layers_to_defuse, debug=self._debug
        ).run(mod["main"])
        flag_flowpass1 = (
            gen_non_mrvl_subgraph and flow_pass == 1 and len(rest_of_subgraph_inputs_en_id) > 0
        )
        flag_flowpass2 = flow_pass == 2 and len(rest_of_subgraph_inputs_en_id) > 0
        if flag_flowpass1 or flag_flowpass2:
            mod_restofsubgraph = mod
            mutator3 = RestOfMrvlLayers(
                mrvl_layers_consecutive=mrvl_layers_consecutive,
                rest_of_subgraph_inputs_en_id=rest_of_subgraph_inputs_en_id,
                debug=self._debug,
            )
            # print("type(mutator3): {}".format(type(mutator3).__name__), flush=True)
            mod_restofsubgraph["main"] = mutator3.get_restof_subgraph(mod["main"])
            if len(mrvl_layers_to_defuse_keys) > 0:
                # revert Mrvl layers, which are not in consecutive Mrvl layers,
                #   back as in-line functions
                mod_restofsubgraph = revert_mrvl_mod_to_orig(
                    mod_restofsubgraph, mrvl_layers_to_defuse_keys
                )
        else:
            mod_restofsubgraph = None
        return mod_mrvl, mod_restofsubgraph, mrvl_layers_consecutive_keys
