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
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging
from functools import reduce

import tvm.ir
from tvm.ir import Op
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr import GlobalVar
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.expr import const

from tvm.relay.analysis import analysis as _analysis
from tvm.relay import expr as _expr

from tvm.relay.expr import Call, TupleGetItem
from ... import _ffi_api
from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr, rewrite, DFPatternCallback
from .register import register_pattern_table


logger = logging.getLogger("DNNL")
supported_post_elts = ["nn.relu", "tanh", "sigmoid", "clip", "gelu", "swish", "mish", None]


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        args = expr.args
        if any([x.checked_type.dtype == "int64" for x in args]):
            logger.info("DNNL does not support int64.")
            return False
        # DNNL does not support pooling with ceil_mode = True.
        if "pool" in op_name:
            attrs = dict(get_attrs(expr))
            if "ceil_mode" in attrs.keys() and attrs["ceil_mode"]:
                return False
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv1d")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.conv3d_transpose")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.global_avg_pool2d")
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("round")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("add")
_register_external_op_helper("multiply")
_register_external_op_helper("nn.layer_norm")
_register_external_op_helper("nn.batch_matmul")


def append_eltwise_ops(op, eltwise):
    """Append element-wise post-ops to conv / conv_transpose / dense

    Parameters
    ----------
    op : str
        The op name to be attached with element-wise post-op.
    eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : CallPattern
        Call node sequence.
    """
    if eltwise == "gelu":
        const1 = wildcard()
        const2 = wildcard()
        const3 = wildcard()
        div = is_op("divide")(op, const1)
        erf_val = is_op("erf")(div)
        added_erf_val = is_op("add")(erf_val, const2)
        mul_val = is_op("multiply")(op, added_erf_val)
        op = is_op("multiply")(mul_val, const3)
    elif eltwise == "swish":
        sig_out = is_op("sigmoid")(op)
        op = is_op("multiply")(op, sig_out)
    elif eltwise == "mish":
        const1 = wildcard()
        exp = is_op("exp")(op)
        add = is_op("add")(exp, const1)
        log = is_op("log")(add)
        tanh = is_op("tanh")(log)
        op = is_op("multiply")(op, tanh)
    elif eltwise:
        op = is_op(eltwise)(op)
    return op


def make_conv_pattern(conv_name, with_bias=True, with_eltwise=None):
    """Create patterns related to conv and conv_transpose.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `conv / conv_transpose`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    conv_out : CallPattern
        Call node sequence.
    """
    if with_eltwise not in supported_post_elts:
        raise ValueError("Unsupported eltwise post-op: %s" % with_eltwise)
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op(conv_name)(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return append_eltwise_ops(conv_out, with_eltwise)


def make_conv_bias_sum_relu_pattern(conv_type, has_relu=True):
    """Create patterns with sum op.

    Parameters
    ----------
    conv_type : str
        Should be nn.conv1d / nn.conv2d / nn.conv3d.
    has_relu : bool
        Whether attach relu.
    Returns
    -------
    out : CallPattern
        Call node sequence.
    """
    data1 = wildcard()
    weight = wildcard()
    bias = wildcard()
    data2 = wildcard()
    out = is_op(conv_type)(data1, weight)
    out = is_op("add")(out, bias)
    out = is_op("add")(out, data2)
    if has_relu:
        out = is_op("nn.relu")(out)
    return out


def get_op_name(expr):
    """Get the operator name from an expression."""
    if isinstance(expr, Op):
        return expr.name
    if isinstance(expr, Call):
        return get_op_name(expr.op)
    if isinstance(expr, TupleGetItem):
        return get_op_name(expr.tuple_value)
    if isinstance(expr, relay.Tuple):
        return get_op_name(expr.fields[0])
    return ""


def get_args(expr):
    """Get the arguments from an expression."""
    if isinstance(expr, Call):
        return expr.args
    if isinstance(expr, TupleGetItem):
        return get_args(expr.tuple_value)
    if isinstance(expr, relay.Tuple):
        return [arg for args in map(get_args, expr.fields) for arg in args]
    return []


def get_attrs(expr):
    """Get the attributes from an expression."""
    if isinstance(expr, Call):
        return expr.attrs
    if isinstance(expr, TupleGetItem):
        return get_attrs(expr.tuple_value)
    return {}


def make_sum_pattren_predicate(checker):
    """Check whether the conv_bias_add_sum pattern is as expected."""

    def predicate(expr):
        if get_op_name(expr) == "nn.relu":
            expr = expr.args[0]
        for e, op_name in zip([expr, expr.args[0]], ["sum", "bias_add"]):
            args = get_args(e)
            attrs = get_attrs(e.args[0])
            if not checker(attrs, args, op_name):
                return False
        return True

    return predicate


def make_bias_add_pattren_predicate(checker):
    """Check whether the conv_bias pattern is as expected."""

    def predicate(expr):
        if get_op_name(expr) == "nn.relu":
            expr = expr.args[0]
        if get_op_name(expr) == "add":
            args = get_args(expr)
            attrs = get_attrs(expr.args[0])
            if not checker(attrs, args, "bias_add"):
                return False
        return True

    return predicate


def add_checker(attrs, args, op_name):
    """Check if add is aligned with elementwise_add and bias_add."""
    if op_name == "sum":
        if not isinstance(args[0].op, tvm.ir.op.Op):
            return False
        if args[0].op.name != "add":
            return False
        if tuple(get_shape(args[0])) != tuple(get_shape(args[1])):
            return False
    if op_name == "bias_add":
        if attrs is None:
            return False
        if not isinstance(args[0].op, tvm.ir.op.Op):
            return False
        if args[0].op.name != "nn.conv2d":
            return False
        channel = dict(attrs)["channels"]
        const_shape = get_shape(args[1])
        if channel != reduce(lambda x, y: x * y, const_shape):
            return False
    return True


def make_dense_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.dense.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    if with_eltwise not in supported_post_elts:
        raise ValueError("Unsupported eltwise post-op: %s" % with_eltwise)
    data = wildcard()
    weight = wildcard()
    bias = wildcard()

    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    return append_eltwise_ops(dense_out, with_eltwise)


def make_dnnl_pattern(op_name, with_bias, with_eltwise):
    """Create dnnl patterns.

    Parameters
    ----------
    op_name : str
        The first call node's op name.
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat_name = op_name.replace("nn", "dnnl")
    if "_transpose" in op_name:
        pat_name = "dnnl.deconv" + op_name.split("_")[0][-2::]
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if "conv" in op_name:
        dnnl_pattern = (
            pat_name,
            make_conv_pattern(op_name, with_bias, with_eltwise),
            make_bias_add_pattren_predicate(add_checker),
        )
    elif op_name == "nn.dense":
        dnnl_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning(
            "Currently, only conv1d, conv2d, conv2d_transpose, conv3d_transpose, "
            "dense op are supported, but got %s.",
            op_name,
        )
        dnnl_pattern = ()
    return dnnl_pattern


def make_qnn_conv2d_pattern():
    """Make qnn.conv2d based pattern supported by DNNL

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    o_scl = is_constant()
    dst_zp = is_constant()
    act_scl = is_constant()
    sum_scl = is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.conv2d")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. Ex: act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)

    return "dnnl.qnn.conv2d", pat


def make_qnn_dense_pattern():
    """Make qnn.dense based pattern supported by DNNL

    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    data = wildcard()
    weight = is_constant()
    bias = is_constant()
    o_scl = is_constant()
    dst_zp = is_constant()
    act_scl = is_constant()
    sum_scl = is_constant()
    sum_src = wildcard()

    zero_zp = is_expr(const(0, dtype="int32"))

    pat = is_op("qnn.dense")(data, weight, zero_zp, zero_zp, is_constant(), is_constant())
    pat = is_op("cast")(pat)
    pat = is_op("add")(pat, bias) | pat  # optional bias
    pat = is_op("multiply")(pat, o_scl)
    pat = is_op("clip")(pat)  # TBD, not only clip
    pat = is_op("multiply")(pat, act_scl) | pat  # optional multiply. ex act_scl == 1
    pat = is_op("add")(pat, sum_scl * is_op("cast")(sum_src)) | pat  # optional sum
    pat = is_op("add")(pat, dst_zp) | pat  # optional dst_zp, can be dst_zp == 0
    pat = is_op("cast")(pat)

    return "dnnl.qnn.dense", pat


@register_pattern_table("dnnl")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    dnnl_patterns : List[dnnl_pattern]
        Created patterns.
    """
    dnnl_patterns = list()
    dnnl_patterns.append(make_qnn_conv2d_pattern())
    dnnl_patterns.append(make_qnn_dense_pattern())
    dnnl_patterns.append(
        (
            "dnnl.conv2d_bias_sum_relu",
            make_conv_bias_sum_relu_pattern("nn.conv2d"),
            make_sum_pattren_predicate(add_checker),
        )
    )
    dnnl_patterns.append(
        (
            "dnnl.conv2d_bias_sum",
            make_conv_bias_sum_relu_pattern("nn.conv2d", False),
            make_sum_pattren_predicate(add_checker),
        )
    )

    elt_list = ["nn.relu", "tanh", "sigmoid", "clip", "gelu", "swish", "mish", None]
    for with_bias in [True, False]:
        for elt in elt_list:
            if not with_bias and not elt:
                continue
            for conv_name in [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv3d",
                "nn.conv2d_transpose",
                "nn.conv3d_transpose",
            ]:
                dnnl_patterns.append(make_dnnl_pattern(conv_name, with_bias, elt))
            dnnl_patterns.append(make_dnnl_pattern("nn.dense", with_bias, elt))
    return dnnl_patterns


def get_optimal_layout_for_conv(
    data_layout, kernel_layout, weight_shape, out_shape, paddings, strides, dilates, groups, dtype
):
    """Get the optimal layout of dnnl, given shape of conv2d.

    Parameters
    ----------
    data_layout, kernel_layout,weight_shape, out_shape, paddings, strides, dilates, groups
        : String
          Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv(
        data_layout,
        kernel_layout,
        weight_shape,
        out_shape,
        paddings,
        strides,
        dilates,
        groups,
        dtype,
    )


def get_optimal_layout_for_conv_transpose(
    data_layout,
    kernel_layout,
    weight_shape,
    out_shape,
    paddings,
    output_paddings,
    strides,
    dilates,
    groups,
    dtype,
):
    """Get the optimal layout of dnnl, given shape of tranposed conv2d.

    Parameters
    ----------
    data_layout, kernel_layout, weight_shape, out_shape, paddings, output_paddings, strides,
    dilates, groups
        : Int, String
          Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv_transpose(
        data_layout,
        kernel_layout,
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        groups,
        dtype,
    )


def get_shape(tensor):
    """Get tensor's shape."""
    if isinstance(tensor, relay.expr.Var):
        return tensor.type_annotation.concrete_shape
    if isinstance(tensor, relay.expr.Constant):
        return tensor.data.shape
    if isinstance(tensor, tvm.ir.tensor_type.TensorType):
        return tensor.concrete_shape
    if isinstance(tensor, tvm.ir.container.Array):
        return tensor[-1].shape
    if isinstance(tensor, relay.expr.Call):
        if tensor.op.name == "multiply":
            return tensor.type_args[0].shape
        return tensor.checked_type.shape
    raise TypeError("Unsupport data type: %s" % type(tensor))


def get_dtype(tensor):
    """Get tensor's dtype."""
    if isinstance(tensor, relay.expr.Var):
        return tensor.type_annotation.dtype
    if isinstance(tensor, relay.expr.Constant):
        return tensor.data.dtype
    if isinstance(tensor, tvm.ir.tensor_type.TensorType):
        return tensor.dtype
    if isinstance(tensor, tvm.ir.container.Array):
        return tensor[-1].dtype
    if isinstance(tensor, relay.expr.Call):
        if tensor.op.name == "multiply":
            return tensor.type_args[0].dtype
        return tensor.checked_type.dtype
    raise TypeError("Unsupport data type: %s" % type(tensor))


def tag2layout(input_data, is_weight=False, conv_type="Conv1D"):
    """Transfer layout, denoted with `a, b, c, d, e`,
    into valid layout (NCHW / OIHW) of TVM."""
    if "Conv1D" in conv_type:
        data_dic = {"a": "N", "b": "C", "c": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "W", "d": "G"}
    elif "Conv2D" in conv_type:
        data_dic = {"a": "N", "b": "C", "c": "H", "d": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "H", "d": "W"}
        if "e" in input_data:
            weight_dic = {"a": "G", "b": "O", "c": "I", "d": "H", "e": "W"}
    elif "Conv3D" in conv_type:
        data_dic = {"a": "N", "b": "C", "c": "D", "d": "H", "e": "W"}
        weight_dic = {"a": "O", "b": "I", "c": "D", "d": "H", "e": "W", "f": "G"}

    dic = weight_dic if is_weight else data_dic
    res = ""

    for i in input_data:
        if i.isupper():
            i = i.lower()
            res += dic[i]
            dic[i] = dic[i].lower()
        elif i.islower():
            res += dic[i]
        elif i.isdigit():
            res += i
        else:
            raise ValueError("Unsupport layout format: %s" % input_data)

    return res


def legalize_pad_avg_pool(attrs, inputs, types):
    """Legalize pad->avg_pool2d pattern.
    Fuse this pattern into one avg_pool2d with padding = (1, 1),
    and count_include_pad = True"""
    data = inputs[0]
    new_attrs = dict(attrs)
    if isinstance(data, relay.expr.Call) and data.op.name == "nn.pad":
        new_attrs["padding"] = (1, 1)
        new_attrs["count_include_pad"] = True
        return relay.nn.avg_pool2d(data.args[0], **new_attrs)
    return relay.nn.avg_pool2d(data, **attrs)


def legalize_group_conv(attrs, inputs, types):
    """Legalize group conv / conv_transpose calculation.
    Alter weight layout from OIHW to GOIHW / IOHW to GIOHW"""
    groups = attrs.groups
    data, weight = inputs
    if groups == 1:
        if "Transpose" not in type(attrs).__name__:
            return relay.nn.conv2d(data, weight, **attrs)
        return relay.nn.conv2d_transpose(data, weight, **attrs)
    OC, IC, H, W = get_shape(weight)
    new_attrs = dict(attrs)
    weight = relay.reshape(weight, (groups, OC // groups, IC, H, W))
    if "Transpose" not in type(attrs).__name__:
        new_attrs["kernel_layout"] = "GOIHW"
        return relay.nn.conv2d(data, weight, **new_attrs)
    new_attrs["kernel_layout"] = "GIOHW"
    return relay.nn.conv2d_transpose(data, weight, **new_attrs)


def alter_conv(attrs, inputs, tinfos, out_type):
    """The convolution's layout auto-query func for dnnl."""

    data, weight = inputs
    groups = str(attrs.groups)
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    dtype = get_dtype(weight)
    new_attrs = dict(attrs)
    conv_type = type(attrs).__name__.split("Attrs")[0]

    res = get_optimal_layout_for_conv(
        attrs["data_layout"],
        attrs["kernel_layout"],
        weight_shape,
        out_shape,
        paddings,
        strides,
        dilates,
        groups,
        dtype,
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = tag2layout(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = tag2layout(weight_df, is_weight=True, conv_type=conv_type)
    new_attrs["out_layout"] = tag2layout(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == "Conv1D":
        return relay.nn.conv1d(data, weight, **new_attrs)
    if conv_type == "Conv2D":
        return relay.nn.conv2d(data, weight, **new_attrs)
    return relay.nn.conv3d(data, weight, **new_attrs)


def alter_conv_transpose(attrs, inputs, tinfos, out_type):
    """The transposed convolution's layout auto-query func for dnnl."""

    data, weight = inputs
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    output_paddings = ",".join([str(x) for x in attrs.get_int_tuple("output_padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    groups = str(attrs.groups)
    dtype = get_dtype(weight)
    new_attrs = dict(attrs)
    conv_type = type(attrs).__name__.split("Attrs")[0]

    res = get_optimal_layout_for_conv_transpose(
        attrs["data_layout"],
        attrs["kernel_layout"],
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        groups,
        dtype,
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = tag2layout(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = tag2layout(weight_df, is_weight=True, conv_type=conv_type)
    new_attrs["out_layout"] = tag2layout(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == "Conv1DTranspose":
        return relay.nn.conv1d_transpose(data, weight, **new_attrs)
    if conv_type == "Conv2DTranspose":
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)
    return relay.nn.conv3d_transpose(data, weight, **new_attrs)


class IsComputeIntensiveGraph(ExprVisitor):
    """
    Visits the Graph recursively and checks if it contains compute heavy ops like convolutions and
    its transpose and dense.
    """

    def __init__(self):
        ExprVisitor.__init__(self)
        self.is_compute_intensive = False

    def visit_call(self, call):
        compute_intensive_ops = set(
            [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv2d_transpose",
                "nn.conv3d",
                "nn.conv3d_transpose",
                "nn.dense",
                "nn.layer_norm",
                "nn.batch_matmul",
                "nn.global_avg_pool2d",
            ]
        )
        if isinstance(call.op, tvm.tir.op.Op):
            if str(call.op) in compute_intensive_ops:
                self.is_compute_intensive = True

        return super().visit_call(call)

    def is_graph_compute_intensive(self, subgraph) -> bool:
        """
        This function recursively visits the graph and checks if it's compute intensive"
        """
        self.visit(subgraph)
        return self.is_compute_intensive


def is_valid_subgraph(body):
    """Final check on whether the subgraph is valid and should be offloaded to DNNL."""
    return IsComputeIntensiveGraph().is_graph_compute_intensive(body)


def prune_dnnl_subgraphs(mod):
    """
    Removes invalid subgraphs, which does not contain compute intensive dnnl ops.
    """

    class SubgraphRemover(ExprMutator):
        """
        Reverts subgraphs in subgraphs_to_remove back to TVM instead of using an external codegen.
        """

        def __init__(self, subgraphs_to_remove, mod, new_mod):
            ExprMutator.__init__(self)
            self.subgraphs_to_remove = subgraphs_to_remove
            self.mod = mod
            self.new_mod = new_mod

        def visit_call(self, call):
            if isinstance(call.op, GlobalVar):
                name = call.op.name_hint
                if name in self.subgraphs_to_remove:
                    # "Inline" the subgraph back into new main function.
                    func = self.mod[name]
                    var_map = {}
                    for arg, param in zip(call.args, func.params):
                        var_map[param] = super().visit(arg)
                    new_body = relay.bind(func.body, var_map)
                    return new_body
                if name != "main":
                    args = []
                    for arg in call.args:
                        args.append(super().visit(arg))
                    return call.op(*args)
            return super().visit_call(call)

    subgraphs_to_remove = []
    # If only one subgraph, do nothing.
    if len(mod.get_global_vars()) <= 2:
        return mod
    # Remove invalid subgraphs
    for subgraph in mod.get_global_vars():
        name = subgraph.name_hint
        if not mod[name].attrs or mod[name].attrs["Compiler"] != "dnnl":
            continue
        if not is_valid_subgraph(mod[name].body):
            subgraphs_to_remove.append(name)
    # Create new pruned module
    new_mod = tvm.IRModule(mod.functions, mod.type_definitions)
    new_mod["main"] = SubgraphRemover(subgraphs_to_remove, mod, new_mod).visit(mod["main"])
    new_mod = transform.RemoveUnusedFunctions()(new_mod)
    return new_mod


class LayerNormRewrite(DFPatternCallback):
    """
    A callback to rewrite the following operators into a single layer normalization operator.

    Pattern #1:
    1   %4 = mean(%3, axis=[-1], keepdims=True) /* ty=Tensor[(1, 3136, 1), float32] */;
    2   %5 = subtract(%3, %4) /* ty=Tensor[(1, 3136, 64), float32] */;
    3   %6 = cast(%5, dtype="float32") /* ty=Tensor[(1, 3136, 64), float32] */;
    4   %7 = power(%6, 2f /* ty=float32 */) /* ty=Tensor[(1, 3136, 64), float32] */;
    5   %8 = mean(%7, axis=[-1], keepdims=True) /* ty=Tensor[(1, 3136, 1), float32] */;
    6   %9 = add(%8, 1e-05f /* ty=float32 */) /* ty=Tensor[(1, 3136, 1), float32] */;
    7   %10 = sqrt(%9) /* ty=Tensor[(1, 3136, 1), float32] */;
    8   %11 = divide(%5, %10) /* ty=Tensor[(1, 3136, 64), float32] */;
    9   %12 = multiply(%11, meta[relay.Constant][2] /* ty=Tensor[(64), float32] */)
            /* ty=Tensor[(1, 3136, 64), float32] */;
    10   %13 = add(%12, meta[relay.Constant][3] /* ty=Tensor[(64), float32] */)
            /* ty=Tensor[(1, 3136, 64), float32] */;

    Pattern #2:
    1   %0 = mean(%input, axis=[-1], keepdims=True);
    2   %1 = variance(%input, %0, axis=[-1], keepdims=True);
    3   %2 = add(%1, 1e-05f /* ty=float32 */) /* ty=Tensor[(1, 49, 1), float32] */;
    4   %3 = subtract(%input, %0);
    5   %4 = sqrt(%2) /* ty=Tensor[(1, 49, 1), float32] */;
    6   %5 = divide(%3, %4);
    7   %6 = multiply(%5, meta[relay.Constant][0] /* ty=Tensor[(64), float32] */)
            /* ty=Tensor[(1, 49, 64), float32] */;
    8   %7 = add(%6, meta[relay.Constant][1] /* ty=Tensor[(64), float32] */)
            /* ty=Tensor[(1, 49, 64), float32] */

    """

    def __init__(self):
        super(LayerNormRewrite, self).__init__()
        self.data = wildcard()
        self.gamma = wildcard()
        self.beta = wildcard()
        mu = is_op("mean")(self.data)
        diff = is_op("subtract")(self.data, mu)
        cdiff = diff | is_op("cast")(diff)
        const_two = is_expr(relay.const(2)) | is_expr(relay.const(2.0))
        p1 = is_op("power")(cdiff, const_two)
        mp1 = is_op("mean")(p1) | is_op("variance")(self.data, mu)
        eps = is_expr(relay.const(1e-5)) | is_expr(relay.const(1e-6))
        added_eps = is_op("add")(mp1, eps)
        deno = is_op("sqrt")(added_eps)
        div_out = is_op("divide")(diff, deno)
        div_out2 = diff * is_op("rsqrt")(added_eps)
        weighted = is_op("multiply")(div_out | div_out2, self.gamma)
        added_bias = is_op("add")(weighted, self.beta)
        self.pattern = added_bias

    def callback(self, pre, post, node_map):
        data = node_map[self.data][0]
        gamma = node_map[self.gamma][0]
        beta = node_map[self.beta][0]
        return relay.op.nn.layer_norm(data=data, gamma=gamma, beta=beta)


def rewrite_layer_norm(mod):
    """Rewrite the input graph to replace multiple operators with a TVM native layer normalization
    operator so that we can offload them to dnnl layer normalization byoc part.
    """
    mod["main"] = rewrite(LayerNormRewrite(), mod["main"])
    return mod


class DenseReshapeBiasGeluRewrite(DFPatternCallback):
    """
    A callback to reorder reshape operators when the patterns are as below:

    Pattern #1:
    1   %62 = nn.dense(%61, meta[relay.Constant][13] /* ty=Tensor[(64, 64), float32] */,
                units=None, out_dtype="float32") /* ty=Tensor[(3136, 64), float32] */;
    2   %63 = reshape(%62, newshape=[1, 3136, 64]) /* ty=Tensor[(1, 3136, 64), float32] */;
    3   %64 = add(meta[relay.Constant][4] /* ty=Tensor[(64), float32] */, %63)
                /* ty=Tensor[(1, 3136, 64), float32] */;

    Pattern #2:
    1   %76 = nn.dense(%75, meta[relay.Constant][18] /* ty=Tensor[(512, 64), float32] */,
                units=None, out_dtype="float32") /*  ty=Tensor[(3136, 512), float32] */;
    2   %77 = reshape(%76, newshape=[1, 3136, 512]) /* ty=Tensor[(1, 3136, 512), float32] */;
    3   %78 = add(meta[relay.Constant][15] /* ty=Tensor[(512), float32] */, %77)
                /* ty=Tensor[(1, 3136, 512), float32] */;
    4   %79 = divide(%78, 1.41421f /* ty=float32 */) /* ty=Tensor[(1, 3136, 512), float32] */;
    5   %80 = erf(%79) /* ty=Tensor[(1, 3136, 512), float32] */;
    6   %81 = add(%80, 1f /* ty=float32 */) /* ty=Tensor[(1, 3136, 512), float32] */;
    7   %82 = multiply(%78, %81) /* ty=Tensor[(1, 3136, 512), float32] */;
    8   %83 = multiply(%82, 0.5f /* ty=float32 */) /* ty=Tensor[(1, 3136, 512), float32] */;
    """

    def __init__(self, has_gelu=True):
        super(DenseReshapeBiasGeluRewrite, self).__init__()
        self.data = wildcard()
        self.weight = wildcard()
        self.bias = wildcard()
        self.const1 = wildcard()
        self.const2 = wildcard()
        self.const3 = wildcard()

        self.attr_map = {}
        self.has_gelu = has_gelu

        den = is_op("nn.dense")(self.data, self.weight)
        re_den = is_op("reshape")(den)
        added = is_op("add")(self.bias, re_den)
        if self.has_gelu:
            divisor = is_op("divide")(added, self.const1)
            val_erf = is_op("erf")(divisor)
            added_erf = is_op("add")(val_erf, self.const2)
            mul1 = is_op("multiply")(added, added_erf)
            mul2 = is_op("multiply")(mul1, self.const3)
            self.pattern = mul2
        else:
            self.pattern = added

    def get_attr(self, pre):
        """Recursively retrieve attributes from reshape operator."""

        def visit_func(expr):
            if isinstance(expr, _expr.Call) and expr.op == relay.op.get("reshape"):
                new_attrs = {}
                for k in expr.attrs.keys():
                    new_attrs[k] = expr.attrs[k]
                self.attr_map["reshape"] = new_attrs

        _analysis.post_order_visit(pre, visit_func)

    def callback(self, pre, post, node_map):
        self.get_attr(pre)

        data = node_map[self.data][0]
        weight = node_map[self.weight][0]
        bias = node_map[self.bias][0]

        den = relay.op.nn.dense(data, weight)
        added = relay.op.add(bias, den)
        if not self.has_gelu:
            return relay.op.reshape(added, self.attr_map["reshape"]["newshape"])

        const1 = node_map[self.const1][0]
        const2 = node_map[self.const2][0]
        const3 = node_map[self.const3][0]

        divisor = relay.op.divide(added, const1)
        val_erf = relay.op.erf(divisor)
        added_erf = relay.op.add(val_erf, const2)
        mul1 = relay.op.multiply(added, added_erf)
        mul2 = relay.op.multiply(mul1, const3)
        return relay.op.reshape(mul2, self.attr_map["reshape"]["newshape"])


def rewrite_dense_bias_gelu_reshape_last(mod):
    """Rewrite the input graph to reorder reshape operators so that
    we can perform dense_bias_gelu/dense_bias fusion and then offload
    them to byoc part.
    """
    mod["main"] = rewrite(
        [DenseReshapeBiasGeluRewrite(), DenseReshapeBiasGeluRewrite(has_gelu=False)], mod["main"]
    )
    return mod


class ResNetV1Rewrite(DFPatternCallback):
    """
    A callback to advance downsize operation when the patterns are as pattern1,
    and the result is written in pattern2:
    Pattern #1:
    %26 = nn.conv2d(%25, ty=Tensor[(64, 256, 1, 1));
    %27 = add(%26, ty=Tensor[(64, 1, 1));
    %28 = nn.relu(%27);

    %29 = nn.conv2d(%28, ty=Tensor[(64, 64, 3, 3));
    %30 = add(%29, ty=Tensor[(64, 1, 1));
    %31 = nn.relu(%30);

    %32 = nn.conv2d(%31, ty=Tensor[(256, 64, 1, 1));
    %33 = add(%32, ty=Tensor[(256, 1, 1));
    %34 = add(%33, %25);
    %35 = nn.relu(%34);

    %36 = nn.conv2d(%35, ty=Tensor[(128, 256, 1, 1), strides=[2, 2]);
    %37 = add(%36, ty=Tensor[(128, 1, 1));
    %38 = nn.relu(%37);

    %39 = nn.conv2d(%38, ty=Tensor[(128, 128, 3, 3));
    %40 = add(%39, ty=Tensor[(128, 1, 1)]);
    %41 = nn.relu(%40);

    %42 = nn.conv2d(%41, ty=Tensor[(512, 128, 1, 1));
    %43 = nn.conv2d(%35, ty=Tensor[(512, 256, 1, 1), strides=[2, 2]);
    %44 = add(%42, ty=Tensor[(512, 1, 1));
    %45 = add(%43, ty=Tensor[(512, 1, 1));

    %46 = add(%44, %45);
    %47 = nn.relu(%46);
    Pattern #2:
    %26 = nn.conv2d(%25, ty=Tensor[(64, 256, 1, 1));
    %27 = add(%26, ty=Tensor[(64, 1, 1));
    %28 = nn.relu(%27);

    %29 = nn.conv2d(%28, ty=Tensor[(64, 64, 3, 3), strides=[2, 2]);
    %30 = add(%29, ty=Tensor[(64, 1, 1));
    %31 = nn.relu(%30);

    %32 = nn.conv2d(%31, ty=Tensor[(256, 64, 1, 1));
    %33 = add(%32, ty=Tensor[(256, 1, 1));
    %34 = nn.max_pool2d(%25, pool_size=[1, 1], strides=[2, 2], padding=[0, 0, 0, 0]);
    %35 = add(%33, %34);
    %36 = nn.relu(%35);

    %37 = nn.conv2d(%36, ty=Tensor[(128, 256, 1, 1));
    %38 = add(%37, ty=Tensor[(128, 1, 1));
    %39 = nn.relu(%38);

    %40 = nn.conv2d(%39, ty=Tensor[(128, 128, 3, 3));
    %41 = add(%40, ty=Tensor[(128, 1, 1));
    %42 = nn.relu(%41);

    %43 = nn.conv2d(%42, ty=Tensor[(512, 128, 1, 1));
    %44 = nn.conv2d(%36, ty=Tensor[(512, 256, 1, 1));
    %45 = add(%43, ty=Tensor[(512, 1, 1));
    %46 = add(%44, ty=Tensor[(512, 1, 1));
    %47 = add(%45, %46);
    %48 = nn.relu(%47);
    """

    def __init__(self):
        super(ResNetV1Rewrite, self).__init__()
        self.attr_lst = []
        self.data = wildcard()
        self.w1, self.b1 = wildcard(), wildcard()
        self.w2, self.b2 = wildcard(), wildcard()
        self.w3, self.b3 = wildcard(), wildcard()
        self.w4, self.b4 = wildcard(), wildcard()
        self.w5, self.b5 = wildcard(), wildcard()
        self.w6, self.b6 = wildcard(), wildcard()
        self.w7, self.b7 = wildcard(), wildcard()

        conv1 = is_op("nn.conv2d")(self.data, self.w1).has_attr({"kernel_size": [1, 1]})
        conv1 = is_op("add")(conv1, self.b1)
        conv1 = is_op("nn.relu")(conv1)

        conv2 = is_op("nn.conv2d")(conv1, self.w2).has_attr({"kernel_size": [3, 3]})
        conv2 = is_op("add")(conv2, self.b2)
        conv2 = is_op("nn.relu")(conv2)

        conv3 = is_op("nn.conv2d")(conv2, self.w3).has_attr({"kernel_size": [1, 1]})
        conv3 = is_op("add")(conv3, self.b3)
        conv3 = is_op("add")(conv3, self.data)
        conv3 = is_op("nn.relu")(conv3)

        left_conv4 = is_op("nn.conv2d")(conv3, self.w4).has_attr({"strides": [2, 2]})
        left_conv4 = is_op("add")(left_conv4, self.b4)
        left_conv4 = is_op("nn.relu")(left_conv4)

        left_conv5 = is_op("nn.conv2d")(left_conv4, self.w5).has_attr({"kernel_size": [3, 3]})
        left_conv5 = is_op("add")(left_conv5, self.b5)
        left_conv5 = is_op("nn.relu")(left_conv5)

        left_conv6 = is_op("nn.conv2d")(left_conv5, self.w6).has_attr({"kernel_size": [1, 1]})
        left_conv6 = is_op("add")(left_conv6, self.b6)

        right_conv7 = is_op("nn.conv2d")(conv3, self.w7).has_attr({"strides": [2, 2]})
        right_conv7 = is_op("add")(right_conv7, self.b7)

        out = is_op("add")(left_conv6, right_conv7)
        out = is_op("nn.relu")(out)
        self.pattern = out

    def get_attr(self, pre):
        """Recursively retrieve attributes from reshape operator."""

        def visit_func(expr):
            if isinstance(expr, _expr.Call) and expr.op == relay.op.get("nn.conv2d"):
                self.attr_lst.append(expr.attrs)

        _analysis.post_order_visit(pre, visit_func)

    def callback(self, pre, post, node_map):
        self.get_attr(pre)
        data = node_map[self.data][0]
        w1, b1 = node_map[self.w1][0], node_map[self.b1][0]
        w2, b2 = node_map[self.w2][0], node_map[self.b2][0]
        w3, b3 = node_map[self.w3][0], node_map[self.b3][0]
        w4, b4 = node_map[self.w4][0], node_map[self.b4][0]
        w5, b5 = node_map[self.w5][0], node_map[self.b5][0]
        w6, b6 = node_map[self.w6][0], node_map[self.b6][0]
        w7, b7 = node_map[self.w7][0], node_map[self.b7][0]

        new_attrs = self.attr_lst[-7]
        conv1 = relay.op.nn.conv2d(data, w1, **new_attrs)
        conv1 = relay.op.add(conv1, b1)
        conv1 = relay.op.nn.relu(conv1)

        new_attrs = dict(self.attr_lst[-6])
        new_attrs["strides"] = [2, 2]
        conv2 = relay.op.nn.conv2d(conv1, w2, **new_attrs)
        conv2 = relay.op.add(conv2, b2)
        conv2 = relay.op.nn.relu(conv2)

        new_attrs = self.attr_lst[-5]
        conv3 = relay.op.nn.conv2d(conv2, w3, **new_attrs)
        conv3 = relay.op.add(conv3, b3)
        max_pool = relay.op.nn.max_pool2d(
            data, pool_size=(1, 1), strides=(2, 2), layout=new_attrs["data_layout"]
        )
        conv3 = relay.op.add(conv3, max_pool)
        conv3 = relay.op.nn.relu(conv3)

        new_attrs = dict(self.attr_lst[-4])
        new_attrs["strides"] = [1, 1]
        left_conv4 = relay.op.nn.conv2d(conv3, w4, **new_attrs)
        left_conv4 = relay.op.add(left_conv4, b4)
        left_conv4 = relay.op.nn.relu(left_conv4)

        new_attrs = self.attr_lst[-3]
        left_conv5 = relay.op.nn.conv2d(left_conv4, w5, **new_attrs)
        left_conv5 = relay.op.add(left_conv5, b5)
        left_conv5 = relay.op.nn.relu(left_conv5)

        new_attrs = self.attr_lst[-2]
        left_conv6 = relay.op.nn.conv2d(left_conv5, w6, **new_attrs)
        left_conv6 = relay.op.add(left_conv6, b6)

        new_attrs = dict(self.attr_lst[-1])
        new_attrs["strides"] = [1, 1]
        right_conv7 = relay.op.nn.conv2d(conv3, w7, **new_attrs)
        right_conv7 = relay.op.add(right_conv7, b7)

        out = relay.op.add(left_conv6, right_conv7)
        out = relay.op.nn.relu(out)
        self.attr_lst = []
        return out


def rewrite_resnetv1(mod):
    """Rewrite the the ResNetV1 downsize block to reduce the computation complexity."""
    mod["main"] = rewrite(ResNetV1Rewrite(), mod["main"])
    return mod


class LegalizeQnnOpForDnnl(DFPatternCallback):
    """Legalize QNN based patterns to match DNNL

    original pattern:
      OP = qnn.dense | qnn.conv2d
      %1 = OP<int>(SRC, WGH) - OP<int>(src_zp, WGH)   // qnn.conv2d
      %2 = %1 + orig_bias                             // bias
      %2 = (%1 - rq_in_zp) * rq_in_scl / rq_out_scl + rq_out_zp  // qnn.requantize
      %3 = act(%2)                                               // activation == clip
      %4 = ((%3 - sum_lh_zp) * sum_lh_scl + (SRC2 - sum_rh_zp) * sum_rh_scl)  // qnn.add
           / sum_out_scl + sum_out_zp

    transform to DNNL compatible:
      %1 = OP<int>(SRC, WGH)
      %2 = cast(%1, dtype="float")
      %2 = (%1 + bias) * o_scl
      %3 = act(%2) * act_scl
      %4 = %3 + SRC2 * sum_scl
      %5 = %4 + dst_zp
      %6 = cast(%5, dtype="float")

    where:
      o_scl = rq_in_scl / rq_out_scl
      act_scl = sum_lhs_scl / sum_out_scl
      sum_scl = sum_rhs_scl / sum_out_scl
      bias = orig_bias - OP(src_zp, WGH) - rq_in_zp + rq_out_zp * rq_out_scl / rq_in_scl
      dst_zp = sum_out_zp - sum_lhs_zp * sum_lhs_scl / sum_out_scl -
               sum_rhs_zp * sum_rhs_scl / sum_out_scl
    """

    def __init__(self):
        super(LegalizeQnnOpForDnnl, self).__init__()
        self.src = wildcard()
        self.wgh = wildcard()
        self.bias = wildcard()
        self.sum_src = wildcard()

        self.src_scl = is_constant()
        self.src_zp = is_constant()
        self.wgh_scl = is_constant()
        self.wgh_zp = is_expr(const(0))

        self.rq_in_scl = is_constant()
        self.rq_in_zp = is_constant()
        self.rq_out_scl = is_constant()
        self.rq_out_zp = is_constant()

        self.sum_lhs_scl = is_constant()
        self.sum_lhs_zp = is_constant()
        self.sum_rhs_scl = is_constant()
        self.sum_rhs_zp = is_constant()
        self.sum_out_scl = is_constant()
        self.sum_out_zp = is_constant()

        self.root = (is_op("qnn.conv2d") | is_op("qnn.dense"))(
            self.src, self.wgh, self.src_zp, self.wgh_zp, self.src_scl, self.wgh_scl
        )
        pat = is_op("add")(self.root, self.bias) | self.root  # optional bias
        pat = is_op("qnn.requantize")(
            pat, self.rq_in_scl, self.rq_in_zp, self.rq_out_scl, self.rq_out_zp
        )
        pat = is_op("clip")(pat)
        cast = is_op("cast")(pat)
        pat = is_op("qnn.add")(
            cast,
            self.sum_src,
            self.sum_lhs_scl,
            self.sum_lhs_zp,
            self.sum_rhs_scl,
            self.sum_rhs_zp,
            self.sum_out_scl,
            self.sum_out_zp,
        )
        pat = is_op("clip")(pat)
        self.pattern = pat | cast

    def callback(self, pre, post, node_map):
        root = node_map[self.root][0]
        src = node_map[self.src][0]
        wgh = node_map[self.wgh][0]
        bias = node_map.get(self.bias, default=[relay.const(0, dtype="int32")])[0]
        src_zp = node_map[self.src_zp][0]
        rq_in_scl = node_map[self.rq_in_scl][0]
        rq_in_zp = node_map[self.rq_in_zp][0]
        rq_out_scl = node_map[self.rq_out_scl][0]
        rq_out_zp = node_map[self.rq_out_zp][0]

        final_dtype = node_map[self.pattern][0].checked_type.dtype

        if root.op == relay.op.get("qnn.conv2d"):
            dst_layout = root.attrs.out_layout
            dst_layout = root.attrs.data_layout if dst_layout == "" else dst_layout
            wgh_layout = root.attrs.kernel_layout
        else:
            # qnn.dense has no layout attributes. Assume that is plain
            dst_layout = "NC"
            wgh_layout = "OI"

        # TODO(@apeskov): dst_layout may ne blocked
        bias_rank = len(dst_layout) - dst_layout.index("C")

        sum_src = node_map[self.sum_src][0] if self.sum_src in node_map else None
        # Default values if qnn.sum is not present
        sum_lhs_scl = node_map[self.sum_lhs_scl][0] if sum_src else relay.const(1, dtype="float32")
        sum_lhs_zp = node_map[self.sum_lhs_zp][0] if sum_src else relay.const(0, dtype="int32")
        sum_rhs_scl = node_map[self.sum_rhs_scl][0] if sum_src else relay.const(0, dtype="float32")
        sum_rhs_zp = node_map[self.sum_rhs_zp][0] if sum_src else relay.const(0, dtype="int32")
        sum_out_scl = node_map[self.sum_out_scl][0] if sum_src else relay.const(1, dtype="float32")
        sum_out_zp = node_map[self.sum_out_zp][0] if sum_src else relay.const(0, dtype="int32")

        def cast_fp(op):
            return relay.op.cast(op, dtype="float32")

        # recalculate some factors
        o_scl = rq_in_scl / rq_out_scl
        act_scl = sum_lhs_scl / sum_out_scl
        sum_scl = sum_rhs_scl / sum_out_scl
        dst_zp = (
            cast_fp(sum_out_zp)
            - cast_fp(sum_lhs_zp) * sum_lhs_scl / sum_out_scl
            - cast_fp(sum_rhs_zp) * sum_rhs_scl / sum_out_scl
        )
        bias = self.squeeze_bias(bias, dst_layout)
        bias = (
            cast_fp(bias)
            - cast_fp(self.fake_op(src_zp, wgh, wgh_layout))
            - cast_fp(rq_in_zp)
            + cast_fp(rq_out_zp) * rq_out_scl / rq_in_scl
        )
        bias = self.broadcast_to_rank(bias, bias_rank)

        zero_zp = relay.const(0, dtype="int32")
        one_scl = relay.const(1.0, dtype="float32")

        # construct new graph with proper post op ordering
        gr = tvm.relay.Call(
            root.op,
            [src, wgh, zero_zp, zero_zp, one_scl, one_scl],
            root.attrs,
            root.type_args,
            root.span,
        )
        gr = relay.op.cast(gr, dtype="float32")
        gr = gr + bias
        gr = gr * o_scl
        gr = relay.op.clip(gr, 0, 255) * act_scl
        gr = gr + sum_scl * cast_fp(sum_src) if sum_src else gr
        gr = gr + dst_zp
        gr = relay.op.cast(gr, dtype=final_dtype)
        return gr

    @staticmethod
    def fake_op(zp, wgh, layout):
        """Fake operator implementation for zp broadcast input"""
        # Conv:  reduce kernel {OC, IC, KH, KW} -> {OC} in case of group that is still correct
        # Dense: reduce kernel {OC, IC} -> {OC}
        wgh_int = relay.op.cast(wgh, dtype="int32")
        reduced_kernel = relay.op.sum(
            wgh_int, axis=[layout.index("O")], keepdims=False, exclude=True
        )
        return zp * reduced_kernel

    @staticmethod
    def squeeze_bias(bias, layout):
        shape = transform.InferTypeLocal(bias).concrete_shape
        c_position = layout.index("C") - len(layout) + len(shape)
        squeeze_idxs = [i for i in range(len(shape)) if i != c_position]
        return relay.op.squeeze(bias, squeeze_idxs)

    @staticmethod
    def broadcast_to_rank(op, rank):
        """Scalar or 1D tensor are supported"""
        shape = transform.InferTypeLocal(op).concrete_shape
        if len(shape) == 0:
            return op
        if len(shape) == 1:
            return relay.op.expand_dims(op, 1, rank - 1)
        raise ValueError("Unexpected bias rank to broadcast. Only 0 and 1 are supported.")


def legalize_qnn_for_dnnl(mod):
    """Transform qnn primitives to DNNL compatible form. Eliminate source zero point and apply
    strict sequence of post ops."""
    mod["main"] = rewrite(LegalizeQnnOpForDnnl(), mod["main"])

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            # transform.SimplifyInference(),  # TODO: this pass decompose nn.layer_norm
            # transform.FoldScaleAxis(),  # TODO: fail inside TVM in case of grouped convolutions.
            transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
