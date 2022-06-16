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

import tvm.ir
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr import GlobalVar
from tvm.relay.expr_functor import ExprMutator, ExprVisitor

from tvm.relay.analysis import analysis as _analysis
from tvm.relay import expr as _expr


from ... import _ffi_api
from ...dataflow_pattern import wildcard, is_op, is_expr, rewrite, DFPatternCallback
from .register import register_pattern_table


logger = logging.getLogger("DNNL")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
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
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("add")
_register_external_op_helper("multiply")
_register_external_op_helper("nn.layer_norm")


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
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op(conv_name)(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_eltwise:
        return is_op(with_eltwise)(conv_out)
    return conv_out


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
    data = wildcard()
    weight = wildcard()
    bias = wildcard()

    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_eltwise == "gelu":
        const1 = wildcard()
        const2 = wildcard()
        const3 = wildcard()
        div = is_op("divide")(dense_out, const1)
        erf_val = is_op("erf")(div)
        added_erf_val = is_op("add")(erf_val, const2)
        mul_val = is_op("multiply")(dense_out, added_erf_val)
        dense_out = is_op("multiply")(mul_val, const3)
    elif with_eltwise:
        dense_out = is_op(with_eltwise)(dense_out)
    return dense_out


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
        dnnl_pattern = (pat_name, make_conv_pattern(op_name, with_bias, with_eltwise))
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


@register_pattern_table("dnnl")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    dnnl_patterns : List[dnnl_pattern]
        Created patterns.
    """
    elt_list = ["nn.relu", "tanh", "sigmoid", "gelu", None]
    dnnl_patterns = []
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
                if elt != "gelu":
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
        eps = is_expr(relay.const(1e-5))
        added_eps = is_op("add")(mp1, eps)
        deno = is_op("sqrt")(added_eps)
        div_out = is_op("divide")(diff, deno)
        weighted = is_op("multiply")(div_out, self.gamma)
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
