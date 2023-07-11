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
# pylint: disable=invalid-name
"""Common patterns used in BYOC"""

from typing import Dict, Mapping, Tuple, Union
from tvm.script import relax as R, tir as T
from tvm.relax.dpl.pattern import (
    DFPattern,
    is_const,
    is_op,
    is_tuple_get_item,
    wildcard,
    GlobalVarPattern,
    TuplePattern,
)


def _with_bias_activation_pattern(
    out: DFPattern,
    annotations: Dict[str, DFPattern],
    with_bias: bool = False,
    activation: str = None,
) -> Tuple[DFPattern, Mapping[str, DFPattern]]:
    if with_bias:
        annotations["bias"] = bias = wildcard()
        out = is_op("relax.add")(out, bias)

    if activation:
        out = is_op(activation)(out)

    return out, annotations


def make_fused_bias_activation_pattern(
    op_name: str,
    with_bias: bool = False,
    activation: str = None,
) -> Tuple[DFPattern, Mapping[str, DFPattern]]:
    """
    A simple utility to create patterns for an operation fused with bias addition and activation.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"

    with_bias: bool
        Whether or not to include bias addition

    activation: str
        The name of an activation Relax op, such as "relax.nn.relu"

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation

    annotations: Mapping[str, DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """
    lhs = wildcard()
    rhs = wildcard()
    out = is_op(op_name)(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": out}

    return _with_bias_activation_pattern(out, annotations, with_bias, activation)


def make_residual_block_pattern(
    node_output: Union[DFPattern, Tuple[DFPattern, Mapping[str, DFPattern]]],
    binary_op="relax.add",
    activation=None,
) -> Tuple[DFPattern, Mapping[str, DFPattern]]:
    """
    Create pattern for residual block.

    Parameters
    ----------
    node_output: Union[DFPattern, Tuple[DFPattern, Mapping[str, DFPattern]]]
        The output of previous node.

    binary_op: str
        The op used to combine previous node output and residual input.

    activation: str
        The activation function of this residual block. It should be a name of
        activation Relax op, such as "relax.nn.relu".

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a matrix multiplication.

    annotations: Mapping[str, DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    if isinstance(node_output, tuple):
        node_output, arg_patterns = node_output
    else:
        arg_patterns = {}

    residual_input = wildcard()
    op = is_op(binary_op)
    output = op(node_output, residual_input) | op(residual_input, node_output)

    if activation is not None:
        output = is_op(activation)(output)

    return output, {**arg_patterns, "residual": residual_input}


def make_matmul_pattern(
    with_bias: bool = False,
    activation: str = None,
    transposed_rhs: bool = False,
) -> Tuple[DFPattern, Mapping[str, DFPattern]]:
    """
    Create pattern for matrix multiplication.

    Parameters
    ----------
    with_bias: bool
        Whether or not to include bias addition

    activation: str
        The name of an activation Relax op, such as "relax.nn.relu"

    transposed_rhs: bool
        Whether the right hand side of multiplication is transposed.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a matrix multiplication.

    annotations: Mapping[str, DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """

    lhs = wildcard()
    rhs = wildcard()
    annotations = {"lhs": lhs, "rhs": rhs}

    if transposed_rhs:
        rhs = is_op("relax.permute_dims")(rhs)

    out = is_op("relax.matmul")(lhs, rhs)
    annotations["root"] = out

    return _with_bias_activation_pattern(out, annotations, with_bias, activation)


def make_attention_pattern(with_bias: bool = False):
    """
    Create pattern for fused multi head attention.

    Parameters
    ----------
    with_bias: bool
        Whether or not to include bias addition

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused multi head attention.

    annotations: Mapping[str, DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """
    query = wildcard()
    key = wildcard()
    value = wildcard()
    annotations = {"query": query, "key": key, "value": value}
    if with_bias:
        bias = wildcard()
        annotations["bias"] = bias
        out = is_op("relax.nn.attention_bias")(query, key, value, bias)
    else:
        out = is_op("relax.nn.attention")(query, key, value)

    return out, annotations


def make_stacked_attention_pattern(start_op: str, with_bias: bool = False):
    """
    Create pattern for fused multi head attention with stacked input.

    Parameters
    ----------
    start_op: str
        The starting op for pattern, i.e. `R.split` or `R.strided_slice`.

    with_bias: bool
        Whether or not to include bias addition

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused multi head attention.

    annotations: Mapping[str, DFPattern]
        A mapping from name to sub pattern. It can be used to extract
        important expressions from match result, to power the partition
        check function and codegen.
    """
    stacked_qkv = wildcard()
    ops = {}
    if start_op == "split":
        ops["split"] = qkv_tuple = is_op("relax.split")(stacked_qkv)
        query_raw = is_tuple_get_item(qkv_tuple, 0)
        key_raw = is_tuple_get_item(qkv_tuple, 1)
        value_raw = is_tuple_get_item(qkv_tuple, 2)
    elif start_op == "strided_slice":
        ops["strided_slice_query"] = query_raw = is_op("relax.strided_slice")(stacked_qkv)
        ops["strided_slice_key"] = key_raw = is_op("relax.strided_slice")(stacked_qkv)
        ops["strided_slice_value"] = value_raw = is_op("relax.strided_slice")(stacked_qkv)
    else:
        raise NotImplementedError()
    query_reshape_list = wildcard()
    key_reshape_list = wildcard()
    value_reshape_list = wildcard()
    query = is_op("relax.reshape")(query_raw, query_reshape_list)
    key = is_op("relax.reshape")(key_raw, key_reshape_list)
    value = is_op("relax.reshape")(value_raw, value_reshape_list)
    annotations = {
        "stacked_qkv": stacked_qkv,
        "query_reshape_list": query_reshape_list,
        "key_reshape_list": key_reshape_list,
        "value_reshape_list": value_reshape_list,
        **ops,
    }
    if with_bias:
        bias = wildcard()
        annotations["bias"] = bias
        out = is_op("relax.nn.attention_bias")(query, key, value, bias)
    else:
        out = is_op("relax.nn.attention")(query, key, value)
    return out, annotations


def make_layer_norm_pattern():
    """Create a layer norm pattern."""
    inp = wildcard()
    gamma = wildcard()
    beta = wildcard()

    return is_op("relax.nn.layer_norm")(inp, gamma, beta), {}


def make_rms_norm_pattern():
    """Create a layer norm pattern."""
    inp = wildcard()
    weight = wildcard()
    gv = GlobalVarPattern()
    out = is_op("relax.call_tir")(gv, TuplePattern([inp, weight]))
    annotations = {"gv": gv, "inp": inp, "rms_norm": out}
    return out, annotations


def make_attention_rewrite_pattern(
    qkv_layout: str, out_layout: str, with_bias: bool, with_cast: bool
):
    """
    Create pattern for implicit fused multi head attention rewriting.

    Parameters
    ----------
    qkv_layout: str
        The layout of the query, key and value tensor, i.e. BSNH or BSH.

    out_layout: str
        The layout of the output tensor, i.e. BSNH or BSH.

    with_bias: bool
        Whether or not to include bias addition.

    with_cast: bool
        Whether or not rewriting is intended to be applied to a module after the FP16 conversion
        pass.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing an implicit fused multi head attention.

    rewriter: Callable[[Expr, Dict[DFPattern, Expr]], Expr]
        The rewriter for the pattern. It will check the matched patterns, and rewrite.
        If the matched pattern is not able to be rewritten to `R.nn.attention`, the rewriter
        returns the original IR.
    """

    # pylint: disable=invalid-name
    def handle_input(tensor, layout, transpose):
        if layout == "BSNH":
            permuted = is_op("relax.permute_dims")(tensor)
            shape = wildcard()
            reshaped = is_op("relax.reshape")(permuted, shape)
            if transpose:
                transposed = is_op("relax.permute_dims")(reshaped)

            def rewriter(matchings, x):
                if matchings[tensor].struct_info.ndim != 4:
                    return None
                if list(matchings[permuted].attrs.axes) != [0, 2, 1, 3]:
                    return None
                before_reshape = matchings[permuted].struct_info.shape.values
                after_reshape = matchings[shape].struct_info.values
                if not (
                    len(before_reshape) == 4
                    and len(after_reshape) == 3
                    and before_reshape[-2:] == after_reshape[-2:]
                ):
                    return None
                if transpose and list(matchings[transposed].attrs.axes) != [0, 2, 1]:
                    return None
                return x, x.struct_info.shape

            if transpose:
                return transposed, rewriter
            else:
                return reshaped, rewriter
        elif layout == "BSH":
            if transpose:
                transposed = is_op("relax.permute_dims")(tensor)

            def rewriter(matchings, x):
                if matchings[tensor].struct_info.ndim != 3:
                    return None
                if transpose and list(matchings[transposed].attrs.axes) != [0, 2, 1]:
                    return None
                before_reshape = x.struct_info.shape.values
                after_reshape = [before_reshape[0], before_reshape[1], 1, before_reshape[2]]
                return R.reshape(x, after_reshape), after_reshape

            if transpose:
                return transposed, rewriter
            else:
                return tensor, rewriter
        else:
            raise NotImplementedError()

    def handle_output(tensor, layout):
        if layout == "BSNH":
            shape = wildcard()
            reshaped = is_op("relax.reshape")(tensor, shape)
            permuted = is_op("relax.permute_dims")(reshaped)

            def rewriter(matchings, x):
                if matchings[tensor].struct_info.ndim != 3:
                    return None
                before_reshape = matchings[tensor].struct_info.shape.values
                after_reshape = matchings[shape].struct_info.values
                if not (
                    len(before_reshape) == 3
                    and len(after_reshape) == 4
                    and before_reshape[-2:] == after_reshape[-2:]
                ):
                    return None
                if list(matchings[permuted].attrs.axes) != [0, 2, 1, 3]:
                    return None
                return x

            return permuted, rewriter
        elif layout == "BSH":

            def rewriter(matchings, x):
                if matchings[tensor].struct_info.ndim != 3:
                    return None
                return R.reshape(x, matchings[tensor].struct_info.shape.values)

            return tensor, rewriter
        else:
            raise NotImplementedError()

    q_raw, k_raw, v_raw = wildcard(), wildcard(), wildcard()
    q, q_rewriter = handle_input(q_raw, qkv_layout, False)
    k, k_rewriter = handle_input(k_raw, qkv_layout, True)
    v, v_rewriter = handle_input(v_raw, qkv_layout, False)
    matmul_1 = is_op("relax.matmul")(q, k)
    scale = is_const()

    if with_cast:
        multiply = is_op("relax.multiply")(matmul_1, is_op("relax.astype")(scale))
    else:
        multiply = is_op("relax.multiply")(matmul_1, scale)

    if with_bias:
        bias_raw = wildcard()
        add = is_op("relax.add")(multiply, bias_raw)
        softmax_input = add
    else:
        softmax_input = multiply

    if with_cast:
        softmax_input = is_op("relax.astype")(softmax_input)

    softmax = is_op("relax.nn.softmax")(softmax_input)

    if with_cast:
        softmax_output = is_op("relax.astype")(softmax)
    else:
        softmax_output = softmax

    matmul_2 = is_op("relax.matmul")(softmax_output, v)

    out, out_rewriter = handle_output(matmul_2, out_layout)

    def rewriter(original, matchings):
        query, query_shape = q_rewriter(matchings, matchings[q_raw])
        key, key_shape = k_rewriter(matchings, matchings[k_raw])
        value, _ = v_rewriter(matchings, matchings[v_raw])
        if query is None or key is None or value is None:
            return original
        softmax_axis = matchings[softmax].attrs.axis
        softmax_input_rank = len(matchings[softmax].struct_info.shape)
        if softmax_axis == -1:
            softmax_axis += softmax_input_rank
        if softmax_axis != softmax_input_rank - 1:
            return original
        b, s, n, _ = query_shape
        _, s_kv, _, _ = key_shape
        if with_bias:
            bias = matchings[bias_raw]
            bias_shape = list(bias.struct_info.shape)
            if bias_shape == [b * n, s, s_kv]:
                bias = R.reshape(bias, [b, n, s, s_kv])
            elif bias_shape == [b * n, 1, s_kv]:
                bias = R.reshape(bias, [b, n, 1, s_kv])
            elif bias_shape == [b, s, s_kv]:
                bias = R.reshape(bias, [b, 1, s, s_kv])
            elif bias_shape == [b, 1, s_kv]:
                bias = R.reshape(bias, [b, 1, 1, s_kv])
            elif bias_shape in [[1, s, s_kv], [s, s_kv]]:
                bias = R.reshape(bias, [1, 1, s, s_kv])
            elif bias_shape in [[1, 1, s_kv], [1, s_kv], [s_kv]]:
                bias = R.reshape(bias, [1, 1, 1, s_kv])
            else:
                return original
        else:
            bias = None
        out = out_rewriter(
            matchings,
            R.nn.attention(
                query,
                key,
                value,
                bias,
                T.FloatImm(matchings[scale].data.dtype, float(matchings[scale].data.numpy())),
            ),
        )
        return out

    return out, rewriter
