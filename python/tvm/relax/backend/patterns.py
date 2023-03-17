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

"""Common patterns used in BYOC"""

from typing import Dict, Mapping, Tuple, Union

from tvm.relax.dpl.pattern import DFPattern, is_op, wildcard


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
