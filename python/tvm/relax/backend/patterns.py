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

from typing import Dict, Mapping, Tuple

from tvm.relax.dpl.pattern import DFPattern, is_op, wildcard


def _with_bias_activation_pattern(
    out: DFPattern,
    args: Dict[str, DFPattern],
    with_bias: bool = False,
    activation: str = None,
) -> Tuple[DFPattern, Mapping[str, DFPattern]]:
    if with_bias:
        args["bias"] = bias = wildcard()
        out = is_op("relax.add")(out, bias)

    if activation:
        out = is_op(activation)(out)

    return out, args


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

    args: Mapping[str, DFPattern]
        The mapping from arg name to its pattern. It can be used to extract
        arg expression from match result.
    """
    lhs = wildcard()
    rhs = wildcard()
    args = {"lhs": lhs, "rhs": rhs}
    out = is_op(op_name)(lhs, rhs)

    return _with_bias_activation_pattern(out, args, with_bias, activation)


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

    args: Mapping[str, DFPattern]
        The mapping from arg name to its pattern. It can be used to extract
        arg expression from match result.
    """

    lhs = wildcard()
    rhs = wildcard()
    args = {"lhs": lhs, "rhs": rhs}

    if transposed_rhs:
        rhs = is_op("relax.permute_dims")(rhs)

    out = is_op("relax.matmul")(lhs, rhs)

    return _with_bias_activation_pattern(out, args, with_bias, activation)


def make_attention_pattern():
    """
    Create pattern for fused multi head attention.

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused multi head attention.

    args: Mapping[str, DFPattern]
        The mapping from arg name to its pattern. It can be used to extract
        arg expression from match result.
    """
    query = wildcard()
    key = wildcard()
    value = wildcard()
    out = is_op("relax.nn.attention")(query, key, value)

    return out
