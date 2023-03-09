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

"""Pattern table for CUTLASS backend"""

from typing import Mapping, Optional, Tuple

import tvm
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.relax import Call, Expr, ShapeExpr, transform
from tvm.relax.dpl import CallPattern, DFPattern

from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import (
    make_attention_pattern,
    make_fused_bias_activation_pattern,
    make_matmul_pattern,
    make_residual_block_pattern,
)


def _get_static_shape(shape: ShapeExpr) -> Optional[Tuple[int]]:
    result = []
    for dim in shape.values:
        if isinstance(dim, tvm.tir.expr.IntImm):
            result.append(int(dim))
        else:
            return None
    return result


def _is_supported_dtype(lhs_dtype, rhs_dtype):
    """Check if dtypes in the given workload are supported by CUTLASS."""
    return (
        (lhs_dtype == "float16" and rhs_dtype == "float16")
        or (lhs_dtype == "float32" and rhs_dtype == "float32")
        or (lhs_dtype in ("int8", "uint8") and rhs_dtype in ("int8", "uint8"))
    )


def _find_call(op_name: str, match_result: Mapping[DFPattern, Expr]) -> Optional[Expr]:
    result = None

    for pattern, expr in match_result.items():
        if (
            isinstance(expr, Call)
            and isinstance(pattern, CallPattern)
            and isinstance(expr.op, tvm.ir.Op)
            and expr.op.name == op_name
        ):
            if result is not None:
                raise ValueError(f"Found multiple matched call node for {op_name}")
            result = expr

    return result


def _check_conv2d(
    match_result: Mapping[DFPattern, Expr],
    _: Expr,
):
    """Check if the given conv2d workload can be offloaded to CUTLASS."""

    conv2d_call = _find_call("relax.nn.conv2d", match_result)
    if conv2d_call is None:
        return False

    data_layout = conv2d_call.attrs.data_layout
    kernel_layout = conv2d_call.attrs.kernel_layout
    data, weight, *_ = conv2d_call.args
    if (
        data_layout != "NHWC"
        or kernel_layout != "OHWI"
        or not _is_supported_dtype(data.struct_info.dtype, weight.struct_info.dtype)
    ):
        return False

    # pylint: disable=invalid-name
    IC = data.struct_info.shape.values[3]
    OC = weight.struct_info.shape.values[0]
    # not depthwise conv2d
    return not IC == OC == conv2d_call.attrs.groups


def _check_matmul(
    match_result: Mapping[DFPattern, Expr],
    _: Expr,
) -> bool:
    """Check if the given matmul workload can be offloaded to CUTLASS."""

    matmul_call: Call = _find_call("relax.matmul", match_result)
    if matmul_call is None:
        return False

    lhs, rhs, *_ = matmul_call.args

    lhs_dtype = lhs.struct_info.dtype
    rhs_dtype = rhs.struct_info.dtype
    if not _is_supported_dtype(lhs_dtype, rhs_dtype):
        return False

    lhs_shape = lhs.struct_info.shape.values
    rhs_shape = rhs.struct_info.shape.values
    return is_shape_valid_for_cutlass_matmul(lhs_shape, rhs_shape)


def _get_activation_from_name(pattern_name):
    if "_relu" in pattern_name:
        return "relax.nn.relu"
    elif "_gelu" in pattern_name:
        return "relax.nn.gelu"
    elif "_silu" in pattern_name:
        return "relax.nn.silu"
    else:
        return None


def matmul_patterns():
    """
    Returns a list of all matmul patterns in cutlass BYOC backend.
    """

    def _matmul_pattern(pattern_name):
        transposed_rhs = "_transposed" in pattern_name
        with_bias = "_bias" in pattern_name
        activation = _get_activation_from_name(pattern_name)

        return (
            pattern_name,
            *make_matmul_pattern(
                transposed_rhs=transposed_rhs,
                with_bias=with_bias,
                activation=activation,
            ),
            _check_matmul,
        )

    return [
        _matmul_pattern("cutlass.matmul"),
        _matmul_pattern("cutlass.matmul_bias"),
        _matmul_pattern("cutlass.matmul_bias_relu"),
        _matmul_pattern("cutlass.matmul_bias_gelu"),
        _matmul_pattern("cutlass.matmul_transposed"),
        _matmul_pattern("cutlass.matmul_transposed_bias"),
        _matmul_pattern("cutlass.matmul_transposed_bias_relu"),
        _matmul_pattern("cutlass.matmul_transposed_bias_gelu"),
    ]


def conv2d_patterns():
    """
    Returns a list of all conv2d patterns in cutlass BYOC backend.
    """

    def _conv2d_pattern(pattern_name):
        with_bias = "_bias" in pattern_name
        activation = _get_activation_from_name(pattern_name)

        return (
            pattern_name,
            *make_fused_bias_activation_pattern(
                "relax.nn.conv2d",
                with_bias=with_bias,
                activation=activation,
            ),
            _check_conv2d,
        )

    return [
        _conv2d_pattern("cutlass.conv2d"),
        _conv2d_pattern("cutlass.conv2d_bias"),
        _conv2d_pattern("cutlass.conv2d_bias_relu"),
        _conv2d_pattern("cutlass.conv2d_bias_silu"),
    ]


def residual_block_patterns():
    """
    Returns a list of all residual block patterns in cutlass BYOC backend.
    """
    patterns = []

    for activation, name_postfix in [(None, ""), ("relax.nn.relu", "_relu")]:
        for name, pat, arg_pat, _ in conv2d_patterns()[1:]:
            for bin_op in ["relax.add", "relax.multiply"]:
                patterns.append(
                    (
                        name + "_residual_" + bin_op.split(".")[-1] + name_postfix,
                        *make_residual_block_pattern(
                            (pat, arg_pat), binary_op=bin_op, activation=activation
                        ),
                        _check_conv2d,
                    )
                )

    return patterns


def attention_patterns():
    """
    Returns a list of all attention patterns in cutlass BYOC backend.
    """
    return [
        (
            "cutlass.attention",
            *make_attention_pattern(),
        ),
        (
            "cutlass.attention_bias",
            *make_attention_pattern(with_bias=True),
        ),
    ]


register_patterns(
    [
        *conv2d_patterns(),
        *matmul_patterns(),
        *residual_block_patterns(),
        *attention_patterns(),
    ]
)


def partition_for_cutlass(mod):
    """
    Partition the input module into CUTLASS-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        compiled by the CUTLASS backend.
    """

    cutlass_pattern_entries = get_patterns_with_prefix("cutlass")
    patterns = [(e.name, e.pattern, e.check) for e in cutlass_pattern_entries]
    return transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
