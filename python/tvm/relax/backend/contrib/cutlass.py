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
from tvm.contrib.cutlass.build import is_valid_for_cutlass_matmul
from tvm.relax import Call, Expr, ShapeExpr, transform
from tvm.relax.dpl import DFPattern

from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_fused_bias_activation_pattern, make_matmul_pattern


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


def _check_matmul(
    match_result: Mapping[DFPattern, Expr],
    _: Expr,
) -> bool:
    matmul_call: Call = None
    for _, expr in match_result.items():
        if (
            isinstance(expr, Call)
            and isinstance(expr.op, tvm.ir.Op)
            and expr.op.name == "relax.matmul"
        ):
            matmul_call = expr
    if matmul_call is None:
        raise ValueError("Cannot find call to matmul from match_result.")

    lhs_shape = _get_static_shape(matmul_call.args[0].struct_info.shape)
    rhs_shape = _get_static_shape(matmul_call.args[1].struct_info.shape)
    if len(lhs_shape) < 2 or len(rhs_shape) < 2:
        return False

    lhs_dtype = matmul_call.args[0].struct_info.dtype
    rhs_dtype = matmul_call.args[1].struct_info.dtype
    if not _is_supported_dtype(lhs_dtype, rhs_dtype):
        return False

    return is_valid_for_cutlass_matmul(lhs_shape, rhs_shape)


register_patterns(
    [
        (
            "cutlass.conv2d",
            *make_fused_bias_activation_pattern(
                "relax.nn.conv2d",
                with_bias=False,
                activation=None,
            ),
        ),
        (
            "cutlass.conv2d_bias_relu",
            *make_fused_bias_activation_pattern(
                "relax.nn.conv2d",
                with_bias=True,
                activation="relax.nn.relu",
            ),
        ),
        (
            "cutlass.matmul",
            *make_matmul_pattern(
                with_bias=False,
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_bias",
            *make_matmul_pattern(
                with_bias=True,
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_bias_relu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_bias_gelu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_transposed",
            *make_matmul_pattern(
                with_bias=False,
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_transposed_bias",
            *make_matmul_pattern(
                with_bias=True,
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_transposed_bias_relu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
        (
            "cutlass.matmul_transposed_bias_gelu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
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
