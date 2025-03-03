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

"""Pattern table for hipblas backend"""
import operator
from functools import reduce

import tvm
from tvm.relax import transform
from tvm.relax.transform import PatternCheckContext

from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import make_matmul_pattern
from ..utils import has_leaking_intermediate_variables


def _is_supported_dtype(lhs_dtype, rhs_dtype, out_dtype):  # pylint: disable=unused-argument
    """Check if dtypes in the given workload are supported by hipblas BYOC."""
    if lhs_dtype == "e4m3_float8" and rhs_dtype == "e4m3_float8":
        # The output cannot be 'e5m2_float8' if inputs are 'e4m3_float8'
        # return out_dtype != "e5m2_float8"
        return False
    return (lhs_dtype == "float16" and rhs_dtype == "float16") or (
        lhs_dtype == "int8" and rhs_dtype == "int8"
    )


def _check_matmul(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]
    matmul_call = context.annotated_expr["root"]

    lhs_dtype = lhs.struct_info.dtype
    rhs_dtype = rhs.struct_info.dtype
    out_dtype = matmul_call.struct_info.dtype
    if not _is_supported_dtype(lhs_dtype, rhs_dtype, out_dtype):
        return False

    lhs_shape = lhs.struct_info.shape.values
    rhs_shape = rhs.struct_info.shape.values

    if not isinstance(lhs_shape[-1], (tvm.tir.expr.IntImm, int)):
        # Reduction axis must be constant
        return False

    if lhs_dtype == "int8" and rhs_dtype == "int8":
        return False
    elif lhs_dtype == "e4m3_float8" and rhs_dtype == "e4m3_float8":
        return False

    lhs_batches = reduce(operator.mul, lhs_shape[:-2], 1)
    rhs_batches = reduce(operator.mul, rhs_shape[:-2], 1)

    if "bias" in context.annotated_expr:
        if lhs_dtype == "int8" and rhs_dtype == "int8":
            # Non-default epilogue not supported for IGEMM
            return False
        bias = context.annotated_expr["bias"]
        bias_shape = bias.struct_info.shape.values
        bias_batches = reduce(operator.mul, bias_shape[:-1], 1)
        if not isinstance(bias_batches, (tvm.tir.expr.IntImm, int)) or int(bias_batches) > 1:
            # hipblas only supports bias vector
            return False

    # hipblasLt does not seem to support batched GEMM with one of matrices having
    # one batch (with batch_stride 0). So for batched GEMM, the two batch counts
    # must be equal. If lhs is batched but rhs is not, we can use the regular GEMM by
    # flattening all batch axes into the M axis.
    return (
        isinstance(lhs_batches, tvm.tir.Var)
        or isinstance(rhs_batches, tvm.tir.Var)
        or (int(lhs_batches) == int(rhs_batches))
        or (lhs_batches >= 1 and rhs_batches == 1)
    )


register_patterns(
    [
        (
            "hipblas.matmul",
            *make_matmul_pattern(
                with_bias=False,
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_bias",
            *make_matmul_pattern(
                with_bias=True,
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_bias_relu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_bias_gelu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_transposed",
            *make_matmul_pattern(
                with_bias=False,
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_transposed_bias",
            *make_matmul_pattern(
                with_bias=True,
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_transposed_bias_relu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.relu",
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
        (
            "hipblas.matmul_transposed_bias_gelu",
            *make_matmul_pattern(
                with_bias=True,
                activation="relax.nn.gelu",
                transposed_rhs=True,
            ),
            _check_matmul,
        ),
    ]
)


def partition_for_hipblas(mod):
    """
    Partition the input module into hipblas-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        offloaded to the hipblas backend.
    """

    patterns = get_patterns_with_prefix("hipblas")
    return transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
