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

from typing import Mapping, Optional, Sequence, Tuple

import tvm
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.relax import DataflowVar, ShapeExpr, Var, transform
from tvm.relax.transform import PatternCheckContext

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


def _has_leaking_intermediate_variables(context: PatternCheckContext) -> bool:
    """
    Check whether intermediate variables in the region to be fused are used outside
    the fused region.
    """
    defined_vars = set(context.matched_bindings.keys())
    output_var = context.value_to_bound_var[context.matched_expr]
    intermediate_vars = {v for v in context.matched_bindings if v != output_var}

    if any(not isinstance(v, DataflowVar) for v in intermediate_vars):
        # If intermediate variable is not a DataflowVar, it can be accessed and potentially
        # used outside the DataflowBlock.
        return True

    # Check whether all users of an intermediate variable are inside the fused region.
    for var in intermediate_vars:
        if any(var_user not in defined_vars for var_user in context.var_usages[var]):
            return True

    return False


def _has_dependency(from_var: Var, to_var: Var, var_usages: Mapping[Var, Sequence[Var]]):
    if from_var == to_var:
        return True

    checked = set()
    vars_to_check = [to_var]
    while vars_to_check:
        current_var = vars_to_check.pop()
        for user in var_usages.get(current_var, []):
            if user == from_var:
                return True
            if user not in checked:
                checked.add(user)
                vars_to_check.append(user)

    return False


def _check_conv2d(context: PatternCheckContext) -> bool:
    """Check if the given conv2d workload can be offloaded to CUTLASS."""
    if _has_leaking_intermediate_variables(context):
        return False

    conv2d_call = context.annotated_expr["root"]
    data_layout = conv2d_call.attrs.data_layout
    kernel_layout = conv2d_call.attrs.kernel_layout
    data, weight, *_ = conv2d_call.args
    if (
        data_layout != "NHWC"
        or kernel_layout != "OHWI"
        or not _is_supported_dtype(data.struct_info.dtype, weight.struct_info.dtype)
    ):
        return False

    if "residual" in context.annotated_expr:
        residual = context.annotated_expr["residual"]
        if not isinstance(residual, Var):
            residual = context.value_to_bound_var[residual]
        conv2d_var = context.value_to_bound_var[conv2d_call]
        if _has_dependency(from_var=residual, to_var=conv2d_var, var_usages=context.var_usages):
            # If residual depends on the result of conv2d, this cannot be handled by cutlass.
            return False

    # pylint: disable=invalid-name
    IC = data.struct_info.shape.values[3]
    OC = weight.struct_info.shape.values[0]
    # not depthwise conv2d
    return not IC == OC == conv2d_call.attrs.groups


def _check_matmul(context: PatternCheckContext) -> bool:
    """Check if the given matmul workload can be offloaded to CUTLASS."""
    if _has_leaking_intermediate_variables(context):
        return False

    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]

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
        for check, base_patterns in [
            (_check_conv2d, conv2d_patterns()),
            (_check_matmul, matmul_patterns()),
        ]:
            for name, pat, arg_pat, _ in base_patterns:
                # Append residual patterns only to those base patterns with bias add,
                # since conv2d or matmul + residual add without bias is already supported
                # via conv2d or matmul + bias patterns (the residual input is treated as "bias").
                if "bias" in name:
                    for bin_op in ["relax.add", "relax.multiply"]:
                        patterns.append(
                            (
                                name + "_residual_" + bin_op.split(".")[-1] + name_postfix,
                                *make_residual_block_pattern(
                                    (pat, arg_pat), binary_op=bin_op, activation=activation
                                ),
                                check,
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


def partition_for_cutlass(mod, annotate_codegen=True):
    """
    Partition the input module into CUTLASS-supported subgraphs.

    Parameters
    ----------
    mod: tvm.IRModule
        The IRModule to be partitioned.

    annotate_codegen: bool
        Whether to wrap each created composite function with another function, whose
        body consists only of a call to the composite function. See the doc of FuseOpsByPattern
        for more detail.

    Returns
    -------
    mod: tvm.IRModule
        The resulting IRModule, containing partitioned subgraphs to be
        compiled by the CUTLASS backend.
    """

    patterns = get_patterns_with_prefix("cutlass")
    return transform.FuseOpsByPattern(
        patterns, bind_constants=False, annotate_codegen=annotate_codegen
    )(mod)
