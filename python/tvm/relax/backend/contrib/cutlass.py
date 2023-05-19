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
import operator
from functools import reduce
from typing import Mapping, Sequence

import tvm
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.relax import Call, DataflowVar, Function, PyExprMutator, Var, expr_functor, transform
from tvm.relax.dpl import rewrite_call
from tvm.relax.transform import PatternCheckContext

from ..pattern_registry import get_patterns_with_prefix, register_patterns
from ..patterns import (
    make_attention_pattern,
    make_attention_rewrite_pattern,
    make_fused_bias_activation_pattern,
    make_layer_norm_pattern,
    make_matmul_pattern,
    make_residual_block_pattern,
    make_stacked_attention_pattern,
)


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


def _check_residual(root_call: Call, context: PatternCheckContext) -> bool:
    if "residual" in context.annotated_expr:
        residual = context.annotated_expr["residual"]
        if not isinstance(residual, Var):
            if not residual in context.value_to_bound_var:
                return False

            residual = context.value_to_bound_var[residual]

        root_var = context.value_to_bound_var[root_call]
        if _has_dependency(from_var=residual, to_var=root_var, var_usages=context.var_usages):
            # If residual depends on the result of the root call, this cannot be handled by cutlass.
            return False

        shape1 = [int(s) for s in root_var.struct_info.shape]
        shape2 = [int(s) for s in residual.struct_info.shape]

        if shape1 != shape2:
            return False

    return True


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

    if not _check_residual(conv2d_call, context):
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

    if not _check_residual(context.annotated_expr["root"], context):
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


def _check_stacked_attention(context: PatternCheckContext) -> bool:
    """Check if the given stacked attention workload can be offloaded to CUTLASS."""
    if _has_leaking_intermediate_variables(context):
        return False
    if not context.annotated_expr["stacked_qkv"].struct_info.ndim == 3:
        return False
    if "split" in context.annotated_expr:
        split_op = context.annotated_expr["split"]
        if not split_op.attrs.axis == 2:
            return False
    else:
        last_end = 0
        for name in ["query", "key", "value"]:
            assert f"strided_slice_{name}" in context.annotated_expr
            strided_slice_op = context.annotated_expr[f"strided_slice_{name}"]
            if list(strided_slice_op.attrs.axes) != [2]:
                return False
            if list(strided_slice_op.attrs.begin) != [last_end]:
                return False
            if not len(strided_slice_op.attrs.end) == 1:
                return False
            last_end = strided_slice_op.attrs.end[0]
            if list(strided_slice_op.attrs.strides) != [1]:
                return False
    return True


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
        (
            "cutlass.stacked_attention",
            *make_stacked_attention_pattern(start_op="split"),
            _check_stacked_attention,
        ),
        (
            "cutlass.stacked_attention",
            *make_stacked_attention_pattern(start_op="split", with_bias=True),
            _check_stacked_attention,
        ),
        (
            "cutlass.stacked_attention",
            *make_stacked_attention_pattern(start_op="strided_slice"),
            _check_stacked_attention,
        ),
        (
            "cutlass.stacked_attention",
            *make_stacked_attention_pattern(start_op="strided_slice", with_bias=True),
            _check_stacked_attention,
        ),
    ]


def _check_layer_norm(context: PatternCheckContext) -> bool:
    attrs = context.matched_expr.attrs

    if not attrs.center or not attrs.scale:
        return False

    if len(attrs.axes) != 1:
        # Contiguous inner-most axes can be supported, but reject it for now for simplicity.
        return False

    axis = int(attrs.axes[0])
    rank = len(context.matched_expr.struct_info.shape)

    if axis < 0:
        axis += rank

    return axis == rank - 1


def layer_norm_pattern():
    """Create a layer norm pattern for CUTLASS."""
    return [
        (
            "cutlass.layer_norm",
            *make_layer_norm_pattern(),
            _check_layer_norm,
        ),
    ]


def attention_rewrite_patterns():
    """
    Returns a list of all attention rewriting patterns in cutlass BYOC backend.
    """
    patterns = []
    for qkv_layout in ["BSNH", "BSH"]:
        for out_layout in ["BSNH", "BSH"]:
            for with_bias in [True, False]:
                patterns.append(make_attention_rewrite_pattern(qkv_layout, out_layout, with_bias))
    return patterns


register_patterns(
    [
        *conv2d_patterns(),
        *matmul_patterns(),
        *residual_block_patterns(),
        *attention_patterns(),
        *layer_norm_pattern(),
    ]
)

_REWRITE_PATTERNS = [*attention_rewrite_patterns()]


@expr_functor.mutator
class WorkspaceAnnotator(PyExprMutator):
    """Annotate a workspace requirement for each CUTLASS-offloaded function."""

    def __init__(self, mod):
        super().__init__(mod)

    def visit_function_(self, f):
        if f.attrs is None or "Composite" not in f.attrs:
            body = super().visit_expr(f.body)
            new_f = Function(f.params, body, f.ret_struct_info, f.attrs, f.span)

            if f.attrs and "global_symbol" in f.attrs and "cutlass" in f.attrs["global_symbol"]:
                composite_func = body.blocks[0].bindings[0].value
                if "WorkspaceSize" in composite_func.attrs:
                    return new_f.with_attr("WorkspaceSize", composite_func.attrs["WorkspaceSize"])

            return new_f

        if "attention" in f.attrs["Composite"]:
            # Workspace is needed only for larger head sizes, but for simplicity we always allocate.
            out_dtype = f.ret_struct_info.dtype
            out_size_1d = reduce(operator.mul, f.ret_struct_info.shape, 1)
            # This needs to be in sync with the actual value that the kernel expects.
            workspace_size_bytes = out_size_1d * {"float16": 2, "float32": 4}[out_dtype]
            return f.with_attr("WorkspaceSize", workspace_size_bytes)

        return f


@tvm.transform.module_pass(opt_level=0)
def annotate_workspace(mod, _):
    """Pass to annotate a workspace requirement for each CUTLASS-offloaded function."""
    annotator = WorkspaceAnnotator(mod)
    for name, f in mod.functions.items():
        if isinstance(f, Function):
            new_f = annotator.visit_expr(f)
            mod.update_func(name, new_f)
    return mod


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
    for func_name, func in mod.functions.items():
        if isinstance(func, Function):
            for pattern, rewriter in _REWRITE_PATTERNS:
                mod[func_name] = rewrite_call(pattern, rewriter, func)
    patterns = get_patterns_with_prefix("cutlass")
    return tvm.transform.Sequential(
        [
            transform.FuseOpsByPattern(
                patterns, bind_constants=False, annotate_codegen=annotate_codegen
            ),
            annotate_workspace,
            transform.AllocateWorkspace(),
        ]
    )(mod)
