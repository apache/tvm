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

"""Pattern table for the XNNPACK Relax backend."""

from collections.abc import Callable

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm import relax
from tvm.relax.dpl.pattern import is_const, is_op, wildcard
from tvm.relax.transform import FuseOpsByPattern, FusionPattern, PatternCheckContext

from .pattern_registry import get_patterns_with_prefix, register_patterns
from .utils import has_leaking_intermediate_variables

_SUPPORTED_PRECISIONS = ("fp32", "fp16_hint", "fp16_force")
_SUPPORTED_PARTITION_POLICIES = ("greedy", "cost", "debug_all_supported")
_SUPPORTED_LAYOUT_POLICIES = ("auto", "NHWC", "preserve")
_XNN_EXTRA_BYTES = 16
_DTYPE_BYTES = {"float16": 2, "float32": 4, "int8": 1, "uint8": 1, "int32": 4}
_QPARAM_SCALE_RTOL = 1e-5
_QPARAM_SCALE_ATOL = 1e-8


def _get_static_shape(expr: relax.Expr) -> list[int] | None:
    sinfo = expr.struct_info
    if not isinstance(sinfo, relax.TensorStructInfo):
        return None
    if sinfo.shape is None or not hasattr(sinfo.shape, "values"):
        return None

    shape = []
    for dim in sinfo.shape.values:
        if not isinstance(dim, (tvm.tirx.expr.IntImm, int)):
            return None
        dim = int(dim)
        if dim <= 0:
            return None
        shape.append(dim)
    return shape


def _is_float32_tensor(expr: relax.Expr) -> bool:
    sinfo = expr.struct_info
    return isinstance(sinfo, relax.TensorStructInfo) and sinfo.dtype == "float32"


def _is_static_float32(expr: relax.Expr) -> bool:
    return _is_float32_tensor(expr) and _get_static_shape(expr) is not None


def _tensor_dtype(expr: relax.Expr) -> str | None:
    sinfo = expr.struct_info
    if isinstance(sinfo, relax.TensorStructInfo):
        return str(sinfo.dtype)
    return None


def _num_elements(expr: relax.Expr) -> int | None:
    shape = _get_static_shape(expr)
    if shape is None:
        return None
    result = 1
    for dim in shape:
        result *= dim
    return result


def _tensor_nbytes(expr: relax.Expr) -> int:
    num_elements = _num_elements(expr)
    dtype = _tensor_dtype(expr)
    if num_elements is None or dtype not in _DTYPE_BYTES:
        return 0
    return num_elements * _DTYPE_BYTES[dtype]


def _const_numpy(expr: relax.Expr) -> np.ndarray | None:
    if not isinstance(expr, relax.Constant):
        return None
    return expr.data.numpy()


def _const_scalar_float(expr: relax.Expr) -> float | None:
    arr = _const_numpy(expr)
    if arr is None or arr.size != 1:
        return None
    value = float(arr.reshape(-1)[0])
    if not np.isfinite(value):
        return None
    return value


def _const_int_array(expr: relax.Expr) -> np.ndarray | None:
    arr = _const_numpy(expr)
    if arr is None:
        return None
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    return arr.astype("int64")


def _const_float_array(expr: relax.Expr) -> np.ndarray | None:
    arr = _const_numpy(expr)
    if arr is None:
        return None
    if not np.issubdtype(arr.dtype, np.floating):
        return None
    arr = arr.astype("float64")
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _const_scalar_int(expr: relax.Expr) -> int | None:
    arr = _const_int_array(expr)
    if arr is None or arr.size != 1:
        return None
    return int(arr.reshape(-1)[0])


def _same_static_shape(lhs: relax.Expr, rhs: relax.Expr) -> bool:
    lhs_shape = _get_static_shape(lhs)
    rhs_shape = _get_static_shape(rhs)
    return lhs_shape is not None and lhs_shape == rhs_shape


def _is_external_input(expr: relax.Expr) -> bool:
    return not isinstance(expr, relax.Constant)


def _as_float_prim_value(expr: relax.Expr) -> float | None:
    if not isinstance(expr, relax.PrimValue):
        return None
    value = expr.value
    if isinstance(value, tvm.tirx.expr.FloatImm):
        return float(value.value)
    if isinstance(value, tvm.tirx.expr.IntImm):
        return float(value.value)
    return None


def _call_op_name(expr: relax.Expr) -> str | None:
    if not isinstance(expr, relax.Call):
        return None
    if hasattr(expr.op, "name"):
        return expr.op.name
    return None


def _attrs_axis(attrs) -> int:
    return int(attrs.axis) if attrs is not None and hasattr(attrs, "axis") else -1


def _normalize_axis(axis: int, rank: int) -> int | None:
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    return axis


def _qscheme_from_scale(scale: relax.Expr) -> str | None:
    arr = _const_float_array(scale)
    if arr is None:
        return None
    return "per_tensor" if arr.size == 1 else "per_channel"


def _parse_qparams(
    scale: relax.Expr,
    zero_point: relax.Expr,
    dtype: str,
    shape: list[int],
    axis: int,
    *,
    allow_per_channel: bool,
    channel_dim: int | None = None,
    require_zero_point_zero: bool = False,
) -> dict[str, object] | None:
    scale_arr = _const_float_array(scale)
    zp_arr = _const_int_array(zero_point)
    if scale_arr is None or zp_arr is None:
        return None
    if not np.all(scale_arr > 0):
        return None
    if dtype == "int8":
        if np.any(zp_arr < -128) or np.any(zp_arr > 127):
            return None
    elif dtype == "int32":
        if np.any(zp_arr < np.iinfo("int32").min) or np.any(zp_arr > np.iinfo("int32").max):
            return None
    else:
        return None
    if require_zero_point_zero and np.any(zp_arr != 0):
        return None

    rank = len(shape)
    normalized_axis = _normalize_axis(axis, rank)
    if normalized_axis is None:
        return None
    if channel_dim is None:
        channel_dim = normalized_axis
    if channel_dim < 0 or channel_dim >= rank:
        return None

    if scale_arr.size == 1 and zp_arr.size == 1:
        return {
            "qscheme": "per_tensor",
            "scale": scale_arr.reshape(-1).astype("float64"),
            "zero_point": int(zp_arr.reshape(-1)[0]),
            "axis": normalized_axis,
            "channel_dim": channel_dim,
        }

    if not allow_per_channel:
        return None
    if scale_arr.ndim != 1 or scale_arr.size != shape[channel_dim]:
        return None
    if zp_arr.size not in (1, scale_arr.size):
        return None
    if zp_arr.size != 1 and np.any(zp_arr != zp_arr.reshape(-1)[0]):
        return None
    return {
        "qscheme": "per_channel",
        "scale": scale_arr.reshape(-1).astype("float64"),
        "zero_point": int(zp_arr.reshape(-1)[0]),
        "axis": normalized_axis,
        "channel_dim": channel_dim,
    }


def _parse_dequantize(
    expr: relax.Expr,
    *,
    expected_dtype: str,
    allow_per_channel: bool,
    channel_dim: int | None = None,
    require_constant_input: bool = False,
    require_zero_point_zero: bool = False,
    bindings=None,
    input_override: relax.Expr | None = None,
) -> dict[str, object] | None:
    if _call_op_name(expr) != "relax.dequantize":
        return None
    input_expr, scale, zero_point = expr.args[:3]
    if input_override is not None:
        input_expr = input_override
    if isinstance(input_expr, relax.Var) and bindings is not None and input_expr in bindings:
        input_expr = bindings[input_expr]
    if require_constant_input and not isinstance(input_expr, relax.Constant):
        return None
    if _tensor_dtype(input_expr) != expected_dtype or _tensor_dtype(expr) != "float32":
        return None
    shape = _get_static_shape(input_expr)
    if shape is None:
        return None
    qparams = _parse_qparams(
        scale,
        zero_point,
        expected_dtype,
        shape,
        _attrs_axis(expr.attrs),
        allow_per_channel=allow_per_channel,
        channel_dim=channel_dim,
        require_zero_point_zero=require_zero_point_zero,
    )
    if qparams is None:
        return None
    qparams.update({"value": input_expr, "shape": shape, "dtype": expected_dtype})
    return qparams


def _parse_activation_qdq(expr: relax.Expr, bindings=None) -> dict[str, object] | None:
    qdq = _parse_dequantize(
        expr,
        expected_dtype="int8",
        allow_per_channel=False,
        require_constant_input=False,
        bindings=bindings,
    )
    if qdq is None or not _is_external_input(qdq["value"]):
        return None
    return qdq


def _parse_weight_qdq(
    expr: relax.Expr,
    channel_dim: int,
    bindings=None,
    input_override: relax.Expr | None = None,
) -> dict[str, object] | None:
    return _parse_dequantize(
        expr,
        expected_dtype="int8",
        allow_per_channel=True,
        channel_dim=channel_dim,
        require_constant_input=True,
        require_zero_point_zero=True,
        bindings=bindings,
        input_override=input_override,
    )


def _parse_bias_qdq(
    expr: relax.Expr,
    input_scale: np.ndarray,
    weight_scale: np.ndarray,
    output_channels: int,
    bindings=None,
    input_override: relax.Expr | None = None,
) -> dict[str, object] | None:
    qdq = _parse_dequantize(
        expr,
        expected_dtype="int32",
        allow_per_channel=True,
        channel_dim=0,
        require_constant_input=True,
        require_zero_point_zero=True,
        bindings=bindings,
        input_override=input_override,
    )
    if qdq is None or qdq["shape"] != [output_channels]:
        return None
    expected = input_scale.reshape(-1)[0] * weight_scale
    if expected.size == 1 and qdq["scale"].size == output_channels:
        expected = np.full((output_channels,), expected.reshape(-1)[0])
    if qdq["scale"].size == 1 and expected.size == output_channels:
        return None
    if not np.allclose(qdq["scale"], expected, rtol=_QPARAM_SCALE_RTOL, atol=_QPARAM_SCALE_ATOL):
        return None
    return qdq


def _parse_output_quantize(expr: relax.Expr) -> dict[str, object] | None:
    if _call_op_name(expr) != "relax.quantize":
        return None
    input_expr, scale, zero_point = expr.args[:3]
    if _tensor_dtype(input_expr) != "float32" or _tensor_dtype(expr) != "int8":
        return None
    shape = _get_static_shape(expr)
    if shape is None:
        return None
    qparams = _parse_qparams(
        scale,
        zero_point,
        "int8",
        shape,
        _attrs_axis(expr.attrs),
        allow_per_channel=False,
    )
    if qparams is None:
        return None
    qparams.update({"value": input_expr, "shape": shape, "dtype": "int8"})
    return qparams


def _qparams_equal(lhs: dict[str, object], rhs: dict[str, object]) -> bool:
    return (
        lhs["qscheme"] == rhs["qscheme"]
        and lhs["zero_point"] == rhs["zero_point"]
        and lhs["axis"] == rhs["axis"]
        and lhs["channel_dim"] == rhs["channel_dim"]
        and np.array_equal(lhs["scale"], rhs["scale"])
    )


def _qparams_value_equal(lhs: dict[str, object], rhs: dict[str, object]) -> bool:
    return (
        lhs["qscheme"] == rhs["qscheme"]
        and lhs["zero_point"] == rhs["zero_point"]
        and np.array_equal(lhs["scale"], rhs["scale"])
    )


def _activation_bounds(root: relax.Expr, inner: relax.Expr) -> tuple[relax.Expr, float, float] | None:
    if root.same_as(inner) or (
        isinstance(root, relax.Call)
        and isinstance(inner, relax.Call)
        and _call_op_name(root) == _call_op_name(inner)
    ):
        return inner, -float("inf"), float("inf")
    if _call_op_name(root) == "relax.nn.relu" and root.args[0].same_as(inner):
        return root, 0.0, float("inf")
    if _call_op_name(root) == "relax.clip" and root.args[0].same_as(inner):
        min_value = _as_float_prim_value(root.args[1])
        max_value = _as_float_prim_value(root.args[2])
        if min_value is None or max_value is None or min_value > max_value:
            return None
        return root, min_value, max_value
    return None


def _collect_op_names(expr: relax.Expr) -> list[str]:
    names: list[str] = []

    def visit(current):
        if isinstance(current, relax.Call):
            name = _call_op_name(current)
            if name is not None:
                names.append(name)
            for arg in current.args:
                visit(arg)

    visit(expr)
    names.reverse()
    return names


def _find_call_in_expr(expr: relax.Expr, op_name: str) -> relax.Call | None:
    if isinstance(expr, relax.Call):
        if _call_op_name(expr) == op_name:
            return expr
        for arg in expr.args:
            found = _find_call_in_expr(arg, op_name)
            if found is not None:
                return found
    return None


def _find_call_in_expr_resolved(expr: relax.Expr, op_name: str, bindings=None) -> relax.Call | None:
    if isinstance(expr, relax.Var) and bindings is not None and expr in bindings:
        return _find_call_in_expr_resolved(bindings[expr], op_name, bindings)
    if isinstance(expr, relax.Call):
        if _call_op_name(expr) == op_name:
            return expr
        for arg in expr.args:
            found = _find_call_in_expr_resolved(arg, op_name, bindings)
            if found is not None:
                return found
    return None


def _find_bias_dequantize(expr: relax.Expr, weighted: relax.Expr) -> relax.Call | None:
    if isinstance(expr, relax.Call):
        if _call_op_name(expr) == "relax.add":
            lhs, rhs = expr.args
            if lhs.same_as(weighted) and _call_op_name(rhs) == "relax.dequantize":
                return rhs
            if rhs.same_as(weighted) and _call_op_name(lhs) == "relax.dequantize":
                return lhs
        for arg in expr.args:
            found = _find_bias_dequantize(arg, weighted)
            if found is not None:
                return found
    return None


def _resolve_bound_expr(context: PatternCheckContext, expr: relax.Expr | None) -> relax.Expr | None:
    if isinstance(expr, relax.Var) and expr in context.matched_bindings:
        return _resolve_bound_expr(context, context.matched_bindings[expr])
    if isinstance(expr, relax.Var):
        for value, bound_var in context.value_to_bound_var.items():
            if bound_var.same_as(expr):
                return value
    return expr


def _op_list_from_pattern(pattern_name: str, root: relax.Expr) -> list[str]:
    op_list = _collect_op_names(root)
    if "qs8_reshape" in pattern_name:
        return ["relax.dequantize", "relax.reshape", "relax.quantize"]
    if "qs8_flatten" in pattern_name:
        return ["relax.dequantize", "relax.flatten", "relax.quantize"]
    if "qs8_copy" in pattern_name:
        return ["relax.dequantize", "relax.quantize"]
    if "qs8_max_pool2d" in pattern_name:
        return ["relax.dequantize", "relax.nn.max_pool2d", "relax.quantize"]
    if "qs8_avg_pool2d" in pattern_name:
        return ["relax.dequantize", "relax.nn.avg_pool2d", "relax.quantize"]
    if "qs8_add" in pattern_name:
        return [
            "relax.dequantize",
            "relax.add",
            *(["relax.nn.relu"] if "relu" in pattern_name else []),
            *(["relax.clip"] if "clip" in pattern_name else []),
            "relax.quantize",
        ]
    if "qs8_fully_connected" in pattern_name:
        return [
            "relax.dequantize",
            "relax.matmul",
            *(["relax.add"] if "bias" in pattern_name else []),
            *(["relax.nn.relu"] if "relu" in pattern_name else []),
            *(["relax.clip"] if "clip" in pattern_name else []),
            "relax.quantize",
        ]
    if "qs8_conv2d" in pattern_name or "qs8_depthwise_conv2d" in pattern_name:
        return [
            "relax.dequantize",
            "relax.nn.conv2d",
            *(["relax.add"] if "bias" in pattern_name else []),
            *(["relax.nn.relu"] if "relu" in pattern_name else []),
            *(["relax.clip"] if "clip" in pattern_name else []),
            "relax.quantize",
        ]
    if "conv2d" in pattern_name:
        op_list = ["relax.nn.conv2d"]
        if "bias" in pattern_name:
            op_list.append("relax.add")
        if "relu" in pattern_name:
            op_list.append("relax.nn.relu")
        if "clip" in pattern_name:
            op_list.append("relax.clip")
        return op_list
    if op_list:
        return op_list
    if pattern_name.endswith(".add"):
        return ["relax.add"]
    if pattern_name.endswith(".relu"):
        return ["relax.nn.relu"]
    if pattern_name.endswith(".clip"):
        return ["relax.clip"]
    if pattern_name.endswith(".sigmoid"):
        return ["relax.sigmoid"]
    if pattern_name.endswith(".tanh"):
        return ["relax.tanh"]
    if pattern_name.endswith(".max_pool2d"):
        return ["relax.nn.max_pool2d"]
    if pattern_name.endswith(".avg_pool2d"):
        return ["relax.nn.avg_pool2d"]
    return []


def _candidate_layout(context: PatternCheckContext) -> str:
    for expr in context.annotated_expr.values():
        if isinstance(expr, relax.Call) and expr.attrs is not None:
            attrs = expr.attrs
            if hasattr(attrs, "data_layout"):
                return str(attrs.data_layout)
            if hasattr(attrs, "layout"):
                return str(attrs.layout)
    return "none"


def _candidate_dtype(context: PatternCheckContext) -> str:
    for key in ("root", "conv", "weighted", "input", "q_data", "data", "lhs", "rhs", "q_lhs"):
        expr = context.annotated_expr.get(key)
        if expr is not None:
            dtype = _tensor_dtype(expr)
            if dtype is not None:
                return dtype
    return "unknown"


def _padding_2d(padding) -> list[int] | None:
    padding = [int(x) for x in padding]
    if len(padding) == 1:
        return [padding[0], padding[0], padding[0], padding[0]]
    if len(padding) == 2:
        return [padding[0], padding[1], padding[0], padding[1]]
    if len(padding) == 4:
        return padding
    return None


def _check_no_leaks(context: PatternCheckContext) -> bool:
    if has_leaking_intermediate_variables(context):
        return False
    return True


def _check_unary(context: PatternCheckContext) -> bool:
    if not _check_no_leaks(context):
        return False
    input_expr = context.annotated_expr["input"]
    root_expr = context.annotated_expr["root"]

    if not _is_external_input(input_expr):
        return False
    if not _is_static_float32(input_expr) or not _is_static_float32(root_expr):
        return False
    if not _same_static_shape(input_expr, root_expr):
        return False

    if _call_op_name(root_expr) == "relax.clip":
        clip_min = _as_float_prim_value(root_expr.args[1])
        clip_max = _as_float_prim_value(root_expr.args[2])
        return clip_min is not None and clip_max is not None and clip_min <= clip_max
    return True


def _check_add(context: PatternCheckContext) -> bool:
    if not _check_no_leaks(context):
        return False
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]
    root = context.annotated_expr["root"]

    if not _is_static_float32(lhs) or not _is_static_float32(rhs) or not _is_static_float32(root):
        return False
    if not _is_external_input(lhs) or not _is_external_input(rhs):
        return False
    return _same_static_shape(lhs, rhs) and _same_static_shape(lhs, root)


def _check_pool2d(context: PatternCheckContext) -> bool:
    if not _check_no_leaks(context):
        return False
    input_expr = context.annotated_expr["input"]
    root = context.annotated_expr["root"]

    if not _is_external_input(input_expr):
        return False
    if not _is_static_float32(input_expr) or not _is_static_float32(root):
        return False
    if len(_get_static_shape(input_expr)) != 4 or len(_get_static_shape(root)) != 4:
        return False

    attrs = root.attrs
    out_layout = attrs.out_layout if attrs.out_layout else attrs.layout
    if attrs.layout != "NHWC" or out_layout != "NHWC":
        return False
    if [int(x) for x in attrs.dilation] != [1, 1]:
        return False
    if bool(attrs.ceil_mode):
        return False
    if _padding_2d(attrs.padding) != [0, 0, 0, 0]:
        return False
    if _call_op_name(root) == "relax.nn.avg_pool2d" and bool(attrs.count_include_pad):
        return False
    return True


def _check_conv2d(context: PatternCheckContext) -> bool:
    if not _check_no_leaks(context):
        return False

    data = context.annotated_expr["data"]
    weight = context.annotated_expr["weight"]
    conv = context.annotated_expr["conv"]
    root = context.annotated_expr["root"]
    bias = context.annotated_expr.get("bias")

    if not _is_external_input(data) or not isinstance(weight, relax.Constant):
        return False
    if bias is not None and not isinstance(bias, relax.Constant):
        return False
    exprs = [data, weight, conv, root]
    if bias is not None:
        exprs.append(bias)
    for expr in exprs:
        if not _is_static_float32(expr):
            return False

    data_shape = _get_static_shape(data)
    weight_shape = _get_static_shape(weight)
    conv_shape = _get_static_shape(conv)
    root_shape = _get_static_shape(root)
    if len(data_shape) != 4 or len(weight_shape) != 4 or len(conv_shape) != 4:
        return False
    if conv_shape != root_shape:
        return False

    attrs = conv.attrs
    out_layout = attrs.out_layout if attrs.out_layout else attrs.data_layout
    if attrs.data_layout != "NHWC" or out_layout != "NHWC" or attrs.kernel_layout != "OHWI":
        return False
    if int(attrs.groups) != 1:
        return False
    if attrs.out_dtype not in ("", "float32"):
        return False
    if _padding_2d(attrs.padding) is None:
        return False
    if weight_shape[1] <= 0 or weight_shape[2] <= 0:
        return False
    if data_shape[3] != weight_shape[3] or conv_shape[3] != weight_shape[0]:
        return False
    if bias is not None and _get_static_shape(bias) != [weight_shape[0]]:
        return False

    root_name = _call_op_name(root)
    if root_name == "relax.clip":
        clip_min = _as_float_prim_value(root.args[1])
        clip_max = _as_float_prim_value(root.args[2])
        return clip_min is not None and clip_max is not None and clip_min <= clip_max
    return root_name in ("relax.nn.relu", "relax.add", "relax.nn.conv2d")


def _qs8_weighted_parts(context: PatternCheckContext) -> tuple[dict[str, object], ...] | None:
    matched_expr = _resolve_bound_expr(context, context.matched_expr)
    output = _parse_output_quantize(matched_expr)
    if output is None:
        return None
    weighted = _resolve_bound_expr(context, context.annotated_expr.get("weighted"))
    if weighted is None:
        q_root = _resolve_bound_expr(context, output["value"])
        weighted = _find_call_in_expr(q_root, "relax.matmul") or _find_call_in_expr(
            q_root, "relax.nn.conv2d"
        )
    if weighted is None:
        return None

    data_dq = _resolve_bound_expr(context, context.annotated_expr.get("data_dq", weighted.args[0]))
    data = _parse_activation_qdq(data_dq, context.matched_bindings)
    if data is None:
        return None
    return (data, output, {"weighted": weighted})


def _check_qs8_fully_connected(context: PatternCheckContext) -> bool:
    if _tensor_dtype(context.annotated_expr.get("root")) != "int8":
        return False
    parts = _qs8_weighted_parts(context)
    if parts is None:
        return False
    data, _, extra = parts
    matmul = extra["weighted"]
    if _call_op_name(matmul) != "relax.matmul":
        return False
    weight_dq = _resolve_bound_expr(
        context, context.annotated_expr.get("weight_dq", matmul.args[1])
    )
    weight = _parse_weight_qdq(
        weight_dq,
        channel_dim=1,
        bindings=context.matched_bindings,
        input_override=_resolve_bound_expr(context, weight_dq.args[0]),
    )
    if weight is None:
        return False
    if context.annotated_expr.get("bias_dq") is None:
        return True
    data_shape = _get_static_shape(data["value"])
    weight_shape = _get_static_shape(weight["value"])
    out_shape = _get_static_shape(context.matched_expr)
    if data_shape is None or weight_shape is None or out_shape is None:
        return False
    if len(data_shape) != 2 or len(weight_shape) != 2 or len(out_shape) != 2:
        return False
    if data_shape[1] != weight_shape[0] or out_shape != [data_shape[0], weight_shape[1]]:
        return False
    return True


def _check_qs8_conv2d(context: PatternCheckContext) -> bool:
    if _tensor_dtype(context.annotated_expr.get("root")) != "int8":
        return False
    parts = _qs8_weighted_parts(context)
    if parts is None:
        return False
    data, _, extra = parts
    conv = extra["weighted"]
    if _call_op_name(conv) != "relax.nn.conv2d":
        return False
    weight_dq = _resolve_bound_expr(
        context, context.annotated_expr.get("weight_dq", conv.args[1])
    )
    weight = _parse_weight_qdq(
        weight_dq,
        channel_dim=0,
        bindings=context.matched_bindings,
        input_override=_resolve_bound_expr(context, weight_dq.args[0]),
    )
    if weight is None:
        return False
    data_shape = _get_static_shape(data["value"])
    weight_shape = _get_static_shape(weight["value"])
    conv_shape = _get_static_shape(conv)
    root_shape = _get_static_shape(context.matched_expr)
    if data_shape is None or weight_shape is None or conv_shape is None or root_shape is None:
        return False
    if len(data_shape) != 4 or len(weight_shape) != 4 or len(conv_shape) != 4:
        return False
    attrs = conv.attrs
    out_layout = attrs.out_layout if attrs.out_layout else attrs.data_layout
    if attrs.data_layout != "NHWC" or out_layout != "NHWC" or attrs.kernel_layout != "OHWI":
        return False
    if int(attrs.groups) != 1 or attrs.out_dtype not in ("", "float32"):
        return False
    if _padding_2d(attrs.padding) is None:
        return False
    if data_shape[3] != weight_shape[3] or conv_shape[3] != weight_shape[0]:
        return False
    if root_shape != conv_shape:
        return False
    return True


def _check_qs8_depthwise_conv2d(context: PatternCheckContext) -> bool:
    if _tensor_dtype(context.annotated_expr.get("root")) != "int8":
        return False
    parts = _qs8_weighted_parts(context)
    if parts is None:
        return False
    data, _, extra = parts
    conv = extra["weighted"]
    if _call_op_name(conv) != "relax.nn.conv2d":
        return False
    weight_dq = _resolve_bound_expr(
        context, context.annotated_expr.get("weight_dq", conv.args[1])
    )
    weight = _parse_weight_qdq(
        weight_dq,
        channel_dim=2,
        bindings=context.matched_bindings,
        input_override=_resolve_bound_expr(context, weight_dq.args[0]),
    )
    if weight is None:
        return False
    data_shape = _get_static_shape(data["value"])
    weight_shape = _get_static_shape(weight["value"])
    conv_shape = _get_static_shape(conv)
    root_shape = _get_static_shape(context.matched_expr)
    if data_shape is None or weight_shape is None or conv_shape is None or root_shape is None:
        return False
    if len(data_shape) != 4 or len(weight_shape) != 4 or len(conv_shape) != 4:
        return False
    attrs = conv.attrs
    out_layout = attrs.out_layout if attrs.out_layout else attrs.data_layout
    if attrs.data_layout != "NHWC" or out_layout != "NHWC" or attrs.kernel_layout != "HWOI":
        return False
    if attrs.out_dtype not in ("", "float32") or _padding_2d(attrs.padding) is None:
        return False
    input_channels = data_shape[3]
    depth_multiplier = weight_shape[3]
    if depth_multiplier != 1:
        return False
    if int(attrs.groups) != input_channels:
        return False
    if weight_shape[2] != input_channels or conv_shape[3] != input_channels * depth_multiplier:
        return False
    if root_shape != conv_shape:
        return False
    return True


def _qs8_unary_qdq_parts(
    context: PatternCheckContext,
    op_name: str,
) -> tuple[dict[str, object], dict[str, object], relax.Call] | None:
    if _tensor_dtype(context.annotated_expr.get("root")) != "int8":
        return None
    matched_expr = _resolve_bound_expr(context, context.matched_expr)
    output = _parse_output_quantize(matched_expr)
    if output is None:
        return None
    op = _resolve_bound_expr(context, context.annotated_expr.get("op", output["value"]))
    if not isinstance(op, relax.Call) or _call_op_name(op) != op_name:
        return None
    data_dq = _resolve_bound_expr(context, context.annotated_expr.get("data_dq", op.args[0]))
    data = _parse_activation_qdq(data_dq, context.matched_bindings)
    if data is None:
        return None
    return data, output, op


def _check_qs8_reshape_like(context: PatternCheckContext, op_name: str) -> bool:
    if not _check_no_leaks(context):
        return False
    parts = _qs8_unary_qdq_parts(context, op_name)
    if parts is None:
        return False
    data, output, op = parts
    if not _qparams_value_equal(data, output):
        return False
    input_elems = _num_elements(data["value"])
    output_elems = _num_elements(context.matched_expr)
    if input_elems is None or output_elems is None or input_elems != output_elems:
        return False
    if _get_static_shape(op) != output["shape"]:
        return False
    return True


def _check_qs8_copy(context: PatternCheckContext) -> bool:
    if not _check_no_leaks(context):
        return False
    if _tensor_dtype(context.annotated_expr.get("root")) != "int8":
        return False
    matched_expr = _resolve_bound_expr(context, context.matched_expr)
    output = _parse_output_quantize(matched_expr)
    if output is None:
        return False
    data_dq = _resolve_bound_expr(context, context.annotated_expr.get("data_dq", output["value"]))
    data = _parse_activation_qdq(data_dq, context.matched_bindings)
    if data is None:
        return False
    return _qparams_value_equal(data, output) and data["shape"] == output["shape"]


def _check_qs8_pool2d(context: PatternCheckContext, op_name: str) -> bool:
    if not _check_no_leaks(context):
        return False
    parts = _qs8_unary_qdq_parts(context, op_name)
    if parts is None:
        return False
    data, output, pool = parts
    if op_name == "relax.nn.max_pool2d" and not _qparams_value_equal(data, output):
        return False
    data_shape = _get_static_shape(data["value"])
    pool_shape = _get_static_shape(pool)
    out_shape = _get_static_shape(context.matched_expr)
    if data_shape is None or pool_shape is None or out_shape is None:
        return False
    if len(data_shape) != 4 or len(pool_shape) != 4 or pool_shape != out_shape:
        return False
    attrs = pool.attrs
    out_layout = attrs.out_layout if attrs.out_layout else attrs.layout
    if attrs.layout != "NHWC" or out_layout != "NHWC":
        return False
    if [int(x) for x in attrs.dilation] != [1, 1]:
        return False
    if bool(attrs.ceil_mode):
        return False
    if _padding_2d(attrs.padding) is None:
        return False
    pool_size = [int(x) for x in attrs.pool_size]
    strides = [int(x) for x in attrs.strides]
    if pool_size == [1, 1] and strides != [1, 1]:
        return False
    if op_name == "relax.nn.avg_pool2d" and bool(attrs.count_include_pad):
        return False
    return True


def _qs8_add_parts(
    context: PatternCheckContext,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], relax.Call] | None:
    if _tensor_dtype(context.annotated_expr.get("root")) != "int8":
        return None
    matched_expr = _resolve_bound_expr(context, context.matched_expr)
    output = _parse_output_quantize(matched_expr)
    if output is None:
        return None
    q_root = _resolve_bound_expr(context, output["value"])
    add = _find_call_in_expr_resolved(q_root, "relax.add", context.matched_bindings)
    if add is None:
        return None
    lhs_dq = _resolve_bound_expr(context, context.annotated_expr.get("lhs_dq", add.args[0]))
    rhs_dq = _resolve_bound_expr(context, context.annotated_expr.get("rhs_dq", add.args[1]))
    lhs = _parse_activation_qdq(lhs_dq, context.matched_bindings)
    rhs = _parse_activation_qdq(rhs_dq, context.matched_bindings)
    if lhs is None or rhs is None:
        return None
    return lhs, rhs, output, add


def _check_qs8_add(context: PatternCheckContext) -> bool:
    if not _check_no_leaks(context):
        return False
    parts = _qs8_add_parts(context)
    if parts is None:
        return False
    lhs, rhs, output, add = parts
    lhs_shape = _get_static_shape(lhs["value"])
    rhs_shape = _get_static_shape(rhs["value"])
    add_shape = _get_static_shape(add)
    out_shape = _get_static_shape(context.matched_expr)
    if lhs_shape is None or rhs_shape is None or add_shape is None or out_shape is None:
        return False
    if lhs_shape != rhs_shape or lhs_shape != add_shape or lhs_shape != out_shape:
        return False
    root = _resolve_bound_expr(context, output["value"])
    if isinstance(root, relax.Call) and _call_op_name(root) == "relax.clip":
        min_value = _as_float_prim_value(root.args[1])
        max_value = _as_float_prim_value(root.args[2])
        return min_value is not None and max_value is not None and min_value <= max_value
    return True


def _unary_pattern(pattern_name: str, op_name: str):
    input_expr = wildcard()
    root = is_op(op_name)(input_expr)
    return (pattern_name, root, {"input": input_expr, "root": root}, _check_unary)


def _clip_pattern(pattern_name: str):
    input_expr = wildcard()
    min_value = wildcard()
    max_value = wildcard()
    root = is_op("relax.clip")(input_expr, min_value, max_value)
    return (pattern_name, root, {"input": input_expr, "root": root}, _check_unary)


def _add_pattern():
    lhs = wildcard()
    rhs = wildcard()
    root = is_op("relax.add")(lhs, rhs)
    return ("xnnpack.add", root, {"lhs": lhs, "rhs": rhs, "root": root}, _check_add)


def _pool2d_pattern(pattern_name: str, op_name: str):
    input_expr = wildcard()
    root = is_op(op_name)(input_expr)
    return (pattern_name, root, {"input": input_expr, "root": root}, _check_pool2d)


def _conv2d_patterns():
    data = wildcard()
    weight = is_const()
    bias = is_const()
    conv = is_op("relax.nn.conv2d")(data, weight)
    bias_add = is_op("relax.add")(conv, bias)
    conv_relu = is_op("relax.nn.relu")(conv)
    bias_relu = is_op("relax.nn.relu")(bias_add)
    min_value = wildcard()
    max_value = wildcard()
    conv_clip = is_op("relax.clip")(conv, min_value, max_value)
    bias_clip = is_op("relax.clip")(bias_add, min_value, max_value)

    return [
        (
            "xnnpack.conv2d_bias_clip",
            bias_clip,
            {"data": data, "weight": weight, "bias": bias, "conv": conv, "root": bias_clip},
            _check_conv2d,
        ),
        (
            "xnnpack.conv2d_bias_relu",
            bias_relu,
            {"data": data, "weight": weight, "bias": bias, "conv": conv, "root": bias_relu},
            _check_conv2d,
        ),
        (
            "xnnpack.conv2d_clip",
            conv_clip,
            {"data": data, "weight": weight, "conv": conv, "root": conv_clip},
            _check_conv2d,
        ),
        (
            "xnnpack.conv2d_relu",
            conv_relu,
            {"data": data, "weight": weight, "conv": conv, "root": conv_relu},
            _check_conv2d,
        ),
        (
            "xnnpack.conv2d_bias",
            bias_add,
            {"data": data, "weight": weight, "bias": bias, "conv": conv, "root": bias_add},
            _check_conv2d,
        ),
        (
            "xnnpack.conv2d",
            conv,
            {"data": data, "weight": weight, "conv": conv, "root": conv},
            _check_conv2d,
        ),
    ]


def _qdq_input_pattern():
    q_data = wildcard()
    data_scale = is_const()
    data_zp = is_const()
    return q_data, is_op("relax.dequantize")(q_data, data_scale, data_zp)


def _qdq_const_pattern():
    q_const = is_const()
    scale = is_const()
    zero_point = is_const()
    return q_const, is_op("relax.dequantize")(q_const, scale, zero_point)


def _qs8_weighted_patterns(prefix: str, weighted, check):
    q_data, data_dq = _qdq_input_pattern()
    q_weight, weight_dq = _qdq_const_pattern()
    base_weighted = weighted(data_dq, weight_dq)
    q_bias, bias_dq = _qdq_const_pattern()
    bias_add = is_op("relax.add")(base_weighted, bias_dq)
    relu = is_op("relax.nn.relu")(base_weighted)
    bias_relu = is_op("relax.nn.relu")(bias_add)
    min_value = wildcard()
    max_value = wildcard()
    clip = is_op("relax.clip")(base_weighted, min_value, max_value)
    bias_clip = is_op("relax.clip")(bias_add, min_value, max_value)
    out_scale = is_const()
    out_zp = is_const()

    def make(name_suffix, expr, has_bias=False):
        root = is_op("relax.quantize")(expr, out_scale, out_zp)
        annotations = {
            "data": q_data,
            "data_dq": data_dq,
            "weighted": base_weighted,
            "root": root,
        }
        return (f"xnnpack.{prefix}{name_suffix}", root, annotations, check)

    return [
        make("_bias_clip", bias_clip, True),
        make("_bias_relu", bias_relu, True),
        make("_clip", clip),
        make("_relu", relu),
        make("_bias", bias_add, True),
        make("", base_weighted),
    ]


def _qs8_fully_connected_patterns():
    return _qs8_weighted_patterns(
        "qs8_fully_connected",
        lambda data, weight: is_op("relax.matmul")(data, weight),
        _check_qs8_fully_connected,
    )


def _qs8_conv2d_patterns():
    return _qs8_weighted_patterns(
        "qs8_conv2d",
        lambda data, weight: is_op("relax.nn.conv2d")(data, weight),
        _check_qs8_conv2d,
    )


def _qs8_depthwise_conv2d_patterns():
    return _qs8_weighted_patterns(
        "qs8_depthwise_conv2d",
        lambda data, weight: is_op("relax.nn.conv2d")(data, weight),
        _check_qs8_depthwise_conv2d,
    )


def _qs8_reshape_pattern(pattern_name: str, op_name: str, check):
    q_data, data_dq = _qdq_input_pattern()
    if op_name == "relax.reshape":
        shape = wildcard()
        op = is_op(op_name)(data_dq, shape)
    else:
        op = is_op(op_name)(data_dq)
    out_scale = is_const()
    out_zp = is_const()
    root = is_op("relax.quantize")(op, out_scale, out_zp)
    return (
        pattern_name,
        root,
        {"q_data": q_data, "data_dq": data_dq, "op": op, "root": root},
        lambda context: check(context, op_name),
    )


def _qs8_copy_pattern():
    q_data, data_dq = _qdq_input_pattern()
    out_scale = is_const()
    out_zp = is_const()
    root = is_op("relax.quantize")(data_dq, out_scale, out_zp)
    return (
        "xnnpack.qs8_copy",
        root,
        {"q_data": q_data, "data_dq": data_dq, "root": root},
        _check_qs8_copy,
    )


def _qs8_pool2d_pattern(pattern_name: str, op_name: str):
    q_data, data_dq = _qdq_input_pattern()
    op = is_op(op_name)(data_dq)
    out_scale = is_const()
    out_zp = is_const()
    root = is_op("relax.quantize")(op, out_scale, out_zp)
    return (
        pattern_name,
        root,
        {"q_data": q_data, "data_dq": data_dq, "op": op, "root": root},
        lambda context: _check_qs8_pool2d(context, op_name),
    )


def _qs8_add_patterns():
    q_lhs, lhs_dq = _qdq_input_pattern()
    q_rhs, rhs_dq = _qdq_input_pattern()
    add = is_op("relax.add")(lhs_dq, rhs_dq)
    relu = is_op("relax.nn.relu")(add)
    min_value = wildcard()
    max_value = wildcard()
    clip = is_op("relax.clip")(add, min_value, max_value)
    out_scale = is_const()
    out_zp = is_const()

    def make(suffix, expr):
        root = is_op("relax.quantize")(expr, out_scale, out_zp)
        return (
            f"xnnpack.qs8_add{suffix}",
            root,
            {"q_lhs": q_lhs, "lhs_dq": lhs_dq, "q_rhs": q_rhs, "rhs_dq": rhs_dq,
             "op": add, "root": root},
            _check_qs8_add,
        )

    return [
        make("_clip", clip),
        make("_relu", relu),
        make("", add),
    ]


def _conv2d_flops(conv: relax.Expr) -> int:
    if not isinstance(conv, relax.Call):
        return 0
    data_shape = _get_static_shape(conv.args[0])
    weight_shape = _get_static_shape(conv.args[1])
    out_shape = _get_static_shape(conv)
    if data_shape is None or weight_shape is None or out_shape is None:
        return 0
    if len(data_shape) != 4 or len(weight_shape) != 4 or len(out_shape) != 4:
        return 0
    out_elems = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    kernel_h, kernel_w, in_channels = weight_shape[1], weight_shape[2], weight_shape[3]
    return int(out_elems * kernel_h * kernel_w * in_channels * 2)


def _depthwise_conv2d_flops(conv: relax.Expr) -> int:
    if not isinstance(conv, relax.Call):
        return 0
    weight_shape = _get_static_shape(conv.args[1])
    out_shape = _get_static_shape(conv)
    if weight_shape is None or out_shape is None or len(weight_shape) != 4 or len(out_shape) != 4:
        return 0
    out_elems = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3]
    return int(out_elems * weight_shape[0] * weight_shape[1] * 2)


def _matmul_flops(matmul: relax.Expr) -> int:
    if not isinstance(matmul, relax.Call):
        return 0
    lhs_shape = _get_static_shape(matmul.args[0])
    rhs_shape = _get_static_shape(matmul.args[1])
    out_shape = _get_static_shape(matmul)
    if lhs_shape is None or rhs_shape is None or out_shape is None:
        return 0
    if len(lhs_shape) != 2 or len(rhs_shape) != 2 or len(out_shape) != 2:
        return 0
    return int(out_shape[0] * out_shape[1] * lhs_shape[1] * 2)


def _pool2d_flops(pool: relax.Expr) -> int:
    if not isinstance(pool, relax.Call):
        return 0
    out_elems = _num_elements(pool)
    if out_elems is None:
        return 0
    attrs = pool.attrs
    kernel = [int(x) for x in attrs.pool_size]
    return int(out_elems * kernel[0] * kernel[1])


def _quantized_op_type(pattern_name: str) -> str:
    name = pattern_name.removeprefix("xnnpack.")
    if not name.startswith("qs8_"):
        return "none"
    for suffix in ("_bias_clip", "_bias_relu", "_clip", "_relu", "_bias"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _estimate_flops(context: PatternCheckContext, pattern_name: str) -> int:
    root = context.annotated_expr.get("root", context.matched_expr)
    op_names = _collect_op_names(root)
    if "qs8_fully_connected" in pattern_name:
        return _matmul_flops(context.annotated_expr.get("weighted", root))
    if "qs8_depthwise_conv2d" in pattern_name:
        return _depthwise_conv2d_flops(context.annotated_expr.get("weighted", root))
    if "qs8_conv2d" in pattern_name:
        return _conv2d_flops(context.annotated_expr.get("weighted", root))
    if "qs8_max_pool2d" in pattern_name or "qs8_avg_pool2d" in pattern_name:
        return _pool2d_flops(context.annotated_expr.get("op", root))
    if "qs8_reshape" in pattern_name or "qs8_flatten" in pattern_name or "qs8_copy" in pattern_name:
        return 0
    if "relax.nn.conv2d" in op_names or "conv2d" in pattern_name:
        return _conv2d_flops(context.annotated_expr.get("conv", root))
    if _call_op_name(root) in ("relax.nn.max_pool2d", "relax.nn.avg_pool2d"):
        return _pool2d_flops(root)
    out_elems = _num_elements(root)
    if out_elems is None:
        return 0
    return int(out_elems * max(1, len(op_names)))


def _is_compute_heavy(pattern_name: str, context: PatternCheckContext, flops: int) -> bool:
    if "conv2d" in pattern_name or "fully_connected" in pattern_name:
        return True
    if "qs8_max_pool2d" in pattern_name or "qs8_avg_pool2d" in pattern_name:
        return flops >= 4096
    root = context.annotated_expr.get("root", context.matched_expr)
    if _call_op_name(root) in ("relax.nn.max_pool2d", "relax.nn.avg_pool2d"):
        return flops >= 4096
    return False


def _external_input_exprs(context: PatternCheckContext) -> list[relax.Expr]:
    exprs = []
    for key, expr in context.annotated_expr.items():
        if key in ("root", "conv"):
            continue
        if isinstance(expr, relax.Constant):
            continue
        if isinstance(expr, relax.Expr) and _tensor_dtype(expr) is not None:
            if all(not expr.same_as(existing) for existing in exprs):
                exprs.append(expr)
    return exprs


def _constant_exprs(context: PatternCheckContext) -> list[relax.Expr]:
    exprs = []
    for expr in context.annotated_expr.values():
        if isinstance(expr, relax.Constant) and all(
            not expr.same_as(existing) for existing in exprs
        ):
            exprs.append(expr)
    return exprs


def _make_report_entry(
    context: PatternCheckContext,
    pattern_name: str,
    policy: str,
    accepted: bool,
    reason: str,
) -> dict[str, object]:
    root = context.annotated_expr.get("root", context.matched_expr)
    op_list = _op_list_from_pattern(pattern_name, root)
    external_inputs = _external_input_exprs(context)
    constants = _constant_exprs(context)
    output_bytes = _tensor_nbytes(root)
    input_bytes = sum(_tensor_nbytes(expr) for expr in external_inputs)
    constant_bytes = sum(_tensor_nbytes(expr) for expr in constants)
    copy_bytes = input_bytes + output_bytes + constant_bytes
    padded_copy_bytes = copy_bytes + (len(external_inputs) + len(constants) + 1) * _XNN_EXTRA_BYTES
    flops = _estimate_flops(context, pattern_name)
    ratio = float("inf") if padded_copy_bytes == 0 and flops > 0 else 0.0
    if padded_copy_bytes > 0:
        ratio = float(flops) / float(padded_copy_bytes)
    quantized = "qs8_" in pattern_name
    qscheme = "none"
    if quantized:
        weighted = _find_call_in_expr(context.matched_expr, "relax.matmul") or _find_call_in_expr(
            context.matched_expr, "relax.nn.conv2d"
        )
        qscheme = _qscheme_from_scale(weighted.args[1].args[1]) if weighted is not None else None
        if qscheme is None:
            root_q = _parse_output_quantize(context.matched_expr)
            qscheme = root_q["qscheme"] if root_q is not None else None
        qscheme = qscheme or "unknown"
    qdq_count = sum(1 for op in op_list if op in ("relax.quantize", "relax.dequantize"))
    quantized_op_type = _quantized_op_type(pattern_name)
    qparam_equality_required = quantized_op_type in (
        "qs8_reshape",
        "qs8_flatten",
        "qs8_copy",
        "qs8_max_pool2d",
    )
    return {
        "candidate_id": -1,
        "accepted": accepted,
        "reason": reason,
        "op_list": op_list,
        "dtype": _candidate_dtype(context),
        "layout": _candidate_layout(context),
        "estimated_flops": flops,
        "copy_bytes": copy_bytes,
        "padded_copy_bytes": padded_copy_bytes,
        "layout_transform_bytes": 0,
        "cast_bytes": 0,
        "external_input_count": len(external_inputs),
        "external_output_count": 1,
        "boundary_count": len(external_inputs) + 1,
        "compute_to_copy_ratio": ratio,
        "policy": policy,
        "quantized": quantized,
        "qscheme": qscheme,
        "qdq_boundary_count": qdq_count,
        "qparam_source": "constant" if quantized else "none",
        "qparam_validation_result": "ok" if quantized and accepted else reason,
        "quantized_op_type": quantized_op_type,
        "qparams_summary": qscheme if quantized else "none",
        "qparam_equality_required": qparam_equality_required,
        "qparam_rejection_reason": reason if quantized and not accepted else "none",
    }


def _validate_partition_options(
    precision: str,
    partition_policy: str,
    layout: str,
    min_subgraph_size: int,
    min_compute_to_copy_ratio: float,
):
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            "Unsupported XNNPACK precision. Expected one of "
            f"{_SUPPORTED_PRECISIONS}, but got {precision!r}."
        )
    if partition_policy not in _SUPPORTED_PARTITION_POLICIES:
        raise ValueError(
            "Unsupported XNNPACK partition_policy. Expected one of "
            f"{_SUPPORTED_PARTITION_POLICIES}, but got {partition_policy!r}."
        )
    if layout not in _SUPPORTED_LAYOUT_POLICIES:
        raise ValueError(
            "Unsupported XNNPACK layout policy. Expected one of "
            f"{_SUPPORTED_LAYOUT_POLICIES}, but got {layout!r}."
        )
    if min_subgraph_size < 1:
        raise ValueError("min_subgraph_size must be at least 1.")
    if min_compute_to_copy_ratio < 0:
        raise ValueError("min_compute_to_copy_ratio must be non-negative.")


def _cost_accepts(
    context: PatternCheckContext,
    pattern_name: str,
    layout_policy: str,
    min_subgraph_size: int,
    min_compute_to_copy_ratio: float,
    allow_isolated_elementwise: bool,
    allow_layout_rewrite: bool,
    allow_cast_boundary: bool,
) -> tuple[bool, str]:
    del allow_cast_boundary  # Explicit fp16 and cast-boundary lowering are not implemented yet.
    entry = _make_report_entry(context, pattern_name, "cost", True, "")
    op_count = len(entry["op_list"])
    dtype = entry["dtype"]
    layout = entry["layout"]
    ratio = float(entry["compute_to_copy_ratio"])
    flops = int(entry["estimated_flops"])

    if dtype != "float32" and not ("qs8_" in pattern_name and dtype == "int8"):
        return False, "rejected_unsupported_dtype"
    if layout_policy == "NHWC" and layout not in ("NHWC", "none") and not allow_layout_rewrite:
        return False, "rejected_layout_rewrite_overhead"
    if layout_policy == "NHWC" and layout not in ("NHWC", "none") and op_count <= 1:
        return False, "rejected_layout_rewrite_overhead"
    if not allow_isolated_elementwise and (
        ("qs8_add" in pattern_name)
        or ("qs8_reshape" in pattern_name)
        or ("qs8_flatten" in pattern_name)
        or ("qs8_copy" in pattern_name)
    ):
        if "qs8_add" in pattern_name:
            return False, "rejected_isolated_elementwise"
        return False, "rejected_low_compute_to_copy_ratio"
    if not allow_isolated_elementwise and op_count <= 1 and "conv2d" not in pattern_name:
        root_name = _call_op_name(context.annotated_expr.get("root", context.matched_expr))
        if root_name not in ("relax.nn.max_pool2d", "relax.nn.avg_pool2d"):
            return False, "rejected_isolated_elementwise"
    if _is_compute_heavy(pattern_name, context, flops):
        return True, "accepted_compute_heavy"
    if op_count >= min_subgraph_size and ratio >= min_compute_to_copy_ratio:
        return True, "accepted_ratio"
    return False, "rejected_low_compute_to_copy_ratio"


def _wrap_patterns_for_policy(
    patterns: list[FusionPattern],
    partition_policy: str,
    layout_policy: str,
    min_subgraph_size: int,
    min_compute_to_copy_ratio: float,
    allow_isolated_elementwise: bool,
    allow_layout_rewrite: bool,
    allow_cast_boundary: bool,
    report: list[dict[str, object]] | None,
) -> list[FusionPattern]:
    if partition_policy == "greedy" and report is None:
        return patterns

    wrapped = []

    for pattern in patterns:
        original_check: Callable[[PatternCheckContext], bool] | None = pattern.check

        def make_check(pattern_name, check):
            def check_with_policy(context: PatternCheckContext) -> bool:
                supported = True if check is None else bool(check(context))
                if not supported:
                    candidate_dtype = _candidate_dtype(context)
                    if candidate_dtype not in ("float32", "int8"):
                        reason = "rejected_unsupported_dtype"
                    elif layout_policy == "NHWC" and _candidate_layout(context) not in (
                        "NHWC",
                        "none",
                    ):
                        reason = "rejected_layout_rewrite_overhead"
                    else:
                        reason = "rejected_existing_support_check"
                    accepted = False
                elif partition_policy in ("greedy", "debug_all_supported"):
                    reason = (
                        "accepted_debug_all_supported"
                        if partition_policy == "debug_all_supported"
                        else "accepted_supported"
                    )
                    accepted = True
                else:
                    accepted, reason = _cost_accepts(
                        context,
                        pattern_name,
                        layout_policy,
                        min_subgraph_size,
                        min_compute_to_copy_ratio,
                        allow_isolated_elementwise,
                        allow_layout_rewrite,
                        allow_cast_boundary,
                    )
                if report is not None:
                    entry = _make_report_entry(
                        context, pattern_name, partition_policy, accepted, reason
                    )
                    entry["candidate_id"] = len(report)
                    report.append(entry)
                return accepted

            return check_with_policy

        wrapped.append(
            FusionPattern(
                pattern.name,
                pattern.pattern,
                pattern.annotation_patterns,
                make_check(pattern.name, original_check),
                pattern.attrs_getter,
            )
        )
    return wrapped


register_patterns(
    [
        *_qs8_fully_connected_patterns(),
        *_qs8_conv2d_patterns(),
        *_qs8_depthwise_conv2d_patterns(),
        _qs8_reshape_pattern("xnnpack.qs8_reshape", "relax.reshape", _check_qs8_reshape_like),
        _qs8_reshape_pattern("xnnpack.qs8_flatten", "relax.flatten", _check_qs8_reshape_like),
        _qs8_copy_pattern(),
        _qs8_pool2d_pattern("xnnpack.qs8_max_pool2d", "relax.nn.max_pool2d"),
        _qs8_pool2d_pattern("xnnpack.qs8_avg_pool2d", "relax.nn.avg_pool2d"),
        *_qs8_add_patterns(),
        *_conv2d_patterns(),
        _pool2d_pattern("xnnpack.max_pool2d", "relax.nn.max_pool2d"),
        _pool2d_pattern("xnnpack.avg_pool2d", "relax.nn.avg_pool2d"),
        _add_pattern(),
        _clip_pattern("xnnpack.clip"),
        _unary_pattern("xnnpack.relu", "relax.nn.relu"),
        _unary_pattern("xnnpack.sigmoid", "relax.sigmoid"),
        _unary_pattern("xnnpack.tanh", "relax.tanh"),
    ]
)


def partition_for_xnnpack(
    mod: IRModule,
    precision: str = "fp32",
    partition_policy: str = "greedy",
    layout: str = "auto",
    min_subgraph_size: int = 2,
    min_compute_to_copy_ratio: float = 8.0,
    allow_isolated_elementwise: bool = False,
    allow_layout_rewrite: bool = True,
    allow_cast_boundary: bool = False,
    report_partition_decisions: bool = False,
) -> IRModule | tuple[IRModule, list[dict[str, object]]]:
    """Partition the input module into XNNPACK-supported subgraphs.

    Phase 3 supports a small static-shape float32 NHWC CNN subset.
    """

    _validate_partition_options(
        precision,
        partition_policy,
        layout,
        min_subgraph_size,
        min_compute_to_copy_ratio,
    )

    patterns = list(reversed(get_patterns_with_prefix("xnnpack")))
    report = [] if report_partition_decisions else None
    patterns = _wrap_patterns_for_policy(
        patterns,
        partition_policy,
        layout,
        min_subgraph_size,
        min_compute_to_copy_ratio,
        allow_isolated_elementwise,
        allow_layout_rewrite,
        allow_cast_boundary,
        report,
    )
    mod = FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=True)(mod)

    for gv, func in list(mod.functions.items()):
        if (
            isinstance(func, relax.Function)
            and func.attrs
            and func.attrs.get("Codegen") == "xnnpack"
        ):
            mod[gv] = func.with_attr("xnnpack_precision", precision)
    if report is not None:
        return mod, report
    return mod
