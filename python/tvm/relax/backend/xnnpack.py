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

import tvm
from tvm.ir import IRModule
from tvm import relax
from tvm.relax.dpl.pattern import is_const, is_op, wildcard
from tvm.relax.transform import FuseOpsByPattern, PatternCheckContext

from .pattern_registry import get_patterns_with_prefix, register_patterns
from .utils import has_leaking_intermediate_variables

_SUPPORTED_PRECISIONS = ("fp32", "fp16_hint", "fp16_force")


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


register_patterns(
    [
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


def partition_for_xnnpack(mod: IRModule, precision: str = "fp32") -> IRModule:
    """Partition the input module into XNNPACK-supported subgraphs.

    Phase 3 supports a small static-shape float32 NHWC CNN subset.
    """

    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            "Unsupported XNNPACK precision. Expected one of "
            f"{_SUPPORTED_PRECISIONS}, but got {precision!r}."
        )

    patterns = list(reversed(get_patterns_with_prefix("xnnpack")))
    mod = FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=True)(mod)

    for gv, func in list(mod.functions.items()):
        if (
            isinstance(func, relax.Function)
            and func.attrs
            and func.attrs.get("Codegen") == "xnnpack"
        ):
            mod[gv] = func.with_attr("xnnpack_precision", precision)
    return mod
