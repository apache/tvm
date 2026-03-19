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

"""Pattern table for the RK3588 NPU (RKNPU) backend.

Patterns are registered in priority order: most-fused first.
FuseOpsByPattern tries patterns top-to-bottom and takes the first match,
so matmul_bias_relu must come before matmul_bias or matmul_relu.
"""

import tvm

from tvm.relax.backend.patterns import (
    make_matmul_pattern,
    make_conv2d_pattern,
)
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext

from ...pattern_registry import register_patterns


# ---------------------------------------------------------------------------
# Custom pattern builders for ops without built-in helpers
# ---------------------------------------------------------------------------


def _make_add_pattern():
    """Create a pattern matching ``relax.add(lhs, rhs)``."""
    lhs = wildcard()
    rhs = wildcard()
    out = is_op("relax.add")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": out}
    return out, annotations


def _make_multiply_pattern():
    """Create a pattern matching ``relax.multiply(lhs, rhs)``."""
    lhs = wildcard()
    rhs = wildcard()
    out = is_op("relax.multiply")(lhs, rhs)
    annotations = {"lhs": lhs, "rhs": rhs, "root": out}
    return out, annotations


def _make_max_pool2d_pattern():
    """Create a pattern matching ``relax.nn.max_pool2d(input)``."""
    inp = wildcard()
    out = is_op("relax.nn.max_pool2d")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_avg_pool2d_pattern():
    """Create a pattern matching ``relax.nn.avg_pool2d(input)``."""
    inp = wildcard()
    out = is_op("relax.nn.avg_pool2d")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_exp_pattern():
    """Create a pattern matching ``relax.exp(input)``."""
    inp = wildcard()
    out = is_op("relax.exp")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_sigmoid_pattern():
    """Create a pattern matching ``relax.sigmoid(input)``."""
    inp = wildcard()
    out = is_op("relax.sigmoid")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_gelu_pattern():
    """Create a pattern matching ``relax.nn.gelu(input)``."""
    inp = wildcard()
    out = is_op("relax.nn.gelu")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_rsqrt_pattern():
    """Create a pattern matching ``relax.rsqrt(input)``."""
    inp = wildcard()
    out = is_op("relax.rsqrt")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_softmax_pattern():
    """Create a pattern matching ``relax.nn.softmax(input)``."""
    inp = wildcard()
    out = is_op("relax.nn.softmax")(inp)
    annotations = {"input": inp, "root": out}
    return out, annotations


def _make_layer_norm_pattern():
    """Create a pattern matching ``relax.nn.layer_norm(input, gamma, beta)``."""
    inp = wildcard()
    gamma = wildcard()
    beta = wildcard()
    out = is_op("relax.nn.layer_norm")(inp, gamma, beta)
    annotations = {"input": inp, "gamma": gamma, "beta": beta, "root": out}
    return out, annotations


# ---------------------------------------------------------------------------
# Pattern check callbacks
# ---------------------------------------------------------------------------


def _check_matmul(context: PatternCheckContext) -> bool:
    """Validate that a matched matmul pattern is supported by the RKNPU backend."""
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]

    # Only FP16 is supported.
    if lhs.struct_info.dtype != "float16" or rhs.struct_info.dtype != "float16":
        return False

    # Shapes must be static.
    lhs_shape = lhs.struct_info.shape
    rhs_shape = rhs.struct_info.shape
    if lhs_shape is None or rhs_shape is None:
        return False

    # Must be 2D matmul (no batch dimension support yet).
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        return False

    # Validate bias if present.
    if "bias" in context.annotated_expr:
        bias = context.annotated_expr["bias"]
        if bias.struct_info.dtype != "float16":
            return False
        bias_shape = bias.struct_info.shape
        if bias_shape is None:
            return False
        # Bias must be 1D (N,) or broadcastable (1, N).
        if len(bias_shape) == 1:
            pass  # OK: (N,)
        elif len(bias_shape) == 2 and int(bias_shape[0]) == 1:
            pass  # OK: (1, N)
        else:
            return False

    return True


def _check_conv2d_common(context):
    """Shared validation for conv2d and depthwise conv2d patterns.

    Returns (inp_shape, wt_shape) on success, or None on failure.
    """
    inp = context.annotated_expr["input"]
    weight = context.annotated_expr["weight"]

    if inp.struct_info.dtype != "float16" or weight.struct_info.dtype != "float16":
        return None

    inp_shape = inp.struct_info.shape
    wt_shape = weight.struct_info.shape
    if inp_shape is None or wt_shape is None:
        return None

    if len(inp_shape) != 4 or len(wt_shape) != 4:
        return None

    if "bias" in context.annotated_expr:
        bias = context.annotated_expr["bias"]
        if bias.struct_info.dtype != "float16":
            return None
        if bias.struct_info.shape is None:
            return None

    return inp_shape, wt_shape


def _check_conv2d(context: PatternCheckContext) -> bool:
    """Validate that a matched conv2d pattern is supported by the RKNPU backend."""
    result = _check_conv2d_common(context)
    if result is None:
        return False
    inp_shape, wt_shape = result
    # Reject depthwise (groups=C) — handled by separate depthwise patterns.
    if int(wt_shape[1]) == 1 and int(wt_shape[0]) == int(inp_shape[1]):
        return False
    # Reject grouped conv2d (1 < groups < C): weight C_in must equal input C_in.
    # Grouped conv2d has weight shape [N, C/G, kH, kW] where C/G < C.
    if int(wt_shape[1]) != int(inp_shape[1]):
        return False
    return True


def _check_depthwise_conv2d(context: PatternCheckContext) -> bool:
    """Validate that a matched conv2d is depthwise (groups=C) for RKNPU."""
    result = _check_conv2d_common(context)
    if result is None:
        return False
    inp_shape, wt_shape = result
    # Must be depthwise: weight [C, 1, kH, kW] with C == input channels
    return int(wt_shape[1]) == 1 and int(wt_shape[0]) == int(inp_shape[1])


def _check_elementwise(context: PatternCheckContext) -> bool:
    """Validate elementwise add/multiply for RKNPU."""
    lhs = context.annotated_expr["lhs"]
    rhs = context.annotated_expr["rhs"]

    if lhs.struct_info.dtype != "float16" or rhs.struct_info.dtype != "float16":
        return False

    lhs_shape = lhs.struct_info.shape
    rhs_shape = rhs.struct_info.shape
    if lhs_shape is None or rhs_shape is None:
        return False

    return True


def _check_pool2d(context: PatternCheckContext) -> bool:
    """Validate pool2d (max or avg) for RKNPU."""
    inp = context.annotated_expr["input"]

    if inp.struct_info.dtype != "float16":
        return False

    inp_shape = inp.struct_info.shape
    if inp_shape is None:
        return False

    # Must be 4D: [N, C, H, W]
    if len(inp_shape) != 4:
        return False

    return True


def _check_unary(context: PatternCheckContext) -> bool:
    """Validate unary op (exp, sigmoid, etc.) for RKNPU LUT."""
    inp = context.annotated_expr["input"]

    if inp.struct_info.dtype != "float16":
        return False

    inp_shape = inp.struct_info.shape
    if inp_shape is None:
        return False

    # Must be 1D or 2D
    if len(inp_shape) not in (1, 2):
        return False

    return True


def _check_softmax(context: PatternCheckContext) -> bool:
    """Validate softmax for RKNPU graph-level compilation.

    Requires 2D FP16 input with softmax over the last axis.
    """
    inp = context.annotated_expr["input"]

    if inp.struct_info.dtype != "float16":
        return False

    inp_shape = inp.struct_info.shape
    if inp_shape is None:
        return False

    if len(inp_shape) != 2:
        return False

    # Check that softmax is over the last axis
    root = context.annotated_expr["root"]
    if isinstance(root, tvm.relax.Call) and root.attrs is not None:
        axis = int(root.attrs.axis)
        if axis not in (-1, 1):
            return False

    return True


def _check_layer_norm(context: PatternCheckContext) -> bool:
    """Validate layer_norm for RKNPU graph-level compilation.

    Requires 2D FP16 input with normalization over the last axis.
    K must be divisible by 32 (NPU alignment).
    """
    inp = context.annotated_expr["input"]
    gamma = context.annotated_expr["gamma"]
    beta = context.annotated_expr["beta"]

    if inp.struct_info.dtype != "float16":
        return False
    if gamma.struct_info.dtype != "float16":
        return False
    if beta.struct_info.dtype != "float16":
        return False

    inp_shape = inp.struct_info.shape
    if inp_shape is None:
        return False

    if len(inp_shape) != 2:
        return False

    # K must be 32-aligned for NPU matmul
    try:
        K = int(inp_shape[1])
    except (TypeError, ValueError):
        return False
    if K % 32 != 0:
        return False

    # Check that normalization is over the last axis
    root = context.annotated_expr["root"]
    if isinstance(root, tvm.relax.Call) and root.attrs is not None:
        axes = [int(a) for a in root.attrs.axes]
        ndim = len(inp_shape)
        if axes != [-1] and axes != [ndim - 1]:
            return False

    return True


# ---------------------------------------------------------------------------
# Register all patterns
# ---------------------------------------------------------------------------
# Patterns later in the list have higher priority, so most-fused goes last.
register_patterns(
    [
        # Patterns later in the list have higher priority.
        # Fused patterns (matmul_bias_relu, etc.) must have higher priority
        # than standalone elementwise patterns (add, multiply) so that the
        # fused pattern is attempted first and the add is not consumed separately.
        #
        # ---- Unary activation patterns (lowest priority) ----
        (
            "rknpu.exp",
            *_make_exp_pattern(),
            _check_unary,
        ),
        (
            "rknpu.sigmoid",
            *_make_sigmoid_pattern(),
            _check_unary,
        ),
        (
            "rknpu.gelu",
            *_make_gelu_pattern(),
            _check_unary,
        ),
        (
            "rknpu.rsqrt",
            *_make_rsqrt_pattern(),
            _check_unary,
        ),
        # ---- Elementwise patterns (low priority -- fallback) ----
        (
            "rknpu.add",
            *_make_add_pattern(),
            _check_elementwise,
        ),
        (
            "rknpu.multiply",
            *_make_multiply_pattern(),
            _check_elementwise,
        ),
        # ---- Pooling patterns ----
        (
            "rknpu.max_pool2d",
            *_make_max_pool2d_pattern(),
            _check_pool2d,
        ),
        (
            "rknpu.avg_pool2d",
            *_make_avg_pool2d_pattern(),
            _check_pool2d,
        ),
        # ---- Matmul patterns (unfused → most-fused) ----
        (
            "rknpu.matmul",
            *make_matmul_pattern(),
            _check_matmul,
        ),
        (
            "rknpu.matmul_relu",
            *make_matmul_pattern(activation="relax.nn.relu"),
            _check_matmul,
        ),
        (
            "rknpu.matmul_bias",
            *make_matmul_pattern(with_bias=True),
            _check_matmul,
        ),
        (
            "rknpu.matmul_bias_relu",
            *make_matmul_pattern(with_bias=True, activation="relax.nn.relu"),
            _check_matmul,
        ),
        # ---- Conv2D patterns (unfused → most-fused, highest priority) ----
        (
            "rknpu.conv2d",
            *make_conv2d_pattern(),
            _check_conv2d,
        ),
        (
            "rknpu.conv2d_relu",
            *make_conv2d_pattern(activation="relax.nn.relu"),
            _check_conv2d,
        ),
        (
            "rknpu.conv2d_bias",
            *make_conv2d_pattern(with_bias=True),
            _check_conv2d,
        ),
        (
            "rknpu.conv2d_bias_relu",
            *make_conv2d_pattern(with_bias=True, activation="relax.nn.relu"),
            _check_conv2d,
        ),
        # ---- Depthwise Conv2D patterns (highest priority) ----
        (
            "rknpu.depthwise_conv2d",
            *make_conv2d_pattern(),
            _check_depthwise_conv2d,
        ),
        (
            "rknpu.depthwise_conv2d_relu",
            *make_conv2d_pattern(activation="relax.nn.relu"),
            _check_depthwise_conv2d,
        ),
        (
            "rknpu.depthwise_conv2d_bias",
            *make_conv2d_pattern(with_bias=True),
            _check_depthwise_conv2d,
        ),
        (
            "rknpu.depthwise_conv2d_bias_relu",
            *make_conv2d_pattern(with_bias=True, activation="relax.nn.relu"),
            _check_depthwise_conv2d,
        ),
        # ---- High-level patterns for graph-level compilation (highest priority) ----
        (
            "rknpu.softmax",
            *_make_softmax_pattern(),
            _check_softmax,
        ),
        (
            "rknpu.layer_norm",
            *_make_layer_norm_pattern(),
            _check_layer_norm,
        ),
    ]
)
