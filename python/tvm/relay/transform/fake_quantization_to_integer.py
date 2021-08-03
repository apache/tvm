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
"""Relay functions for rewriting fake quantized ops."""
import tvm
from tvm import relay
from tvm.ir import TensorAffineType, TupleAffineType
from ..op import register_fake_quantization_to_integer


def fold_constant(expr):
    return relay.transform.FoldConstantExpr(expr, tvm.IRModule())


@register_fake_quantization_to_integer("qnn.dequantize")
def dequantize(expr, type_map):
    """Remove dequantize op"""
    out = expr.args[0]
    t = type_map[expr]
    return [out, t]


@register_fake_quantization_to_integer("qnn.quantize")
def quantize(expr, type_map):
    """Turn a quantize op into requantize or remove it"""
    out = expr.args[0]
    t = type_map[out]
    in_scale = fold_constant(t.scale)
    in_zero_point = fold_constant(t.zero_point)
    if not (
        tvm.ir.structural_equal(in_scale, expr.args[1])
        and tvm.ir.structural_equal(in_zero_point, expr.args[2])
        and tvm.ir.structural_equal(t.dtype, expr.attrs.out_dtype)
    ):
        out = relay.qnn.op.requantize(
            out,
            in_scale,
            in_zero_point,
            expr.args[1],
            expr.args[2],
            out_dtype=expr.attrs.out_dtype,
        )
    return [out, TensorAffineType(expr.args[1], expr.args[2], expr.attrs.out_dtype)]


def register_unary_identity(op_name):
    def identity(expr, type_map):
        assert len(expr.args) == 1
        arg = expr.args[0]
        t = type_map[arg]
        return [expr, t]

    return register_fake_quantization_to_integer(op_name, identity)


register_unary_identity("reshape")
register_unary_identity("squeeze")
register_unary_identity("strided_slice")
register_unary_identity("transpose")
register_unary_identity("expand_dims")
register_unary_identity("nn.max_pool2d")
register_unary_identity("nn.batch_flatten")


@register_fake_quantization_to_integer("nn.avg_pool2d")
def avgpool2d(expr, type_map):
    """Rewrite a avgpool op"""
    arg = expr.args[0]
    t = type_map[arg]
    arg = relay.op.cast(arg, "int32")
    out = relay.op.nn.avg_pool2d(arg, **expr.attrs)
    out = relay.op.cast(out, t.dtype)
    return [out, t]


@register_fake_quantization_to_integer("nn.bias_add")
def bias_add(expr, type_map):
    """Rewrite a bias_add op"""
    x, b = expr.args
    x_t = type_map[x]
    b_t = type_map[b]
    in_scale = fold_constant(x_t.scale)
    in_zero_point = fold_constant(x_t.zero_point)
    if not tvm.ir.structural_equal(x_t, b_t):
        b = relay.qnn.op.requantize(
            b,
            b_t.scale,
            b_t.zero_point,
            in_scale,
            in_zero_point,
            out_dtype=x_t.dtype,
        )
    out = relay.op.nn.bias_add(x, b, **expr.attrs)
    return [out, x_t]


@register_fake_quantization_to_integer("nn.conv2d")
def conv2d(expr, type_map):
    """Rewrite a conv2d op"""
    attrs = {**expr.attrs}
    attrs.pop("out_dtype")
    x, weight = expr.args
    x_t = type_map[x]
    w_t = type_map[weight]
    conv_scale = fold_constant(x_t.scale * w_t.scale)
    conv_zp = relay.const(0)
    out = relay.qnn.op.conv2d(
        x, weight, x_t.zero_point, w_t.zero_point, x_t.scale, w_t.scale, **attrs
    )
    return [out, TensorAffineType(conv_scale, conv_zp, out.attrs.out_dtype)]


@register_fake_quantization_to_integer("nn.dense")
def dense(expr, type_map):
    """Rewrite a dense op"""
    attrs = {**expr.attrs}
    attrs.pop("out_dtype")
    x, weight = expr.args
    x_t = type_map[x]
    w_t = type_map[weight]
    dense_scale = fold_constant(x_t.scale * w_t.scale)
    dense_zp = relay.const(0)
    out = relay.qnn.op.dense(
        x, weight, x_t.zero_point, w_t.zero_point, x_t.scale, w_t.scale, **attrs
    )
    return [out, TensorAffineType(dense_scale, dense_zp, out.attrs.out_dtype)]


@register_fake_quantization_to_integer("concatenate")
def concat(expr, type_map):
    """Rewrite a concat op"""
    scales = []
    zps = []

    tuple_type = type_map[expr.args[0]]
    for t in tuple_type.types:
        scales.append(t.scale)
        zps.append(t.zero_point)

    out_type = type_map[expr]

    out = relay.qnn.op.concatenate(
        expr.args[0],
        relay.Tuple(scales),
        relay.Tuple(zps),
        out_type.scale,
        out_type.zero_point,
        **expr.attrs,
    )
    return [out, out_type]


@register_fake_quantization_to_integer("split")
def split(expr, type_map):
    """Rewrite a split op"""
    arg = expr.args[0]
    t = type_map[arg]
    attrs = {**expr.attrs}
    if isinstance(attrs["indices_or_sections"], tvm.tir.IntImm):
        num_split = attrs["indices_or_sections"].value
        attrs["indices_or_sections"] = num_split
    else:
        num_split = len(attrs["indices_or_sections"]) + 1
    return [expr, TupleAffineType([t] * num_split)]


@register_fake_quantization_to_integer("clip")
def clip(expr, type_map):
    """Rewrite a clip op"""
    arg = expr.args[0]
    t = type_map[arg]
    amin = expr.attrs.a_min
    amax = expr.attrs.a_max
    scale = fold_constant(t.scale)
    z_p = fold_constant(t.zero_point)
    if isinstance(scale, relay.expr.Constant) and isinstance(z_p, relay.expr.Constant):
        scale = scale.data.numpy().item()
        z_p = z_p.data.numpy().item()
        new_min = int(amin / scale + z_p)
        new_max = int(amax / scale + z_p)
        out = relay.op.clip(arg, new_min, new_max)
    else:
        amin = relay.op.round(relay.op.const(amin) / scale + z_p)
        amax = relay.op.round(relay.op.const(amax) / scale + z_p)
        out = relay.op.minimum(relay.op.maximum(arg, amin), amax)
    return [out, t]


@register_fake_quantization_to_integer("nn.pad")
def pad(expr, type_map):
    """Rewite an nn.pad op"""
    arg = expr.args[0]
    t = type_map[arg]
    pad_value = expr.args[1]
    ## TF2ONNX will sometimes implement the pad_value as a constant without a quantize
    ## To support that, the pass lets branches that terminate in a constant through
    if pad_value in type_map:
        ## if the pad value is calcuated from a dequantize op, it should be in the type map
        ## and we need to make sure it's affine type matches the arg
        pad_t = type_map[pad_value]
        if not tvm.ir.structural_equal(t, pad_t):
            pad_value = relay.qnn.op.requantize(
                pad_value,
                pad_t.scale,
                pad_t.zero_point,
                t.scale,
                t.zero_point,
                out_dtype=t.dtype,
            )
    else:
        ## If the pad-value is a constant, we need to quantize it
        assert isinstance(pad_value, relay.expr.Constant)
        pad_value = relay.qnn.op.quantize(pad_value, t.scale, t.zero_point)

    out = relay.op.nn.pad(arg, pad_value=pad_value, **expr.attrs)
    return [out, t]


def get_binary_types(expr, type_map):
    """Get Affine types of a binary op's inputs and unify them"""
    ##Support the case where one input is quantized and the other is a constant float
    left = expr.args[0]
    right = expr.args[1]
    left_t = None
    right_t = None

    if left in type_map:
        left_t = type_map[left]
    if right in type_map:
        right_t = type_map[right]

    out_t = type_map[expr]
    if left_t is None and right_t is None:
        raise TypeError("neither input is quantized!")
    if left_t is None:
        assert isinstance(left, relay.expr.Constant)
        left = relay.qnn.op.quantize(
            left, right_t.scale, right_t.zero_point, out_dtype=right_t.dtype
        )
        left_t = right_t
        out_t = right_t
    if right_t is None:
        assert isinstance(right, relay.expr.Constant)
        right = relay.qnn.op.quantize(
            right, left_t.scale, left_t.zero_point, out_dtype=left_t.dtype
        )
        right_t = left_t
        out_t = left_t

    # Handle the case of mismatched inputs
    if not left_t.dtype == out_t.dtype:
        out_t = left_t

    return left, right, left_t, right_t, out_t


def register_binary_qnn(op_name, op):
    """Register a Binary Op that converts to QNN"""

    def binary(expr, type_map):
        left, right, left_t, right_t, out_t = get_binary_types(expr, type_map)
        out = op(
            left,
            right,
            left_t.scale,
            left_t.zero_point,
            right_t.scale,
            right_t.zero_point,
            out_t.scale,
            out_t.zero_point,
        )
        return [out, out_t]

    return register_fake_quantization_to_integer(op_name, binary)


# Use lambdas here to avoid a circular import problem
# pylint: disable=unnecessary-lambda
register_binary_qnn("add", lambda *args: relay.qnn.op.add(*args))
register_binary_qnn("multiply", lambda *args: relay.qnn.op.mul(*args))
register_binary_qnn("subtract", lambda *args: relay.qnn.op.subtract(*args))


def register_binary_identity(op_name, op):
    """Register a binary op that works directly on int8"""

    def binary(expr, type_map):
        left, right, left_t, right_t, out_t = get_binary_types(expr, type_map)
        if left_t != out_t:
            left = relay.qnn.op.requantize(
                left,
                left_t.scale,
                left_t.zero_point,
                out_t.scale,
                out_t.zero_point,
                out_dtype=out_t.dtype,
            )

        if right_t != out_t:
            right = relay.qnn.op.requantize(
                right,
                right_t.scale,
                right_t.zero_point,
                out_t.scale,
                out_t.zero_point,
                out_dtype=out_t.dtype,
            )
        out = op(left, right)
        return [out, out_t]

    return register_fake_quantization_to_integer(op_name, binary)


register_binary_identity("minimum", relay.op.minimum)
register_binary_identity("maximum", relay.op.maximum)
