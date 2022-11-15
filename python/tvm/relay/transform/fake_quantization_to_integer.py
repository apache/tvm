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
import numpy as np
import tvm
from tvm import relay
from tvm.ir import TensorAffineType, TupleAffineType

# import to register canonicalization funcs for fq2i
# pylint: disable=unused-import
from tvm.relay.qnn.op import canonicalizations
from tvm.tir import bijective_layout

from ..op import register_fake_quantization_to_integer


def fold_constant(expr):
    return relay.transform.FoldConstantExpr(expr, tvm.IRModule())


def get_zeros(scale):
    return fold_constant(relay.op.cast(relay.op.zeros_like(scale), "int32"))


def infer_shape(expr):
    return relay.transform.InferType()(tvm.IRModule.from_expr(expr))["main"].body.checked_type.shape


def approx_equal(x, y):
    x = fold_constant(x)
    y = fold_constant(y)
    if isinstance(x, relay.Constant) and isinstance(y, relay.Constant):
        equal = np.allclose(x.data.asnumpy(), y.data.asnumpy())
    else:
        equal = tvm.ir.structural_equal(x, y)
    return equal


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
        approx_equal(in_scale, expr.args[1])
        and approx_equal(in_zero_point, expr.args[2])
        and tvm.ir.structural_equal(t.dtype, expr.attrs.out_dtype)
    ):
        out = relay.qnn.op.requantize(
            out,
            in_scale,
            in_zero_point,
            expr.args[1],
            expr.args[2],
            out_dtype=expr.attrs.out_dtype,
            axis=t.axis,
        )
    return [
        out,
        TensorAffineType(expr.args[1], expr.args[2], expr.attrs.out_dtype, expr.attrs.axis),
    ]


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
register_unary_identity("nn.depth_to_space")
register_unary_identity("max")
register_unary_identity("min")
register_unary_identity("image.resize2d")


@register_fake_quantization_to_integer("nn.adaptive_avg_pool1d")
def adaptive_avgpool1d(expr, type_map):
    """Rewrite an adaptive avgpool op"""
    arg = expr.args[0]
    t = type_map[arg]
    out_t = type_map[expr]
    if not (
        approx_equal(t.scale, out_t.scale)
        and approx_equal(t.zero_point, out_t.zero_point)
        and tvm.ir.structural_equal(t.dtype, out_t.dtype)
    ):
        arg = relay.qnn.op.requantize(
            arg,
            t.scale,
            t.zero_point,
            out_t.scale,
            out_t.zero_point,
            out_dtype="int32",
            axis=t.axis,
        )
    else:
        arg = relay.op.cast(arg, "int32")
    output_size = expr.attrs.output_size
    out = relay.op.nn.adaptive_avg_pool1d(arg, output_size)
    return [out, TensorAffineType(out_t.scale, out_t.zero_point, "int32", out_t.axis)]


@register_fake_quantization_to_integer("nn.avg_pool2d")
def avgpool2d(expr, type_map):
    """Rewrite a avgpool op"""
    arg = expr.args[0]
    t = type_map[arg]
    out_t = type_map[expr]
    # Cast (or requantize) to int32.
    if not (
        approx_equal(t.scale, out_t.scale)
        and approx_equal(t.zero_point, out_t.zero_point)
        and tvm.ir.structural_equal(t.dtype, out_t.dtype)
    ):
        arg = relay.qnn.op.requantize(
            arg,
            t.scale,
            t.zero_point,
            out_t.scale,
            out_t.zero_point,
            out_dtype="int32",
            axis=t.axis,
        )
    else:
        arg = relay.op.cast(arg, "int32")
    out = relay.op.nn.avg_pool2d(arg, **expr.attrs)
    if out_t.dtype != "int32":
        # Cast back to output dtype to preserve input dtype == output dtype for AvgPool2d.
        out = relay.op.clip(out, a_min=np.iinfo(out_t.dtype).min, a_max=np.iinfo(out_t.dtype).max)
        out = relay.op.cast(out, out_t.dtype)
    return [out, TensorAffineType(out_t.scale, out_t.zero_point, out_t.dtype, out_t.axis)]


@register_fake_quantization_to_integer("nn.global_avg_pool2d")
def global_avgpool2d(expr, type_map):
    """Rewrite a global_avgpool op"""
    arg = expr.args[0]
    t = type_map[arg]
    out_t = type_map[expr]
    out_t = type_map[expr]
    if not (
        approx_equal(t.scale, out_t.scale)
        and approx_equal(t.zero_point, out_t.zero_point)
        and tvm.ir.structural_equal(t.dtype, out_t.dtype)
    ):
        arg = relay.qnn.op.requantize(
            arg,
            t.scale,
            t.zero_point,
            out_t.scale,
            out_t.zero_point,
            out_dtype="int32",
            axis=t.axis,
        )
    else:
        arg = relay.op.cast(arg, "int32")
    out = relay.op.nn.global_avg_pool2d(arg)
    return [out, TensorAffineType(out_t.scale, out_t.zero_point, "int32", out_t.axis)]


@register_fake_quantization_to_integer("broadcast_to")
def broadcast_to(expr, type_map):
    """Rewrite a broadcast_to op"""
    arg = expr.args[0]
    t = type_map[arg]
    shape = expr.attrs.shape
    out = relay.op.broadcast_to(arg, shape)
    return [out, t]


@register_fake_quantization_to_integer("nn.bias_add")
def bias_add(expr, type_map):
    """Rewrite a bias_add op"""
    x, b = expr.args
    x_t = type_map[x]
    if b in type_map:
        # Ensure bias matches the previous op
        b_t = type_map[b]
        in_scale = fold_constant(x_t.scale)
        in_zero_point = fold_constant(x_t.zero_point)
        if not (
            approx_equal(x_t.scale, b_t.scale)
            and approx_equal(x_t.zero_point, b_t.zero_point)
            and tvm.ir.structural_equal(x_t.dtype, b_t.dtype)
        ):
            b = relay.qnn.op.requantize(
                b,
                b_t.scale,
                b_t.zero_point,
                in_scale,
                in_zero_point,
                out_dtype=x_t.dtype,
                axis=0,
            )
    else:
        # If the bias is a constant, we need to quantize it
        assert isinstance(b, relay.expr.Constant)
        assert b.checked_type.dtype in ["float32", "float64", "float16", "bfloat16"]
        b = relay.qnn.op.quantize(b, x_t.scale, x_t.zero_point, axis=0, out_dtype=x_t.dtype)
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
    conv_zp = get_zeros(conv_scale)
    out = relay.qnn.op.conv2d(
        x, weight, x_t.zero_point, w_t.zero_point, x_t.scale, w_t.scale, **attrs
    )
    out_layout = attrs["out_layout"] if attrs["out_layout"] != "" else attrs["data_layout"]
    out_axis = bijective_layout(out_layout, "NCHW").backward_index(list(range(4)))[1]
    return [out, TensorAffineType(conv_scale, conv_zp, out.attrs.out_dtype, out_axis.value)]


@register_fake_quantization_to_integer("nn.conv2d_transpose")
def conv2d_transpose(expr, type_map):
    """Rewrite a conv2d_transpose op"""
    attrs = {**expr.attrs}
    attrs.pop("out_dtype")
    x, weight = expr.args
    x_t = type_map[x]
    w_t = type_map[weight]
    conv_scale = fold_constant(x_t.scale * w_t.scale)
    conv_zp = get_zeros(conv_scale)

    out = relay.qnn.op.conv2d_transpose(
        x, weight, x_t.zero_point, w_t.zero_point, x_t.scale, w_t.scale, **attrs
    )
    out_layout = attrs["out_layout"] if attrs["out_layout"] != "" else attrs["data_layout"]
    out_axis = bijective_layout(out_layout, "NCHW").backward_index(list(range(4)))[1]
    return [out, TensorAffineType(conv_scale, conv_zp, out.attrs.out_dtype, out_axis.value)]


@register_fake_quantization_to_integer("nn.dense")
def dense(expr, type_map):
    """Rewrite a dense op"""
    attrs = {**expr.attrs}
    attrs.pop("out_dtype")
    x, weight = expr.args
    x_t = type_map[x]
    w_t = type_map[weight]
    dense_scale = fold_constant(x_t.scale * w_t.scale)
    dense_zp = get_zeros(dense_scale)
    out = relay.qnn.op.dense(
        x, weight, x_t.zero_point, w_t.zero_point, x_t.scale, w_t.scale, **attrs
    )
    return [out, TensorAffineType(dense_scale, dense_zp, out.attrs.out_dtype, 1)]


@register_fake_quantization_to_integer("nn.batch_matmul")
def batch_matmul(expr, type_map):
    """Rewrite a batch_matmul op"""
    x, y = expr.args
    x_t = type_map[x]
    y_t = type_map[y]
    matmul_scale = fold_constant(x_t.scale * y_t.scale)
    matmul_zp = relay.const(0)
    out = relay.qnn.op.batch_matmul(x, y, x_t.zero_point, y_t.zero_point, x_t.scale, y_t.scale)
    return [out, TensorAffineType(matmul_scale, matmul_zp, out.attrs.out_dtype, x_t.axis)]


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


@register_fake_quantization_to_integer("topk")
def topk(expr, type_map):
    """Rewrite a topk op"""
    arg = expr.args[0]
    t = type_map[arg]
    attrs = {**expr.attrs}
    assert "ret_type" in attrs and attrs["ret_type"] == "values"
    return [expr, t]


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
    if (
        isinstance(scale, relay.expr.Constant)
        and scale.data.numpy().size == 1
        and isinstance(z_p, relay.expr.Constant)
        and z_p.data.numpy().size == 1
    ):
        scale = scale.data.numpy().item()
        z_p = z_p.data.numpy().item()
        new_min = int(amin / scale + z_p)
        new_max = int(amax / scale + z_p)
        out = relay.op.clip(arg, new_min, new_max)
    else:
        if not isinstance(amin, relay.expr.Constant):
            amin = relay.op.const(amin)
        if not isinstance(amax, relay.expr.Constant):
            amax = relay.op.const(amax)

        scale_shape = infer_shape(scale)
        if len(scale_shape) > 0 and scale_shape[0] > 1:
            b_shape = [1] * len(infer_shape(arg))
            b_shape[t.axis] = -1
            amin = relay.op.reshape(relay.op.broadcast_to(amin, scale_shape), b_shape)
            amax = relay.op.reshape(relay.op.broadcast_to(amax, scale_shape), b_shape)
        amin = relay.qnn.op.quantize(amin, scale, z_p, t.axis, t.dtype)
        amax = relay.qnn.op.quantize(amax, scale, z_p, t.axis, t.dtype)
        out = relay.op.minimum(relay.op.maximum(arg, fold_constant(amin)), fold_constant(amax))

    return [out, t]


@register_fake_quantization_to_integer("nn.relu")
def relu(expr, type_map):
    """Rewrite a relu op"""
    arg = expr.args[0]
    t = type_map[arg]
    scale_shape = infer_shape(t.scale)
    z_p = t.zero_point
    assert len(scale_shape) <= 1
    if len(scale_shape) == 1 and scale_shape[0] > 1:
        b_shape = [1] * len(infer_shape(arg))
        b_shape[t.axis] = -1
        z_p = relay.op.reshape(relay.op.broadcast_to(z_p, scale_shape), b_shape)
    zero = relay.op.cast(z_p, t.dtype)
    return [relay.op.maximum(arg, fold_constant(zero)), t]


@register_fake_quantization_to_integer("nn.leaky_relu")
def leaky_relu(expr, type_map):
    """Rewrite a leaky relu op"""
    arg = expr.args[0]
    x_t = type_map[arg]
    out_t = type_map[expr]
    alpha = expr.attrs.alpha
    output = relay.qnn.op.leaky_relu(
        expr, alpha, x_t.scale, x_t.zero_point, out_t.scale, out_t.zero_point
    )
    return [output, x_t]


@register_fake_quantization_to_integer("nn.pad")
def pad(expr, type_map):
    """Rewite an nn.pad op"""
    arg = expr.args[0]
    t = type_map[arg]
    pad_value = expr.args[1]
    # TF2ONNX will sometimes implement the pad_value as a constant without a quantize
    # To support that, the pass lets branches that terminate in a constant through
    if pad_value in type_map:
        # if the pad value is calcuated from a dequantize op, it should be in the type map
        # and we need to make sure it's affine type matches the arg
        pad_t = type_map[pad_value]
        if not tvm.ir.structural_equal(t, pad_t):
            pad_value = relay.qnn.op.requantize(
                pad_value,
                pad_t.scale,
                pad_t.zero_point,
                t.scale,
                t.zero_point,
                out_dtype=t.dtype,
                axis=pad_t.axis,
            )
    else:
        # If the pad-value is a constant, we need to quantize it
        assert isinstance(pad_value, relay.expr.Constant)
        assert pad_value.checked_type.dtype in ["float32", "float64", "float16", "bfloat16"]
        pad_value = relay.qnn.op.quantize(pad_value, t.scale, t.zero_point)

    out = relay.op.nn.pad(arg, pad_value=pad_value, **expr.attrs)
    return [out, t]


@register_fake_quantization_to_integer("mean")
def mean(expr, type_map):
    """Rewrite a mean op"""
    arg = expr.args[0]
    t = type_map[arg]

    arg = relay.op.cast(arg, "int32")
    out = relay.op.mean(arg, **expr.attrs)
    out = relay.op.cast(out, t.dtype)
    return [out, t]


def get_binary_types(expr, type_map):
    """Get Affine types of a binary op's inputs and unify them"""
    # Support the case where one input is quantized and the other is a constant float
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
    if right_t is None:
        assert isinstance(right, relay.expr.Constant)
        right = relay.qnn.op.quantize(
            right, left_t.scale, left_t.zero_point, out_dtype=left_t.dtype
        )
        right_t = left_t

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
            left_t.axis,
            right_t.axis,
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
                axis=left_t.axis,
            )

        if right_t != out_t:
            right = relay.qnn.op.requantize(
                right,
                right_t.scale,
                right_t.zero_point,
                out_t.scale,
                out_t.zero_point,
                out_dtype=out_t.dtype,
                axis=right_t.axis,
            )
        out = op(left, right)
        return [out, out_t]

    return register_fake_quantization_to_integer(op_name, binary)


register_binary_identity("minimum", relay.op.minimum)
register_binary_identity("maximum", relay.op.maximum)


def register_unary_qnn(op_name, op):
    """Rewrite a unary op"""

    def unary(expr, type_map):
        arg = expr.args[0]
        x_t = type_map[arg]
        out_t = type_map[expr]
        out = op(
            arg,
            x_t.scale,
            x_t.zero_point,
            out_t.scale,
            out_t.zero_point,
        )
        return [out, out_t]

    return register_fake_quantization_to_integer(op_name, unary)


register_unary_qnn("sqrt", relay.qnn.op.sqrt)
register_unary_qnn("rsqrt", relay.qnn.op.rsqrt)
register_unary_qnn("exp", relay.qnn.op.exp)
register_unary_qnn("erf", relay.qnn.op.erf)
register_unary_qnn("sigmoid", relay.qnn.op.sigmoid)
register_unary_qnn("hardswish", relay.qnn.op.hardswish)
register_unary_qnn("tanh", relay.qnn.op.tanh)
register_unary_qnn("abs", relay.qnn.op.abs)
register_unary_qnn("log", relay.qnn.op.log)
