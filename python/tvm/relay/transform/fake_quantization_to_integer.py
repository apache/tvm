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
from ..op import register_fake_quantization_to_integer


def fold_constant(expr):
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.FoldConstant()(mod)
    return mod["main"].body


@register_fake_quantization_to_integer("qnn.dequantize")
def dequantize(expr, type_map):
    """Remove dequantize op"""
    out = expr.args[0]
    t = type_map[expr]
    return [out, t.scale, t.zero_point, t.dtype]


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
    return [out, expr.args[1], expr.args[2], expr.attrs.out_dtype]


def register_unary_identity(op_name, op):
    def identity(expr, type_map):
        assert len(expr.args) == 1
        arg = expr.args[0]
        t = type_map[arg]
        out = op(arg, **expr.attrs)
        return [out, t.scale, t.zero_point, t.dtype]

    return register_fake_quantization_to_integer(op_name, identity)


register_unary_identity("reshape", relay.op.reshape)
register_unary_identity("transpose", relay.op.transpose)
register_unary_identity("nn.max_pool2d", relay.op.nn.max_pool2d)


@register_fake_quantization_to_integer("nn.avg_pool2d")
def avgpool2d(expr, type_map):
    """Rewrite a avgpool op"""
    arg = expr.args[0]
    t = type_map[arg]
    arg = relay.op.cast(arg, "int32")
    out = relay.op.nn.avg_pool2d(arg, **expr.attrs)
    out = relay.op.cast(out, t.dtype)
    return [out, t.scale, t.zero_point, t.dtype]


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
            out_dtype=xt.dtype,
        )
    out = relay.op.nn.bias_add(x, b, **expr.attrs)
    return [out, x_t.scale, x_t.zero_point, x_t.dtype]


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
    return [out, conv_scale, conv_zp, out.attrs.out_dtype]


@register_fake_quantization_to_integer("concatenate")
def concat(expr, type_map):
    """Rewrite a concat op"""
    scales = []
    zps = []
    for arg in expr.args[0].fields:
        t = type_map[arg]
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
    return [out, out_type.scale, out_type.zero_point, out_type.dtype]


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
    return [out, t.scale, t.zero_point, t.dtype]
