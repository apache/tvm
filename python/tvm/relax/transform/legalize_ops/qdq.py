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
# pylint: disable=invalid-name
"""Default legalization function for quantize/dequantize operators."""

from typing import Union
import tvm
from tvm import te, tir
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize, _try_convert_to_scalar_const


def clip_cast(val, dtype):
    const_min = tvm.tir.min_value(dtype)
    const_max = tvm.tir.max_value(dtype)
    return te.max(te.min(val, const_max), const_min).astype(dtype)


def is_const_scalar(x):
    return isinstance(x, (tvm.tir.IntImm, tvm.tir.FloatImm))


@register_legalize("relax.quantize")
def _quantize(bb: BlockBuilder, call: Call) -> Expr:
    """
    Lower relax.quantize into the sequence of simple operations.
    Quantization formula is defined as: out = clip(round(input / scale) + zp, min_val, max_val)
    """
    axis = call.attrs.axis
    out_dtype = call.attrs.out_dtype

    def te_quantize(
        data: te.Tensor,
        scale: Union[te.Tensor, tir.IntImm, tir.FloatImm],
        zp: Union[te.Tensor, tir.IntImm, tir.FloatImm],
    ):
        def quantize_compute(*indices):
            scale_value = scale if is_const_scalar(scale) else scale[indices[axis]]
            zp_value = zp if is_const_scalar(zp) else zp[indices[axis]]
            scaled = data[indices] / scale_value
            round_val = (te.round(scaled) if "int" in out_dtype else scaled) + zp_value
            return clip_cast(round_val, out_dtype)

        output_shape = data.shape
        return te.compute(output_shape, quantize_compute, name="quantized")

    return bb.call_te(
        te_quantize,
        call.args[0],
        _try_convert_to_scalar_const(call.args[1]),
        _try_convert_to_scalar_const(call.args[2]),
        primfunc_name_hint="quantize",
    )


@register_legalize("relax.dequantize")
def _dequantize(bb: BlockBuilder, call: Call) -> Expr:
    """
    Lower relax.dequantize into the sequence of simple operations.
    Dequantization formula is defined as: out = scale * (input - zp)
    Compute datatype: float32

    Example of lowering:

        dtype = ["int32"|"float32"]

        qnn.dequantize(data, scale, zp, "float32") -->
            sub = subtract(cast(data, dtype), zp)
            out = multiply(cast(sub, "float32"), scale)

        qnn.dequantize(data, scale, zp, "float16") -->
            sub = subtract(cast(data, dtype), zp)
            mul = multiply(cast(sub, "float32"), cast(scale, "float32"))
            clipped_out = clip(mul, float32(-65504.0), float32(65504.0))
            out = cast(clipped_out, "float16")
    """
    axis = call.attrs.axis
    out_dtype = call.attrs.out_dtype

    def te_dequantize(
        data: te.Tensor,
        scale: Union[te.Tensor, tir.IntImm, tir.FloatImm],
        zp: Union[te.Tensor, tir.IntImm, tir.FloatImm],
    ):
        def dequantize_compute(*indices):
            scale_value = scale if is_const_scalar(scale) else scale[indices[axis]]
            zp_value = zp if is_const_scalar(zp) else zp[indices[axis]]
            dtype = "float32" if "float" in data.dtype else "int32"
            sub = te.subtract(data[indices].astype(dtype), zp_value)
            out = te.multiply(sub, scale_value.astype("float32"))
            if out_dtype == "float32":
                return out
            return clip_cast(out, out_dtype)

        output_shape = data.shape
        return te.compute(output_shape, dequantize_compute, name="dequantized")

    return bb.call_te(
        te_dequantize,
        call.args[0],
        _try_convert_to_scalar_const(call.args[1]),
        _try_convert_to_scalar_const(call.args[2]),
        primfunc_name_hint="dequantize",
    )
