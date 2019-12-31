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
#pylint: disable=unused-argument
"""Internal module for quantization."""
from __future__ import absolute_import
import math

import tvm
import topi
from .. import expr as _expr
from ..op import op as _reg
from .quantize import QAnnotateKind

@tvm.register_func("tvm.quantize.check_overflow")
def check_overflow(data, overflow_min, overflow_max, out):
    print(data.asnumpy())
    # raise ValueError
    data.copyto(out) 


@_reg.register_compute("relay.op.annotation.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type, target):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 6
    assert attrs.sign
    assert attrs.rounding == "round"

    data, dscale, clip_min, clip_max, overflow_min, overflow_max = inputs
    # return [topi.identity(data)]

    # check overflow
    # data = tvm.extern(data.shape,
    #     [data, overflow_min, overflow_max],
    #     lambda ins, outs: tvm.call_packed(
    #     "tvm.quantize.check_overflow", ins[0], ins[1], ins[2], outs[0]),
    #     name='check_overflow')

    # simulate rounding error
    scaled_data = topi.divide(data, dscale)
    clipped_data = topi.maximum(topi.minimum(scaled_data, clip_max), clip_min)
    round_data = topi.round(clipped_data)

    # recover data
    ret = topi.multiply(round_data, dscale)
    return [ret]


_reg.register_schedule("relay.op.annotation.simulated_quantize",
                       _reg.schedule_injective)
_reg.register_pattern("relay.op.annotation.simulated_quantize",
                      _reg.OpPattern.OPAQUE)


# dom_scale = scale / valid_range
# qdata * dom_scale = fdata

def adjust_scale(data, from_scale, to_scale):
    if from_scale == to_scale:
        return data

    factor = from_scale / to_scale
    shift_factor = math.log2(factor)
    assert shift_factor > 0
    if isinstance(shift_factor, int):
        out = topi.left_shift(data, shift_factor)
    elif isinstance(factor, int):
        out = topi.mulitply(data, factor)
    else:
        dtype = data.dtype
        out = topi.cast(data, "float32")
        out = topi.mulitply(data, factor)
        out = topi.cast(out, dtype)
    return out


def extract_scalar(tensor):
    assert isinstance(tensor, _expr.Constant)
    arr = tensor.value
    assert arr.size == 1
    return arr[0]


# @_reg.register_compute("relay.op.quantize.quantized_add")
def quantized_add_compute(attrs, inputs, out_type, target):
    """Compiler for simulated_quantize."""

    assert len(inputs) == 5

    lhs, rhs, dom_lscale, dom_rscale, dom_oscale = inputs
    dom_lscale = extract_scalar(dom_lscale)
    dom_rscale = extract_scalar(dom_rscale)
    dom_oscale = extract_scalar(dom_oscale)

    lhs = adjust_scale(lhs, dom_lscale, dom_oscale)
    rhs = adjust_scale(rhs, dom_rscale, dom_oscale)
    out = lhs + rhs
    return out


# _reg.register_schedule("relay.op.quantize.quantized_add",
#                        _reg.schedule_injective)
# _reg.register_pattern("relay.op.quantize.quantized_add",
#                       _reg.OpPattern.ELEMWISE)
