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

import tvm
import topi
from ..relay.op import op as _reg
from .. import relay

import math
import functools
import numpy as np

@tvm.register_func("tvm.contrib.debug")
def debug_func(x, y):
    print('counter: {}'.format(debug_func.cnt))
    print('maximum value: ')
    print(np.max(np.abs(x.asnumpy())))
    x.copyto(y)
    debug_func.cnt += 1
debug_func.cnt = 0


def isclose(old, new, rtol, atol):
    # compare two arrays under quantized situation
    thold = np.max(np.abs(old))
    condition = (np.abs(old - new) <= (atol + rtol * thold))
    return condition

def allclose(old, new, rtol, atol):
    # compare two arrays under quantized situation
    cond = isclose(old, new, rtol, atol)
    return np.all(cond)

@tvm.register_func("tvm.contrib.check_overflow")
def check_overflow(data, in_dtype, output):
    arr = data.asnumpy()
    # print(in_dtype)
    # print('before: {}'.format(np.max(arr)))
    arr = arr.astype('int64')
    arr = arr.astype(in_dtype)
    # print('after: {}'.format(np.max(arr)))

    if 'float' in in_dtype:
        # skip overflow check for float input dtype
        data.copyto(output)
        return 

    if not allclose(arr, data.asnumpy(), rtol=1e-03, atol=1e-08):
        print('overflow happens')
        is_close = isclose(arr, data.asnumpy(), rtol=1e-03, atol=1e-08)
        indexes = np.where(np.logical_not(is_close))
        print(arr[indexes])
        print(data.asnumpy()[indexes])
        raise ValueError
    tvm.nd.array(arr.astype('float32')).copyto(output)


@_reg.register_compute("relay.op.annotation.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type, target):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 5
    assert attrs.sign
    assert attrs.rounding == "round"

    data, out_scale, in_scale, clip_min, clip_max = inputs
    if attrs.in_dtype == 'float32' and attrs.out_dtype == 'float32':
        return [topi.identity(data)]

    # simulate overflow
    data = topi.divide(data, in_scale)
    data = tvm.extern(data.shape, [data], lambda ins, outs: tvm.call_packed(
        "tvm.contrib.check_overflow", ins[0], str(attrs.in_dtype), outs[0]))
    data = topi.multiply(data, in_scale)

    # if 'float' not in attrs.in_dtype:
    #     data = topi.divide(data, in_scale)
    #     data = topi.cast(topi.cast(data, 'int64'), attrs.in_dtype)
    #     data = topi.multiply(data, in_scale)

    # simulate rounding error
    scaled_data = topi.divide(data, out_scale)
    clipped_data = topi.maximum(topi.minimum(scaled_data, clip_max), clip_min)
    round_data = topi.round(clipped_data)

    # recover data
    ret = topi.multiply(round_data, out_scale)
    return [ret]


def quantize(data, out_scale, in_scale, clip_min, clip_max, out_dtype, in_dtype):
    casted = False
    if out_dtype.bits > in_dtype.bits:
        casted = True
        data = topi.cast(data, out_dtype)
    data = data * (in_scale / out_scale)
    data = topi.maximum(topi.minimum(data, clip_max), clip_min)
    if idtype != odtype and not casted:
        data = topi.cast(odata, out_dtype)
    return data


@_reg.register_schedule("relay.op.annotation.simulated_quantize")
def simulated_quantize_schedule(attrs, outputs, target):
    s = tvm.create_schedule([x.op for x in outputs])
    return s

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
    assert isinstance(tensor, relay.Constant)
    arr = tensor.value
    assert arr.size == 1
    return arr[0]


# infer scale function registered for ops

def register_infer_scale(op_name, finfer_scale=None, level=10):
    return _reg.register(op_name, "FHagoInferScale", finfer_scale, level)

def product_scale(input_scales):
    input_scales = [scale.value for scale in input_scales]
    return functools.reduce(lambda x, y: x*y, input_scales, 1.0)

register_infer_scale("nn.conv2d", product_scale)
register_infer_scale("nn.dense", product_scale)

def identity_scale(input_scales):
    input_scales = [scale.value for scale in input_scales]
    scale0 = input_scales[0] 
    for scale in input_scales:
        assert math.isclose(scale, scale, rel_tol=1e-6)
    return scale0

register_infer_scale("add", identity_scale)
register_infer_scale("nn.relu", identity_scale)
register_infer_scale("nn.max_pool2d", identity_scale)
register_infer_scale("nn.global_avg_pool2d", identity_scale)
register_infer_scale("nn.batch_flatten", identity_scale)

# threshold rectify function registered for ops

def register_threshold_rectify(op_name, frectify=None, level=10):
    return _reg.register(op_name, "FHagoRectify", frectify, level)

@register_threshold_rectify("add")
def threshold_rectify_for_add(input_bits, output_bits, input_thresholds, output_thresholds):
    sign_bit = 1
    # convert from tvm object to POD
    ibits = [bit.value for bit in input_bits]
    obits = [bit.value for bit in output_bits]
    itholds = [thold.value for thold in input_thresholds]
    otholds = [thold.value for thold in output_thresholds]

    # choose scale of the one with max threshold
    idx = np.argmax(itholds)
    unified_scale = itholds[idx] / (2 ** (ibits[idx] - sign_bit) - 1)

    print('input bits: {}'.format(ibits))
    print('output bits: {}'.format(obits))
    print('input thresholds: {}'.format(', '.join(["{:.3f}".format(thold) for thold in itholds])))
    print('output thresholds: {}'.format(', '.join(["{:.3f}".format(thold) for thold in otholds])))
    print('choose unifed scale {:.3e} for op add'.format(unified_scale))

    new_tholds = []
    for i, bit in enumerate(ibits):
        integer_range = 2 ** (bit - sign_bit) - 1
        thold = integer_range * unified_scale
        print('rectify threshold from {:.3e} to {:.3e} for op add'.format(itholds[i], thold))
        new_tholds.append(integer_range * unified_scale)
    for thold in otholds:
        new_tholds.append(thold)
    return new_tholds




# @_reg.register_compute("relay.op.quantize.quantized_add")
# def quantized_add_compute(attrs, inputs, out_type, target):
#     """Compiler for simulated_quantize."""
# 
#     assert len(inputs) == 5
# 
#     lhs, rhs, dom_lscale, dom_rscale, dom_oscale = inputs
#     dom_lscale = extract_scalar(dom_lscale)
#     dom_rscale = extract_scalar(dom_rscale)
#     dom_oscale = extract_scalar(dom_oscale)
# 
#     lhs = adjust_scale(lhs, dom_lscale, dom_oscale)
#     rhs = adjust_scale(rhs, dom_rscale, dom_oscale)
#     out = lhs + rhs
#     return out
# 
# 
# _reg.register_schedule("relay.op.quantize.quantized_add",
#                        _reg.schedule_injective)
# _reg.register_pattern("relay.op.quantize.quantized_add",
#                       _reg.OpPattern.ELEMWISE)
