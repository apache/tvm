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

from .._ffi.runtime_ctypes import TVMType
from ..relay.op import op as _reg

import tvm
from tvm import relay
import topi
import math
import functools
import numpy as np
import logging

RUNTIME_DEBUG = False

@tvm.register_func("tvm.contrib.print")
def print_func(x, y, msg):
    if RUNTIME_DEBUG:
        print(msg)
    x.copyto(y)

def my_print(data, msg):
    if not RUNTIME_DEBUG:
        return data
    ret = tvm.extern(data.shape, [data], lambda ins, outs: tvm.call_packed(
        "tvm.contrib.print", ins[0], outs[0], msg))
    return ret

@tvm.register_func("tvm.contrib.inspect")
def inspect_func(x, y, msg):
    if RUNTIME_DEBUG:
        print('------------------------------')
        print(msg)
        np_x = x.asnumpy()
        print('max value: {}'.format(np.max(np.abs(np_x))))
        print('mean: {}'.format(np.mean(np_x)))
        print('var: {}'.format(np.var(np_x)))
    x.copyto(y)

def inspect(data, msg):
    if not RUNTIME_DEBUG:
        return data
    ret = tvm.extern(data.shape, [data], lambda ins, outs: tvm.call_packed(
        "tvm.contrib.inspect", ins[0], outs[0], msg))
    return ret


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

    if not allclose(arr, data.asnumpy(), rtol=1e-03, atol=1.0):
        logging.warning('overflow happens')
        is_close = isclose(arr, data.asnumpy(), rtol=1e-03, atol=1.0)
        indexes = np.where(np.logical_not(is_close))
        print('cast to: {}'.format(in_dtype))
        print('thresholds: {}'.format(np.max(np.abs(data.asnumpy()))))
        print('original:\n{}'.format(data.asnumpy()[indexes]))
        print('after overflow:\n{}'.format(arr[indexes]))
        print('')
    tvm.nd.array(arr.astype('float32')).copyto(output)


@_reg.register_compute("hago.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type, target):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 1
    assert attrs.sign
    assert attrs.rounding == "round"

    data = inputs[0]
    in_scale = float(attrs.in_scale)
    out_scale = float(attrs.out_scale)
    clip_min = attrs.clip_min
    clip_max = attrs.clip_max

    # simulate overflow truncate error
    if attrs.in_dtype != 'float32':
        # data = topi.divide(data, in_scale)
        # data = tvm.extern(data.shape, [data], lambda ins, outs: tvm.call_packed(
        #     "tvm.contrib.check_overflow", ins[0], str(attrs.in_dtype), outs[0]))
        # data = topi.multiply(data, in_scale)

        data = topi.divide(data, in_scale)
        data = topi.cast(topi.round(data), 'int64')
        data = topi.cast(data, attrs.in_dtype)
        data = topi.multiply(data, in_scale)
    data = my_print(data, '*******************************************')
    data = my_print(data, "[in_scale={}, out_scale={}, clip_min={}, clip_max={}, in_dtype={}, out_dtype={}".format(in_scale, out_scale, clip_min, clip_max, attrs.in_dtype, attrs.out_dtype))

    # dequantize, directly return real value
    if attrs.out_dtype == 'float32':
        return [topi.identity(data)]

    # simulate rounding error
    # data = inspect(data, 'original data')
    scaled_data = topi.divide(data, out_scale)
    # scaled_data = inspect(scaled_data, 'scaled data')
    scaled_data = topi.round(scaled_data)
    clipped_data = topi.clip(scaled_data, float(clip_min), float(clip_max))
    # clipped_data = inspect(clipped_data, 'clipped data')
    round_data = topi.cast(topi.cast(clipped_data, attrs.out_dtype), 'float32')
    # round_data = inspect(round_data, 'round data')
    ret = topi.multiply(round_data, out_scale)

    ret = inspect(ret, 'return data')
    ret = my_print(ret, '*******************************************')
    return [ret]



@_reg.register_schedule("hago.simulated_quantize")
def simulated_quantize_schedule(attrs, outputs, target):
    s = tvm.create_schedule([x.op for x in outputs])
    return s

_reg.register_pattern("hago.simulated_quantize",
                      _reg.OpPattern.OPAQUE)


# constraint function registered for ops
# used for inferring output data type
# out_dtype.bits >= out_bit

def register_infer_bit(op_name, finfer_bit=None, level=10):
    return _reg.register(op_name, "FHagoInferBit", finfer_bit, level)

def max_bit(attrs, in_bits):
    in_bits = [bit.value for bit in in_bits]
    return [max(in_bits)]

register_infer_bit("nn.relu", max_bit)
register_infer_bit("nn.max_pool2d", max_bit)

def carry_one_bit(attrs, in_bits):
    # max(in_bits[0] - 1, in_bits[1] - 1) + 1 <= (out_bit - 1)
    assert len(in_bits) == 2
    in_bits = [bit.value for bit in in_bits]
    return [max(in_bits) + 1] 

register_infer_bit("add", carry_one_bit)

@register_infer_bit("nn.conv2d")
def infer_bit_for_conv2d(attrs, in_bits):
    # (in_bits[0] - 1) + (in_bits[1] - 1) + 1 <= (out_bit - 1)
    assert len(in_bits) == 2
    kernel_size = [k.value for k in attrs["kernel_size"]]
    size = functools.reduce(lambda x, y: x*y, kernel_size, 1.0) 
    extra_bit = np.ceil(np.math.log(size, 2))
    print('kernel_size: {}'.format(kernel_size))
    print('extra bit for conv2d: {}'.format(extra_bit))
    assert extra_bit >= 0
    in_bits = [bit.value for bit in in_bits]
    out_bit = in_bits[0] + in_bits[1] + int(extra_bit)
    print('out bit: {}'.format(out_bit))
    return [out_bit]


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
    chosen_thold = itholds[idx]
    chosen_bit = ibits[idx]
    unified_scale = itholds[idx] / (2 ** (ibits[idx] - sign_bit))

    print('  in bits   : {}'.format(ibits))
    print('  out bits  : {}'.format(obits))
    print('  in tholds : {}'.format(', '.join(["{:.3f}".format(thold) for thold in itholds])))
    print('  out tholds: {}'.format(', '.join(["{:.3f}".format(thold) for thold in otholds])))
    print('  choose unifed scale {:.3e} for op add'.format(unified_scale))
    new_tholds = []
    for i, bit in enumerate(ibits):
        # integer_range = 2 ** (bit - sign_bit) - 1
        # thold = integer_range * unified_scale
        thold = (2 ** (bit - chosen_bit)) * chosen_thold 
        print('  rectify threshold from {} to {} for op add'.format(itholds[i], thold))
        new_tholds.append(thold)
    for thold in otholds:
        new_tholds.append(thold)

    print('  new tholds: {}'.format(', '.join(["{:.3f}".format(thold) for thold in new_tholds])))

    return new_tholds


# realize registration for ops

def register_realize(op_name, frealize=None, level=10):
    return _reg.register(op_name, "FHagoRealize", frealize, level)

def forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return relay.Call(
        ref_call.op, args, ref_call.attrs, ref_call.type_args)


def to_scalar(constant):
    assert isinstance(constant, relay.Constant)
    scalar = constant.data.asnumpy()
    assert scalar.size == 1
    return scalar.item()

# TODO(ziheng) change to op_desc in the future
@register_realize("add")
def realize_addition(node, in_types, out_types):
    lhs, rhs = node.args
    dtype = out_types[0].value
    if in_types[0] != dtype:
        lhs = relay.cast(lhs, dtype)
    if in_types[1] != dtype:
        rhs = relay.cast(rhs, dtype)
    return forward_op(node, [lhs, rhs])


@register_realize("nn.conv2d")
def realize_conv2d(node, in_types, out_types):
    attrs_dict = {key: getattr(node.attrs, key) for key in dir(node.attrs)}
    attrs_dict['out_dtype'] = out_types[0].value
    attrs = tvm.make.node("relay.attrs.Conv2DAttrs", **attrs_dict)
    return relay.Call(node.op, node.args, attrs, node.type_args)
