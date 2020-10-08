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
from tvm import relay
from tvm._ffi.runtime_ctypes import DataType
from ..relay.op import op as _reg
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
    ret = tvm.te.extern(data.shape, [data], lambda ins, outs: tvm.tir.call_packed(
        "tvm.contrib.print", ins[0], outs[0], msg))
    return ret

@tvm.register_func("tvm.contrib.print_info")
def print_info(data, in_scale, out_scale, clip_min, clip_max, out):
    if RUNTIME_DEBUG:
        print('in_scale:  {}'.format(in_scale.asnumpy()))
        print('out_scale: {}'.format(out_scale.asnumpy()))
        print('clip_min:  {}'.format(clip_min.asnumpy()))
        print('clip_max:  {}'.format(clip_max.asnumpy()))
    data.copyto(out)

def print_info(data, in_scale, out_scale, clip_min, clip_max):
    if not RUNTIME_DEBUG:
        return data
    ret = tvm.te.extern(data.shape, [data, in_scale, out_scale, clip_min, clip_max],
        lambda ins, outs: tvm.tir.call_packed(
        "tvm.contrib.print_info", ins[0], ins[1], ins[2], ins[3], ins[4], outs[0]))
    return ret


@tvm.register_func("tvm.contrib.inspect")
def inspect_func(x, y, msg):
    if RUNTIME_DEBUG:
        print('------------------------------')
        print(msg)
        # print(x)
        np_x = x.asnumpy()
        print('max value: {:.4f}'.format(np.max(np.abs(np_x))))
        print('mean: {:.4f}'.format(np.mean(np_x)))
        print('var: {:.4f}'.format(np.var(np_x)))
    x.copyto(y)

def inspect(data, msg):
    if not RUNTIME_DEBUG:
        return data
    ret = tvm.te.extern(data.shape, [data], lambda ins, outs: tvm.tir.call_packed(
        "tvm.contrib.inspect", ins[0], outs[0], msg))
    return ret

@tvm.register_func("tvm.contrib.compare")
def compare_func(a, b, out, msg):
    print('------------------------------')
    print('------------COMPARE-----------')
    print(msg)
    np_x = a.asnumpy()
    np_y = b.asnumpy()
    print('max value : {:.4f}, {:.4f}'.format(np.max(np_x), np.max(np_y)))
    print('min value : {:.4f}, {:.4f}'.format(np.min(np_x), np.min(np_y)))
    print('max abs   : {:.4f}, {:.4f}'.format(np.max(np.abs(np_x)), np.max(np.abs(np_y))))
    print('mean      : {:.4f}, {:.4f}'.format(np.mean(np_x), np.mean(np_y)))
    print('var       : {:.4f}, {:.4f}'.format(np.var(np_x), np.var(np_y)))
    abs_err = np.abs(np_x - np_y)
    rel_err = (np_x - np_y) / np.max(np.abs(np_y))
    idx = np.unravel_index(np.argmax(abs_err, axis=None), abs_err.shape)
    print('maximum absolute error: {:.4f}({:.2f}%), compare {:.4f} with {:.4f}'
          .format(np.max(abs_err), rel_err[idx] * 100, np_x[idx], np_y[idx]))
    a.copyto(out)


def compare(origin, ret, msg):
    if not RUNTIME_DEBUG:
        return ret
    ret = tvm.te.extern(ret.shape, [origin, ret],
        lambda ins, outs: tvm.tir.call_packed(
        "tvm.contrib.compare", ins[0], ins[1], outs[0], msg))
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


@_reg.register_compute("nn.simulated_quantize")
def simulated_quantize_compute(attrs, inputs, out_type):
    """Compiler for simulated_quantize."""
    assert len(inputs) == 5
    assert attrs.sign
    assert attrs.rounding == "round"
    axis = attrs.axis

    data, in_scale, out_scale, clip_min, clip_max = inputs
    data = my_print(data, '\n\n*******************************************')
    data = print_info(data, in_scale, out_scale, clip_min, clip_max)
    # data = my_print(data, "[in_scale={}, out_scale={}, clip_min={}, clip_max={}, in_dtype={}, out_dtype={}".format(in_scale, out_scale, clip_min, clip_max, attrs.in_dtype, attrs.out_dtype))
    origin = data
    data = inspect(data, 'original data')

    ##################################
    # simulate overflow truncate error
    if attrs.in_dtype != 'float32':
        # data = topi.divide(data, in_scale)
        # data = tvm.extern(data.shape, [data], lambda ins, outs: tvm.call_packed(
        #     "tvm.contrib.check_overflow", ins[0], str(attrs.in_dtype), outs[0]))
        # data = topi.multiply(data, in_scale)

        if len(in_scale.shape) == 1:
            assert axis is not None
            assert len(out_scale.shape) == 0
            # per-channel dequantize
            expand_axes = [i for i in range(len(data.shape)) if i != axis]
            in_scale = topi.expand_like(in_scale, data, expand_axes)
            data = topi.divide(data, in_scale)
        else:
            data = topi.divide(data, in_scale)
        data = topi.cast(topi.round(data), 'int64')
        data = topi.cast(data, attrs.in_dtype)
        data = topi.multiply(data, in_scale)

    ########################################
    # dequantize, directly return real value
    if attrs.out_dtype == 'float32':
        data = my_print(data, '*******************************************\n\n')
        return [topi.identity(data)]


    #########################
    # simulate rounding error
    if len(out_scale.shape) == 1:
        assert axis is not None
        assert len(in_scale.shape) == 0
        # per-channel quantize
        expand_axes = [i for i in range(len(data.shape)) if i != axis]
        out_scale = topi.expand_like(out_scale, data, expand_axes)
        scaled_data = topi.divide(data, out_scale)
    else:
        scaled_data = topi.divide(data, out_scale)
    scaled_data = inspect(scaled_data, 'scaled data')

    round_data = topi.round(scaled_data)
    round_data = inspect(round_data, 'round data')

    #########################
    # simulate clipping error
    clipped_data = topi.maximum(topi.minimum(round_data, clip_max), clip_min)
    clipped_data = inspect(clipped_data, 'clipped data')

    cast_data = topi.cast(topi.cast(clipped_data, attrs.out_dtype), 'float32')
    cast_data = inspect(cast_data, 'cast_data')

    ret = topi.multiply(cast_data, out_scale)
    ret = inspect(ret, 'return data')
    ret = compare(origin, ret, "compare origin with return data")
    ret = my_print(ret, '*******************************************\n\n')
    return [ret]


_reg.register_schedule("nn.simulated_quantize", tvm.relay.op.strategy.schedule_simulated_quantize)
_reg.register_pattern("nn.simulated_quantize", _reg.OpPattern.OPAQUE)


# infer scale function registered for ops

def register_infer_scale(op_name, finfer_scale=None, level=10):
    return tvm.ir.register_op_attr(op_name, "FHagoInferScale", finfer_scale, level)

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
register_infer_scale("mean", identity_scale)
register_infer_scale("nn.softmax", identity_scale)
register_infer_scale("layout_transform", identity_scale)
register_infer_scale("nn.pad", identity_scale)
register_infer_scale("nn.relu", identity_scale)
register_infer_scale("nn.max_pool2d", identity_scale)
register_infer_scale("nn.avg_pool2d", identity_scale)
register_infer_scale("nn.global_avg_pool2d", identity_scale)
register_infer_scale("nn.adaptive_avg_pool2d", identity_scale)
register_infer_scale("nn.batch_flatten", identity_scale)

# threshold rectify function registered for ops

def register_threshold_rectify(op_name, frectify=None, level=10):
    return tvm.ir.register_op_attr(op_name, "FHagoRectify", frectify, level)

@register_threshold_rectify("add")
def unify_scale(input_bits, output_bits, input_thresholds, output_thresholds):
    sign_bit = 1
    # convert from tvm object to POD
    ibits = [bit.value for bit in input_bits]
    # FIXME - Uncommenting next line can cause failures when add is followed by a non-quantized op.
    # Exmaple is add --> clip and model is MXNet mobilenetv2
    # obits = [32 if bit is None else bit.value for bit in output_bits]
    itholds = [thold.value for thold in input_thresholds]
    otholds = [thold.value for thold in output_thresholds]

    # choose scale of the one with max threshold
    idx = np.argmax(itholds)
    chosen_thold = itholds[idx]
    chosen_bit = ibits[idx]
    unified_scale = itholds[idx] / (2 ** (ibits[idx] - sign_bit))

    print('  in bits   : {}'.format(ibits))
    # print('  out bits  : {}'.format(obits))
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
    return tvm.ir.register_op_attr(op_name, "FHagoRealize", frealize, level)

def forward_op(ref_call, args):
    """forward the operator of ref_call with provided arguments"""
    return relay.Call(
        ref_call.op, args, ref_call.attrs, ref_call.type_args)


# TODO(ziheng) change to op_desc in the future
@register_realize("add")
def realize_addition(node, in_types, out_types):
    lhs, rhs = node.args
    from tvm.runtime import DataType
    dtype = out_types[0]
    if in_types[0] != dtype:
        lhs = relay.cast(lhs, DataType(dtype))
    if in_types[1] != dtype:
        rhs = relay.cast(rhs, DataType(dtype))
    return forward_op(node, [lhs, rhs])


@register_realize("nn.conv2d")
def realize_conv2d(node, in_types, out_types):
    data, weight = node.args
    fields = node.attrs.list_field_info()
    attrs_dict = {}
    for field in fields:
        key = field.name
        attrs_dict[str(key)] = getattr(node.attrs, key)
    attrs_dict['out_dtype'] = DataType(out_types[0])
    attrs = tvm.ir.make_node("relay.attrs.Conv2DAttrs", **attrs_dict)
    return relay.Call(node.op, node.args, attrs, node.type_args)
