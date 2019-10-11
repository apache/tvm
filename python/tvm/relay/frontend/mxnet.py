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
# pylint: disable=invalid-name, import-self, len-as-condition, no-else-return
"""MXNet symbol frontend."""
from __future__ import absolute_import as _abs

import json
import tvm
from .. import analysis
from .. import expr as _expr
from .. import op as _op
from .. import module as _module
from ... import nd as _nd

from .common import StrAttrsDict
from .common import infer_type as _infer_type
from .nnvm_common import _rename, _binop_scalar, _rbinop_scalar, _reduce
from .nnvm_common import _arg_reduce, _init_op, _softmax_op, _cast
from .nnvm_common import _clip, _transpose, _upsampling
from .nnvm_common import _elemwise_sum, _reshape
from .nnvm_common import _warn_not_used

__all__ = ['from_mxnet']

_activation_map = {
    "sigmoid": _op.sigmoid,
    "tanh"   : _op.tanh,
    "relu"   : _op.nn.relu
}

def _mx_fully_connected(inputs, attrs):
    import mxnet as mx
    units = attrs.get_int("num_hidden")
    use_bias = not attrs.get_bool("no_bias", False)
    try:
        _ = mx.sym.FullyConnected(mx.sym.var("x"), num_hidden=1, flatten=True)
        has_flatten = True
    except mx.base.MXNetError:
        # no flatten attribute in old mxnet
        has_flatten = False
    use_flatten = attrs.get_bool("flatten", True)
    if has_flatten and use_flatten:
        inputs[0] = _op.nn.batch_flatten(inputs[0])
    data_shape = _infer_type(inputs[0]).checked_type.shape
    if len(data_shape) > 2:
        inputs[0] = _op.reverse_reshape(inputs[0], [-1, 0])
    res = _op.nn.dense(inputs[0], inputs[1], units=units)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=-1)
    if len(data_shape) > 2:
        new_shape = data_shape[:-1]
        new_shape.append(units)
        res = _op.reshape(res, new_shape)
    return res


def _get_channel_axis(layout, op_name):
    if layout == "NCHW":
        return 1
    if layout == "NHWC":
        return 3
    raise tvm.error.OpAttributeInvalid(
        'Value {} in attribute "layout" of operator {} is not valid.'.format(layout, op_name))


def _mx_activations(inputs, attrs):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 1
    if act_type == "softrelu":
        def _stable_softrelu(x):
            # log(1 + exp(-abs(x))) + relu(x)
            one = _expr.const(1, dtype="float32")
            exp_neg_abs_x = _op.exp(_op.negative(_op.abs(x)))
            return _op.add(_op.log(_op.add(one, exp_neg_abs_x)),
                           _op.nn.relu(x))
        return _stable_softrelu(inputs[0])
    if act_type not in _activation_map:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend MXNet.'.format(act_type))
    return _activation_map[act_type](inputs[0])


def _mx_compare(new_op, wrapper):
    def impl(inputs, attrs):
        expr = _infer_type(inputs[0])
        dtype = expr.checked_type.dtype
        return wrapper(new_op)(inputs, attrs).astype(dtype)
    return impl


def _mx_zeros(inputs, attrs):
    assert len(inputs) == 0
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_str("dtype", "float32")
    if 0 in shape:
        return None
    return _op.zeros(shape=shape, dtype=dtype)


def _mx_conv(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) == 2:
        return _mx_conv2d(inputs, attrs)
    elif len(kernel_size) == 1:
        return _mx_conv1d(inputs, attrs)
    else:
        raise tvm.error.OpAttributeInvalid(
            '1D or 2D kernels only are supported for operator Convolution')

def _mx_conv1d(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 1:
        raise tvm.error.OpAttributeInvalid(
            'Non 1D or 2D kernels are not supported for operator Convolution')
    data_layout = attrs.get_str("layout", "NCW")
    # MXNet Conv1D only supports ‘NCW’ layout for now.
    if data_layout != "NCW":
        raise tvm.error.OpAttributeInvalid(
            'Only "NCW" data layout is supported for 1D Convolution')
    data_layout = "NCHW"
    channel_axis = 1
    kernel_layout = "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = (1,) + kernel_size
    new_attrs["strides"] = (1,) + attrs.get_int_tuple("stride", (1,))
    new_attrs["padding"] = (0,) + attrs.get_int_tuple("pad", (0,))
    new_attrs["dilation"] = (1,) +  attrs.get_int_tuple("dilate", (1,))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", False)
    data = _op.expand_dims(inputs[0], axis=2)
    kernel = _op.expand_dims(inputs[1], axis=2)
    res = _op.nn.conv2d(data, kernel, **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    res = _op.squeeze(res, axis=[2])
    return res


def _mx_conv2d(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise tvm.error.OpAttributeInvalid(
            'Non 1D or 2D kernels are not supported for operator Convolution')
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv2d(inputs[0], inputs[1], **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_conv_transpose(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) == 2:
        return _mx_conv2d_transpose(inputs, attrs)
    elif len(kernel_size) == 1:
        return _mx_conv1d_transpose(inputs, attrs)
    else:
        raise tvm.error.OpAttributeInvalid(
            '1D or 2D kernels only are supported for operator Convolution')


def _mx_conv1d_transpose(inputs, attrs):
    if "target_shape" in attrs.attrs:
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "target_shape" is not supported for operator Conv2D-transpose.')
    data_layout = attrs.get_str("layout", "NCW")
    if data_layout != "NCW":
        raise tvm.error.OpAttributeInvalid(
            'Only "NCW" data layout is supported for 1D Convolution')
    data_layout = "NCHW"
    channel_axis = 1
    kernel_layout = "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = (1,) + attrs.get_int_tuple("kernel")
    new_attrs["strides"] = (1,) + attrs.get_int_tuple("stride", (1,))
    new_attrs["output_padding"] = (0,) + attrs.get_int_tuple("adj", (0,))
    new_attrs["padding"] = (0,) + attrs.get_int_tuple("pad", (0,))
    new_attrs["dilation"] = (1,) +  attrs.get_int_tuple("dilate", (1,))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", True)
    data = _op.expand_dims(inputs[0], axis=2)
    kernel = _op.expand_dims(inputs[1], axis=2)
    res = _op.nn.conv2d_transpose(data, kernel, **new_attrs)

    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    res = _op.squeeze(res, axis=[2])
    return res


def _mx_conv2d_transpose(inputs, attrs):
    if "target_shape" in attrs.attrs:
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "target_shape" is not supported for operator Conv2D-transpose.')
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise tvm.error.OpAttributeInvalid(
            'Non-2D kernels are not supported for operator Conv2D-transpose.')
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d_transpose")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
    new_attrs["output_padding"] = attrs.get_int_tuple("adj", (0, 0))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", True)
    res = _op.nn.conv2d_transpose(inputs[0], inputs[1], **new_attrs)

    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_pooling(inputs, attrs):
    global_pool = attrs.get_bool("global_pool", False)
    pool_type = attrs.get_str("pool_type")

    def _pool2d(new_op, is_avg):
        kernel_size = attrs.get_int_tuple("kernel")
        if len(kernel_size) != 2:
            raise tvm.error.OpAttributeInvalid(
                'Only 2D kernels are supported for operator Pool2D.')
        new_attrs = {}
        new_attrs["pool_size"] = kernel_size
        new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["ceil_mode"] = (attrs.get_str("pooling_convention", "valid") == "full")
        if is_avg:
            new_attrs["count_include_pad"] = attrs.get_bool("count_include_pad", True)
        return new_op(inputs[0], **new_attrs)

    if pool_type == "max":
        if global_pool:
            return _op.nn.global_max_pool2d(inputs[0])
        return _pool2d(_op.nn.max_pool2d, False)
    if pool_type == "avg":
        if global_pool:
            return _op.nn.global_avg_pool2d(inputs[0])
        return _pool2d(_op.nn.avg_pool2d, True)
    raise tvm.error.OpNotImplemented(
        'Operator {} Pooling is not supported for frontend MXNet.'.format(pool_type.capitalize()))


def _mx_adaptive_avg_pooling(inputs, attrs):
    output_size = attrs.get_int_tuple("output_size", [])
    return _op.contrib.adaptive_avg_pool2d(inputs[0], output_size)


def _mx_dropout(inputs, attrs):
    rate = attrs.get_float("p", 0.5)
    return _op.nn.dropout(inputs[0], rate=rate)


def _mx_BlockGrad(inputs, attrs): #pylint: disable=unused-argument
    return inputs


def _mx_batch_norm(inputs, attrs):
    if attrs.get_bool("output_mean_var", False):
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "output_mean_var" is not supported for operator Batch Norm.')
    if attrs.get_bool("use_global_stats", False):
        _warn_not_used("use_global_stats", "batch_norm")
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", 1)
    new_attrs["epsilon"] = attrs.get_float("eps", 0.001)
    new_attrs["center"] = True
    new_attrs["scale"] = not attrs.get_bool("fix_gamma", True)
    return _op.nn.batch_norm(*inputs, **new_attrs)


def _mx_instance_norm(inputs, attrs):
    assert len(inputs) == 3
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", 1)
    new_attrs["epsilon"] = attrs.get_float("eps", 1e-5)
    return _op.nn.instance_norm(*inputs, **new_attrs)


def _mx_layer_norm(inputs, attrs):
    assert len(inputs) == 3
    if attrs.get_bool("output_mean_var", False):
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "output_mean_var" is not supported for operator Layer Norm.')
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", -1)
    new_attrs["epsilon"] = attrs.get_float("eps", 1e-5)
    return _op.nn.layer_norm(*inputs, **new_attrs)


def _mx_slice(inputs, attrs):
    new_attrs = {}
    begin = list(attrs.get_int_tuple('begin', None))
    end = list(attrs.get_int_tuple('end', None))
    stride = attrs.get_int_tuple('step', None)
    if begin is None:
        raise tvm.error.OpAttributeRequired(
            'Attribute "begin" not found in operator Slice.')
    if end is None:
        raise tvm.error.OpAttributeRequired(
            'Attribute "end" not found in operator Slice.')
    begin = tuple(x if x is not None else 0 for x in begin)
    new_attrs = {'begin': begin, 'end': end}
    if stride is not None:
        new_attrs['strides'] = stride
    return _op.strided_slice(inputs[0], **new_attrs)


def _mx_slice_like(inputs, attrs):
    assert len(inputs) == 2
    new_attrs = {}
    new_attrs["axes"] = attrs.get_int_tuple("axes", None)
    return _op.slice_like(*inputs, **new_attrs)


def _mx_slice_axis(inputs, attrs):
    assert len(inputs) == 1
    expr = _infer_type(inputs[0])
    shape = expr.checked_type.shape
    axis = attrs.get_int("axis")
    ax_beg = attrs.get_int("begin")
    ax_end = attrs.get_str("end")
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape)
    if ax_end == "None":
        ax_end = int(shape[axis])
    else:
        ax_end = int(ax_end)
    if ax_beg < 0:
        ax_beg += int(shape[axis])
    if ax_end < 0:
        ax_end += int(shape[axis])
    assert 0 <= ax_beg < int(shape[axis])
    assert ax_beg < ax_end <= int(shape[axis])
    begin = []
    end = []
    for i, dim in enumerate(shape):
        if i != axis:
            begin.append(0)
            end.append(dim)
        else:
            begin.append(ax_beg)
            end.append(ax_end)
    return _op.strided_slice(inputs[0], begin, end)


def _mx_crop_like(inputs, attrs):
    if len(inputs) < 2:
        raise tvm.error.OpAttributeUnimplemented(
            "Only support crop_like pattern for operator Crop.")
    if attrs.get_bool("center_crop", False):
        raise tvm.error.OpAttributeUnimplemented(
            "Center crop is not supported in operator Crop.")
    if attrs.get_int_tuple("h_w", (0, 0)) != (0, 0):
        raise tvm.error.OpAttributeUnimplemented(
            "Doesn't support h_w in operator Crop.")
    offset = attrs.get_int_tuple("offset", (0, 0))
    new_attrs = {}
    if offset == (0, 0):
        new_attrs["axes"] = (2, 3)
        return _op.slice_like(*inputs, **new_attrs)
    expr = _infer_type(inputs[1])
    like_shape = expr.checked_type.shape
    new_attrs['begin'] = [0, 0, offset[0], offset[1]]
    new_attrs['end'] = [like_shape[0], like_shape[1], offset[0]+like_shape[2],
                        offset[1]+like_shape[3]]
    return _op.strided_slice(inputs[0], **new_attrs)


def _mx_split(inputs, attrs):
    axis = attrs.get_int("axis", 1)
    new_attrs = {}
    new_attrs["indices_or_sections"] = attrs.get_int("num_outputs")
    new_attrs["axis"] = axis
    res = _op.split(inputs[0], **new_attrs)
    if attrs.get_bool("squeeze_axis", False):
        return tuple([_op.squeeze(x, axis=[axis]) for x in res])
    return res


def _mx_softmax_activation(inputs, attrs):
    mode = attrs.get_str("mode", "instance")
    axis = 0 if mode == "instance" else 1
    return _op.nn.softmax(inputs[0], axis=axis)


def _mx_softmax_output(inputs, attrs):
    if attrs.get_bool("multi_output", False):
        return _op.nn.softmax(inputs[0], axis=1)
    return _op.nn.softmax(inputs[0])


def _mx_linear_regression_output(inputs, _):
    return inputs[0]


def _mx_concat(inputs, attrs):
    axis = attrs.get_int("dim", 1)
    return _op.concatenate(tuple(inputs), axis=axis)


def _mx_stack(inputs, attrs):
    axis = attrs.get_int("axis", 0)
    return _op.stack(tuple(inputs), axis=axis)


def _mx_expand_dims(inputs, attrs):
    axis = attrs.get_int("axis")
    return _op.expand_dims(inputs[0], axis=axis)

def _mx_pad(inputs, attrs):
    pad_mode = attrs.get_str('mode', None)
    if pad_mode is None:
        raise tvm.error.OpAttributeRequired(
            'Attribute "mode" not found in operator pad.')
    if pad_mode not in ['constant', 'edge', 'reflect']:
        raise tvm.error.OpAttributeInvalid(
            'Value ' + mode + ' in attribute "mode" is not valid')
    pad_width = attrs.get_int_tuple('pad_width', None)
    if pad_width is None:
        raise tvm.error.OpAttributeRequired(
            'Attribute "pad_width" not found in operator pad.')
    if None in pad_width:
        raise tvm.error.OpAttributeInvalid(
            'Value None in attribute "pad_width" of operator Slice is not valid.')
    constant_value = attrs.get_float('constant_value', 0.0)
    padding = tuple(tuple((b, a)) for b, a in zip(pad_width[::2], pad_width[1::2]))
    return _op.nn.pad(data=inputs[0],
                      pad_width=padding,
                      pad_value=constant_value,
                      pad_mode=pad_mode)

def _mx_leaky_relu(inputs, attrs):
    act_type = attrs.get_str("act_type")
    if act_type == "leaky":
        return _op.nn.leaky_relu(inputs[0], alpha=attrs.get_float("slope", 0.25))
    if act_type == "prelu":
        assert len(inputs) == 2
        return _op.nn.prelu(*inputs)
    if act_type == "elu":
        # -slope * relu(1-exp(x)) + relu(x)
        slope = attrs.get_float("slope", 0.25)
        one = _expr.const(1, dtype="float32")
        x = inputs[0]
        mslope = _op.nn.relu(_op.subtract(one, _op.exp(x)))
        mslope = _op.multiply(mslope, _expr.const(-slope, dtype="float32"))
        return _op.add(mslope, _op.nn.relu(x))
    if act_type == "rrelu":
        # NOTE this is only converted for inference.
        lower_bound = attrs.get_float("lower_bound")
        upper_bound = attrs.get_float("upper_bound")
        alpha = (lower_bound + upper_bound) / 2.0
        return _op.nn.leaky_relu(inputs[0], alpha=alpha)
    raise tvm.error.OpNotImplemented(
        'Operator {} is not supported for frontend MXNet.'.format(act_type))


def _mx_make_power(power):
    def _impl(inputs, _):  # Note: no attrs
        assert len(inputs) == 1
        scalar = _expr.const(power, dtype=None)
        # Note: int maps to "int32", float maps to "float32"
        return _op.power(inputs[0], scalar)
    return _impl


def _mx_make_exponent(base):
    # exp(b, x) = e^b * e^x
    def _impl(inputs, _):  # Note: no attrs
        assert len(inputs) == 1
        scalar = _op.exp(_expr.const(base, dtype="float32"))
        return _op.multiply(inputs[0], scalar)
    return _impl


def _mx_make_logarithm(base):
    # log(b, x) = log(x) / log(b)
    def _impl(inputs, _):  # Note: no attrs
        assert len(inputs) == 1
        scalar = _op.log(_expr.const(base, dtype="float32"))
        return _op.divide(inputs[0], scalar)
    return _impl


def _mx_expm1():
    # exp_minus_1 x = exp(x) - 1
    def _impl(inputs, _):  # Note: no attrs
        assert len(inputs) == 1
        one = _expr.const(1, dtype="float32")
        return _op.log(_op.subtract(inputs[0], one))
    return _impl


def _mx_log1p():
    # 1_plus_log x = log(x + 1)
    def _impl(inputs, _):  # Note: no attrs
        assert len(inputs) == 1
        one = _expr.const(1, dtype="float32")
        return _op.log(_op.add(inputs[0], one))
    return _impl


def _mx_lrn(inputs, attrs):
    new_attrs = {}
    new_attrs["alpha"] = attrs.get_float("alpha", 0.0001)
    new_attrs["beta"] = attrs.get_float("beta", 0.75)
    new_attrs["bias"] = attrs.get_float("knorm", 2)
    # NCHW format and normalization along channel axis
    new_attrs["axis"] = 1
    new_attrs["size"] = attrs.get_int("nsize")
    assert len(inputs) == 1
    return _op.nn.lrn(inputs[0], **new_attrs)


def _mx_multibox_prior(inputs, attrs):
    new_attrs = {}
    new_attrs["sizes"] = attrs.get_float_tuple("sizes", (1.0, ))
    new_attrs["steps"] = attrs.get_float_tuple("steps", (-1.0, -1.0))
    new_attrs["offsets"] = attrs.get_float_tuple("offsets", (0.5, 0.5))
    new_attrs["ratios"] = attrs.get_float_tuple("ratios", (1.0, ))
    new_attrs["clip"] = attrs.get_bool("clip", False)
    return _op.vision.multibox_prior(inputs[0], **new_attrs)


def _mx_multibox_detection(inputs, attrs):
    new_attrs0 = {}
    new_attrs0["clip"] = attrs.get_bool("clip", True)
    new_attrs0["threshold"] = attrs.get_float("threshold", 0.01)
    new_attrs0["variances"] = attrs.get_float_tuple("variances", (0.1, 0.1,
                                                                  0.2, 0.2))

    new_attrs1 = {}
    new_attrs1["return_indices"] = False
    new_attrs1["iou_threshold"] = attrs.get_float("nms_threshold", 0.5)
    new_attrs1["force_suppress"] = attrs.get_bool("force_suppress", False)
    new_attrs1["top_k"] = attrs.get_int("nms_topk", -1)

    ret = _op.vision.multibox_transform_loc(inputs[0], inputs[1],
                                            inputs[2], **new_attrs0)
    return _op.vision.non_max_suppression(ret[0], ret[1], **new_attrs1)


def _mx_batch_dot(inputs, attrs):
    assert len(inputs) == 2
    a, b = inputs
    transpose_a = attrs.get_bool("transpose_a", False)
    transpose_b = attrs.get_bool("transpose_b", False)
    if transpose_a is True:
        msg = 'Value {} in attribute "transpose_a" of operator batch_dot ' \
              'is not valid.'
        raise tvm.error.OpAttributeInvalid(msg.format(transpose_a))
    if transpose_b is False:
        b = _op.transpose(b, axes=[0, 2, 1])
    return _op.nn.batch_matmul(a, b)


def _mx_arange(inputs, attrs):
    assert len(inputs) == 0
    if attrs.get_int("repeat", 1) != 1:
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "repeat" is not supported in operator arange.')
    dtype = attrs.get_str("dtype", "float32")
    stop = attrs.get_str("stop", "None")
    if stop == "None":
        stop = None
    else:
        stop = _expr.const(float(stop), dtype=dtype)
    new_attrs = {}
    new_attrs["start"] = _expr.const(attrs.get_float("start", 0.0), dtype=dtype)
    new_attrs["stop"] = stop
    new_attrs["step"] = _expr.const(attrs.get_float("step", 1.0), dtype=dtype)
    new_attrs["dtype"] = dtype
    return _op.arange(**new_attrs)


def _mx_repeat(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["repeats"] = attrs.get_int("repeats")
    new_attrs["axis"] = attrs.get_int("axis", 0)
    return _op.repeat(inputs[0], **new_attrs)


def _mx_tile(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["reps"] = attrs.get_int_tuple("reps")
    return _op.tile(inputs[0], **new_attrs)


def _mx_take(inputs, attrs):
    assert len(inputs) == 2
    mode = attrs.get_str("mode", "clip")
    if mode == "raise":
        raise tvm.error.OpAttributeUnimplemented("take with raise mode is not supported yet")
    axis = attrs.get_int("axis", 0)
    return _op.take(inputs[0], inputs[1].astype("int32"), axis, mode)


def _mx_reverse(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis")
    return _op.reverse(inputs[0], **new_attrs)


def _mx_roi_align(inputs, attrs):
    new_attrs = {}
    new_attrs["pooled_size"] = attrs.get_int_tuple("pooled_size")
    new_attrs["spatial_scale"] = attrs.get_float("spatial_scale")
    new_attrs["sample_ratio"] = attrs.get_int("sample_ratio", -1)
    new_attrs["layout"] = "NCHW"
    return _op.vision.roi_align(inputs[0], inputs[1], **new_attrs)

def _mx_resize(inputs, attrs):
    scale_height = attrs.get_float("scale_height", None)
    scale_width = attrs.get_float("scale_width", None)
    height = attrs.get_int("height", 1)
    width = attrs.get_int("width", 1)
    expr = _infer_type(inputs[0])
    shape = expr.checked_type.shape
    if scale_height is not None:
        height = (scale_height * shape[2]).astype("int32")
    if scale_width is not None:
        width = (scale_width * shape[3]).astype("int32")
    size = (height, width)
    return _op.image.resize(inputs[0], size, align_corners=True)

def _mx_roi_pooling(inputs, attrs):
    new_attrs = {}
    new_attrs["pooled_size"] = attrs.get_int_tuple("pooled_size")
    new_attrs["spatial_scale"] = attrs.get_float("spatial_scale")
    new_attrs["layout"] = "NCHW"
    return _op.vision.roi_pool(inputs[0], inputs[1], **new_attrs)


def _mx_proposal(inputs, attrs):
    new_attrs = {}
    new_attrs["scales"] = attrs.get_float_tuple("scales", (4.0, 8.0, 16.0, 32.0))
    new_attrs["ratios"] = attrs.get_float_tuple("ratios", (0.5, 1.0, 2.0))
    new_attrs["feature_stride"] = attrs.get_int("feature_stride", 16)
    new_attrs["threshold"] = attrs.get_float("threshold", 0.7)
    new_attrs["rpn_pre_nms_top_n"] = attrs.get_int("rpn_pre_nms_top_n", 6000)
    new_attrs["rpn_post_nms_top_n"] = attrs.get_int("rpn_post_nms_top_n", 300)
    new_attrs["rpn_min_size"] = attrs.get_int("rpn_min_size", 16)
    new_attrs["iou_loss"] = attrs.get_bool("iou_loss", False)
    assert not attrs.get_bool("output_score", False), "proposal doesn't support output score"
    return _op.vision.proposal(inputs[0], inputs[1], inputs[2], **new_attrs)


def _mx_box_nms(inputs, attrs):
    force_suppress = attrs.get_bool("force_suppress", False)
    iou_thresh = attrs.get_float('overlap_thresh', 0.5)
    top_k = attrs.get_int('topk', -1)
    valid_thresh = attrs.get_float('valid_thresh', 0)
    coord_start = attrs.get_int('coord_start', 2)
    score_index = attrs.get_int('score_index', 1)
    id_index = attrs.get_int('id_index', -1)
    in_format = attrs.get_str('in_format', 'corner')
    out_format = attrs.get_str('out_format', 'corner')
    if in_format != 'corner':
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "in_format" must equal "corner" for operator box_nms.')
    if out_format != 'corner':
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "out_format" must equal "corner" for operator box_nms.')

    ret = _op.vision.get_valid_counts(inputs[0], score_threshold=valid_thresh,
                                      id_index=id_index, score_index=score_index)
    nms_out = _op.vision.non_max_suppression(ret[1],
                                             ret[0],
                                             iou_threshold=iou_thresh,
                                             force_suppress=force_suppress,
                                             top_k=top_k,
                                             coord_start=coord_start,
                                             score_index=score_index,
                                             id_index=id_index,
                                             return_indices=False,
                                             invalid_to_bottom=True)
    return nms_out


def _mx_l2_normalize(inputs, attrs):
    new_attrs = {}
    mode = attrs.get_str('mode', 'instance')
    if mode != 'channel':
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "mode" must equal "channel" for operator l2_normalize.')
    new_attrs['eps'] = attrs.get_float('eps', 1e-10)
    new_attrs['axis'] = [1]
    return _op.nn.l2_normalize(inputs[0], **new_attrs)


def _mx_shape_array(inputs, attrs):
    assert len(inputs) == 1
    if attrs.get_int("lhs_begin", None) is not None:
        raise tvm.error.OpAttributeUnimplemented("shape_array doesn't support lhs_begin")
    if attrs.get_int("lhs_end", None) is not None:
        raise tvm.error.OpAttributeUnimplemented("shape_array doesn't support lhs_end")
    if attrs.get_int("rhs_begin", None) is not None:
        raise tvm.error.OpAttributeUnimplemented("shape_array doesn't support rhs_begin")
    if attrs.get_int("rhs_end", None) is not None:
        raise tvm.error.OpAttributeUnimplemented("shape_array doesn't support rhs_end")
    return _op.shape_of(inputs[0], dtype='int64')


def _mx_full(inputs, attrs):
    assert len(inputs) == 0
    val = attrs.get_float("value")
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_str("dtype", "float32")
    return _op.full(_expr.const(val, dtype), shape, dtype)


def _mx_squeeze(inputs, attrs):
    assert len(inputs) == 1
    axis = attrs.get_int_tuple("axis", None)
    return _op.squeeze(inputs[0], axis)


def _mx_broadcast_axis(inputs, attrs):
    assert len(inputs) == 1
    axis = attrs.get_int_tuple("axis", [])
    size = attrs.get_int_tuple("size", [])
    assert len(axis) == len(size)
    if len(axis) == 0:
        return inputs[0]
    expr = _infer_type(inputs[0])
    src_shape = expr.checked_type.shape
    tgt_shape = []
    for i, dim in enumerate(src_shape):
        if i not in axis:
            tgt_shape.append(dim)
        else:
            assert int(dim) == 1
            idx = axis.index(i)
            tgt_shape.append(size[idx])
    return _op.broadcast_to(inputs[0], tgt_shape)


def _mx_embedding(inputs, _):
    assert len(inputs) == 2
    indices, weight = inputs
    return _op.take(weight, indices.astype('int32'), axis=0)


def _mx_smooth_l1(inputs, attrs):
    scalar = attrs.get_float("scalar", 1.0)
    scalar_sq = scalar * scalar
    mask = _op.less(inputs[0], _expr.const(1.0 / scalar_sq, dtype='float32'))
    return _op.where(mask,
                     _expr.const(scalar_sq / 2.0, dtype='float32') * inputs[0] * inputs[0],
                     _op.abs(inputs[0]) - _expr.const(0.5 / scalar_sq))


def _mx_deformable_convolution(inputs, attrs):
    new_attrs = {}
    assert attrs.get_bool("no_bias")
    new_attrs["kernel_size"] = attrs.get_int_tuple("kernel")
    new_attrs["strides"] = attrs.get_int_tuple("stride")
    new_attrs["padding"] = attrs.get_int_tuple("pad")
    new_attrs["dilation"] = attrs.get_int_tuple("dilate")
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["deformable_groups"] = attrs.get_int("num_deformable_group", 1)
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    assert attrs.get_str("layout", "NCHW") == "NCHW", "Deformable conv2d only supports NCHW layout"
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.deformable_conv2d(inputs[0], inputs[1], inputs[2], **new_attrs)
    if use_bias:
        assert len(inputs) == 4
        res = _op.nn.bias_add(res, inputs[3])
    return res


def _mx_argsort(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", -1)
    new_attrs["is_ascend"] = attrs.get_bool("is_ascend", True)
    new_attrs["dtype"] = attrs.get_str("dtype", "float32")
    return _op.argsort(inputs[0], **new_attrs)


def _mx_topk(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["k"] = attrs.get_int("k", 1)
    new_attrs["axis"] = attrs.get_int("axis", -1)
    new_attrs["is_ascend"] = attrs.get_bool("is_ascend", True)
    ret_type = attrs.get_str("ret_typ", "indices")
    if ret_type == "mask":
        raise tvm.error.OpAttributeUnimplemented(
            "Attribute ret_type=mask is not supported in topk operator")
    new_attrs["ret_type"] = "values" if ret_type == "value" else ret_type
    new_attrs["dtype"] = attrs.get_str("dtype", "float32")
    return _op.topk(inputs[0], **new_attrs)


def _mx_sequence_mask(inputs, attrs):
    assert len(inputs) == 1 or len(inputs) == 2
    new_attrs = {}
    use_sequence_length = attrs.get_bool('use_sequence_length', False)
    new_attrs['mask_value'] = attrs.get_float('value', 0.0)
    new_attrs['axis'] = attrs.get_int('axis', 0)
    if use_sequence_length:
        return _op.sequence_mask(*inputs, **new_attrs)
    else:
        return inputs[0]


def _mx_contrib_div_sqrt_dim(inputs, _):
    assert len(inputs) == 1
    ndim = len(_infer_type(inputs[0]).checked_type.shape)
    dim = _op.take(_op.shape_of(inputs[0]), _expr.const(ndim-1, dtype="int32"))
    dtype = _infer_type(inputs[0]).checked_type.dtype
    sqrt_dim = _op.sqrt(dim.astype(dtype))
    out = inputs[0] / sqrt_dim
    return out


def _mx_rnn_param_concat(inputs, _):
    # We don't need to concatenate RNN params because we will unravel the RNN op
    return [inputs]


def _mx_rnn_layer(inputs, attrs):
    def _rnn_cell(data, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias, activation):
        i2h = _op.nn.bias_add(_op.nn.dense(data, i2h_weight), i2h_bias, axis=-1)
        h2h = _op.nn.bias_add(_op.nn.dense(states[0], h2h_weight), h2h_bias, axis=-1)
        out = _activation_map[activation](i2h + h2h)
        return out, [out]

    def _gru_cell(data, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        expr = _infer_type(data)
        dtype = expr.checked_type.dtype
        i2h = _op.nn.bias_add(_op.nn.dense(data, i2h_weight), i2h_bias, axis=-1)
        h2h = _op.nn.bias_add(_op.nn.dense(states[0], h2h_weight), h2h_bias, axis=-1)
        i2h_r, i2h_z, i2h = _op.split(i2h, indices_or_sections=3, axis=1)
        h2h_r, h2h_z, h2h = _op.split(h2h, indices_or_sections=3, axis=1)
        reset_gate = _activation_map["sigmoid"](i2h_r + h2h_r)
        update_gate = _activation_map["sigmoid"](i2h_z + h2h_z)
        next_h_tmp = _activation_map["tanh"](reset_gate * h2h + i2h)
        next_h = (_expr.const(1, dtype) - update_gate) * next_h_tmp + update_gate * states[0]
        return next_h, [next_h]

    def _lstm_cell(data, states, i2h_weight, h2h_weight, i2h_bias, h2h_bias):
        i2h = _op.nn.bias_add(_op.nn.dense(data, i2h_weight), i2h_bias, axis=-1)
        h2h = _op.nn.bias_add(_op.nn.dense(states[0], h2h_weight), h2h_bias, axis=-1)
        gates = i2h + h2h
        slice_gates = _op.split(gates, indices_or_sections=4, axis=1)
        in_gate = _activation_map["sigmoid"](slice_gates[0])
        forget_gate = _activation_map["sigmoid"](slice_gates[1])
        in_transform = _activation_map["tanh"](slice_gates[2])
        out_gate = _activation_map["sigmoid"](slice_gates[3])
        next_c = forget_gate * states[1] + in_gate * in_transform
        next_h = out_gate * _activation_map["tanh"](next_c)
        return next_h, [next_h, next_c]

    num_layers = attrs.get_int("num_layers", 1)
    mode = attrs.get_str("mode")
    output_states = attrs.get_bool("state_outputs", False)
    if mode.startswith("rnn"):
        mode, activation = mode.split('_')
    assert mode in ["rnn", "gru", "lstm"]
    bidirectional = attrs.get_bool("bidirectional", False)
    direct = 2 if bidirectional else 1
    layout = attrs.get_str("layout", "TNC")
    if layout != "TNC":
        raise tvm.error.OpAttributeUnimplemented(
            "RNN with layout other than TNC is not supported yet")
    num_states = 2 if mode == 'lstm' else 1
    assert len(inputs) == num_states + 2

    seq_data = inputs[0]
    concat_weight = inputs[1]
    init_states = inputs[2:]
    expr = _infer_type(seq_data)
    data_shape = expr.checked_type.shape
    seq_len = int(data_shape[0])
    assert len(concat_weight) == num_layers * 4 * direct

    for idx, state in enumerate(init_states[:]):
        if isinstance(state, dict):
            node = state
            attrs = StrAttrsDict(node.get("attrs", {}))
            op_name = node["op"]
            # by default, RNN layer uses zeros to initialize states
            assert op_name == "_zeros"
            shape = attrs.get_int_tuple("shape")
            dtype = attrs.get_str("dtype", "float32")
            init_layout = attrs.get_str("__layout__")
            new_shape = list(shape)
            for i, dim in enumerate(shape):
                if dim == 0:
                    axis = layout.find(init_layout[i])
                    assert axis >= 0
                    new_shape[i] = int(data_shape[axis])
            init_states[idx] = _op.zeros(new_shape, dtype)

    weights = []
    bias = []
    states = []
    back_weights = []
    back_bias = []
    back_states = []
    for i in range(num_layers):
        weights.append([concat_weight[i*2*direct].args[0],
                        concat_weight[i*2*direct + 1].args[0]])
        bias.append([concat_weight[(num_layers+i)*2*direct].args[0],
                     concat_weight[(num_layers+i)*2*direct + 1].args[0]])
        s = []
        for state in init_states:
            s.append(_op.take(state, _expr.const(i*direct, "int32"), axis=0))
        states.append(s)
        if bidirectional:
            back_weights.append([concat_weight[i*2*direct + 2].args[0],
                                 concat_weight[i*2*direct + 3].args[0]])
            back_bias.append([concat_weight[(num_layers+i)*2*direct + 2].args[0],
                              concat_weight[(num_layers+i)*2*direct + 3].args[0]])
            s = []
            for state in init_states:
                s.append(_op.take(state, _expr.const(i*direct+1, "int32"), axis=0))
            back_states.append(s)

    xs = [_op.take(seq_data, _expr.const(t, "int32"), axis=0) for t in range(seq_len)]
    for l in range(num_layers):
        outputs = []
        back_outputs = []
        for x in xs:
            if mode == "rnn":
                out, new_states = _rnn_cell(x, states[l], *weights[l], *bias[l], activation)
            elif mode == "gru":
                out, new_states = _gru_cell(x, states[l], *weights[l], *bias[l])
            else: # mode == "lstm"
                out, new_states = _lstm_cell(x, states[l], *weights[l], *bias[l])
            states[l] = new_states
            outputs.append(out)
        if bidirectional:
            for x in reversed(xs):
                if mode == "rnn":
                    out, new_states = _rnn_cell(
                        x, back_states[l], *back_weights[l], *back_bias[l], activation)
                elif mode == "gru":
                    out, new_states = _gru_cell(
                        x, back_states[l], *back_weights[l], *back_bias[l])
                else: # mode == "lstm"
                    out, new_states = _lstm_cell(
                        x, back_states[l], *back_weights[l], *back_bias[l])
                back_states[l] = new_states
                back_outputs.append(out)
            back_outputs.reverse()
            concat_outputs = []
            for t, out in enumerate(outputs):
                new_out = _op.concatenate([out, back_outputs[t]], axis=-1)
                concat_outputs.append(new_out)
            outputs = concat_outputs
        xs = outputs

    ret = [_op.stack(outputs, axis=0)]
    if output_states:
        for i in range(num_states):
            inputs = []
            for l, s in enumerate(states):
                inputs.append(s[i])
                if bidirectional:
                    inputs.append(back_states[l][i])
            ret.append(_op.stack(inputs, axis=0))
    return ret

def _mx_one_hot(inputs, attrs):
    indices = inputs[0].astype('int32')
    depth = attrs.get_int('depth', 0)
    dtype = attrs.get_str('dtype', 'int32')
    on_value = tvm.relay.const(attrs.get_float('on_value', 1.0), dtype)
    off_value = tvm.relay.const(attrs.get_float('off_value', 0.0), dtype)
    return _op.one_hot(indices, on_value, off_value, depth, -1, dtype)


def _mx_contrib_fifo_buffer(inputs, attrs):
    new_attrs = {}
    new_attrs['axis'] = attrs.get_int('axis')
    return _op.nn.fifo_buffer(*inputs, **new_attrs)


# Note: due to attribute conversion constraint
# ops in the identity set must be attribute free
_identity_list = [
    "log",
    "exp",
    "erf",
    "sqrt",
    "floor",
    "ceil",
    "sigmoid",
    "tanh",
    "negative",
    "reshape_like",
    "zeros_like",
    "ones_like",
    "where",
    "gather_nd",
    "cos",
    "sin"
]

_convert_map = {
    "_copy"                  : _rename(_op.copy),
    "relu"                   : _rename(_op.nn.relu),
    "broadcast_add"          : _rename(_op.add),
    "broadcast_sub"          : _rename(_op.subtract),
    "broadcast_mul"          : _rename(_op.multiply),
    "broadcast_div"          : _rename(_op.divide),
    "broadcast_mod"          : _rename(_op.mod),
    "broadcast_maximum"      : _rename(_op.maximum),
    "broadcast_minimum"      : _rename(_op.minimum),
    "arctan"                 : _rename(_op.atan),
    "broadcast_equal"        : _mx_compare(_op.equal, _rename),
    "broadcast_not_equal"    : _mx_compare(_op.not_equal, _rename),
    "broadcast_greater"      : _mx_compare(_op.greater, _rename),
    "broadcast_greater_equal": _mx_compare(_op.greater_equal, _rename),
    "broadcast_lesser"       : _mx_compare(_op.less, _rename),
    "broadcast_lesser_equal" : _mx_compare(_op.less_equal, _rename),
    "elemwise_add"           : _rename(_op.add),
    "elemwise_sub"           : _rename(_op.subtract),
    "elemwise_mul"           : _rename(_op.multiply),
    "elemwise_div"           : _rename(_op.divide),
    "_maximum"               : _rename(_op.maximum),
    "_minimum"               : _rename(_op.minimum),
    "flatten"                : _rename(_op.nn.batch_flatten),
    "Flatten"                : _rename(_op.nn.batch_flatten),
    # scalar power
    "square"                 : _mx_make_power(2),
    "rsqrt"                  : _mx_make_power(-1/2),
    "cbrt"                   : _mx_make_power(1/3),
    "rcbrt"                  : _mx_make_power(-1/3),
    "__pow_scalar__"         : _binop_scalar(_op.power),
    "_power_scalar"          : _binop_scalar(_op.power),
    "__rsub_scalar__"        : _rbinop_scalar(_op.subtract),
    "_rminus_scalar"         : _rbinop_scalar(_op.subtract),
    "__rdiv_scalar__"        : _rbinop_scalar(_op.divide),
    "_rdiv_scalar"           : _rbinop_scalar(_op.divide),
    "__rpow_scalar__"        : _rbinop_scalar(_op.power),
    # scalar op
    "__add_scalar__"         : _binop_scalar(_op.add),
    "_plus_scalar"           : _binop_scalar(_op.add),
    "__sub_scalar__"         : _binop_scalar(_op.subtract),
    "_minus_scalar"          : _binop_scalar(_op.subtract),
    "__mul_scalar__"         : _binop_scalar(_op.multiply),
    "_mul_scalar"            : _binop_scalar(_op.multiply),
    "__div_scalar__"         : _binop_scalar(_op.divide),
    "_div_scalar"            : _binop_scalar(_op.divide),
    "log2"                   : _mx_make_logarithm(2),
    "log10"                  : _mx_make_logarithm(10),
    "log1p"                  : _mx_log1p,
    "expm1"                  : _mx_expm1,
    "_equal_scalar"          : _mx_compare(_op.equal, _binop_scalar),
    "_not_equal_scalar"      : _mx_compare(_op.not_equal, _binop_scalar),
    "_greater_scalar"        : _mx_compare(_op.greater, _binop_scalar),
    "_greater_equal_scalar"  : _mx_compare(_op.greater_equal, _binop_scalar),
    "_lesser_scalar"         : _mx_compare(_op.less, _binop_scalar),
    "_lesser_equal_scalar"   : _mx_compare(_op.less_equal, _binop_scalar),
    "_maximum_scalar"        : _binop_scalar(_op.maximum),
    "_minimum_scalar"        : _binop_scalar(_op.minimum),
    # reduction ops
    "mean"          : _reduce(_op.mean),
    "max"           : _reduce(_op.max),
    "min"           : _reduce(_op.min),
    "sum"           : _reduce(_op.sum),
    "max_axis"      : _reduce(_op.max),
    "min_axis"      : _reduce(_op.min),
    "sum_axis"      : _reduce(_op.sum),
    "argmax"        : _arg_reduce(_op.argmax),
    "argmin"        : _arg_reduce(_op.argmin),
    # init ops
    "_ones"         : _init_op(_op.ones),
    # softmax
    "softmax"       : _softmax_op(_op.nn.softmax),
    "log_softmax"   : _softmax_op(_op.nn.log_softmax),
    "Softmax"       : _softmax_op(_op.nn.softmax),
    # per op specialization
    "Reshape"       : _reshape,
    "reshape"       : _reshape,
    "Cast"          : _cast,
    "clip"          : _clip,
    "transpose"     : _transpose,
    "UpSampling"    : _upsampling,
    "add_n"         : _elemwise_sum,
    # MXNet specific implementations
    "_zeros"        : _mx_zeros,
    "FullyConnected": _mx_fully_connected,
    "Activation"    : _mx_activations,
    "Convolution"   : _mx_conv,
    "Convolution_v1": _mx_conv2d,
    "Deconvolution" : _mx_conv_transpose,
    "Pooling"       : _mx_pooling,
    "Pooling_v1"    : _mx_pooling,
    "Dropout"       : _mx_dropout,
    "BatchNorm"     : _mx_batch_norm,
    "BatchNorm_v1"  : _mx_batch_norm,
    "InstanceNorm"  : _mx_instance_norm,
    "LayerNorm"     : _mx_layer_norm,
    "LRN"           : _mx_lrn,
    "L2Normalization"  : _mx_l2_normalize,
    "slice"         : _mx_slice,
    "slice_like"    : _mx_slice_like,
    "slice_axis"    : _mx_slice_axis,
    "SliceChannel"  : _mx_split,
    "split"         : _mx_split,
    "expand_dims"   : _mx_expand_dims,
    "Concat"        : _mx_concat,
    "concat"        : _mx_concat,
    "stack"         : _mx_stack,
    "batch_dot"     : _mx_batch_dot,
    "LeakyReLU"     : _mx_leaky_relu,
    "_arange"       : _mx_arange,
    "_full"         : _mx_full,
    "repeat"        : _mx_repeat,
    "tile"          : _mx_tile,
    "pad"           : _mx_pad,
    "Pad"           : _mx_pad,
    "take"          : _mx_take,
    "reverse"       : _mx_reverse,
    "squeeze"       : _mx_squeeze,
    "broadcast_axis": _mx_broadcast_axis,
    "BlockGrad"     : _mx_BlockGrad,
    "shape_array"   : _mx_shape_array,
    "Embedding"     : _mx_embedding,
    "argsort"       : _mx_argsort,
    "topk"          : _mx_topk,
    "SequenceMask"  : _mx_sequence_mask,
    "SoftmaxOutput" : _mx_softmax_output,
    "SoftmaxActivation" : _mx_softmax_activation,
    "LinearRegressionOutput" : _mx_linear_regression_output,
    "smooth_l1"     : _mx_smooth_l1,
    "_contrib_div_sqrt_dim": _mx_contrib_div_sqrt_dim,
    "one_hot"           : _mx_one_hot,
    # vision
    "_contrib_BilinearResize2D" : _mx_resize,
    "_contrib_MultiBoxPrior" : _mx_multibox_prior,
    "_contrib_MultiBoxDetection" : _mx_multibox_detection,
    "_contrib_ROIAlign" : _mx_roi_align,
    "ROIPooling"        : _mx_roi_pooling,
    "_contrib_Proposal" : _mx_proposal,
    "_contrib_MultiProposal" : _mx_proposal,
    "_contrib_box_nms" : _mx_box_nms,
    "_contrib_DeformableConvolution" : _mx_deformable_convolution,
    "_contrib_AdaptiveAvgPooling2D" : _mx_adaptive_avg_pooling,
    # NLP
    "RNN"               : _mx_rnn_layer,
    "_rnn_param_concat" : _mx_rnn_param_concat,
    # Depricated:
    "Crop"              : _mx_crop_like,
    # List of missing operators that are present in NNVMv1
    # TODO(tvm-tvm): support all operators.
    #
    # "broadcast_to",
    "contrib_fifo_buffer" : _mx_contrib_fifo_buffer,
}

# set identity list
_convert_map.update({k : _rename(k) for k in _identity_list})


def _from_mxnet_impl(symbol, shape_dict, dtype_info, mod=None):
    #pylint: disable=unused-argument
    """Convert mxnet symbol to compatible relay Function.

    Reconstruct a relay Function by traversing the mxnet symbol.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol
        Incompatible symbol from mxnet.
        The op_name and attrs inside are not always compatible.

    shape_dict : dict
        Known parameter shapes

    dtype_info : dict or str.
        Known parameter dtypes

    mod : tvm.relay.Module
        The module that contains global information. It will be used for
        converting ops that need global information, e.g. control-flow ops.

    Returns:
    -------
    func : tvm.relay.Function
        Converted relay Function
    """
    assert symbol is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}

    for nid, node in enumerate(jnodes):
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = StrAttrsDict(node.get("attrs", {}))
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            shape = shape_dict[node_name] if node_name in shape_dict else None
            if isinstance(dtype_info, dict):
                dtype = dtype_info[node_name] if node_name in dtype_info else "float32"
            else:
                dtype = dtype_info
            node_map[nid] = [_expr.var(node_name, shape=shape, dtype=dtype)]
        elif op_name in _convert_map:
            res = _convert_map[op_name](children, attrs)
            if res is None:
                # defer conversion, used in RNN state initialization
                res = [node]
            elif isinstance(res, (_expr.TupleWrapper, tuple, list)):
                pass
            elif isinstance(res, _expr.Expr):
                res = [res]
            else:
                raise RuntimeError("unexpected type %s" % type(res))
            node_map[nid] = res
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend MXNet.'.format(op_name))

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    func = _expr.Function(analysis.free_vars(outputs), outputs)
    return func


def _update_shape_dtype(shape, dtype, params):
    """Update shape dtype given params information"""
    shape = {} if shape is None else shape
    if not params:
        return shape, dtype
    shape = shape.copy()
    shape.update({k : v.shape for k, v in params.items()})
    if isinstance(dtype, str):
        for k, v in params.items():
            if v.dtype != dtype:
                raise ValueError(
                    "%s: dtype not expected %s vs %s" % (k, dtype, v.dtype))
    else:
        dtype = dtype.copy()
        dtype.update({k : str(v.dtype) for k, v in params.items()})
    return shape, dtype


def from_mxnet(symbol,
               shape=None,
               dtype="float32",
               arg_params=None,
               aux_params=None):
    """Convert from MXNet"s model into compatible relay Function.

    Parameters
    ----------
    symbol : mxnet.Symbol or mxnet.gluon.HybridBlock
        MXNet symbol.

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    arg_params : dict of str to mx.NDArray
        The argument parameters in mxnet

    aux_params : dict of str to mx.NDArray
        The auxiliary parameters in mxnet

    Returns
    -------
    mod : tvm.relay.Module
        The relay module for compilation

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import mxnet as mx
    except ImportError as e:
        raise ImportError("{}. MXNet is required to parse symbols.".format(e))

    mod = _module.Module()
    if isinstance(symbol, mx.sym.Symbol):
        params = {}
        arg_params = arg_params if arg_params else {}
        aux_params = aux_params if aux_params else {}
        for k, v in arg_params.items():
            params[k] = _nd.array(v.asnumpy())
        for k, v in aux_params.items():
            params[k] = _nd.array(v.asnumpy())
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        func = _from_mxnet_impl(symbol, shape, dtype, mod)
    elif isinstance(symbol, mx.gluon.HybridBlock):
        if arg_params is not None or aux_params is not None:
            raise ValueError("arg_params and aux_params ae not used when importing HybridBlock")
        params = {}
        for k, v in symbol.collect_params().items():
            params[k] = _nd.array(v.data().asnumpy())
        inputs = []
        for name in shape:
            inputs.append(mx.sym.Variable(name))
        sym = symbol(*inputs)
        if isinstance(sym, (list, tuple)):
            sym = mx.sym.Group(sym)
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        func = _from_mxnet_impl(sym, shape, dtype, mod)
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    mod["main"] = func
    return mod, params
