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
# pylint: disable=invalid-name, import-self, len-as-condition, no-else-return, too-many-lines
"""MXNet symbol frontend."""
import json
import math

import numpy as np
import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import scope_builder as _scope_builder
from .common import StrAttrsDict
from .common import get_name as _get_name
from .common import infer_shape as _infer_shape
from .common import infer_type as _infer_type
from .common import infer_value as _infer_value
from .mxnet_qnn_op_utils import (
    dequantize_mxnet_min_max,
    get_conv_mkldnn_requantized_scale_outDtype,
    get_mkldnn_int8_scale,
    get_mkldnn_requantize_scale_outDtype,
    get_mkldnn_uint8_scale,
    quantize_conv_bias_mkldnn_from_var,
    quantize_conv_weights_bias_channel_mkldnn_from_var,
    quantize_mxnet_min_max,
)
from .nnvm_common import (
    _arg_reduce,
    _binop_scalar,
    _cast,
    _clip,
    _elemwise_sum,
    _init_op,
    _rbinop_scalar,
    _reduce,
    _rename,
    _reshape,
    _softmax_op,
    _transpose,
    _upsampling,
    _warn_not_used,
)

__all__ = ["from_mxnet"]

_activation_map = {"sigmoid": _op.sigmoid, "tanh": _op.tanh, "relu": _op.nn.relu}


def _mx_fully_connected(inputs, attrs):
    import mxnet as mx  # pylint: disable=import-outside-toplevel

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
    if layout in ["NCHW", "NCDHW"]:
        return 1
    if layout == "NHWC":
        return 3
    if layout == "NDHWC":
        return 4
    raise tvm.error.OpAttributeInvalid(
        'Value {} in attribute "layout" of operator {} is not valid.'.format(layout, op_name)
    )


def _mx_activations(inputs, attrs):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 1
    if act_type == "softrelu":

        def _stable_softrelu(x):
            # log(1 + exp(-abs(x))) + relu(x)
            one = _expr.const(1, dtype="float32")
            exp_neg_abs_x = _op.exp(_op.negative(_op.abs(x)))
            return _op.add(_op.log(_op.add(one, exp_neg_abs_x)), _op.nn.relu(x))

        return _stable_softrelu(inputs[0])
    if act_type not in _activation_map:
        raise tvm.error.OpNotImplemented(
            "Operator {} is not supported for frontend MXNet.".format(act_type)
        )
    return _activation_map[act_type](inputs[0])


def _mx_compare(new_op, wrapper):
    def impl(inputs, attrs):
        expr = _infer_type(inputs[0])
        dtype = expr.checked_type.dtype
        return wrapper(new_op)(inputs, attrs).astype(dtype)

    return impl


def _mx_unravel_index(inputs, attrs):
    assert len(inputs) == 1
    shape = attrs.get_int_tuple("shape")
    shape_expr = _expr.const(list(shape))
    return _op.unravel_index(inputs[0], shape_expr)


def _mx_swap_axis(inputs, attrs):
    assert len(inputs) == 1
    dim1 = attrs.get_int("dim1")
    dim2 = attrs.get_int("dim2")
    shape = _infer_type(inputs[0]).checked_type.shape
    axes = list(range(len(shape)))
    axes[dim1] = dim2
    axes[dim2] = dim1
    return _op.transpose(inputs[0], axes=axes)


def _mx_zeros(inputs, attrs):
    assert len(inputs) == 0
    shape = attrs.get_int_tuple("shape")
    dtype = attrs.get_str("dtype", "float32")
    if 0 in shape:
        return None
    return _op.zeros(shape=shape, dtype=dtype)


def _mx_conv(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) == 3:
        return _mx_conv3d(inputs, attrs)
    elif len(kernel_size) == 2:
        return _mx_conv2d(inputs, attrs)
    elif len(kernel_size) == 1:
        return _mx_conv1d(inputs, attrs)
    else:
        raise tvm.error.OpAttributeInvalid(
            "1D, 2D or 3D kernels only are supported for operator Convolution"
        )


def _mx_conv1d(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 1:
        raise tvm.error.OpAttributeInvalid(
            "Non 1D or 2D kernels are not supported for operator Convolution"
        )
    data_layout = attrs.get_str("layout", "NCW")
    # MXNet Conv1D only supports ‘NCW’ layout for now.
    if data_layout != "NCW":
        raise tvm.error.OpAttributeInvalid('Only "NCW" data layout is supported for 1D Convolution')
    data_layout = "NCHW"
    channel_axis = 1
    kernel_layout = "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = (1,) + kernel_size
    new_attrs["strides"] = (1,) + attrs.get_int_tuple("stride", (1,))
    new_attrs["padding"] = (0,) + attrs.get_int_tuple("pad", (0,))
    new_attrs["dilation"] = (1,) + attrs.get_int_tuple("dilate", (1,))
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


def _get_mx_conv2d_attrs(attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    data_layout = attrs.get_str("layout", "NCHW")
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
    return new_attrs


def _mx_conv2d(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    data_layout = attrs.get_str("layout", "NCHW")
    if len(kernel_size) != 2:
        raise tvm.error.OpAttributeInvalid("Only 2D kernels are supported for operator Convolution")

    new_attrs = _get_mx_conv2d_attrs(attrs)
    channel_axis = _get_channel_axis(data_layout, "conv2d")
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv2d(inputs[0], inputs[1], **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _get_mx_conv3d_attrs(attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    data_layout = attrs.get_str("layout", "NCDHW")
    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "DHWIO" if data_layout == "NDHWC" else "OIDHW"
    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1, 1))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    return new_attrs


def _mx_conv3d(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    data_layout = attrs.get_str("layout", "NCDHW")
    if len(kernel_size) != 3:
        raise tvm.error.OpAttributeInvalid("Only 3D kernels are supported for operator Convolution")

    new_attrs = _get_mx_conv3d_attrs(attrs)
    channel_axis = _get_channel_axis(data_layout, "conv3d")
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv3d(inputs[0], inputs[1], **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_conv_transpose(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) == 3:
        return _mx_conv3d_transpose(inputs, attrs)
    elif len(kernel_size) == 2:
        return _mx_conv2d_transpose(inputs, attrs)
    elif len(kernel_size) == 1:
        return _mx_conv1d_transpose(inputs, attrs)
    else:
        raise tvm.error.OpAttributeInvalid(
            "1D, 2D or 3D kernels only are supported for operator Convolution"
        )


def _mx_conv1d_transpose(inputs, attrs):
    if "target_shape" in attrs.attrs:
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "target_shape" is not supported for operator Conv2D-transpose.'
        )
    data_layout = attrs.get_str("layout", "NCW")
    if data_layout != "NCW":
        raise tvm.error.OpAttributeInvalid('Only "NCW" data layout is supported for 1D Convolution')
    channel_axis = 1
    kernel_layout = "OIW"
    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = attrs.get_int_tuple("kernel")
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1,))
    new_attrs["output_padding"] = attrs.get_int_tuple("adj", (0,))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0,))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1,))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", True)
    res = _op.nn.conv1d_transpose(inputs[0], inputs[1], **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_conv2d_transpose(inputs, attrs):
    if "target_shape" in attrs.attrs:
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "target_shape" is not supported for operator Conv2D-transpose.'
        )
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise tvm.error.OpAttributeInvalid(
            "Non-2D kernels are not supported for operator Conv2D-transpose."
        )
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d_transpose")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "IOHW"

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


def _mx_conv3d_transpose(inputs, attrs):
    if "target_shape" in attrs.attrs:
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "target_shape" is not supported for operator Conv3D-transpose.'
        )
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 3:
        raise tvm.error.OpAttributeInvalid(
            "Non-3D kernels are not supported for operator Conv3D-transpose."
        )
    data_layout = attrs.get_str("layout", "NCDHW")
    channel_axis = _get_channel_axis(data_layout, "conv3d_transpose")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "DHWIO" if data_layout == "NDHWC" else "OIDHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1, 1))
    new_attrs["output_padding"] = attrs.get_int_tuple("adj", (0, 0, 0))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", True)
    res = _op.nn.conv3d_transpose(inputs[0], inputs[1], **new_attrs)

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
            raise tvm.error.OpAttributeInvalid("Only 2D kernels are supported for operator Pool2D.")
        new_attrs = {}
        new_attrs["pool_size"] = kernel_size
        new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["ceil_mode"] = attrs.get_str("pooling_convention", "valid") == "full"
        if is_avg:
            new_attrs["count_include_pad"] = attrs.get_bool("count_include_pad", True)
        return new_op(inputs[0], **new_attrs)

    def _pool3d(new_op, is_avg):
        kernel_size = attrs.get_int_tuple("kernel")
        if len(kernel_size) != 3:
            raise tvm.error.OpAttributeInvalid("Only 3D kernels are supported for operator Pool3D.")
        new_attrs = {}
        new_attrs["pool_size"] = kernel_size
        new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0, 0))
        new_attrs["ceil_mode"] = attrs.get_str("pooling_convention", "valid") == "full"
        if is_avg:
            new_attrs["count_include_pad"] = attrs.get_bool("count_include_pad", True)
        return new_op(inputs[0], **new_attrs)

    # 3D pooling
    if len(_infer_shape(inputs[0])) == 5:
        if pool_type == "max":
            if global_pool:
                return _op.nn.global_max_pool3d(inputs[0])
            return _pool3d(_op.nn.max_pool3d, False)
        if pool_type == "avg":
            if global_pool:
                return _op.nn.global_avg_pool3d(inputs[0])
            return _pool3d(_op.nn.avg_pool3d, True)
        raise tvm.error.OpNotImplemented(
            "Operator {} Pooling is not supported for frontend MXNet.".format(
                pool_type.capitalize()
            )
        )
    # 2D Pooling
    if pool_type == "max":
        if global_pool:
            return _op.nn.global_max_pool2d(inputs[0])
        return _pool2d(_op.nn.max_pool2d, False)
    if pool_type == "avg":
        if global_pool:
            return _op.nn.global_avg_pool2d(inputs[0])
        return _pool2d(_op.nn.avg_pool2d, True)
    raise tvm.error.OpNotImplemented(
        "Operator {} Pooling is not supported for frontend MXNet.".format(pool_type.capitalize())
    )


def _mx_adaptive_avg_pooling(inputs, attrs):
    output_size = attrs.get_int_tuple("output_size", [])
    return _op.nn.adaptive_avg_pool2d(inputs[0], output_size)


def _mx_dropout(inputs, attrs):
    rate = attrs.get_float("p", 0.5)
    return _op.nn.dropout(inputs[0], rate=rate)


def _mx_BlockGrad(inputs, attrs):  # pylint: disable=unused-argument
    return inputs


def _mx_batch_norm(inputs, attrs):
    if attrs.get_bool("output_mean_var", False):
        raise tvm.error.OpAttributeUnImplemented(
            'Attribute "output_mean_var" is not supported for operator Batch Norm.'
        )
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
            'Attribute "output_mean_var" is not supported for operator Layer Norm.'
        )
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", -1)
    new_attrs["epsilon"] = attrs.get_float("eps", 1e-5)
    return _op.nn.layer_norm(*inputs, **new_attrs)


def _mx_group_norm(inputs, attrs):
    assert len(inputs) == 3
    if attrs.get_bool("output_mean_var", False):
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "output_mean_var" is not supported for operator Group Norm.'
        )
    new_attrs = {}
    new_attrs["axis"] = 1
    new_attrs["num_groups"] = attrs.get_int("num_groups", 1)
    new_attrs["epsilon"] = attrs.get_float("eps", 1e-5)
    return _op.nn.group_norm(*inputs, **new_attrs)


def _mx_slice(inputs, attrs):
    new_attrs = {}
    begin = list(attrs.get_int_tuple("begin", None))
    end = list(attrs.get_int_tuple("end", None))
    stride = attrs.get_int_tuple("step", None)
    input_shape = _infer_type(inputs[0]).checked_type.shape
    if begin is None:
        raise tvm.error.OpAttributeRequired('Attribute "begin" not found in operator Slice.')
    if end is None:
        raise tvm.error.OpAttributeRequired('Attribute "end" not found in operator Slice.')
    begin = (x if x is not None else 0 for x in begin)
    for i, ed in enumerate(end):
        if ed is None:
            end[i] = input_shape[i]
    new_attrs = {"begin": list(begin), "end": list(end)}
    if stride is not None:
        stride = (x if x is not None else 1 for x in stride)
        new_attrs["strides"] = list(stride)
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
            "Only support crop_like pattern for operator Crop."
        )
    if attrs.get_bool("center_crop", False):
        raise tvm.error.OpAttributeUnimplemented("Center crop is not supported in operator Crop.")
    if attrs.get_int_tuple("h_w", (0, 0)) != (0, 0):
        raise tvm.error.OpAttributeUnimplemented("Doesn't support h_w in operator Crop.")
    offset = attrs.get_int_tuple("offset", (0, 0))
    new_attrs = {}
    if offset == (0, 0):
        new_attrs["axes"] = (2, 3)
        return _op.slice_like(*inputs, **new_attrs)
    expr = _infer_type(inputs[1])
    like_shape = expr.checked_type.shape
    new_attrs["begin"] = [0, 0, offset[0], offset[1]]
    new_attrs["end"] = [
        like_shape[0],
        like_shape[1],
        offset[0] + like_shape[2],
        offset[1] + like_shape[3],
    ]
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


def _mx_logistic_regression_output(inputs, _):
    return _op.sigmoid(inputs[0])


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
    pad_mode = attrs.get_str("mode", None)
    if pad_mode is None:
        raise tvm.error.OpAttributeRequired('Attribute "mode" not found in operator pad.')
    if pad_mode not in ["constant", "edge", "reflect"]:
        raise tvm.error.OpAttributeInvalid("Value " + mode + ' in attribute "mode" is not valid')
    pad_width = attrs.get_int_tuple("pad_width", None)
    if pad_width is None:
        raise tvm.error.OpAttributeRequired('Attribute "pad_width" not found in operator pad.')
    if None in pad_width:
        raise tvm.error.OpAttributeInvalid(
            'Value None in attribute "pad_width" of operator Slice is not valid.'
        )
    constant_value = attrs.get_float("constant_value", 0.0)
    padding = tuple(tuple((b, a)) for b, a in zip(pad_width[::2], pad_width[1::2]))
    return _op.nn.pad(
        data=inputs[0], pad_width=padding, pad_value=constant_value, pad_mode=pad_mode
    )


def _mx_leaky_relu(inputs, attrs):
    act_type = attrs.get_str("act_type", "leaky")
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
    if act_type == "gelu":
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt2 = _expr.const(math.sqrt(2), dtype="float32")
        erf = _op.erf(_op.divide(inputs[0], sqrt2))
        one = _expr.const(1, dtype="float32")
        erf_plus_one = _op.add(one, erf)
        half = _expr.const(0.5, dtype="float32")
        half_x = _op.multiply(inputs[0], half)
        return _op.multiply(half_x, erf_plus_one)
    raise tvm.error.OpNotImplemented(
        "Operator {} is not supported for frontend MXNet.".format(act_type)
    )


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
    new_attrs["sizes"] = attrs.get_float_tuple("sizes", (1.0,))
    new_attrs["steps"] = attrs.get_float_tuple("steps", (-1.0, -1.0))
    new_attrs["offsets"] = attrs.get_float_tuple("offsets", (0.5, 0.5))
    new_attrs["ratios"] = attrs.get_float_tuple("ratios", (1.0,))
    new_attrs["clip"] = attrs.get_bool("clip", False)
    return _op.vision.multibox_prior(inputs[0], **new_attrs)


def _mx_multibox_detection(inputs, attrs):
    new_attrs0 = {}
    new_attrs0["clip"] = attrs.get_bool("clip", True)
    new_attrs0["threshold"] = attrs.get_float("threshold", 0.01)
    new_attrs0["variances"] = attrs.get_float_tuple("variances", (0.1, 0.1, 0.2, 0.2))

    new_attrs1 = {}
    new_attrs1["return_indices"] = False
    new_attrs1["iou_threshold"] = attrs.get_float("nms_threshold", 0.5)
    new_attrs1["force_suppress"] = attrs.get_bool("force_suppress", False)
    new_attrs1["top_k"] = attrs.get_int("nms_topk", -1)

    ret = _op.vision.multibox_transform_loc(inputs[0], inputs[1], inputs[2], **new_attrs0)
    return _op.vision.non_max_suppression(ret[0], ret[1], ret[1], **new_attrs1)


def _mx_dot(inputs, attrs):
    assert len(inputs) == 2
    a, b = inputs
    rank_a = len(_infer_type(a).checked_type.shape)
    rank_b = len(_infer_type(b).checked_type.shape)
    if rank_a != 2 or rank_b != 2:
        raise tvm.error.OpAttributeUnimplemented("Only 2-D arrays are supported.")
    transpose_a = attrs.get_bool("transpose_a", False)
    transpose_b = attrs.get_bool("transpose_b", False)
    if transpose_a is True:
        msg = 'Value {} in attribute "transpose_a" of operator dot ' "is not valid."
        raise tvm.error.OpAttributeInvalid(msg.format(transpose_a))
    if transpose_b is False:
        b = _op.transpose(b, axes=[1, 0])
    return _op.nn.dense(a, b)


def _mx_batch_dot(inputs, attrs):
    assert len(inputs) == 2
    a, b = inputs
    a_shape = _infer_type(a).checked_type.shape
    batch_shapes = None
    if len(a_shape) > 3:
        batch_shapes = a_shape[:-2]
        a = _op.reverse_reshape(a, newshape=(-1, 0, 0))
    b_shape = _infer_type(b).checked_type.shape
    if len(b_shape) > 3:
        if batch_shapes is None:
            batch_shapes = b_shape[:-2]
        b = _op.reverse_reshape(b, newshape=(-1, 0, 0))
    transpose_a = attrs.get_bool("transpose_a", False)
    transpose_b = attrs.get_bool("transpose_b", False)
    if transpose_a is True:
        msg = 'Value {} in attribute "transpose_a" of operator batch_dot ' "is not valid."
        raise tvm.error.OpAttributeInvalid(msg.format(transpose_a))
    if transpose_b is False:
        b = _op.transpose(b, axes=[0, 2, 1])
    out = _op.nn.batch_matmul(a, b)
    if batch_shapes is not None:
        out = _op.reverse_reshape(out, newshape=tuple(batch_shapes) + (0, 0))
    return out


def _mx_arange(inputs, attrs):
    assert len(inputs) == 0
    if attrs.get_int("repeat", 1) != 1:
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "repeat" is not supported in operator arange.'
        )
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


# pylint: disable=unused-argument
def _mx_make_loss(inputs, attrs):
    # while doing inference make_loss does not have any effect
    # and it should be mapped to identity
    return inputs[0]


def _mx_contrib_arange_like(inputs, attrs):
    assert len(inputs) == 1
    if attrs.get_int("repeat", 1) != 1:
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "repeat" is not supported in operator arange_like.'
        )
    ty = _infer_type(inputs[0]).checked_type
    assert ty
    shape, dtype = get_const_tuple(ty.shape), ty.dtype
    axis = attrs.get_int("axis", None)
    if axis is None:
        n_elems = 1
        for dim in shape:
            if not isinstance(dim, int):
                raise tvm.error.OpError("Don't support arange_like with symbolic input shape.")
            n_elems *= dim
    else:
        axis = axis + len(shape) if axis < 0 else axis
        assert 0 <= axis < len(shape)
        n_elems = shape[axis]
        if not isinstance(n_elems, int):
            raise tvm.error.OpError("Don't support arange_like with symbolic input shape.")
        shape = (n_elems,)
    start = attrs.get_float("start", 0.0)
    step = attrs.get_float("step", 1.0)
    stop = start + step * n_elems
    new_attrs = {}
    new_attrs["start"] = _expr.const(start, dtype=dtype)
    new_attrs["stop"] = _expr.const(stop, dtype=dtype)
    new_attrs["step"] = _expr.const(step, dtype=dtype)
    new_attrs["dtype"] = dtype
    ret = _op.arange(**new_attrs)
    if len(shape) > 1:
        ret = _op.reshape(ret, shape)
    return ret


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
    return _op.take(inputs[0], inputs[1].astype("int32"), axis=axis, mode=mode)


def _mx_gather_nd(inputs, attrs):
    assert len(inputs) == 2
    assert len(_infer_shape(inputs[1])) > 1, "index tensor to have at least 2 dimensions"
    return _op.gather_nd(inputs[0], inputs[1])


def _mx_reverse(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis")
    return _op.reverse(inputs[0], **new_attrs)


def _mx_sequence_reverse(inputs, attrs):
    new_attrs = {}
    use_seq_lengths = attrs.get_bool("use_sequence_length")
    if not use_seq_lengths:
        assert len(inputs) == 1
        new_attrs["axis"] = attrs.get_int("axis")
        return _op.reverse(inputs[0], **new_attrs)

    assert len(inputs) == 2
    new_attrs["seq_axis"] = attrs.get_int("axis")
    # MXNet assumes batch_axis as 1.
    new_attrs["batch_axis"] = 1
    return _op.reverse_sequence(inputs[0], inputs[1], **new_attrs)


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
    return _op.image.resize2d(inputs[0], size, coordinate_transformation_mode="align_corners")


def _mx_amp_multicast(inputs, attrs):
    cast_narrow = attrs.get_bool("cast_narrow", False)
    dtypes = [_infer_type(x).checked_type.dtype for x in inputs]
    supported_dtypes = ["float16", "float32"]
    assert all(
        [x in supported_dtypes for x in dtypes]
    ), "amp_multicast support is limited to float16 and float32 inputs only."
    has_float16 = any(x == "float16" for x in dtypes)
    has_float32 = any(x == "float32" for x in dtypes)
    dtype = dtypes[0]
    if cast_narrow and has_float16:
        dtype = "float16"
    if not cast_narrow and has_float32:
        dtype = "float32"
    return [_op.cast(x, dtype) for x in inputs]


def _mx_grid_generator(inputs, attrs):
    transform_type = attrs.get_str("transform_type")
    if transform_type == "affine":
        target_shape = attrs.get_int_tuple("target_shape")
        return _op.image.affine_grid(_op.reshape(inputs[0], (0, 2, 3)), target_shape)
    if transform_type == "warp":
        checked_type = _infer_type(inputs[0]).checked_type
        batch, _, height, width = get_const_tuple(checked_type.shape)
        dtype = checked_type.dtype
        identity_affine = relay.const(np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=dtype))
        identity_affine = _op.broadcast_to(identity_affine, (batch, 2, 3))
        normalizer = (2.0 / np.array([width - 1, height - 1])).reshape(1, -1, 1, 1).astype(dtype)
        normalized_flow = inputs[0] * relay.const(normalizer)
        grid = _op.image.affine_grid(identity_affine, (height, width))
        return grid + normalized_flow
    raise ValueError("unknown transform type" + transform_type)


def _mx_bilinear_sampler(inputs, attrs):
    return _op.image.grid_sample(inputs[0], inputs[1], "bilinear", "NCHW")


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
    iou_thresh = attrs.get_float("overlap_thresh", 0.5)
    top_k = attrs.get_int("topk", -1)
    valid_thresh = attrs.get_float("valid_thresh", 0)
    coord_start = attrs.get_int("coord_start", 2)
    score_index = attrs.get_int("score_index", 1)
    id_index = attrs.get_int("id_index", -1)
    in_format = attrs.get_str("in_format", "corner")
    out_format = attrs.get_str("out_format", "corner")
    if in_format != "corner":
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "in_format" must equal "corner" for operator box_nms.'
        )
    if out_format != "corner":
        raise tvm.error.OpAttributeInvalid(
            'Value of attribute "out_format" must equal "corner" for operator box_nms.'
        )

    ret = _op.vision.get_valid_counts(
        inputs[0], score_threshold=valid_thresh, id_index=id_index, score_index=score_index
    )
    nms_out = _op.vision.non_max_suppression(
        ret[1],
        ret[0],
        ret[2],
        iou_threshold=iou_thresh,
        force_suppress=force_suppress,
        top_k=top_k,
        coord_start=coord_start,
        score_index=score_index,
        id_index=id_index,
        return_indices=False,
        invalid_to_bottom=True,
    )
    return nms_out


def _mx_box_decode(inputs, attrs):
    std0 = relay.const(attrs.get_float("std0", 1), "float32")
    std1 = relay.const(attrs.get_float("std1", 1), "float32")
    std2 = relay.const(attrs.get_float("std2", 1), "float32")
    std3 = relay.const(attrs.get_float("std3", 1), "float32")
    clip = attrs.get_float("clip", -1)
    in_format = attrs.get_str("format", "corner")

    anchors = inputs[1]  # (1, N, 4) encoded in corner or center
    a = _op.split(anchors, indices_or_sections=4, axis=-1)
    # Convert to format "center".
    if in_format == "corner":
        a_width = a[2] - a[0]
        a_height = a[3] - a[1]
        a_x = a[0] + a_width * relay.const(0.5, "float32")
        a_y = a[1] + a_height * relay.const(0.5, "float32")
    else:
        a_x, a_y, a_width, a_height = a
    data = inputs[0]  # (B, N, 4) predicted bbox offset
    p = _op.split(data, indices_or_sections=4, axis=-1)
    ox = p[0] * std0 * a_width + a_x
    oy = p[1] * std1 * a_height + a_y
    dw = p[2] * std2
    dh = p[3] * std3
    if clip > 0:
        clip = relay.const(clip, "float32")
        dw = _op.minimum(dw, clip)
        dh = _op.minimum(dh, clip)
    dw = _op.exp(dw)
    dh = _op.exp(dh)
    ow = dw * a_width * relay.const(0.5, "float32")
    oh = dh * a_height * relay.const(0.5, "float32")
    out = _op.concatenate([ox - ow, oy - oh, ox + ow, oy + oh], axis=-1)
    return out


def _mx_l2_normalize(inputs, attrs):
    new_attrs = {}
    mode = attrs.get_str("mode", "instance")
    if mode == "channel":
        new_attrs["axis"] = [1]
    elif mode == "instance":
        ndim = len(_infer_type(inputs[0]).checked_type.shape)
        new_attrs["axis"] = list(range(1, ndim))
    elif mode == "spatial":
        ndim = len(_infer_type(inputs[0]).checked_type.shape)
        new_attrs["axis"] = list(range(2, ndim))
    else:
        raise tvm.error.OpAttributeInvalid(
            'Mode "{}" is not supported for operator l2_normalize.'.format(mode)
        )
    new_attrs["eps"] = attrs.get_float("eps", 1e-10)
    return _op.nn.l2_normalize(inputs[0], **new_attrs)


def _mx_softsign(inputs, attrs):
    return inputs[0] / (_expr.const(1.0) + _op.abs(inputs[0]))


def _mx_softmin(inputs, attrs):
    axis = attrs.get_int("axis", -1)
    return _op.nn.softmax(_op.negative(inputs[0]), axis)


def _mx_hard_sigmoid(inputs, attrs):
    x = (_expr.const(0.2) * inputs[0]) + _expr.const(0.5)
    return _op.clip(x, a_min=0.0, a_max=1.0)


def _mx_reciprocal(inputs, attrs):
    return _expr.const(1.0) / inputs[0]


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
    return _op.shape_of(inputs[0], dtype="int64")


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
    return _op.take(weight, indices.astype("int32"), axis=0)


def _mx_smooth_l1(inputs, attrs):
    scalar = attrs.get_float("scalar", 1.0)
    scalar_sq = scalar * scalar
    mask = _op.less(inputs[0], _expr.const(1.0 / scalar_sq, dtype="float32"))
    return _op.where(
        mask,
        _expr.const(scalar_sq / 2.0, dtype="float32") * inputs[0] * inputs[0],
        _op.abs(inputs[0]) - _expr.const(0.5 / scalar_sq),
    )


def _mx_deformable_convolution(inputs, attrs):
    new_attrs = {}
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
    new_attrs["is_ascend"] = attrs.get_bool("is_ascend", False)
    ret_type = attrs.get_str("ret_typ", "indices")
    if ret_type == "mask":
        raise tvm.error.OpAttributeUnimplemented(
            "Attribute ret_type=mask is not supported in topk operator"
        )
    new_attrs["ret_type"] = "values" if ret_type == "value" else ret_type
    new_attrs["dtype"] = attrs.get_str("dtype", "float32")
    return _op.topk(inputs[0], **new_attrs)


def _mx_sequence_mask(inputs, attrs):
    assert len(inputs) == 1 or len(inputs) == 2
    new_attrs = {}
    use_sequence_length = attrs.get_bool("use_sequence_length", False)
    new_attrs["mask_value"] = attrs.get_float("value", 0.0)
    new_attrs["axis"] = attrs.get_int("axis", 0)
    if use_sequence_length:
        return _op.sequence_mask(*inputs, **new_attrs)
    else:
        return inputs[0]


def _mx_contrib_div_sqrt_dim(inputs, _):
    assert len(inputs) == 1
    ndim = len(_infer_type(inputs[0]).checked_type.shape)
    dim = _op.take(_op.shape_of(inputs[0]), _expr.const(ndim - 1, dtype="int32"))
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
        mode, activation = mode.split("_")
    assert mode in ["rnn", "gru", "lstm"]
    bidirectional = attrs.get_bool("bidirectional", False)
    direct = 2 if bidirectional else 1
    layout = attrs.get_str("layout", "TNC")
    if layout != "TNC":
        raise tvm.error.OpAttributeUnimplemented(
            "RNN with layout other than TNC is not supported yet"
        )
    num_states = 2 if mode == "lstm" else 1
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
        weights.append(
            [concat_weight[i * 2 * direct].args[0], concat_weight[i * 2 * direct + 1].args[0]]
        )
        bias.append(
            [
                concat_weight[(num_layers + i) * 2 * direct].args[0],
                concat_weight[(num_layers + i) * 2 * direct + 1].args[0],
            ]
        )
        s = []
        for state in init_states:
            s.append(_op.take(state, _expr.const(i * direct, "int32"), axis=0))
        states.append(s)
        if bidirectional:
            back_weights.append(
                [
                    concat_weight[i * 2 * direct + 2].args[0],
                    concat_weight[i * 2 * direct + 3].args[0],
                ]
            )
            back_bias.append(
                [
                    concat_weight[(num_layers + i) * 2 * direct + 2].args[0],
                    concat_weight[(num_layers + i) * 2 * direct + 3].args[0],
                ]
            )
            s = []
            for state in init_states:
                s.append(_op.take(state, _expr.const(i * direct + 1, "int32"), axis=0))
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
            else:  # mode == "lstm"
                out, new_states = _lstm_cell(x, states[l], *weights[l], *bias[l])
            states[l] = new_states
            outputs.append(out)
        if bidirectional:
            for x in reversed(xs):
                if mode == "rnn":
                    out, new_states = _rnn_cell(
                        x, back_states[l], *back_weights[l], *back_bias[l], activation
                    )
                elif mode == "gru":
                    out, new_states = _gru_cell(x, back_states[l], *back_weights[l], *back_bias[l])
                else:  # mode == "lstm"
                    out, new_states = _lstm_cell(x, back_states[l], *back_weights[l], *back_bias[l])
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
    indices = inputs[0].astype("int32")
    depth = attrs.get_int("depth", 0)
    dtype = attrs.get_str("dtype", "int32")
    on_value = tvm.relay.const(attrs.get_float("on_value", 1.0), dtype)
    off_value = tvm.relay.const(attrs.get_float("off_value", 0.0), dtype)
    return _op.one_hot(indices, on_value, off_value, depth, -1, dtype)


def _mx_depth_to_space(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["block_size"] = attrs.get_int("block_size")
    return _op.nn.depth_to_space(*inputs, **new_attrs)


def _mx_space_to_depth(inputs, attrs):
    assert len(inputs) == 1
    new_attrs = {}
    new_attrs["block_size"] = attrs.get_int("block_size")
    return _op.nn.space_to_depth(*inputs, **new_attrs)


def _mx_correlation(inputs, attrs):
    assert len(inputs) == 2
    new_attrs = {}
    new_attrs["kernel_size"] = attrs.get_int("kernel_size", 1)
    new_attrs["max_displacement"] = attrs.get_int("max_displacement", 1)
    new_attrs["stride1"] = attrs.get_int("stride1", 1)
    new_attrs["stride2"] = attrs.get_int("stride2", 1)
    new_attrs["padding"] = attrs.get_int("pad_size", 0)
    new_attrs["is_multiply"] = attrs.get_bool("is_multiply", True)
    new_attrs["layout"] = "NCHW"
    return _op.nn.correlation(*inputs, **new_attrs)


def _mx_contrib_fifo_buffer(inputs, attrs):
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis")
    return _op.nn.fifo_buffer(*inputs, **new_attrs)


def _mx_contrib_interleaved_matmul_selfatt_qk(inputs, attrs):
    """
    tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
    q_proj = mx.nd.transpose(tmp[:,:,:,0,:], axes=(1, 2, 0, 3))
    q_proj = mx.nd.reshape(q_proj, shape=(-1, 0, 0), reverse=True)
    q_proj = mx.nd.contrib.div_sqrt_dim(q_proj)
    k_proj = mx.nd.transpose(tmp[:,:,:,1,:], axes=(1, 2, 0, 3))
    k_proj = mx.nd.reshape(k_proj, shape=(-1, 0, 0), reverse=True)
    output = mx.nd.batch_dot(q_proj, k_proj, transpose_b=True)
    """
    assert len(inputs) == 1
    qkv = inputs[0]
    num_heads = attrs.get_int("heads")
    qkv = _op.reshape(qkv, newshape=(0, 0, num_heads, 3, -1))
    q_proj = _op.take(qkv, _expr.const(0, "int32"), axis=3)
    q_proj = _op.transpose(q_proj, axes=[1, 2, 0, 3])
    q_proj = _op.reverse_reshape(q_proj, newshape=(-1, 0, 0))
    q_proj = _mx_contrib_div_sqrt_dim([q_proj], None)
    k_proj = _op.take(qkv, _expr.const(1, "int32"), axis=3)
    k_proj = _op.transpose(k_proj, axes=[1, 2, 0, 3])
    k_proj = _op.reverse_reshape(k_proj, newshape=(-1, 0, 0))
    ret = _op.nn.batch_matmul(q_proj, k_proj)
    return ret


def _mx_contrib_interleaved_matmul_selfatt_valatt(inputs, attrs):
    """
    tmp = mx.nd.reshape(queries_keys_values, shape=(0, 0, num_heads, 3, -1))
    v_proj = mx.nd.transpose(tmp[:,:,:,2,:], axes=(1, 2, 0, 3))
    v_proj = mx.nd.reshape(v_proj, shape=(-1, 0, 0), reverse=True)
    output = mx.nd.batch_dot(attention, v_proj)
    output = mx.nd.reshape(output, shape=(-1, num_heads, 0, 0), reverse=True)
    output = mx.nd.transpose(output, axes=(2, 0, 1, 3))
    output = mx.nd.reshape(output, shape=(0, 0, -1))
    """
    assert len(inputs) == 2
    qkv, att = inputs
    num_heads = attrs.get_int("heads")
    qkv = _op.reshape(qkv, newshape=(0, 0, num_heads, 3, -1))
    v_proj = _op.take(qkv, _expr.const(2, "int32"), axis=3)
    v_proj = _op.transpose(v_proj, axes=(1, 2, 0, 3))
    v_proj = _op.reverse_reshape(v_proj, newshape=(-1, 0, 0))
    v_proj = _op.transpose(v_proj, axes=[0, 2, 1])
    out = _op.nn.batch_matmul(att, v_proj)
    out = _op.reverse_reshape(out, newshape=(-1, num_heads, 0, 0))
    out = _op.transpose(out, axes=(2, 0, 1, 3))
    out = _op.reshape(out, newshape=(0, 0, -1))
    return out


def _mx_cond(inputs, attrs, subgraphs):
    assert len(subgraphs) == 3
    cond_input_locs = json.loads(attrs.get_str("cond_input_locs"))
    then_input_locs = json.loads(attrs.get_str("then_input_locs"))
    else_input_locs = json.loads(attrs.get_str("else_input_locs"))
    num_outputs = attrs.get_int("num_outputs")

    input_args = []
    for i, arg in enumerate(inputs):
        var = _expr.var("arg%s" % i, _infer_type(arg).checked_type)
        input_args.append(var)
    cond_args = [input_args[i] for i in cond_input_locs]
    then_args = [input_args[i] for i in then_input_locs]
    else_args = [input_args[i] for i in else_input_locs]

    cond_arg_shapes = [arg.type_annotation.shape for arg in cond_args]
    cond_arg_dtype_info = [arg.type_annotation.dtype for arg in cond_args]
    cond_func = _from_mxnet_impl(subgraphs[0], cond_arg_shapes, cond_arg_dtype_info)
    cond = _expr.Call(cond_func, cond_args).astype("bool")
    cond_shape = get_const_tuple(_infer_type(cond).checked_type.shape)
    if len(cond_shape) > 0:
        assert len(cond_shape) == 1 and cond_shape[0] == 1, "Condition is not scalar"
        cond = _op.take(cond, _expr.const(1, "int"))

    sb = _scope_builder.ScopeBuilder()
    with sb.if_scope(cond):
        then_arg_shapes = [arg.type_annotation.shape for arg in then_args]
        then_arg_dtype_info = [arg.type_annotation.dtype for arg in then_args]
        then_func = _from_mxnet_impl(subgraphs[1], then_arg_shapes, then_arg_dtype_info)
        sb.ret(_expr.Call(then_func, then_args))
    with sb.else_scope():
        else_arg_shapes = [arg.type_annotation.shape for arg in else_args]
        else_arg_dtype_info = [arg.type_annotation.dtype for arg in else_args]
        else_func = _from_mxnet_impl(subgraphs[2], else_arg_shapes, else_arg_dtype_info)
        sb.ret(_expr.Call(else_func, else_args))
    func = _function.Function(input_args, sb.get())
    ret = _expr.Call(func, inputs)
    if num_outputs > 1:
        ret = _expr.TupleWrapper(ret, num_outputs)
    return ret


def _qnn_contrib_concat(inputs, attrs):
    axis = attrs.get_int("dim", 1)
    num_args = attrs.get_int("num_args", -1)
    assert num_args > 0

    input_exprs = inputs[0:num_args]

    min_start_idx = num_args
    max_start_idx = num_args + 1

    mins = list()
    maxs = list()

    for i in range(min_start_idx, len(inputs), 2):
        mins.append(inputs[i])

    for i in range(max_start_idx, len(inputs), 2):
        maxs.append(inputs[i])

    # Check if all the input tensors have same qnn params.
    if len(set(mins)) == 1 and len(set(maxs)) == 1:
        output_min = mins[0]
        output_max = maxs[0]
        concat = _op.concatenate(tuple(input_exprs), axis=axis)
        return concat, output_min, output_max
    else:
        # Get all dtypes. Find input and output scales, call concatenate.
        dtypes = [_infer_type(x).checked_type.dtype for x in input_exprs]
        assert all(
            [x == "uint8" for x in dtypes]
        ), "Current support is limited to uint8 inputs only."
        new_min = min(mins)
        new_max = max(maxs)
        assert new_min == 0

        output_scale = get_mkldnn_uint8_scale(new_min, new_max)
        min_max = zip(mins, maxs)
        input_scales = [get_mkldnn_uint8_scale(x, y) for (x, y) in min_max]
        input_zeros = [0] * len(input_scales)
        output_zero = 0

        input_scales_expr = [relay.const(x, "float32") for x in input_scales]
        input_zeros_expr = [relay.const(x, "int32") for x in input_zeros]

        output_scale_expr = relay.const(output_scale, "float32")
        output_zero_expr = relay.const(output_zero, "int32")

        res = relay.qnn.op.concatenate(
            input_exprs,
            input_scales_expr,
            input_zeros_expr,
            output_scale_expr,
            output_zero_expr,
            axis=axis,
        )
        return res, new_min, new_max


def _qnn_quantize(inputs, attrs):
    out_dtype = "int8"
    out_type = attrs.get_str("out_type")
    if out_type == "auto":
        if attrs.has_attr("min_calib_range") and attrs.has_attr("max_calib_range"):
            if attrs.get_float("min_calib_range") >= 0:
                out_dtype = "uint8"
            else:
                out_dtype = "int8"
    else:
        out_dtype = out_type
    if out_dtype not in {"int8", "uint8"}:
        raise ValueError("Unsupported out_dtype: %s" % out_dtype)
    min_calib_range = attrs.get_float("min_calib_range", 0.0)
    max_calib_range = attrs.get_float("max_calib_range", 0.0)
    quantized_output, _, _ = quantize_mxnet_min_max(
        inputs[0], min_range=min_calib_range, max_range=max_calib_range, out_dtype=out_dtype
    )
    return quantized_output, min_calib_range, max_calib_range


def _qnn_contrib_quantized_fifo_buffer(inputs, attrs, params):
    data = inputs[0]
    buffer = inputs[1]
    min_calib_range = inputs[2]
    max_calib_range = inputs[3]
    data_dtype = _infer_type(data).checked_type.dtype
    buffer_shape = _infer_shape(buffer)
    buffer_name = _get_name(buffer)
    params[buffer_name] = _nd.array(np.zeros(buffer_shape).astype(data_dtype))
    new_buffer = relay.var(buffer_name, relay.TensorType(buffer_shape, data_dtype))
    inputs[1] = new_buffer
    res = _op.nn.fifo_buffer(data=data, buffer=new_buffer, axis=attrs.get_int("axis"))
    return res, min_calib_range, max_calib_range


def _get_subgraph_op(subgraphs, op_name):
    assert len(subgraphs) == 1, "Subgraph should have 1 node but has {}".format(len(subgraphs))
    subgraph = subgraphs[0]
    nodes = subgraph["nodes"]
    assert nodes is not None
    for node in nodes:
        if node["op"] == op_name:
            return node
    raise ValueError("Op {} was not found in the subgraph".format(op_name))


def _qnn_conv(inputs, attrs, subgraphs, params):
    def _has_fused_activation(_attrs, _supported_activations):
        has_fused_activation = False
        if attrs.get_bool("with_act", False) or attrs.get_bool("with_postsum_act", False):
            subgraph_activation_attrs = _get_subgraph_op(subgraphs, "Activation")["attrs"]
            act_type = subgraph_activation_attrs["act_type"]
            if act_type not in _supported_activations:
                raise ValueError(
                    "Fused activation {} is not supported at " "this time".format(act_type)
                )
            has_fused_activation = True
        return has_fused_activation

    def _get_data_scale_and_zp(_data, _inputs, _data_min_idx, _data_max_idx):
        """Finds the Qnn params for the data expr."""
        data_min = _inputs[_data_min_idx]
        data_max = _inputs[_data_max_idx]
        assert data_min <= data_max
        data_dtype = _infer_type(_data).checked_type.dtype
        assert data_dtype in {"int8", "uint8"}
        if data_min < 0.0:
            assert (
                data_dtype == "int8"
            ), "Expect int8 when data_min < 0.0, consider quantize model with int8."
        _data_scale = (
            get_mkldnn_uint8_scale(data_min, data_max)
            if data_dtype == "uint8"
            else get_mkldnn_int8_scale(data_min, data_max)
        )
        _data_zero_point = 0
        return _data_scale, _data_zero_point

    def _get_bn_alpha_coeff(_bn_gamma_idx, _bn_beta_idx, _bn_running_mean_idx, _bn_running_var_idx):
        """Extract the BN coeff. These will be use later for BN folding into convolution."""
        # Extract relevant attrs from bn.
        bn_attrs = _get_subgraph_op(subgraphs, "BatchNorm")["attrs"]
        bn_epsilon_param = float(bn_attrs["eps"])
        bn_scale_param = bn_attrs["fix_gamma"] == "False"
        bn_center_param = True

        # Extract the relevant relay expressions.
        bn_running_var = inputs[_bn_running_var_idx]
        bn_gamma = inputs[_bn_gamma_idx]
        bn_beta = inputs[_bn_beta_idx]
        bn_running_mean = inputs[_bn_running_mean_idx]

        # Get coefficient to multiply to weights.
        bn_epsilon = relay.const(bn_epsilon_param, "float32")
        denominator = relay.sqrt(relay.add(bn_running_var, bn_epsilon))
        _bn_scale = relay.divide(relay.const(1.0, "float32"), denominator)
        if bn_scale_param:
            _bn_scale = relay.multiply(bn_gamma, _bn_scale)

        # Get the shift.
        _bn_shift = relay.negative(relay.multiply(bn_running_mean, _bn_scale))
        if bn_center_param:
            _bn_shift = relay.add(bn_beta, _bn_shift)

        return _bn_scale, _bn_shift

    def _fold_bn(_bn_scale, _bn_shift, _has_bias, _has_bn):
        """Fold BN into kernel and bias. Get new kernel and bias."""
        _kernel = inputs[1]
        if _bn_scale:
            assert attrs.get_bool("with_bn", False)
            # Weights are on OIHW, and _bn_scale is in O.
            exp_bn_scale = relay.expand_dims(_bn_scale, axis=1, num_newaxis=3)
            _kernel = relay.multiply(exp_bn_scale, _kernel)

        _bias = None
        if _has_bias:
            _bias = inputs[2]
            if _has_bn:
                assert _bn_shift is not None
                assert _bn_scale is not None
                _bias = relay.add(relay.multiply(_bn_scale, _bias), _bn_shift)
        elif _has_bn:
            assert _bn_shift is not None
            assert _bn_scale is not None
            _bias = _bn_shift
        return _kernel, _bias

    def _get_quantized_kernel(_kernel, _bias, _data_scale):
        # For quantizing, we need min/max of kernel. So, we have to pre compute this expr.
        np_kernel = _infer_value(_kernel, params).numpy()
        kernel_channel_min = np.amin(np_kernel, axis=(1, 2, 3))
        kernel_channel_max = np.amax(np_kernel, axis=(1, 2, 3))

        np_bias = None
        if _bias is not None:
            np_bias = _infer_value(_bias, params).numpy()
        return quantize_conv_weights_bias_channel_mkldnn_from_var(
            _kernel, np_bias, kernel_channel_min, kernel_channel_max, _data_scale
        )

    def _get_qnn_conv2d(
        _data,
        _kernel,
        _data_zero_point,
        _kernel_zero_point,
        _data_scale,
        _kernel_vector_scale,
        _conv2d_attrs,
    ):
        return relay.qnn.op.conv2d(
            _data,
            _kernel,
            input_zero_point=relay.const(_data_zero_point, "int32"),
            kernel_zero_point=relay.const(_kernel_zero_point, "int32"),
            input_scale=relay.const(_data_scale, "float32"),
            kernel_scale=relay.const(_kernel_vector_scale),
            channels=_conv2d_attrs["channels"],
            groups=_conv2d_attrs["groups"],
            kernel_size=_conv2d_attrs["kernel_size"],
            strides=_conv2d_attrs["strides"],
            dilation=_conv2d_attrs["dilation"],
            padding=_conv2d_attrs["padding"],
            data_layout=_conv2d_attrs["data_layout"],
            kernel_layout=_conv2d_attrs["kernel_layout"],
        )

    def _get_requantized_op(_res, _input_scale, _output_scale, _out_dtype):
        # Requantize to get the output back
        return relay.qnn.op.requantize(
            _res,
            input_scale=relay.const(_input_scale),
            input_zero_point=relay.const(0, "int32"),
            output_scale=relay.const(_output_scale, "float32"),
            output_zero_point=relay.const(0, "int32"),
            axis=1,
            out_dtype=_out_dtype,
        )

    def _get_sum(_res, _output_scale, out_dtype):
        """Handles sum of the second quantized tensor."""
        # This is done in following steps
        #   1) rhs is the add's second operand. First rhs will be requantized to output scale with
        #   dtype int32. The int32 dtype is to keep precision high before adding.
        #   2) Call normal add
        #   3) Depending on final out_dtype, clip and cast (basically requantize).

        _output_scale = relay.const(_output_scale, "float32")
        data_sum = inputs[-5]
        data_sum_min = inputs[-2]
        data_sum_max = inputs[-1]

        data_sum_dtype = _infer_type(data_sum).checked_type.dtype
        data_sum_scale = (
            get_mkldnn_uint8_scale(data_sum_min, data_sum_max)
            if data_sum_dtype == "uint8"
            else get_mkldnn_int8_scale(data_sum_min, data_sum_max)
        )
        data_sum_scale = relay.const(data_sum_scale, "float32")
        zero_point = relay.const(0, "int32")

        # Save one requantize if the previous expr already has a requantize node. This also improves
        # little bit with accuracy.
        if isinstance(data_sum, _expr.Call) and data_sum.op.name == "qnn.requantize":
            prev_input, prev_scale, prev_zero_point = data_sum.args[0:3]
            prev_axis = data_sum.attrs.axis
            data_sum = relay.qnn.op.requantize(
                prev_input,
                input_scale=prev_scale,
                input_zero_point=prev_zero_point,
                output_scale=_output_scale,
                output_zero_point=zero_point,
                axis=prev_axis,
                out_dtype="int32",
            )
        else:
            data_sum = relay.qnn.op.requantize(
                data_sum,
                input_scale=data_sum_scale,
                input_zero_point=zero_point,
                output_scale=_output_scale,
                output_zero_point=zero_point,
                out_dtype="int32",
            )

        # 2) Add two int32 tensors.
        _res = relay.add(_res, data_sum)

        # 3) Clip/cast to change the out dtype.
        _res = relay.clip(
            _res,
            a_min=float(tvm.tir.op.min_value(out_dtype).value),
            a_max=float(tvm.tir.op.max_value(out_dtype).value),
        )
        _res = relay.cast(_res, out_dtype)
        return _res

    def _parse():
        assert len(subgraphs) == 1
        subgraph_conv_attrs = StrAttrsDict(_get_subgraph_op(subgraphs, "Convolution")["attrs"])

        is_quantized = attrs.get_bool("quantized", False)
        if is_quantized:
            # The MKLDNN has a quantized convolution subgraph. There are many different arguments
            # that are taken into account to parse the subgraph.
            #   * no_bias
            #   * with_sum
            #   * with_bn
            #   * with_postsum_relu
            #   * with_act
            #
            # Note - Relu/clip handling is not required because output min/max take care of that.
            #
            # The parsing can be broken down into following steps
            #   1) Get the input data scale and zero points.
            #   2) Extract BN params.
            #   3) Fold the BN params into kernel and bias.
            #   4) Quantize the kernel.
            #   4) Call QNN conv2d op.
            #   5) Quantize bias and call bias_add.
            #   6) Handle sum of quantized tensors if needed. Or just Requantize.

            has_bias = not subgraph_conv_attrs.get_bool("no_bias", False)
            has_sum = attrs.get_bool("with_sum", False)
            has_bn = attrs.get_bool("with_bn", False)

            ###############################################
            #   1) Get the input data scale and zero point.
            ###############################################
            # Last 2 indexes are data min and max. If the conv has a sum, then last 2 indexes are
            # for the second tensor. So, the data min max indexes are last 3 and 4
            data_min_idx = -2
            data_max_idx = -1
            if has_sum:
                data_min_idx = -4
                data_max_idx = -3

            data = inputs[0]
            data_scale, data_zero_point = _get_data_scale_and_zp(
                data, inputs, data_min_idx, data_max_idx
            )

            #############################
            #   2) Extract the BN params.
            #############################
            # Find the indexes to look at for BN.
            bn_scale = bn_shift = None
            if has_bn:
                if has_bias:
                    bn_start_idx = 3
                else:
                    bn_start_idx = 2

                bn_gamma_idx = bn_start_idx
                bn_beta_idx = bn_start_idx + 1
                bn_running_mean_idx = bn_start_idx + 2
                bn_running_var_idx = bn_start_idx + 3

                bn_scale, bn_shift = _get_bn_alpha_coeff(
                    bn_gamma_idx, bn_beta_idx, bn_running_mean_idx, bn_running_var_idx
                )

            ########################################
            #   3) Fold the BN into kernel and bias.
            ########################################
            kernel, bias = _fold_bn(bn_scale, bn_shift, has_bias, has_bn)

            #######################################################################
            #   4) Fold BN params into kernel. Get quantized kernel and QNN params.
            #######################################################################
            kernel, kernel_vector_scale, kernel_zero_point = _get_quantized_kernel(
                kernel, bias, data_scale
            )

            ##########################
            #   5) Call QNN conv2d op.
            ##########################
            conv2d_attrs = _get_mx_conv2d_attrs(subgraph_conv_attrs)
            res = _get_qnn_conv2d(
                data,
                kernel,
                data_zero_point,
                kernel_zero_point,
                data_scale,
                kernel_vector_scale,
                conv2d_attrs,
            )

            ###############################################
            #   6) Fold BN params into bias. Call bias_add.
            ###############################################
            if has_bias or has_bn:
                bias_scale = data_scale * kernel_vector_scale
                int32_bias = quantize_conv_bias_mkldnn_from_var(bias, bias_scale)
                res = _op.nn.bias_add(res, int32_bias, axis=1)

            #####################################################################
            #   7) Handle sum of quantized tensors if needed. Or just Requantize.
            #####################################################################
            min_output_range = attrs.get_float("min_calib_range")
            max_output_range = attrs.get_float("max_calib_range")
            output_scale, out_dtype = get_conv_mkldnn_requantized_scale_outDtype(
                min_output_range, max_output_range
            )

            # QNN conv2d output scale is product of data_scale and kernel_vector_scale
            input_scale = data_scale * kernel_vector_scale
            if attrs.get_bool("with_sum", False):
                # There is a second tensor that has to be added to the QNN conv2d output. Therefore,
                # the QNN conv2d is first requantized to output scale with int32 precision. The
                # second tensor will also be requantized to output scale with int32 precision,
                # followed by an add operator.
                res = _get_requantized_op(res, input_scale, output_scale, "int32")
                res = _get_sum(res, output_scale, out_dtype)
            else:
                # Get the requantized conv output
                res = _get_requantized_op(res, input_scale, output_scale, out_dtype)

            return res, min_output_range, max_output_range
        else:
            res = _mx_conv(inputs, subgraph_conv_attrs)
            has_fused_relu = _has_fused_activation(attrs, ["relu"])
            if has_fused_relu:
                res = _op.nn.relu(res)
            return res

    return _parse()


def _qnn_flatten(inputs, attrs):
    # pylint: disable=unused-argument
    data = inputs[0]
    output_min = inputs[1]
    output_max = inputs[2]
    output = _op.nn.batch_flatten(data)
    return output, output_min, output_max


def _qnn_dequantize(inputs, attrs):
    # pylint: disable=unused-argument
    data = inputs[0]
    input_min = inputs[1]
    input_max = inputs[2]
    in_dtype = _infer_type(data).checked_type.dtype
    result = dequantize_mxnet_min_max(data, input_min, input_max, in_dtype)
    return result


def _qnn_activation(inputs, attrs):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 3
    assert act_type == "relu", "Currently only relu is supported"
    data = inputs[0]
    range_min = inputs[1]
    range_max = inputs[2]
    res = _op.nn.relu(data)
    return res, range_min, range_max


def _qnn_pooling(inputs, attrs):
    input_min = inputs[1]
    input_max = inputs[2]
    data = inputs[0]
    data_dtype = _infer_type(data).checked_type.dtype
    pool_type = attrs.get_str("pool_type")
    if data_dtype in ("int8", "uint8") and pool_type != "max":
        data = _op.cast(data, "int32")
    res = _mx_pooling([data, input_min, input_max], attrs)
    if data_dtype in ("int8", "uint8") and pool_type != "max":
        res = _op.cast(res, data_dtype)
    return res, input_min, input_max


def _qnn_batch_norm(inputs, attrs):
    # Perform batch norm in FP32
    data = inputs[0]

    # Dequantize the data.
    data_min_idx, data_max_idx = (-2, -1)
    data_min, data_max = inputs[data_min_idx], inputs[data_max_idx]
    data_dtype = _infer_type(data).checked_type.dtype
    data_scale = (
        get_mkldnn_uint8_scale(data_min, data_max)
        if data_dtype == "uint8"
        else get_mkldnn_int8_scale(data_min, data_max)
    )
    data_zp = 0
    data = relay.qnn.op.dequantize(
        data, relay.const(data_scale, "float32"), relay.const(data_zp, "int32")
    )

    # Run BN. The last 4 inputs are same as before.
    new_inputs = [data, *inputs[1:5]]
    res = _mx_batch_norm(new_inputs, attrs)

    # Quantize the result
    min_output_range = attrs.get_float("min_calib_range")
    max_output_range = attrs.get_float("max_calib_range")
    output_scale, out_dtype = get_conv_mkldnn_requantized_scale_outDtype(
        min_output_range, max_output_range
    )
    res = relay.qnn.op.quantize(
        res[0], relay.const(output_scale, "float32"), relay.const(0, "int32"), out_dtype=out_dtype
    )
    return res, min_output_range, max_output_range


def _qnn_fully_connected(inputs, attrs, subgraphs, params):
    def _get_input_scale_zp(_data_dtype, _inputs, _has_bias):
        data_min_idx, data_max_idx = (3, 4) if _has_bias else (2, 3)
        data_min, data_max = _inputs[data_min_idx], _inputs[data_max_idx]
        _data_scale = (
            get_mkldnn_uint8_scale(data_min, data_max)
            if _data_dtype == "uint8"
            else get_mkldnn_int8_scale(data_min, data_max)
        )
        _data_zp = 0
        return _data_scale, _data_zp

    def _get_kernel_scale_zp_tensor_quantized(_kernel, _inputs, _has_bias):
        kernel_dtype = _infer_type(_kernel).checked_type.dtype

        if kernel_dtype != "int8":
            raise tvm.error.OpNotImplemented(
                "Tensor wise quantized expects weights in int8 data type"
            )

        if isinstance(_kernel, tvm.relay.Call) and _kernel.op.name == "qnn.quantize":
            _kernel_scale = _kernel.args[1].data.numpy()
            _kernel_zp = _kernel.args[2].data.numpy()
            return _kernel_scale, _kernel_zp

        kernel_min_idx, kernel_max_idx = (5, 6) if _has_bias else (4, 5)
        kernel_min_name = _get_name(_inputs[kernel_min_idx])
        kernel_min = params[kernel_min_name].numpy()[0]
        kernel_max_name = _get_name(_inputs[kernel_max_idx])
        kernel_max = params[kernel_max_name].numpy()[0]
        _kernel_scale = (
            get_mkldnn_uint8_scale(kernel_min, kernel_max)
            if kernel_dtype == "uint8"
            else get_mkldnn_int8_scale(kernel_min, kernel_max)
        )
        _kernel_zp = 0
        return _kernel_scale, _kernel_zp

    def _get_kernel_scale_zp_channel_quantized(_kernel, _bias, _data_scale):
        kernel_dtype = _infer_type(_kernel).checked_type.dtype
        if kernel_dtype != "float32":
            raise tvm.error.OpNotImplemented(
                "Channel wise quantized expects weights in float32 data type"
            )

        # Get the FP32 values, calculate min/max and then channel quantize them
        np_kernel = _infer_value(_kernel, params).numpy()
        kernel_channel_min = np.amin(np_kernel, axis=(1,))
        kernel_channel_max = np.amax(np_kernel, axis=(1,))

        np_bias = None
        if _bias is not None:
            np_bias = _infer_value(_bias, params).numpy()
        return quantize_conv_weights_bias_channel_mkldnn_from_var(
            _kernel, np_bias, kernel_channel_min, kernel_channel_max, _data_scale
        )

    def _get_bias_requantize_scale(_inputs, _data_scale, _kernel_scale):
        _bias = _inputs[2]
        if isinstance(_bias, tvm.relay.Call) and _bias.op.name == "qnn.quantize":
            _bias_scale = _bias.args[1].data.numpy()
            _bias_requantize_scale = _bias_scale / (_data_scale * _kernel_scale)
            _bias_requantize_scale = _expr.const(_bias_requantize_scale, dtype="float32")
            return _bias_requantize_scale

        bias_min_name = _get_name(_inputs[7])
        bias_min = params[bias_min_name].numpy()[0]
        bias_max_name = _get_name(_inputs[8])
        bias_max = params[bias_max_name].numpy()[0]
        bias_scale = get_mkldnn_int8_scale(bias_min, bias_max)
        _bias_requantize_scale = bias_scale / (_data_scale * _kernel_scale)
        _bias_requantize_scale = _expr.const(_bias_requantize_scale, dtype="float32")
        return _bias_requantize_scale

    is_quantized = attrs.get_bool("quantized", False)
    with_relu = attrs.get_bool("with_relu", False)
    subgraph_dense_attrs = StrAttrsDict(_get_subgraph_op(subgraphs, "FullyConnected")["attrs"])
    if not is_quantized:
        res = _mx_fully_connected(inputs, subgraph_dense_attrs)
        if with_relu:
            res = _op.nn.relu(res)
        return res
    else:
        has_bias = not subgraph_dense_attrs.get_bool("no_bias", False)
        units = subgraph_dense_attrs.get_int("num_hidden")
        is_flatten = subgraph_dense_attrs.get_bool("flatten", True)
        enable_float_output = attrs.get_bool("enable_float_output", False)
        is_channel_quantized = attrs.get_bool("channel_wise_quantize", False)

        ########################
        # Get data, kernel, bias
        ########################
        data, kernel = inputs[0], inputs[1]
        bias = None
        if has_bias:
            bias = inputs[2]

        ##############################
        # Handle for shape of data > 2
        ##############################
        if is_flatten:
            data = _op.nn.batch_flatten(data)
        data_shape = _infer_type(data).checked_type.shape
        if len(data_shape) > 2:
            data = _op.reverse_reshape(data, [-1, 0])

        ###############################
        # Get data scale and zero point
        ###############################
        data_dtype = _infer_type(data).checked_type.dtype
        data_scale, data_zp = _get_input_scale_zp(data_dtype, inputs, has_bias)

        #################################
        # Get weight scale and zero point
        #################################
        if is_channel_quantized:
            kernel, kernel_scale, kernel_zp = _get_kernel_scale_zp_channel_quantized(
                kernel, bias, data_scale
            )
        else:
            kernel_scale, kernel_zp = _get_kernel_scale_zp_tensor_quantized(
                kernel, inputs, has_bias
            )

        ################
        # Call QNN dense
        ################
        res = relay.qnn.op.dense(
            data,
            kernel,
            input_zero_point=relay.const(data_zp, "int32"),
            kernel_zero_point=relay.const(kernel_zp, "int32"),
            input_scale=relay.const(data_scale, "float32"),
            kernel_scale=relay.const(kernel_scale, "float32"),
            units=units,
        )

        #################
        # Handle bias add
        #################
        if has_bias:
            if is_channel_quantized:
                bias_scale = data_scale * kernel_scale
                int32_bias = quantize_conv_bias_mkldnn_from_var(bias, bias_scale)
                res = _op.nn.bias_add(res, int32_bias, axis=-1)
            else:
                bias_data = inputs[2]
                bias_requantize_scale = _get_bias_requantize_scale(inputs, data_scale, kernel_scale)
                multiplied_bias = _op.multiply(
                    _op.cast(bias_data, "float32"), bias_requantize_scale
                )
                rounded_bias = _op.round(multiplied_bias)
                clipped_bias = _op.clip(
                    rounded_bias,
                    a_min=tvm.tir.op.min_value("int32").value,
                    a_max=tvm.tir.op.max_value("int32").value,
                )
                requantized_bias = _op.cast(clipped_bias, "int32")
                res = _op.nn.bias_add(res, requantized_bias, axis=-1)

        ##############################################
        # Dequantize if float32 output else Requantize
        ##############################################
        if enable_float_output:
            output_scale = np.float32(data_scale * kernel_scale)
            res = relay.qnn.op.dequantize(
                res, relay.const(output_scale), input_zero_point=relay.const(0, "int32"), axis=1
            )
            if with_relu:
                res = _op.nn.relu(res)
        else:

            if is_channel_quantized:
                raise tvm.error.OpNotImplemented(
                    "Channel wise quantized dense with non float output is not supported yet"
                )
            out_dtype = "uint8" if attrs.get_bool("with_relu", False) else "int8"
            input_scale = np.float32(data_scale * kernel_scale)
            min_output_range = attrs.get_float("min_calib_range")
            max_output_range = attrs.get_float("max_calib_range")
            output_scale = get_mkldnn_requantize_scale_outDtype(
                min_output_range, max_output_range, out_dtype
            )
            res = relay.qnn.op.requantize(
                res,
                input_scale=relay.const(input_scale, "float32"),
                input_zero_point=relay.const(0, "int32"),
                output_scale=relay.const(output_scale, "float32"),
                output_zero_point=relay.const(0, "int32"),
                out_dtype=out_dtype,
            )
            if with_relu:
                res = _op.nn.relu(res)

        ##############################
        # Handle for shape of data > 2
        ##############################
        if len(data_shape) > 2:
            new_shape = data_shape[:-1]
            new_shape.append(units)
            res = _op.reshape(res, new_shape)

        if enable_float_output:
            return res
        return res, min_output_range, max_output_range


def _mx_broadcast_to(inputs, attrs):
    data = inputs[0]
    tgt_shape = attrs.get_int_tuple("shape", [])

    return _op.broadcast_to(data, tgt_shape)


def _mx_broadcast_like(inputs, attrs):
    assert len(inputs) == 2
    for axes in ["lhs_axes", "rhs_axes"]:
        if axes in attrs.attrs:
            raise tvm.error.OpAttributeUnImplemented(
                'Attribute "{}" is not supported for operator broadcast_like.'.format(axes)
            )
    return _op.broadcast_to_like(*inputs)


def _mx_logical_not(inputs, input_types):
    data = inputs[0]
    dtype = _infer_type(data).checked_type.dtype
    data = _op.cast(data, "bool") if dtype != "bool" else data

    return _op.cast(_op.logical_not(data), dtype)


def _mx_broadcast_logical(logical_op):
    def impl(inputs, input_types):
        lhs_type = _infer_type(inputs[0]).checked_type.dtype
        rhs_type = _infer_type(inputs[1]).checked_type.dtype
        lhs = _op.cast(inputs[0], "bool") if lhs_type != "bool" else inputs[0]
        rhs = _op.cast(inputs[1], "bool") if rhs_type != "bool" else inputs[1]

        return _op.cast(logical_op(lhs, rhs), lhs_type)

    return impl


def _mx_npi_transpose(inputs, attrs):
    axes = attrs.get_int_tuple("axes", None)
    # translate default case
    axes = None if len(axes) == 0 or axes[0] is None else axes
    return _op.transpose(inputs[0], axes=axes)


def _mx_npi_pad(inputs, attrs):
    pad_mode = attrs.get_str("mode", None)
    if pad_mode is None:
        raise tvm.error.OpAttributeRequired('Attribute "mode" not found in operator pad.')
    if pad_mode not in ["constant", "edge", "reflect"]:
        raise tvm.error.OpAttributeInvalid("Value " + mode + ' in attribute "mode" is not valid')
    if "pad_width" not in attrs.attrs:
        raise tvm.error.OpAttributeRequired('Attribute "pad_width" not found in operator pad.')
    # Begin to parse tuple of tuple, we cannot use get_int_tuple here because it's a tuple of tuple.
    pad_width = attrs.attrs["pad_width"]
    pad_width = pad_width.replace("(", "[")
    pad_width = pad_width.replace(")", "]")
    pad_width = json.loads(pad_width)
    constant_values = attrs.get_float("constant_values", 0.0)
    return _op.nn.pad(
        data=inputs[0], pad_width=pad_width, pad_value=constant_values, pad_mode=pad_mode
    )


def _mx_npi_concatenate(inputs, attrs):
    axis = attrs.get_str("axis", "0")
    if axis == "None":
        return _op.reshape(_op.concatenate(tuple(inputs), axis=0), (-1,))
    else:
        return _op.concatenate(tuple(inputs), axis=int(axis))


def _mx_npi_stack(inputs, attrs):
    axis = attrs.get_str("axis", "0")
    if axis == "None":
        return _op.reshape(_op.stack(tuple(inputs), axis=0), (-1,))
    else:
        return _op.stack(tuple(inputs), axis=int(axis))


def _mx_npx_reshape(inputs, attrs):
    shape = attrs.get_int_tuple("newshape")
    reverse = attrs.get_bool("reverse", False)
    shape_list = list(shape)
    old_shape = get_const_tuple(_infer_type(inputs[0]).checked_type.shape)
    new_shape = []
    if reverse:
        old_shape = old_shape[::-1]
        shape_list = shape_list[::-1]
    ptr = 0
    unknown_axis = None
    src_ptr = 0
    while src_ptr < len(shape_list):
        ele = shape_list[src_ptr]
        src_ptr += 1
        if ele > 0:
            new_shape.append(ele)
            ptr += 1
        elif ele == -1:
            new_shape.append(-1)
            if unknown_axis is not None:
                raise tvm.error.OpAttributeInvalid("Can only have one -1 in the input shape.")
            unknown_axis = len(new_shape)
            ptr += 1
        elif ele == -2:
            new_shape.append(old_shape[ptr])
            ptr += 1
        elif ele == -3:
            if old_shape[ptr] != 1:
                raise tvm.error.OpAttributeInvalid(
                    "Dimension of the original shape "
                    "that corresponds to -3 must be 1. Received"
                    " {}".format(old_shape[ptr])
                )
            ptr += 1
        elif ele == -4:
            new_shape += old_shape[ptr:]
            break
        elif ele == -5:
            new_shape.append(old_shape[ptr] * old_shape[ptr + 1])
            ptr += 2
        elif ele == -6:
            # Split axis
            lhs = shape_list[src_ptr]
            rhs = shape_list[src_ptr + 1]
            src_ptr += 2
            if lhs == -1 and rhs == -1:
                raise tvm.error.OpAttributeInvalid("The lhs and rhs can not both be -1.")
            if lhs == -1:
                if old_shape[ptr] % rhs != 0:
                    raise tvm.error.OpAttributeInvalid(
                        "When splitting the axis, "
                        "the dimension of the split axis must "
                        "be divisible by the splitted values."
                    )
                lhs = old_shape[ptr] // rhs
            if rhs == -1:
                if old_shape[ptr] % lhs != 0:
                    raise tvm.error.OpAttributeInvalid(
                        "When splitting the axis, "
                        "the dimension of the split axis must "
                        "be divisible by the splitted values."
                    )
                rhs = old_shape[ptr] // lhs
            new_shape.append(lhs)
            new_shape.append(rhs)
            ptr += 1
        else:
            raise tvm.error.OpAttributeInvalid("Shape dimension %d is not supported" % ele)
    if reverse:
        new_shape = new_shape[::-1]
    return _op.reshape(inputs[0], newshape=new_shape)


def _mx_split_v2(inputs, attrs):
    axis = attrs.get_int("axis")
    indices = list(attrs.get_int_tuple("indices", []))
    # remove the prefix '0'
    if len(indices) != 0 and indices[0] == 0:
        indices.remove(0)
    sections = attrs.get_int("sections", 0)
    indices_or_sections = list(indices) if len(indices) != 0 else sections
    res = _op.split(inputs[0], indices_or_sections=indices_or_sections, axis=axis)
    if attrs.get_bool("squeeze_axis", False):
        res = tuple([_op.squeeze(x, axis=[axis]) for x in res])
    return res


def _mx_npi_where_rscalar(inputs, attrs):
    cond, dat = inputs
    scalar = attrs.get_float("scalar")
    cond_shape = get_const_tuple(_infer_type(cond).checked_type.shape)
    dat_shape = get_const_tuple(_infer_type(dat).checked_type.shape)
    dtype = _infer_type(dat).checked_type.dtype
    # Check for broadcasting
    out_shape = np.broadcast(np.empty(cond_shape), np.empty(dat_shape)).shape
    if out_shape != cond_shape:
        cond = _op.broadcast_to(cond, out_shape)
    if out_shape != dat_shape:
        dat = _op.broadcast_to(dat, out_shape)
    scalar = _expr.const(scalar, dtype=dtype)
    ones = _op.ones_like(dat)
    scalar = _op.multiply(ones, scalar)
    return _op.where(cond, dat, scalar)


# Note: due to attribute conversion constraint
# ops in the identity set must be attribute free
_identity_list = [
    "abs",
    "log",
    "exp",
    "erf",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "trunc",
    "sign",
    "sigmoid",
    "negative",
    "reshape_like",
    "zeros_like",
    "ones_like",
    "cos",
    "cosh",
    "sin",
    "sinh",
    "tan",
    "tanh",
    "where",
]

_convert_map = {
    "_copy": _rename(_op.copy),
    "relu": _rename(_op.nn.relu),
    "broadcast_add": _rename(_op.add),
    "broadcast_plus": _rename(_op.add),
    "broadcast_sub": _rename(_op.subtract),
    "broadcast_minus": _rename(_op.subtract),
    "broadcast_mul": _rename(_op.multiply),
    "broadcast_div": _rename(_op.divide),
    "broadcast_mod": _rename(_op.mod),
    "broadcast_maximum": _rename(_op.maximum),
    "broadcast_minimum": _rename(_op.minimum),
    "broadcast_power": _rename(_op.power),
    "arccos": _rename(_op.acos),
    "arcsin": _rename(_op.asin),
    "arctan": _rename(_op.atan),
    "arccosh": _rename(_op.acosh),
    "arcsinh": _rename(_op.asinh),
    "arctanh": _rename(_op.atanh),
    "broadcast_equal": _mx_compare(_op.equal, _rename),
    "broadcast_not_equal": _mx_compare(_op.not_equal, _rename),
    "broadcast_greater": _mx_compare(_op.greater, _rename),
    "broadcast_greater_equal": _mx_compare(_op.greater_equal, _rename),
    "broadcast_lesser": _mx_compare(_op.less, _rename),
    "broadcast_lesser_equal": _mx_compare(_op.less_equal, _rename),
    "broadcast_logical_or": _mx_broadcast_logical(_op.logical_or),
    "broadcast_logical_and": _mx_broadcast_logical(_op.logical_and),
    "broadcast_logical_xor": _mx_broadcast_logical(_op.logical_xor),
    "broadcast_to": _mx_broadcast_to,
    "broadcast_like": _mx_broadcast_like,
    "logical_not": _mx_logical_not,
    "_equal": _mx_compare(_op.equal, _rename),
    "_not_equal": _mx_compare(_op.not_equal, _rename),
    "_greater": _mx_compare(_op.greater, _rename),
    "_greater_equal": _mx_compare(_op.greater_equal, _rename),
    "_lesser": _mx_compare(_op.less, _rename),
    "_lesser_equal": _mx_compare(_op.less_equal, _rename),
    "elemwise_add": _rename(_op.add),
    "elemwise_sub": _rename(_op.subtract),
    "elemwise_mul": _rename(_op.multiply),
    "elemwise_div": _rename(_op.divide),
    "_maximum": _rename(_op.maximum),
    "_minimum": _rename(_op.minimum),
    "flatten": _rename(_op.nn.batch_flatten),
    "Flatten": _rename(_op.nn.batch_flatten),
    # scalar power
    "square": _mx_make_power(2),
    "rsqrt": _mx_make_power(-1 / 2),
    "cbrt": _mx_make_power(1 / 3),
    "rcbrt": _mx_make_power(-1 / 3),
    "__pow_scalar__": _binop_scalar(_op.power),
    "_power_scalar": _binop_scalar(_op.power),
    "__rsub_scalar__": _rbinop_scalar(_op.subtract),
    "_rminus_scalar": _rbinop_scalar(_op.subtract),
    "__rdiv_scalar__": _rbinop_scalar(_op.divide),
    "_rdiv_scalar": _rbinop_scalar(_op.divide),
    "__rpow_scalar__": _rbinop_scalar(_op.power),
    # scalar op
    "__add_scalar__": _binop_scalar(_op.add),
    "_plus_scalar": _binop_scalar(_op.add),
    "__sub_scalar__": _binop_scalar(_op.subtract),
    "_minus_scalar": _binop_scalar(_op.subtract),
    "__mul_scalar__": _binop_scalar(_op.multiply),
    "_mul_scalar": _binop_scalar(_op.multiply),
    "__div_scalar__": _binop_scalar(_op.divide),
    "_div_scalar": _binop_scalar(_op.divide),
    "log2": _mx_make_logarithm(2),
    "log10": _mx_make_logarithm(10),
    "log1p": _mx_log1p,
    "expm1": _mx_expm1,
    "_equal_scalar": _mx_compare(_op.equal, _binop_scalar),
    "_not_equal_scalar": _mx_compare(_op.not_equal, _binop_scalar),
    "_greater_scalar": _mx_compare(_op.greater, _binop_scalar),
    "_greater_equal_scalar": _mx_compare(_op.greater_equal, _binop_scalar),
    "_lesser_scalar": _mx_compare(_op.less, _binop_scalar),
    "_lesser_equal_scalar": _mx_compare(_op.less_equal, _binop_scalar),
    "_maximum_scalar": _binop_scalar(_op.maximum),
    "_minimum_scalar": _binop_scalar(_op.minimum),
    # reduction ops
    "mean": _reduce(_op.mean),
    "max": _reduce(_op.max),
    "min": _reduce(_op.min),
    "sum": _reduce(_op.sum),
    "max_axis": _reduce(_op.max),
    "min_axis": _reduce(_op.min),
    "sum_axis": _reduce(_op.sum),
    "argmax": _arg_reduce(_op.argmax),
    "argmin": _arg_reduce(_op.argmin),
    # init ops
    "_ones": _init_op(_op.ones),
    # softmax
    "softmax": _softmax_op(_op.nn.softmax),
    "log_softmax": _softmax_op(_op.nn.log_softmax),
    "Softmax": _softmax_op(_op.nn.softmax),
    "softsign": _mx_softsign,
    "softmin": _mx_softmin,
    "hard_sigmoid": _mx_hard_sigmoid,
    "reciprocal": _mx_reciprocal,
    # per op specialization
    "Reshape": _reshape,
    "reshape": _reshape,
    "Cast": _cast,
    "amp_cast": _cast,
    "amp_multicast": _mx_amp_multicast,
    "clip": _clip,
    "transpose": _transpose,
    "UpSampling": _upsampling,
    "add_n": _elemwise_sum,
    # MXNet specific implementations
    "_zeros": _mx_zeros,
    "FullyConnected": _mx_fully_connected,
    "Activation": _mx_activations,
    "Convolution": _mx_conv,
    "Convolution_v1": _mx_conv2d,
    "Deconvolution": _mx_conv_transpose,
    "Pooling": _mx_pooling,
    "Pooling_v1": _mx_pooling,
    "Dropout": _mx_dropout,
    "BatchNorm": _mx_batch_norm,
    "BatchNorm_v1": _mx_batch_norm,
    "_contrib_SyncBatchNorm": _mx_batch_norm,
    "InstanceNorm": _mx_instance_norm,
    "LayerNorm": _mx_layer_norm,
    "GroupNorm": _mx_group_norm,
    "LRN": _mx_lrn,
    "L2Normalization": _mx_l2_normalize,
    "slice": _mx_slice,
    "slice_like": _mx_slice_like,
    "slice_axis": _mx_slice_axis,
    "SliceChannel": _mx_split,
    "split": _mx_split,
    "_split_v2": _mx_split_v2,
    "SwapAxis": _mx_swap_axis,
    "expand_dims": _mx_expand_dims,
    "Concat": _mx_concat,
    "concat": _mx_concat,
    "stack": _mx_stack,
    "dot": _mx_dot,
    "batch_dot": _mx_batch_dot,
    "LeakyReLU": _mx_leaky_relu,
    "_arange": _mx_arange,
    "_full": _mx_full,
    "repeat": _mx_repeat,
    "tile": _mx_tile,
    "pad": _mx_pad,
    "Pad": _mx_pad,
    "take": _mx_take,
    "gather_nd": _mx_gather_nd,
    "reverse": _mx_reverse,
    "SequenceReverse": _mx_sequence_reverse,
    "squeeze": _mx_squeeze,
    "broadcast_axis": _mx_broadcast_axis,
    "broadcast_axes": _mx_broadcast_axis,
    "BlockGrad": _mx_BlockGrad,
    "shape_array": _mx_shape_array,
    "Embedding": _mx_embedding,
    "argsort": _mx_argsort,
    "topk": _mx_topk,
    "_unravel_index": _mx_unravel_index,
    "SequenceMask": _mx_sequence_mask,
    "SoftmaxOutput": _mx_softmax_output,
    "SoftmaxActivation": _mx_softmax_activation,
    "LinearRegressionOutput": _mx_linear_regression_output,
    "LogisticRegressionOutput": _mx_logistic_regression_output,
    "smooth_l1": _mx_smooth_l1,
    "make_loss": _mx_make_loss,
    "_contrib_div_sqrt_dim": _mx_contrib_div_sqrt_dim,
    "_contrib_arange_like": _mx_contrib_arange_like,
    "one_hot": _mx_one_hot,
    "depth_to_space": _mx_depth_to_space,
    "space_to_depth": _mx_space_to_depth,
    "Correlation": _mx_correlation,
    # vision
    "_contrib_BilinearResize2D": _mx_resize,
    "_contrib_MultiBoxPrior": _mx_multibox_prior,
    "_contrib_MultiBoxDetection": _mx_multibox_detection,
    "_contrib_ROIAlign": _mx_roi_align,
    "ROIPooling": _mx_roi_pooling,
    "_contrib_Proposal": _mx_proposal,
    "_contrib_MultiProposal": _mx_proposal,
    "_contrib_box_nms": _mx_box_nms,
    "_contrib_box_decode": _mx_box_decode,
    "_contrib_DeformableConvolution": _mx_deformable_convolution,
    "_contrib_AdaptiveAvgPooling2D": _mx_adaptive_avg_pooling,
    "GridGenerator": _mx_grid_generator,
    "BilinearSampler": _mx_bilinear_sampler,
    # NLP
    "RNN": _mx_rnn_layer,
    "_rnn_param_concat": _mx_rnn_param_concat,
    "_contrib_interleaved_matmul_selfatt_qk": _mx_contrib_interleaved_matmul_selfatt_qk,
    "_contrib_interleaved_matmul_selfatt_valatt": _mx_contrib_interleaved_matmul_selfatt_valatt,
    # control flow
    "_cond": _mx_cond,
    # Depricated:
    "Crop": _mx_crop_like,
    # List of missing operators that are present in NNVMv1
    # TODO(tvm-tvm): support all operators.
    #
    # "contrib_fifo_buffer": _mx_contrib_fifo_buffer,
    "ring_buffer": _mx_contrib_fifo_buffer,
    # Qnn ops
    "_contrib_quantize_v2": _qnn_quantize,
    "_contrib_quantized_concat": _qnn_contrib_concat,
    # "_contrib_quantized_fifo_buffer": _qnn_contrib_quantized_fifo_buffer,
    "_contrib_quantized_ring_buffer": _qnn_contrib_quantized_fifo_buffer,
    "_sg_mkldnn_conv": _qnn_conv,
    "_contrib_quantized_flatten": _qnn_flatten,
    "_contrib_dequantize": _qnn_dequantize,
    "_contrib_quantized_act": _qnn_activation,
    "_contrib_quantized_pooling": _qnn_pooling,
    "_contrib_quantized_batch_norm": _qnn_batch_norm,
    "_sg_mkldnn_fully_connected": _qnn_fully_connected,
    # numpy
    "_np_transpose": _mx_npi_transpose,
    "_npi_transpose": _mx_npi_transpose,
    "_npi_pad": _mx_npi_pad,
    "_npi_concatenate": _mx_npi_concatenate,
    "_npx_reshape": _mx_npx_reshape,
    "_np_copy": _rename(_op.copy),
    "_npi_copy": _rename(_op.copy),
    "_npi_power": _rename(_op.power),
    "_npi_power_scalar": _binop_scalar(_op.power),
    "_npi_multiply": _rename(_op.multiply),
    "_npi_multiply_scalar": _binop_scalar(_op.multiply),
    "_npi_add": _rename(_op.add),
    "_npi_add_scalar": _binop_scalar(_op.add),
    "_npi_subtract": _rename(_op.subtract),
    "_npi_subtract_scalar": _binop_scalar(_op.subtract),
    "_npi_where_rscalar": _mx_npi_where_rscalar,
    "_npi_less": _rename(_op.less),
    "_npi_less_equal": _mx_compare(_op.less_equal, _rename),
    "_npi_tanh": _rename(_op.tanh),
    "_npi_true_divide_scalar": _binop_scalar(_op.divide),
    "_npi_stack": _mx_npi_stack,
}

# set identity list
_convert_map.update({k: _rename(k) for k in _identity_list})

_control_flow_ops = ["_cond", "_foreach", "_while_loop"]
_qnn_subgraph_ops = ["_sg_mkldnn_conv", "_sg_mkldnn_fully_connected"]
_subgraph_ops = _control_flow_ops + _qnn_subgraph_ops
_params_ops = ["_contrib_quantized_ring_buffer"]


def _get_op_params(children, attrs, op_name, node, params):
    op_params = [children, attrs]
    if op_name in _subgraph_ops:
        subgraphs = node["subgraphs"]
        op_params.append(subgraphs)
        if op_name in _qnn_subgraph_ops:
            op_params.append(params)
    if op_name in _params_ops:
        op_params.append(params)
    return op_params


def _from_mxnet_impl(symbol, shape_dict, dtype_info, params=None, mod=None):
    # pylint: disable=unused-argument
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

    mod : tvm.IRModule
        The module that contains global information. It will be used for
        converting ops that need global information, e.g. control-flow ops.

    Returns:
    -------
    func : tvm.relay.Function
        Converted relay Function
    """
    assert symbol is not None
    if isinstance(symbol, dict):
        jgraph = symbol
    else:
        jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}
    shape_idx = 0

    # Check if there have any unsupported ops
    unsupported = {}
    for node in jnodes:
        op_name = node["op"]
        if op_name != "null" and op_name not in _convert_map:
            if op_name not in unsupported:
                unsupported[op_name] = 0
            unsupported[op_name] += 1

    if unsupported:
        msg = "\n".join(["{}: {}".format(op_name, cnt) for op_name, cnt in unsupported.items()])
        raise tvm.error.OpNotImplemented(
            "One or more operators are not supported in frontend MXNet:\n{}".format(msg)
        )

    for nid, node in enumerate(jnodes):
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = StrAttrsDict(node.get("attrs", {}))
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            if isinstance(shape_dict, dict):
                shape = shape_dict[node_name] if node_name in shape_dict else None
            elif isinstance(shape_dict, (list, tuple)):
                shape = shape_dict[shape_idx]
            else:
                raise ValueError("Unknown type of shape_dict: %s" + type(shape_dict))
            if isinstance(dtype_info, dict):
                dtype = dtype_info[node_name] if node_name in dtype_info else "float32"
            elif isinstance(dtype_info, (list, tuple)):
                dtype = dtype_info[shape_idx]
            else:
                dtype = dtype_info
            if isinstance(shape_dict, (list, tuple)):
                shape_idx += 1
            node_map[nid] = [_expr.var(node_name, shape=shape, dtype=dtype)]
        else:
            assert op_name in _convert_map
            op_params = _get_op_params(children, attrs, op_name, node, params)
            res = _convert_map[op_name](*op_params)
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
    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    func = _function.Function(analysis.free_vars(outputs), outputs)
    return func


def _update_shape_dtype(shape, dtype, params):
    """Update shape dtype given params information"""
    shape = {} if shape is None else shape
    if not params:
        return shape, dtype
    shape = shape.copy()
    shape.update({k: v.shape for k, v in params.items()})
    if isinstance(dtype, str):
        for k, v in params.items():
            if v.dtype != dtype:
                raise ValueError("%s: dtype not expected %s vs %s" % (k, dtype, v.dtype))
    else:
        dtype = dtype.copy()
        dtype.update({k: str(v.dtype) for k, v in params.items()})
    return shape, dtype


def from_mxnet(symbol, shape=None, dtype="float32", arg_params=None, aux_params=None):
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
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import mxnet as mx  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("{}. MXNet is required to parse symbols.".format(e))

    mod = IRModule()
    if isinstance(symbol, mx.sym.Symbol):
        params = {}
        arg_params = arg_params if arg_params else {}
        aux_params = aux_params if aux_params else {}
        for k, v in arg_params.items():
            params[k] = _nd.array(v.asnumpy())
        for k, v in aux_params.items():
            params[k] = _nd.array(v.asnumpy())
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        func = _from_mxnet_impl(symbol, shape, dtype, params, mod)
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
        func = _from_mxnet_impl(sym, shape, dtype, params, mod)
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    mod["main"] = func
    return mod, params
