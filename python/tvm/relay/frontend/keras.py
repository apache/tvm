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
# pylint: disable=invalid-name, import-self, import-outside-toplevel
"""Keras frontend."""
import dis
import sys
import numpy as np
import tvm
from tvm.ir import IRModule, TensorType, TupleType

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ... import nd as _nd
from .common import ExprTable, new_var

__all__ = ["from_keras"]


def _check_data_format(keras_layer):
    if hasattr(keras_layer, ("data_format")):
        if keras_layer.data_format != "channels_last":
            raise ValueError("Keras frontend currently supports data_format = channels_last only.")


def _get_pad_pair(input1d, kernel1d, stride1d):
    out1d = (input1d + stride1d - 1) // stride1d
    pad = np.maximum((out1d - 1) * stride1d + kernel1d - input1d, 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return [pad_before, pad_after]


def _get_elu(inexpr, alpha):
    """A helper method for elu."""
    return _op.negative(alpha) * _op.nn.relu(
        _expr.const(1.0, dtype="float32") - _op.exp(inexpr)
    ) + _op.nn.relu(inexpr)


def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]


def _convert_recurrent_activation(inexpr, keras_layer):
    act_type = keras_layer.recurrent_activation.__name__
    return _convert_activation(inexpr, act_type, None, None, None)


def _convert_activation(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    if isinstance(keras_layer, str):
        act_type = keras_layer
    else:
        if sys.version_info.major < 3:
            act_type = keras_layer.activation.func_name
        else:
            act_type = keras_layer.activation.__name__
    if act_type == "linear":
        if isinstance(keras_layer, str):
            return inexpr
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") else 1.0
        beta = keras_layer.beta if hasattr(keras_layer, "beta") else 0.0
        alpha = _expr.const(alpha, dtype="float32")
        beta = _expr.const(beta, dtype="float32")
        return _op.add(_op.multiply(inexpr, alpha), beta)
    if act_type == "softmax":
        axis = 1 if data_layout == "NCHW" else -1
        return _op.nn.softmax(inexpr, axis)
    if act_type == "sigmoid":
        return _op.sigmoid(inexpr)
    if act_type == "tanh":
        return _op.tanh(inexpr)
    if act_type == "relu":
        return _op.nn.relu(inexpr)
    if act_type == "softplus":
        return _op.log(_op.add(_op.exp(inexpr), _expr.const(1.0, dtype="float32")))
    if act_type == "elu":
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") else 1.0
        alpha = _expr.const(alpha, dtype="float32")
        return _get_elu(inexpr, alpha)
    if act_type == "selu":
        # Alpha, Gamma values obtained from https://arxiv.org/abs/1706.02515
        alpha = (
            keras_layer.alpha
            if hasattr(keras_layer, "alpha")
            else 1.6732632423543772848170429916717
        )
        gamma = (
            keras_layer.gamma
            if hasattr(keras_layer, "gamma")
            else 1.0507009873554804934193349852946
        )
        alpha = _expr.const(alpha, dtype="float32")
        gamma = _expr.const(gamma, dtype="float32")
        return gamma * _get_elu(inexpr, alpha)
    if act_type == "relu6":
        return _op.clip(inexpr, a_min=0.0, a_max=6.0)
    if act_type == "softsign":
        return inexpr / (_expr.const(1.0, dtype="float32") + _op.abs(inexpr))
    if act_type == "hard_sigmoid":
        x = (_expr.const(0.2, dtype="float32") * inexpr) + _expr.const(0.5, dtype="float32")
        return _op.clip(x, a_min=0.0, a_max=1.0)
    if act_type == "swish":
        return inexpr * _op.sigmoid(inexpr)

    raise tvm.error.OpNotImplemented(f"Operator {act_type} is not supported in frontend Keras.")


def _convert_advanced_activation(inexpr, keras_layer, etab, data_layout, input_shape=None):
    act_type = type(keras_layer).__name__
    if input_shape is None:
        input_shape = keras_layer.input_shape

    if act_type == "Softmax":
        axis = keras_layer.axis
        dims = len(input_shape) if input_shape else 0
        if isinstance(axis, list):
            raise tvm.error.OpAttributeUnImplemented(f"Softmax with axes {axis} is not supported.")
        if data_layout == "NCHW":
            if dims == 0:
                axis = 0
            elif axis == -1:
                axis = 1
            else:
                axis = axis + 1 if axis < dims - 1 else 1
        return _op.nn.softmax(inexpr, axis=axis)
    if act_type == "ReLU":
        if np.isnan(keras_layer.threshold).any():
            raise tvm.error.OpAttributeInvalid("The threshold value of a ReLU cannot be None.")
        threshold = _expr.const(keras_layer.threshold, dtype="float32")
        if keras_layer.max_value and float(keras_layer.threshold) == 0:
            # f(x) = max_value, for x >= max_value
            # f(x) = x,         for threshold <= x < max_value
            return _op.clip(inexpr, a_min=0.0, a_max=float(keras_layer.max_value))
        if keras_layer.max_value and _op.greater(threshold, inexpr).astype("float32"):
            # f(x) = negative_slope * (inexpr - threshold)
            negative_slope = _expr.const(keras_layer.negative_slope, dtype="float32")
            return _op.multiply(negative_slope, _op.subtract(inexpr, threshold))
        return _op.nn.relu(inexpr)
    if act_type == "LeakyReLU":
        if np.isnan(keras_layer.alpha).any():
            raise tvm.error.OpAttributeInvalid("The alpha value of a LeakyReLU cannot be None.")
        return _op.nn.leaky_relu(inexpr, alpha=float(keras_layer.alpha))
    if act_type == "ELU":
        if np.isnan(keras_layer.alpha).any():
            raise tvm.error.OpAttributeInvalid("The alpha value of a ELU cannot be None.")
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") else 1.0
        alpha = _expr.const(alpha, dtype="float32")
        return _get_elu(inexpr, alpha)
    if act_type == "PReLU":
        assert hasattr(keras_layer, "alpha"), "alpha required for PReLU."
        _check_data_format(keras_layer)
        size = len(keras_layer.alpha.shape)
        if data_layout == "NCHW":
            alpha = etab.new_const(keras_layer.get_weights()[0].transpose(np.roll(range(size), 1)))
        else:
            alpha = etab.new_const(keras_layer.get_weights()[0])
        return _op.negative(alpha) * _op.nn.relu(_op.negative(inexpr)) + _op.nn.relu(inexpr)
    if act_type == "ThresholdedReLU":
        theta = keras_layer.theta if hasattr(keras_layer, "theta") else 1.0
        return _op.multiply(
            inexpr, _op.greater(inexpr, _expr.const(theta, dtype="float32")).astype("float32")
        )

    raise tvm.error.OpNotImplemented(f"Operator {act_type} is not supported in frontend Keras.")


def _convert_merge(
    inexpr, keras_layer, _, input_shape=None, data_layout=None
):  # pylint: disable=unused-argument
    merge_type = type(keras_layer).__name__
    ret = inexpr[0]
    if merge_type == "Dot":
        axes = keras_layer.axes
        if isinstance(keras_layer.axes, int):
            axes = [keras_layer.axes, keras_layer.axes]
        if isinstance(axes, list):
            if len(axes) != 2:
                raise tvm.error.OpAttributeUnImplemented(
                    f"Dot with axes {keras_layer.axes} is not supported."
                )
            for i, axis in enumerate(axes):
                if axis not in [1, 2]:
                    raise tvm.error.OpAttributeUnImplemented(
                        f"Dot with axes {keras_layer.axes} is not supported."
                    )
                if axes[i] == 2:
                    inexpr[i] = _op.transpose(inexpr[i], axes=[0, 2, 1])
        else:
            raise tvm.error.OpAttributeUnImplemented(
                f"Dot with axes {keras_layer.axes} is not supported."
            )
        ret_dot = _op.nn.batch_matmul(inexpr[0], inexpr[1])
        ret = _op.transpose(ret_dot, axes=[0, 2, 1])
    elif merge_type == "Subtract":
        assert len(inexpr) == 2, "Subtract merge takes 2 inputs."
        ret = _op.subtract(ret, inexpr[1])
    elif merge_type in ["Add", "Multiply", "Minimum", "Maximum"]:
        op_map = {
            "Add": _op.add,
            "Multiply": _op.multiply,
            "Minimum": _op.minimum,
            "Maximum": _op.maximum,
        }
        for i in range(1, len(inexpr)):
            ret = op_map[merge_type](ret, inexpr[i])
    elif merge_type == "Average":
        for i in range(1, len(inexpr)):
            ret = _op.add(ret, inexpr[i])
        ret = ret / _expr.const(len(inexpr), dtype="float32")
    else:
        raise tvm.error.OpNotImplemented(
            f"Operator {merge_type} is not supported in frontend Keras."
        )
    return ret


def _convert_permute(
    inexpr, keras_layer, _, input_shape=None, data_layout=None
):  # pylint: disable=unused-argument
    return _op.transpose(inexpr, axes=(0,) + keras_layer.dims)


def _convert_embedding(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    indices = inexpr
    weightList = keras_layer.get_weights()
    weight = etab.new_const(weightList[0])
    out = _op.take(weight, indices.astype("int32"), axis=0)

    return out


def _convert_dense(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    weightList = keras_layer.get_weights()
    weight = etab.new_const(weightList[0].transpose([1, 0]))
    params = {"weight": weight, "units": weightList[0].shape[1]}
    units = list(weightList[0].shape)[1]
    assert units > 0, "The value of units must be a positive integer"
    if input_shape is None:
        input_shape = keras_layer.input_shape
    input_dim = len(input_shape)
    # In case of RNN dense, input shape will be (1, 1, n)
    if input_dim > 2:
        input_shape = tuple(dim if dim else 1 for dim in _as_list(input_shape)[0])
        # Keras has no limitations on the shape of the input tensor. But our
        # dense op expects 2D input. All inputs with number of dimensions > 2
        # are reshaped all "batch" axes into one.
        # For example: (N, d1, d2, d3) -> (N * d1 * d2, d3)
        new_batch_size = np.prod(input_shape[:-1])
        inexpr = _op.reshape(inexpr, newshape=(new_batch_size, input_shape[-1]))
    out = _op.nn.dense(data=inexpr, **params)
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        out = _op.nn.bias_add(out, bias)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != "linear":
        out = _convert_activation(out, act_type, etab, data_layout)
    if input_dim > 2:
        out_shape = (*input_shape[:-1], units)
        out = _op.reshape(out, newshape=out_shape)
    return out


def _convert_convolution1d(inexpr, keras_layer, etab, data_layout, input_shape=None):
    is_deconv = type(keras_layer).__name__ == "Conv1DTranspose"

    if input_shape is None:
        input_shape = keras_layer.input_shape
    _check_data_format(keras_layer)
    weightList = keras_layer.get_weights()
    weight = weightList[0]

    if data_layout == "NWC":
        kernel_layout = "WIO"
        if is_deconv:
            kernel_layout = "WOI"
    else:
        kernel_layout = "OIW"
        if is_deconv:
            kernel_layout = "IOW"
        msg = (
            f"Kernel layout with {kernel_layout} is not supported for operator Convolution1D "
            f"in frontend Keras."
        )
        raise tvm.error.OpAttributeUnImplemented(msg)

    if is_deconv:
        if kernel_layout == "IOW":
            weight = weight.transpose([2, 1, 0])
        kernel_w, n_filters, _ = weight.shape
    else:
        kernel_w, _, n_filters = weight.shape

    dilation_rate = keras_layer.dilation_rate
    if isinstance(dilation_rate, (list, tuple)):
        dilation = [dilation_rate[0]]
    else:
        dilation = [dilation_rate]

    dilated_kernel_w = (kernel_w - 1) * dilation[0] + 1
    stride_w = keras_layer.strides[0]
    params = {
        "weight": etab.new_const(weight),
        "kernel_size": [kernel_w],
        "strides": [stride_w],
        "dilation": dilation,
        "padding": [0],
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
    }
    params["channels"] = n_filters

    if keras_layer.padding == "valid":
        pass
    # calculate the padding values
    elif keras_layer.padding == "same":
        in_w = input_shape[1]
        pad_w = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        params["padding"] = [pad_w[0], pad_w[1]]
    else:
        msg = (
            f"Padding with {keras_layer.padding} is not supported for operator Convolution3D "
            f"in frontend Keras."
        )
        raise tvm.error.OpAttributeUnImplemented(msg)

    if is_deconv:
        out = _op.nn.conv1d_transpose(data=inexpr, **params)
    else:
        out = _op.nn.conv1d(data=inexpr, **params)

    channel_axis = -1 if data_layout == "NWC" else 1
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        out = _op.nn.bias_add(out, bias, channel_axis)

    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != "linear":
        out = _convert_activation(out, act_type, etab, data_layout)

    return out


def _convert_convolution(inexpr, keras_layer, etab, data_layout, input_shape=None):
    _check_data_format(keras_layer)
    is_deconv = type(keras_layer).__name__ == "Conv2DTranspose"
    is_depthconv = type(keras_layer).__name__ == "DepthwiseConv2D"
    weightList = keras_layer.get_weights()
    weight = weightList[0]
    if input_shape is None:
        input_shape = keras_layer.input_shape

    if data_layout == "NHWC":
        if is_depthconv:
            kernel_layout = "HWOI"
        elif is_deconv:
            kernel_layout = "HWOI"
        else:
            kernel_layout = "HWIO"
    else:
        if is_deconv:
            kernel_layout = "IOHW"
        else:
            kernel_layout = "OIHW"

    if is_deconv:
        kernel_h, kernel_w, n_filters, in_channels = weight.shape
        if kernel_layout == "IOHW":
            weight = weight.transpose([3, 2, 0, 1])
    elif is_depthconv:
        kernel_h, kernel_w, in_channels, depth_mult = weight.shape
        if kernel_layout == "OIHW":
            weight = weight.transpose([2, 3, 0, 1])
    elif data_layout == "NCHW":
        kernel_h, kernel_w, in_channels, n_filters = weight.shape
        weight = weight.transpose([3, 2, 0, 1])
    else:
        kernel_h, kernel_w, in_channels, n_filters = weight.shape
    if isinstance(keras_layer.dilation_rate, (list, tuple)):
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]
    dilated_kernel_h = (kernel_h - 1) * dilation[0] + 1
    dilated_kernel_w = (kernel_w - 1) * dilation[1] + 1
    stride_h, stride_w = keras_layer.strides
    params = {
        "weight": etab.new_const(weight),
        "kernel_size": [kernel_h, kernel_w],
        "strides": [stride_h, stride_w],
        "dilation": dilation,
        "padding": [0, 0],
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
    }
    if is_depthconv:
        params["channels"] = in_channels * depth_mult
        params["groups"] = in_channels
    else:
        params["channels"] = n_filters
    if is_deconv and keras_layer.output_padding:
        params["output_padding"] = keras_layer.output_padding
    if keras_layer.padding == "valid":
        pass
    # we insert a separate pad operator
    elif keras_layer.padding == "same":
        in_h = input_shape[1]
        in_w = input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        params["padding"] = (pad_t, pad_l, pad_b, pad_r)
    else:
        msg = (
            f"Padding with {keras_layer.padding} is not supported for operator Convolution "
            f"in frontend Keras."
        )
        raise tvm.error.OpAttributeUnImplemented(msg)
    if is_deconv:
        out = _op.nn.conv2d_transpose(data=inexpr, **params)
    else:
        out = _op.nn.conv2d(data=inexpr, **params)

    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        if data_layout == "NCHW":
            out = _op.nn.bias_add(out, bias)
        else:
            out = _op.nn.bias_add(out, bias, axis=-1)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != "linear":
        out = _convert_activation(out, act_type, etab, data_layout)

    return out


def _convert_convolution3d(inexpr, keras_layer, etab, data_layout, input_shape=None):
    _check_data_format(keras_layer)
    is_deconv = type(keras_layer).__name__ == "Conv3DTranspose"
    weightList = keras_layer.get_weights()
    weight = weightList[0]
    if input_shape is None:
        input_shape = keras_layer.input_shape

    if data_layout == "NDHWC":
        kernel_layout = "DHWIO"
        if is_deconv:
            kernel_layout = "DHWOI"
    else:
        kernel_layout = "OIDHW"
        if is_deconv:
            kernel_layout = "IODHW"
        msg = (
            f"Kernel layout with {kernel_layout} is not supported for operator Convolution3D "
            f"in frontend Keras."
        )
        raise tvm.error.OpAttributeUnImplemented(msg)

    if is_deconv:
        kernel_d, kernel_h, kernel_w, n_filters, _ = weight.shape
        if kernel_layout == "IODHW":
            weight = weight.transpose([4, 3, 0, 1, 2])
    else:
        kernel_d, kernel_h, kernel_w, _, n_filters = weight.shape

    dilation_rate = keras_layer.dilation_rate
    if isinstance(dilation_rate, (list, tuple)):
        dilation = [dilation_rate[0], dilation_rate[1], dilation_rate[2]]
    else:
        dilation = [dilation_rate, dilation_rate, dilation_rate]

    dilated_kernel_d = (kernel_d - 1) * dilation[0] + 1
    dilated_kernel_h = (kernel_h - 1) * dilation[1] + 1
    dilated_kernel_w = (kernel_w - 1) * dilation[2] + 1
    stride_d, stride_h, stride_w = keras_layer.strides
    params = {
        "weight": etab.new_const(weight),
        "kernel_size": [kernel_d, kernel_h, kernel_w],
        "strides": [stride_d, stride_h, stride_w],
        "dilation": dilation,
        "padding": [0, 0, 0],
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
    }
    params["channels"] = n_filters
    if is_deconv and keras_layer.output_padding:
        params["output_padding"] = keras_layer.output_padding

    if keras_layer.padding == "valid":
        pass
    # calculate the padding values
    elif keras_layer.padding == "same":
        in_d = input_shape[1]
        in_h = input_shape[2]
        in_w = input_shape[3]
        pad_d = _get_pad_pair(in_d, dilated_kernel_d, stride_d)
        pad_h = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        pad_w = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        params["padding"] = [pad_d[0], pad_h[0], pad_w[0], pad_d[1], pad_h[1], pad_w[1]]
    else:
        msg = (
            f"Padding with {keras_layer.padding} is not supported for operator Convolution3D "
            f"in frontend Keras."
        )
        raise tvm.error.OpAttributeUnImplemented(msg)
    if is_deconv:
        out = _op.nn.conv3d_transpose(data=inexpr, **params)
    else:
        out = _op.nn.conv3d(data=inexpr, **params)

    channel_axis = -1 if data_layout == "NDHWC" else 1
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[1])
        out = _op.nn.bias_add(out, bias, channel_axis)

    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != "linear":
        out = _convert_activation(out, act_type, etab, None)

    return out


def _convert_separable_convolution(inexpr, keras_layer, etab, data_layout, input_shape=None):
    _check_data_format(keras_layer)

    if data_layout == "NHWC":
        kernel_layout = "HWOI"
    else:
        kernel_layout = "OIHW"

    if input_shape is None:
        input_shape = keras_layer.input_shape

    weightList = keras_layer.get_weights()
    # depthwise conv
    kernel_h, kernel_w, in_channels, depth_mult = weightList[0].shape
    stride_h, stride_w = keras_layer.strides
    if kernel_layout == "OIHW":
        weight0 = weightList[0].transpose([2, 3, 0, 1])
    else:
        weight0 = weightList[0]
    if isinstance(keras_layer.dilation_rate, (list, tuple)):
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]
    params0 = {
        "weight": etab.new_const(weight0),
        "channels": in_channels * depth_mult,
        "groups": in_channels,
        "kernel_size": [kernel_h, kernel_w],
        "strides": [stride_h, stride_w],
        "dilation": dilation,
        "padding": [0, 0],
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
    }
    if keras_layer.padding == "valid":
        pass
    # we insert a separate pad operator
    elif keras_layer.padding == "same":
        in_h = input_shape[1]
        in_w = input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, kernel_w, stride_w)
        params0["padding"] = (pad_t, pad_l, pad_b, pad_r)
    else:
        msg = (
            f"Padding with {keras_layer.padding} is not supported for operator Separable "
            f"Convolution in frontend Keras."
        )
        raise tvm.error.OpAttributeUnImplemented(msg)
    depthconv = _op.nn.conv2d(data=inexpr, **params0)
    # pointwise conv
    if kernel_layout == "OIHW":
        weight1 = weightList[1].transpose([3, 2, 0, 1])
    else:
        weight1 = weightList[1]
        kernel_layout = "HWIO"
    params1 = {
        "weight": etab.new_const(weight1),
        "channels": weightList[1].shape[3],
        "groups": 1,
        "kernel_size": [1, 1],
        "strides": [1, 1],
        "dilation": [1, 1],
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
    }
    out = _op.nn.conv2d(data=depthconv, **params1)
    if keras_layer.use_bias:
        bias = etab.new_const(weightList[2])
        if data_layout == "NCHW":
            out = _op.nn.bias_add(out, bias)
        else:
            out = _op.nn.bias_add(out, bias, axis=-1)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != "linear":
        out = _convert_activation(out, act_type, etab, data_layout)
    return out


def _convert_flatten(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)

    # NCHW -> NHWC so that dense can be correctly converted
    if data_layout == "NCHW":
        inexpr = _op.transpose(inexpr, axes=[0, 2, 3, 1])
    return _op.nn.batch_flatten(inexpr)


def _convert_pooling(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)

    pool_type = type(keras_layer).__name__
    # global pool in keras = global pool + flatten in relay
    global_pool_params = {"layout": data_layout}

    if input_shape is None:
        input_shape = keras_layer.input_shape

    if pool_type == "GlobalMaxPooling2D":
        return _convert_flatten(
            _op.nn.global_max_pool2d(inexpr, **global_pool_params), keras_layer, etab, data_layout
        )
    if pool_type == "GlobalAveragePooling2D":
        global_avg_pool2d = _op.nn.global_avg_pool2d(inexpr, **global_pool_params)
        keep_dims = len(keras_layer.input.shape) == len(keras_layer.output.shape)
        if keep_dims:
            return global_avg_pool2d
        return _convert_flatten(global_avg_pool2d, keras_layer, etab, data_layout)
    pool_h, pool_w = keras_layer.pool_size
    stride_h, stride_w = keras_layer.strides
    params = {
        "pool_size": [pool_h, pool_w],
        "strides": [stride_h, stride_w],
        "padding": [0, 0],
        "layout": data_layout,
    }
    if keras_layer.padding == "valid":
        pass
    elif keras_layer.padding == "same":
        in_h = input_shape[1]
        in_w = input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, pool_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, pool_w, stride_w)
        params["padding"] = [pad_t, pad_l, pad_b, pad_r]
    else:
        raise tvm.error.OpAttributeUnImplemented(
            f"Padding with {keras_layer.padding} is not supported in operator Pooling."
        )
    if pool_type == "MaxPooling2D":
        return _op.nn.max_pool2d(inexpr, **params)
    if pool_type == "AveragePooling2D":
        params["count_include_pad"] = False
        return _op.nn.avg_pool2d(inexpr, **params)
    raise tvm.error.OpNotImplemented(f"Operator {keras_layer} is not supported for frontend Keras.")


def _convert_pooling3d(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__
    if input_shape is None:
        input_shape = keras_layer.input_shape

    if pool_type not in ["MaxPooling3D", "AveragePooling3D"]:
        raise tvm.error.OpNotImplemented(
            f"Operator {keras_layer} is not supported for frontend Keras."
        )

    pool_d1, pool_d2, pool_d3 = keras_layer.pool_size
    stride_d1, stride_d2, stride_d3 = keras_layer.strides
    params = {
        "pool_size": [pool_d1, pool_d2, pool_d3],
        "strides": [stride_d1, stride_d2, stride_d3],
        "padding": [0, 0, 0],
        "layout": data_layout,
    }

    if keras_layer.padding == "valid":
        pass
    elif keras_layer.padding == "same":
        in_d1 = input_shape[1]
        in_d2 = input_shape[2]
        in_d3 = input_shape[3]
        pad_d1 = _get_pad_pair(in_d1, pool_d1, stride_d1)
        pad_d2 = _get_pad_pair(in_d2, pool_d2, stride_d2)
        pad_d3 = _get_pad_pair(in_d3, pool_d3, stride_d3)
        params["padding"] = [pad_d1[0], pad_d2[0], pad_d3[0], pad_d1[1], pad_d2[1], pad_d3[1]]
    else:
        raise tvm.error.OpAttributeUnImplemented(
            f"Padding with {keras_layer.padding} is not supported in operator Pooling3D."
        )

    out = _op.transpose(inexpr, axes=(0, 4, 1, 2, 3))
    params["layout"] = "NCDHW"
    if pool_type == "MaxPooling3D":
        out = _op.nn.max_pool3d(out, **params)
    elif pool_type == "AveragePooling3D":
        out = _op.nn.avg_pool3d(out, **params)

    return _op.transpose(out, axes=(0, 2, 3, 4, 1))


def _convert_global_pooling3d(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__

    global_pool_params = {"layout": data_layout}
    if pool_type == "GlobalMaxPooling3D":
        out = _op.nn.global_max_pool3d(inexpr, **global_pool_params)
    elif pool_type == "GlobalAveragePooling3D":
        out = _op.nn.global_avg_pool3d(inexpr, **global_pool_params)
    else:
        raise tvm.error.OpNotImplemented(
            f"Operator {keras_layer} is not supported for frontend Keras."
        )

    return _convert_flatten(out, keras_layer, etab, input_shape, data_layout)


def _convert_upsample(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    upsample_type = type(keras_layer).__name__
    params = {}
    if upsample_type == "UpSampling1D":
        h = keras_layer.size
        params["scale_h"] = h
    elif upsample_type == "UpSampling2D":
        h, w = keras_layer.size
        params["scale_h"] = h
        params["scale_w"] = w

        if hasattr(keras_layer, "interpolation"):
            interpolation = keras_layer.interpolation
            if interpolation == "nearest":
                params["method"] = "nearest_neighbor"
            else:
                params["method"] = "bilinear"
    else:
        raise tvm.error.OpNotImplemented(
            f"Operator {upsample_type} is not supported for frontend Keras."
        )
    params["layout"] = data_layout
    out = _op.nn.upsampling(inexpr, **params)
    return out


def _convert_upsample3d(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)

    params = {}
    d, h, w = keras_layer.size
    params["scale_d"] = d
    params["scale_h"] = h
    params["scale_w"] = w
    params["layout"] = data_layout
    params["coordinate_transformation_mode"] = "asymmetric"
    out = _op.nn.upsampling3d(inexpr, **params)
    return out


def _convert_cropping(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    crop_type = type(keras_layer).__name__
    if input_shape is None:
        input_shape = keras_layer.input_shape
    if crop_type == "Cropping2D":
        (_, in_h, in_w, _) = input_shape
        ((crop_t, crop_b), (crop_l, crop_r)) = keras_layer.cropping
    else:
        raise tvm.error.OpNotImplemented(
            f"Operator {crop_type} is not supported for frontend Keras."
        )
    int32_max = np.iinfo(np.int32).max
    if data_layout == "NHWC":
        begin = [0, crop_t, crop_l, 0]
        end = [int32_max, in_h - crop_b, in_w - crop_r, int32_max]
    else:
        begin = [0, 0, crop_t, crop_l]
        end = [int32_max, int32_max, in_h - crop_b, in_w - crop_r]
    return _op.strided_slice(
        inexpr,
        begin=begin,
        end=end,
    )


def _convert_batchnorm(inexpr, keras_layer, etab, data_layout, input_shape=None):
    if input_shape is None:
        input_shape = keras_layer.input_shape
    if data_layout == "NCHW" or len(input_shape) < 4:
        axis = 1
    else:
        axis = 3

    params = {"scale": False, "center": False, "epsilon": keras_layer.epsilon, "axis": axis}
    idx = 0
    if keras_layer.scale:
        params["scale"] = True
        gamma = keras_layer.get_weights()[idx]
        params["gamma"] = etab.new_const(gamma)
        idx += 1
    if keras_layer.center:
        params["center"] = True
        beta = keras_layer.get_weights()[idx]
        params["beta"] = etab.new_const(beta)
        idx += 1
    moving_mean = keras_layer.get_weights()[idx]
    moving_var = keras_layer.get_weights()[idx + 1]
    params["moving_mean"] = etab.new_const(moving_mean)
    params["moving_var"] = etab.new_const(moving_var)
    # in case beta or gamma is not defined
    params["beta"] = (
        etab.new_const(np.zeros(moving_mean.shape)) if "beta" not in params else params["beta"]
    )
    params["gamma"] = (
        etab.new_const(np.ones(moving_mean.shape)) if "gamma" not in params else params["gamma"]
    )
    result, moving_mean, moving_var = _op.nn.batch_norm(inexpr, **params)
    return result


def _convert_padding(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)

    padding_type = type(keras_layer).__name__
    padding = keras_layer.padding
    top = left = bottom = right = 0
    if padding_type == "ZeroPadding2D":
        if isinstance(padding, int):
            top = left = bottom = right = padding
        elif isinstance(padding, tuple):
            if isinstance(padding[0], int):
                top, left = padding
                bottom, right = padding
            elif isinstance(padding[0], tuple):
                top, bottom = padding[0]
                left, right = padding[1]
            else:
                msg = (
                    f'Value {str(padding)} in attribute "padding" of operator Padding is '
                    f"not valid."
                )
                raise tvm.error.OpAttributeInvalid(msg)
        else:
            msg = f'Value {str(padding)} in attribute "padding" of operator Padding is not valid.'
            raise tvm.error.OpAttributeInvalid(msg)
    else:
        msg = f"Operator {padding_type} is not supported in frontend Keras."
        raise tvm.error.OpNotImplemented(msg)
    if data_layout == "NCHW":
        return _op.nn.pad(data=inexpr, pad_width=((0, 0), (0, 0), (top, bottom), (left, right)))
    return _op.nn.pad(data=inexpr, pad_width=((0, 0), (top, bottom), (left, right), (0, 0)))


def _convert_padding3d(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)

    padding = keras_layer.padding

    d_pad = h_pad = w_pad = [0, 0]

    # padding can be 'int' or 'tuple of 3 ints' or 'tuple of 3 tuples of 2 ints' or 'tuple
    # of 3 tuples of 2 ints different values'. In all these scenarios keras will send 3
    # tuples of 2 ints.
    if isinstance(padding, tuple) and isinstance(padding[0], tuple):
        d_pad = padding[0]
        h_pad = padding[1]
        w_pad = padding[2]
    else:
        msg = f'Value {str(padding)} in attribute "padding" of operator ZeroPadding3D is not valid.'
        raise tvm.error.OpAttributeInvalid(msg)

    if data_layout == "NCDHW":
        out = _op.nn.pad(
            data=inexpr,
            pad_width=(
                (0, 0),
                (0, 0),
                (d_pad[0], d_pad[1]),
                (h_pad[0], h_pad[1]),
                (w_pad[0], w_pad[1]),
            ),
        )
    else:
        out = _op.nn.pad(
            data=inexpr,
            pad_width=(
                (0, 0),
                (d_pad[0], d_pad[1]),
                (h_pad[0], h_pad[1]),
                (w_pad[0], w_pad[1]),
                (0, 0),
            ),
        )
    return out


def _convert_concat(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    if input_shape is None:
        input_shape = keras_layer.input_shape

    axis = keras_layer.axis
    dims = len(input_shape[0])
    if data_layout == "NCHW":  # need_transpose
        if axis == -1:
            axis = 1
        else:
            axis = axis + 1 if axis < (dims - 1) else 1
    return _op.concatenate(_as_list(inexpr), axis=axis)


def _convert_reshape(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    if input_shape is None:
        input_shape = keras_layer.input_shape

    inshape = input_shape  # includes batch
    tshape = keras_layer.target_shape  # no batch
    shape = (-1,) + tshape

    if data_layout == "NCHW" and (len(inshape) > 3 or len(tshape) > 2):
        # Perform reshape in original NHWC format.
        inexpr = _op.transpose(inexpr, [0] + list(range(2, len(inshape))) + [1])
        inexpr = _op.reshape(inexpr, newshape=shape)
        return _op.transpose(inexpr, axes=[0, -1] + list(range(1, len(shape) - 1)))

    return _op.reshape(inexpr, newshape=shape)


def _convert_lstm(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    if input_shape is None:
        input_shape = keras_layer.input_shape
    if not isinstance(inexpr, list):
        buf = np.zeros((1, keras_layer.units), "float32")
        c_op = etab.new_const(buf)
        h_op = etab.new_const(buf)
        inexpr = [inexpr, h_op, c_op]
    in_data = inexpr[0]
    next_h = inexpr[1]
    next_c = inexpr[2]
    weightList = keras_layer.get_weights()
    in_shape = tuple(dim if dim else 1 for dim in _as_list(input_shape)[0])
    kernel_weight = etab.new_const(weightList[0].transpose([1, 0]))
    recurrent_weight = etab.new_const(weightList[1].transpose([1, 0]))
    if keras_layer.use_bias:
        in_bias = etab.new_const(weightList[2])
    if keras_layer.go_backwards:
        in_data = _op.reverse(in_data, axis=1)
    units = list(weightList[0].shape)[1]
    assert units > 0, "The value of units must be a positive integer"
    time_steps = in_shape[1]
    in_data = _op.squeeze(in_data, axis=[0])
    in_data = _op.split(in_data, indices_or_sections=time_steps, axis=0)
    # loop for the number of time_steps
    out_list = []  # store h outputs in case return_sequences is True
    for data in in_data:
        ixh1 = _op.nn.dense(data, kernel_weight, units=units)
        ixh2 = _op.nn.dense(next_h, recurrent_weight, units=units)
        if keras_layer.use_bias:
            ixh2 = _op.nn.bias_add(ixh2, bias=in_bias)
        gate = ixh1 + ixh2
        gates = _op.split(gate, indices_or_sections=4, axis=1)
        in_gate = _convert_recurrent_activation(gates[0], keras_layer)
        in_transform = _convert_recurrent_activation(gates[1], keras_layer)
        next_c = in_transform * next_c + in_gate * _convert_activation(
            gates[2], keras_layer, etab, data_layout
        )
        out_gate = _convert_recurrent_activation(gates[3], keras_layer)
        next_h = out_gate * _convert_activation(next_c, keras_layer, etab, data_layout)
        if keras_layer.return_sequences:
            out_list.append(_op.expand_dims(next_h, axis=1))
    out = _op.concatenate(out_list, axis=1) if keras_layer.return_sequences else next_h
    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    out = _op.reshape(out, newshape=out_shape)
    return [out, next_h, next_c]


def _convert_simple_rnn(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    if not isinstance(inexpr, list):
        buf = np.zeros((1, keras_layer.units), "float32")
        prev_op = etab.new_const(buf)
        inexpr = [inexpr, prev_op]
    in_data = inexpr[0]
    prev_op = inexpr[1]
    prev_op = _op.nn.batch_flatten(prev_op)
    weightList = keras_layer.get_weights()
    kernel_weight = etab.new_const(weightList[0].transpose([1, 0]))
    recurrent_weight = etab.new_const(weightList[1].transpose([1, 0]))
    units = list(weightList[0].shape)[1]
    assert units > 0, "The value of units must be a positive integer"
    if keras_layer.use_bias:
        in_bias = etab.new_const(weightList[2])
    assert len(in_data.type_annotation.shape) == 3
    timeDim = in_data.type_annotation.shape[1].value
    if keras_layer.go_backwards:
        in_data = _op.reverse(in_data, axis=1)
    in_data_split = _op.split(in_data, indices_or_sections=timeDim, axis=1)
    for i in range(len(in_data_split)):
        in_data_split_i = _op.nn.batch_flatten(in_data_split[i])
        ixh = _op.nn.dense(in_data_split_i, kernel_weight, units=units)
        if keras_layer.use_bias:
            ixh = _op.nn.bias_add(ixh, bias=in_bias)
        ixh2 = _op.nn.dense(prev_op, recurrent_weight, units=units)
        output = ixh + ixh2
        output = _convert_activation(output, keras_layer, etab, data_layout)
        prev_op = output
    return [output, output]


def _convert_gru(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    _check_data_format(keras_layer)
    if not isinstance(inexpr, list):
        buf = np.zeros((1, keras_layer.units), "float32")
        h_tm1 = etab.new_const(buf)
        inexpr = [inexpr, h_tm1]
    in_data = inexpr[0]
    h_tm1_op = inexpr[1]
    weightList = keras_layer.get_weights()
    kernel_weight = etab.new_const(weightList[0].transpose([1, 0]))
    recurrent_weight = etab.new_const(weightList[1].transpose([1, 0]))
    if keras_layer.use_bias:
        in_bias = etab.new_const(weightList[2])
    if keras_layer.go_backwards:
        in_data = _op.reverse(in_data, axis=1)
    units = list(weightList[0].shape)[1]
    assert units > 0, "The value of units must be a positive integer"
    in_data = _op.nn.batch_flatten(in_data)
    matrix_x = _op.nn.dense(in_data, kernel_weight, units=units)
    if keras_layer.use_bias:
        matrix_x = _op.nn.bias_add(matrix_x, in_bias)
    # inputs projected by all gate matrices at once
    split_indices = [keras_layer.units, 2 * keras_layer.units]
    gates = _op.split(matrix_x, indices_or_sections=split_indices, axis=1)
    x_z = gates[0]
    x_r = gates[1]
    x_h = gates[2]
    # hidden state projected separately for update/reset and new
    units = 2 * keras_layer.units
    split_indices = [units]
    rec_weights = _op.split(recurrent_weight, indices_or_sections=split_indices, axis=0)
    h_tm1_op = _op.nn.batch_flatten(h_tm1_op)
    matrix_inner = _op.nn.dense(h_tm1_op, rec_weights[0], units=units)
    split_indices = [keras_layer.units]
    recurrent = _op.split(matrix_inner, indices_or_sections=split_indices, axis=1)
    recurrent_z = recurrent[0]
    recurrent_r = recurrent[1]
    rec_act_z = _convert_recurrent_activation(x_z + recurrent_z, keras_layer)
    rec_act_r = _convert_recurrent_activation(x_r + recurrent_r, keras_layer)
    units = keras_layer.units
    recurrent_h = _op.nn.dense(rec_act_r * h_tm1_op, rec_weights[1], units=units)
    act_hh = _convert_activation(x_h + recurrent_h, keras_layer, etab, data_layout)
    # previous and candidate state mixed by update gate
    output = rec_act_z * h_tm1_op + (_expr.const(1.0, dtype="float32") - rec_act_z) * act_hh
    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    output = _op.reshape(output, newshape=out_shape)
    return [output, output]


def _convert_repeat_vector(
    inexpr, keras_layer, etab, data_layout, input_shape=None
):  # pylint: disable=unused-argument
    if input_shape is None:
        input_shape = keras_layer.input_shape
    input_shape = list(input_shape)
    repeats = keras_layer.n
    out_shape = [-1, repeats] + input_shape[1:]
    out = _op.repeat(inexpr, repeats=repeats, axis=0)
    out = _op.reshape(out, out_shape)
    return out


def _convert_l2_normalize(inexpr, keras_layer, data_layout):
    l2_normalize_is_loaded = False
    param_list = []
    for i in dis.get_instructions(keras_layer.function):
        if i.opname in ["LOAD_GLOBAL", "LOAD_DEREF"]:
            continue
        if i.opname in ["LOAD_ATTR", "LOAD_METHOD"]:
            if i.argval == "l2_normalize":
                assert not l2_normalize_is_loaded, "l2_normalize was already LOADED"
                l2_normalize_is_loaded = True
        elif i.opname in ["LOAD_CONST", "LOAD_FAST"] and l2_normalize_is_loaded:
            param_list.append(i.argval)
        elif i.opname == "BUILD_LIST":
            sz = i.argval
            assert len(param_list) >= sz
            new_list = param_list[-sz:]
            param_list = param_list[:-sz]
            param_list.append(new_list)
        elif i.opname in ["CALL_FUNCTION_KW", "CALL_METHOD"]:
            break

    axis = None
    is_param_list_parsed = False
    if l2_normalize_is_loaded and len(param_list) > 0:
        # last param_list item is tuple of strings means that
        # lambda uses named parameters when calling l2_normalize
        if (
            isinstance(param_list[-1], tuple)
            and len(param_list[-1]) > 0
            and isinstance(param_list[-1][0], str)
        ):
            param_names = param_list[-1]
            if len(param_names) == 1 and param_names[0] == "x":
                # lambda v: K.l2_normalize(x=v)
                axis = None
                is_param_list_parsed = True
            elif len(param_names) == 1 and param_names[0] == "axis" and len(param_list) == 3:
                # lambda x: K.l2_normalize(x, axis=(2,3))
                axis = param_list[1]
                is_param_list_parsed = True
            elif len(param_names) == 2 and len(param_list) == 3:
                # lambda x: K.l2_normalize(x=x, axis=(2,3))
                # lambda x: K.l2_normalize(axis=(2,3), x=x)
                axis = param_list[param_names.index("axis")]
                is_param_list_parsed = True
        else:
            # lambda x: K.l2_normalize(x)
            if len(param_list) == 1:
                axis = None
                is_param_list_parsed = True
            # lambda x: K.l2_normalize(x, (2,3))
            elif len(param_list) == 2:
                axis = param_list[1]
                is_param_list_parsed = True

    def is_int_or_tuple_of_ints(v):
        if isinstance(v, list) and len(v) > 0:
            for i in v:
                if not isinstance(i, int):
                    return False
            return True
        if isinstance(v, tuple) and len(v) > 0:
            return isinstance(v[0], int)
        return isinstance(v, int)

    assert is_param_list_parsed and (
        axis is None or is_int_or_tuple_of_ints(axis)
    ), "Can not parse l2_normalize lambda function found in Lambda layer"
    if isinstance(axis, int):
        axis = [axis]

    if data_layout == "NCHW":
        dims = len(keras_layer.input_shape)

        def fix_axis_for_nchw(axis):
            if axis == 0:
                return 0
            if axis in [(dims - 1), -1]:
                return 1
            return axis + 1

        axis = [fix_axis_for_nchw(x) for x in axis]
    return _op.nn.l2_normalize(inexpr, eps=1e-12, axis=axis)


def _convert_lambda(inexpr, keras_layer, _, data_layout):
    fcode = keras_layer.function.__code__
    # Convert l2_normalize
    if (
        fcode.co_name == "<lambda>"
        and len(fcode.co_names) > 0
        and fcode.co_names[-1] == "l2_normalize"
    ):
        return _convert_l2_normalize(inexpr, keras_layer, data_layout)
    raise tvm.error.OpNotImplemented(
        f"Function {fcode.co_names} used in Lambda layer is not supported in frontend Keras."
    )


def _convert_time_distributed(inexpr, keras_layer, etab, data_layout, input_shape=None):
    # TimeDistributed: split input tensor along the second dimension (assumed to be time),
    # apply inner layer to each split individually,
    # and then combine the results
    if input_shape is None:
        input_shape = keras_layer.input_shape

    assert len(input_shape) >= 2, "Input to TimeDistributed must have at least two dimensions"

    inner_layer = keras_layer.layer
    inner_input_shape = [d for (i, d) in enumerate(input_shape) if i != 1]

    # for NDHWC, inner data layout will drop the D
    inner_data_layout = data_layout
    if data_layout == "NDHWC":
        inner_data_layout = "NHWC"

    # some code duplication from keras_op_to_relay
    # but it's useful to avoid cluttering the etab
    inner_layer_op_name = type(keras_layer.layer).__name__
    if inner_layer_op_name not in _convert_map:
        raise tvm.error.OpNotImplemented(
            f"The inner layer for TimeDistributed {inner_layer_op_name} is not supported for"
            f" frontend Keras."
        )

    conversion_func = lambda expr: _convert_map[inner_layer_op_name](
        expr, inner_layer, etab, inner_data_layout, input_shape=inner_input_shape
    )

    split_dim = input_shape[1]
    split_input = _op.split(inexpr, split_dim, 1)

    split_shape = list(input_shape)
    if split_shape[0] is None:
        split_shape[0] = 1
    split_shape[1] = 1

    split_var = new_var(
        "time_distributed_split",
        type_annotation=TupleType(
            [TensorType(split_shape, dtype="float32") for i in range(split_dim)]
        ),
    )

    # For each split, squeeze away the second dimension,
    # apply the inner layer.
    # Afterwards, combine the transformed splits back along
    # the second dimension using stack
    splits = [
        conversion_func(_op.squeeze(_expr.TupleGetItem(split_var, i), axis=[1]))
        for i in range(split_dim)
    ]

    return _expr.Let(split_var, split_input.astuple(), _op.stack(splits, axis=1))


def _default_skip(inexpr, keras_layer, etab, data_layout):  # pylint: disable=unused-argument
    """Layers that can be skipped because they are train time only."""
    return inexpr


_convert_map = {
    "Dense": _convert_dense,
    "Activation": _convert_activation,
    "Softmax": _convert_advanced_activation,
    "ReLU": _convert_advanced_activation,
    "LeakyReLU": _convert_advanced_activation,
    "PReLU": _convert_advanced_activation,
    "ELU": _convert_advanced_activation,
    "ThresholdedReLU": _convert_advanced_activation,
    "AveragePooling2D": _convert_pooling,
    "MaxPooling2D": _convert_pooling,
    "GlobalAveragePooling2D": _convert_pooling,
    "GlobalMaxPooling2D": _convert_pooling,
    "Conv2D": _convert_convolution,
    "Conv2DTranspose": _convert_convolution,
    "DepthwiseConv2D": _convert_convolution,
    "SeparableConv2D": _convert_separable_convolution,
    "Flatten": _convert_flatten,
    "Reshape": _convert_reshape,
    "Concatenate": _convert_concat,
    "BatchNormalization": _convert_batchnorm,
    # Specific tf.Keras terminology for batch normalization
    "BatchNormalizationV1": _convert_batchnorm,
    "Add": _convert_merge,
    "Subtract": _convert_merge,
    "Multiply": _convert_merge,
    "ZeroPadding2D": _convert_padding,
    "UpSampling2D": _convert_upsample,
    "Cropping2D": _convert_cropping,
    # 'ZeroPadding1D'          : _convert_padding,
    # 'AveragePooling1D'       : _convert_pooling,
    # 'MaxPooling1D'           : _convert_pooling,
    # 'GlobalAveragePooling1D' : _convert_pooling,
    # 'GlobalMaxPooling1D'     : _convert_pooling,
    # 'Cropping1D'             : _convert_cropping,
    # 'UpSampling1D'           : _convert_upsample,
    "Conv1D": _convert_convolution1d,
    # "Conv1DTranspose": _convert_convolution1d,
    "Conv3D": _convert_convolution3d,
    "Conv3DTranspose": _convert_convolution3d,
    # 'SeparableConv3D'        : _convert_convolution3d,
    "MaxPooling3D": _convert_pooling3d,
    "AveragePooling3D": _convert_pooling3d,
    "GlobalMaxPooling3D": _convert_global_pooling3d,
    "GlobalAveragePooling3D": _convert_global_pooling3d,
    "UpSampling3D": _convert_upsample3d,
    "ZeroPadding3D": _convert_padding3d,
    "SimpleRNN": _convert_simple_rnn,
    "LSTM": _convert_lstm,
    "GRU": _convert_gru,
    # 'Bidirectional'          : _convert_bidirectional,
    "TimeDistributed": _convert_time_distributed,
    "Average": _convert_merge,
    "Minimum": _convert_merge,
    "Maximum": _convert_merge,
    "Dot": _convert_merge,
    "Permute": _convert_permute,
    "Embedding": _convert_embedding,
    "RepeatVector": _convert_repeat_vector,
    "Lambda": _convert_lambda,
    "InputLayer": _default_skip,
    "Dropout": _default_skip,
    "AlphaDropout": _default_skip,
    "SpatialDropout2D": _default_skip,
    "SpatialDropout1D": _default_skip,
    "GaussianDropout": _default_skip,
    "GaussianNoise": _default_skip,
}


def _check_unsupported_layers(model):
    missing_ops = set()
    for layer in model.layers:
        op_name = type(layer).__name__
        if op_name not in _convert_map:
            missing_ops.add(op_name)

    if missing_ops:
        raise NotImplementedError(f"The following operators are not implemented: {missing_ops}")


def keras_op_to_relay(inexpr, keras_layer, outname, etab, data_layout):
    """Convert a Keras layer to a Relay expression and update the expression table.

    Parameters
    ----------
    inexpr : relay.expr.Expr or a list of it
        The input Relay expression(s).

    keras_layer : keras.layers
        The Keras layer to be converted.

    outname : str
        Name of the output Relay expression.

    etab : relay.frontend.common.ExprTable
        The global expression table to be updated.

    data_layout : str
        The input data layout
    """
    op_name = type(keras_layer).__name__
    if op_name not in _convert_map:
        raise tvm.error.OpNotImplemented(f"Operator {op_name} is not supported for frontend Keras.")
    outs = _convert_map[op_name](inexpr, keras_layer, etab, data_layout)
    outs = _as_list(outs)
    for t_idx, out in enumerate(outs):
        name = outname + ":" + str(t_idx)
        etab.set_expr(name, out)
    return outs


def from_keras(model, shape=None, layout="NCHW"):
    """Convert keras model to relay Function.

    Parameters
    ----------
    model : keras.engine.training.Model or tensorflow.keras.models.Model
        The keras model to be converted.

    shape: dict of str to int list/tuple
        Input shapes of the model, optional

    layout: str
        One of 'NWC', 'NCHW', 'NHWC', 'NDHWC' indicates how data should
        be arranged in the output model. Default layout is 'NCHW' as it
        in general performs better across TVM.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by Relay.
    """

    def _check_model_is_tf_keras():
        return type(model).__module__.startswith("tensorflow.python.keras")

    def _convert_input_layer(keras_layer):
        input_name = keras_layer.name
        input_shape = shape[input_name] if shape is not None and input_name in shape else None
        if input_shape and len(input_shape) > 1 and any(dim <= 0 for dim in input_shape[1:]):
            msg = (
                "Expected input's non-batch dimensions to have positive length, "
                f"but the input has a shape of {input_shape}"
            )
            raise ValueError(msg)
        etab.set_expr(input_name, new_var(input_name, shape=input_shape))

    def _convert_layer(keras_layer, etab, scope=""):
        inbound_nodes = (
            keras_layer.inbound_nodes
            if hasattr(keras_layer, "inbound_nodes")
            else keras_layer._inbound_nodes
            if hasattr(keras_layer, "_inbound_nodes")
            else None
        )
        if inbound_nodes is None:
            raise TypeError(f"Unknown layer type or unsupported Keras version : {keras_layer}")
        outs = []
        for node_idx, node in enumerate(inbound_nodes):
            # If some nodes in imported model are not relevant to the current model,
            # skip such layers.
            # - In Keras, model._network_nodes contains keys of all nodes relevant to the
            #   current model;
            # - In tf.Keras, this is already done as part of tensorflow.keras.network.get_config
            if not is_tf_keras:
                if (
                    hasattr(model, "_node_key")
                    and not model._node_key(keras_layer, node_idx) in model._network_nodes
                ):
                    continue
            inexpr = []
            # Since Keras allows creating multiple layers from the same name instance,
            # we append node index to the expr name to make it unique.
            # The one exception is InputLayer. Changing input variable names after conversion
            # would confuse users, so we should keep them as far as possible. Fortunately,
            # they are named uniquely to input_1, input_2, input_3... by default.
            # node_indices attribute removed in tensorflow 2.3, however iterate_inbound() can
            # be used
            if hasattr(node, "node_indices"):
                zip_node = zip(
                    _as_list(node.inbound_layers),
                    _as_list(node.node_indices),
                    _as_list(node.tensor_indices),
                    _as_list(node.input_tensors),
                )
                node_attributes = zip_node
            else:
                node_attributes = node.iterate_inbound()

            for inbound_layer, n_idx, t_idx, _ in node_attributes:
                if isinstance(inbound_layer, input_layer_class):
                    expr_name = inbound_layer.name
                    _convert_input_layer(inbound_layer)
                else:
                    expr_name = scope + inbound_layer.name + ":" + str(n_idx) + ":" + str(t_idx)
                expr = etab.get_expr(expr_name)
                inexpr.append(expr)

            # Handle nested layers
            if hasattr(keras_layer, "layers"):
                input_index = 0
                for layer in keras_layer.layers:
                    if isinstance(layer, input_layer_class):
                        # Replace input layer with inbound node
                        etab.set_expr(layer.name, inexpr[input_index])
                        input_index += 1
                    else:
                        # Convert child layer. Prepend scope with parent layer name.
                        layer_outs = _convert_layer(layer, etab, keras_layer.name + "_" + scope)

                # Get output of last child layer and mark as output of parent.
                outname = keras_layer.name + ":" + str(node_idx)
                for t_idx, out in enumerate(layer_outs):
                    name = outname + ":" + str(t_idx)
                    etab.set_expr(name, out)
                outs.extend(layer_outs)
            else:
                if len(inexpr) == 1:
                    inexpr = inexpr[0]
                outs.extend(
                    keras_op_to_relay(
                        inexpr,
                        keras_layer,
                        scope + keras_layer.name + ":" + str(node_idx),
                        etab,
                        layout,
                    )
                )
        return outs

    is_tf_keras = _check_model_is_tf_keras()

    if not is_tf_keras:
        # Importing from Keras
        try:
            import keras
        except ImportError:
            raise ImportError("Keras must be installed")
        if keras.backend.backend() != "tensorflow":
            raise ValueError("Keras frontend currently supports tensorflow backend only.")
        if keras.backend.image_data_format() != "channels_last":
            raise ValueError("Keras frontend currently supports data_format = channels_last only.")
        try:
            import keras.engine as E
        except ImportError:
            try:
                import keras.src.engine as E
            except ImportError:
                raise ImportError("Cannot find Keras's engine")
        expected_model_class = E.training.Model
        if hasattr(E, "InputLayer"):
            input_layer_class = E.InputLayer
        else:
            # TFlite >=2.6
            input_layer_class = E.input_layer.InputLayer
    else:
        # Importing from Tensorflow Keras (tf.keras)
        try:
            from tensorflow import keras as tf_keras
        except ImportError:
            raise ImportError("Tensorflow must be installed")
        expected_model_class = tf_keras.models.Model
        input_layer_class = tf_keras.layers.InputLayer

    assert isinstance(model, expected_model_class)

    etab = ExprTable()
    # Set global data format.
    assert layout in [
        "NWC",
        "NCHW",
        "NHWC",
        "NDHWC",
    ], "Layout must be one of 'NWC', 'NCHW', NHWC or NDHWC"
    for keras_layer in model.layers:
        if isinstance(keras_layer, input_layer_class):
            _convert_input_layer(keras_layer)
        else:
            _convert_layer(keras_layer, etab)

    # model._output_coordinates contains out_node(oc[0]), node_index(oc[1]) and tensor_index(oc[2])
    # Get all output nodes in etab using the name made from above values.
    # The out exprs were added to etab in keras_op_to_relay using this name.
    outexpr = [
        etab.get_expr(oc[0].name + ":" + str(oc[1]) + ":" + str(oc[2]))
        for oc in model._output_coordinates
    ]
    outexpr = outexpr[0] if len(outexpr) == 1 else _expr.Tuple(outexpr)
    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    params = {k: _nd.array(np.array(v, dtype=np.float32)) for k, v in etab.params.items()}
    return IRModule.from_expr(func), params
