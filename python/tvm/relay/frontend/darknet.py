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
# pylint: disable=unused-argument
"""
DarkNet symbol frontend for Relay.
"""

from enum import Enum
import numpy as np
import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .common import get_relay_op, new_var

__all__ = ["from_darknet"]


def _darknet_not_support(attr, op="relay"):
    """Raise error if any operation is not supported."""
    raise NotImplementedError(f"{attr} is not supported in {op}.")


def _get_params_prefix(opname, layer_num):
    """Makes the params prefix name from opname and layer number."""
    return str(opname).replace(".", "_") + str(layer_num)


def _get_params_name(prefix, item):
    """Makes the params name for the k,v pair."""
    return prefix + "_" + item


def _get_param_var(params, prefix, item):
    name = _get_params_name(prefix, item)
    if name not in params:
        raise AttributeError(f"{name} not found in params dict.")
    return new_var(name, shape=params[name].shape, dtype=params[name].dtype)


def _darknet_maxpooling(inputs, params, attrs, prefix):
    """Process the max pool 2d operation."""
    new_attrs = {}
    kernel = attrs.get("kernel")
    strides = attrs.get("stride", 1)
    pads = attrs.get("pad", 1)
    new_attrs["pool_size"] = (kernel, kernel)
    new_attrs["strides"] = (strides, strides)
    new_attrs["padding"] = (pads, pads)
    extra_pad_size = attrs.get("extra_pad_size", 0)
    if extra_pad_size:
        pad_width = ((0, 0), (0, 0), (0, extra_pad_size), (0, extra_pad_size))
        inputs = [
            get_relay_op("pad")(*inputs, pad_width=pad_width, pad_value=np.finfo(np.float32).min)
        ]
    return get_relay_op("max_pool2d")(*inputs, **new_attrs)


def _darknet_avgpooling(inputs, params, attrs, prefix):
    """Process the average pool 2d operation."""
    new_attrs = {}
    kernel = attrs.get("kernel")
    strides = attrs.get("stride", 1)
    pads = attrs.get("pad", 0)

    new_attrs["pool_size"] = (kernel, kernel)
    new_attrs["strides"] = (strides, strides)
    new_attrs["padding"] = (pads, pads)
    return get_relay_op("avg_pool2d")(*inputs, **new_attrs)


def _darknet_conv2d(inputs, params, attrs, prefix):
    """Process the convolution 2d operation."""
    new_attrs = {}
    kernel = attrs.get("kernel")
    strides = attrs.get("stride", 1)
    pads = attrs.get("pad", 0)

    new_attrs["channels"] = attrs.get("num_filter")
    new_attrs["kernel_size"] = (kernel, kernel)
    new_attrs["strides"] = (strides, strides)
    new_attrs["padding"] = (pads, pads)
    new_attrs["dilation"] = attrs.get("dilate", (1, 1))
    new_attrs["groups"] = attrs.get("num_group", 1)

    weight = _get_param_var(params, prefix, "weight")
    out = get_relay_op("conv2d")(*inputs, weight=weight, **new_attrs)

    use_bias = not attrs.get("use_batchNorm", False)
    if use_bias:
        new_attrs = {}
        new_attrs["axis"] = 1
        bias = _get_param_var(params, prefix, "bias")
        out = get_relay_op("bias_add")(out, bias=bias, **new_attrs)
    else:
        new_attrs = {}
        new_attrs["epsilon"] = 0.000001
        gamma = _get_param_var(params, prefix, "gamma")
        beta = _get_param_var(params, prefix, "beta")
        moving_mean = _get_param_var(params, prefix, "moving_mean")
        moving_var = _get_param_var(params, prefix, "moving_var")
        out = get_relay_op("batch_norm")(out, gamma, beta, moving_mean, moving_var, **new_attrs)

    if "activation" in attrs:
        new_attrs = {}
        new_attrs["activation"] = attrs["activation"]
        new_attrs["slope"] = 0.1
        out = _darknet_activations(out, None, new_attrs)
    return out


def _darknet_shortcut(inputs, params, attrs, prefix):
    """Process the shortcut operation."""
    input_0 = inputs[0]
    input_1 = inputs[1]

    input_0_channel = int(attrs["out_channel"])
    input_1_channel = int(attrs["add_out_channel"])
    input_0_size = int(attrs["out_size"])
    input_1_size = int(attrs["add_out_size"])

    if input_0_size > input_1_size:
        scale = int(input_0_size / input_1_size)
        input_1 = get_relay_op("upsampling")(input_1, scale_h=scale, scale_w=scale)

    elif input_0_size < input_1_size:
        stride = int(input_1_size / input_0_size)
        input_1 = get_relay_op("avg_pool2d")(
            input_1, pool_size=(1, 1), strides=(stride, stride), padding=(0, 0)
        )

    if input_0_channel != input_1_channel:
        pad_channel = input_0_channel - input_1_channel
        input_1 = get_relay_op("pad")(
            input_1, pad_width=((0, 0), (0, pad_channel), (0, 0), (0, 0)), pad_value=0.0
        )
    sym = input_0 + input_1
    if "activation" in attrs:
        new_attrs = {}
        new_attrs["activation"] = attrs["activation"]
        sym = _darknet_activations(sym, None, new_attrs)
    return sym


def _darknet_dense(inputs, params, attrs, prefix):
    """Process the dense operation."""
    new_attrs = {}
    new_attrs["units"] = attrs.get("num_hidden")
    data = inputs[0]

    if attrs.get("use_flatten", False) is True:
        data = get_relay_op("batch_flatten")(data)

    weight = _get_param_var(params, prefix, "weight")
    data = get_relay_op("dense")(data, weight, **new_attrs)

    use_bias = attrs.get("use_bias", False)
    if use_bias:
        bias = _get_param_var(params, prefix, "bias")
        data = get_relay_op("bias_add")(data, bias, axis=1)

    if "use_batchNorm" in attrs:
        new_attrs = {}
        new_attrs["epsilon"] = 0.000001
        gamma = _get_param_var(params, prefix, "gamma")
        beta = _get_param_var(params, prefix, "beta")
        moving_mean = _get_param_var(params, prefix, "moving_mean")
        moving_var = _get_param_var(params, prefix, "moving_var")
        data = get_relay_op("batch_norm")(data, gamma, beta, moving_mean, moving_var, **new_attrs)
    if "activation" in attrs:
        new_attrs = {}
        new_attrs["activation"] = attrs["activation"]
        data = _darknet_activations(data, None, new_attrs)
    return data


def _darknet_dropout(inputs, params, attrs, prefix):
    """Process the dropout operation, its a blank operation."""
    new_attrs = {}
    new_attrs["rate"] = attrs.get("p", 0.5)
    return get_relay_op("dropout")(*inputs, **new_attrs)


def _darknet_reshape(inputs, params, attrs, prefix):
    """Process the reshape operation."""
    new_attrs = {}
    new_attrs["shape"] = attrs.get("shape")
    return get_relay_op("reshape")(*inputs, **new_attrs)


def _darknet_upsampling(inputs, params, attrs, prefix):
    """Process the upsampling operation."""
    new_attrs = {}
    new_attrs["scale_h"] = attrs.get("scale", 1)
    new_attrs["scale_w"] = attrs.get("scale", 1)
    return get_relay_op("upsampling")(*inputs, **new_attrs)


def _darknet_l2normalize(inputs, params, attrs, prefix):
    """Process the l2 normalization operation."""
    new_attrs = {}
    new_attrs["eps"] = attrs.get("eps", 0.0)
    new_attrs["axis"] = [attrs.get("axis", 1)]
    return get_relay_op("l2_normalize")(*inputs, **new_attrs)


def _darknet_softmax_output(inputs, params, attrs, prefix):
    """Process the softmax operation."""
    temperature = attrs.get("temperature", 1)
    data = inputs[0]
    if temperature != 1:
        data = data / _expr.const(float(temperature))

    if attrs.get("use_flatten", False) is True:
        data = get_relay_op("batch_flatten")(data)

    new_attrs = {}
    if attrs.get("multi_output", False):
        new_attrs["axis"] = 1
    return get_relay_op("softmax")(data, **new_attrs)


def _darknet_route(inputs, params, attrs, prefix):
    """Process the route operation, which is equivalent to concat."""
    new_attrs = {"axis": attrs.get("dim", 1)}
    return get_relay_op("concatenate")((inputs[0], inputs[1]), **new_attrs)


def _darknet_reorg(inputs, params, attrs, prefix):
    """Process the reorg operation."""
    new_attrs = {}
    if "stride" in attrs:
        new_attrs = {"stride": attrs.get("stride", 1)}
    return get_relay_op("yolo_reorg")(*inputs, **new_attrs)


def _darknet_region(inputs, params, attrs, prefix):
    """Process the region operation."""
    num = attrs.get("n", 1)
    classes = attrs.get("classes", 1)
    coords = attrs.get("coords", 0)
    background = attrs.get("background", 0)
    softmax = attrs.get("softmax", True)
    input_shape = attrs.get("shape")

    split_size = classes + coords + 1
    intermediate_shape = (input_shape[0], num, split_size, input_shape[2], input_shape[3])
    data_block = get_relay_op("reshape")(inputs[0], newshape=intermediate_shape)
    split_indices = (2, 4, 5)
    split_res = get_relay_op("split")(data_block, indices_or_sections=split_indices, axis=2)
    split_res0 = get_relay_op("sigmoid")(split_res[0])
    split_res2 = split_res[2] if background else get_relay_op("sigmoid")(split_res[2])
    split_res3 = get_relay_op("softmax")(split_res[3], axis=2) if softmax else split_res[3]
    out = get_relay_op("concatenate")((split_res0, split_res[1], split_res2, split_res3), axis=2)
    return get_relay_op("reshape")(out, newshape=input_shape)


def _darknet_yolo(inputs, params, attrs, prefix):
    """Process the yolo operation."""
    num = attrs.get("n", 1)
    classes = attrs.get("classes", 1)
    input_shape = attrs.get("shape")
    split_size = classes + 5
    intermediate_shape = (input_shape[0], num, split_size, input_shape[2], input_shape[3])
    data_block = get_relay_op("reshape")(inputs[0], newshape=intermediate_shape)
    split_indices = (2, 4)
    split_res = get_relay_op("split")(data_block, indices_or_sections=split_indices, axis=2)
    split_res0 = get_relay_op("sigmoid")(split_res[0])
    split_res2 = get_relay_op("sigmoid")(split_res[2])
    out = get_relay_op("concatenate")((split_res0, split_res[1], split_res2), axis=2)
    return get_relay_op("reshape")(out, newshape=input_shape)


class ACTIVATION(object):
    """Darknet ACTIVATION Class constant."""

    LOGISTIC = 0
    RELU = 1
    RELIE = 2
    LINEAR = 3
    RAMP = 4
    TANH = 5
    PLSE = 6
    LEAKY = 7
    ELU = 8
    LOGGY = 9
    STAIR = 10
    HARDTAN = 11
    LHTAN = 12


def _darknet_activations(inputs, params, attrs):
    """Process the activation function."""
    act = attrs.get("activation")
    data = inputs[0] if isinstance(inputs, _expr.TupleWrapper) else inputs

    def _const(val):
        return _expr.const(val)

    def _relu(data):
        return get_relay_op("relu")(data)

    def _exp(data):
        return get_relay_op("exp")(data)

    def _tanh(data):
        return get_relay_op("tanh")(data)

    def _sigmoid(data):
        return get_relay_op("sigmoid")(data)

    def _elu(data):
        alpha = _const(-1.0)
        return alpha * _relu(_const(1.0) - _exp(data)) + _relu(data)

    def _leaky_relu(data, slope):
        new_attrs = {}
        new_attrs["alpha"] = slope
        return get_relay_op("leaky_relu")(data, **new_attrs)

    if ACTIVATION.LOGISTIC == act:
        data = _sigmoid(data)
    elif ACTIVATION.RELU == act:
        data = _relu(data)
    elif ACTIVATION.TANH == act:
        data = _tanh(data)
    elif ACTIVATION.LINEAR == act:
        return data
    elif ACTIVATION.LEAKY == act:
        data = _leaky_relu(data, attrs.get("slope", 0.1))
    elif ACTIVATION.ELU == act:
        data = _elu(data)
    else:
        _darknet_not_support("act: " + attrs)
    return data


class LAYERTYPE(Enum):
    """Darknet LAYERTYPE Class constant."""

    CONVOLUTIONAL = 0
    DECONVOLUTIONAL = 1
    CONNECTED = 2
    MAXPOOL = 3
    SOFTMAX = 4
    DETECTION = 5
    DROPOUT = 6
    CROP = 7
    ROUTE = 8
    COST = 9
    NORMALIZATION = 10
    AVGPOOL = 11
    LOCAL = 12
    SHORTCUT = 13
    ACTIVE = 14
    RNN = 15
    GRU = 16
    LSTM = 17
    CRNN = 18
    BATCHNORM = 19
    NETWORK = 20
    XNOR = 21
    REGION = 22
    YOLO = 23
    REORG = 24
    UPSAMPLE = 25
    LOGXENT = 26
    L2NORM = 27
    BLANK = 28


_DARKNET_CONVERT_MAP = {
    LAYERTYPE.CONVOLUTIONAL: _darknet_conv2d,
    LAYERTYPE.CONNECTED: _darknet_dense,
    LAYERTYPE.MAXPOOL: _darknet_maxpooling,
    LAYERTYPE.SOFTMAX: _darknet_softmax_output,
    LAYERTYPE.DROPOUT: _darknet_dropout,
    LAYERTYPE.AVGPOOL: _darknet_avgpooling,
    LAYERTYPE.ROUTE: _darknet_route,
    LAYERTYPE.REORG: _darknet_reorg,
    LAYERTYPE.REGION: _darknet_region,
    LAYERTYPE.SHORTCUT: _darknet_shortcut,
    LAYERTYPE.UPSAMPLE: _darknet_upsampling,
    LAYERTYPE.L2NORM: _darknet_l2normalize,
    LAYERTYPE.YOLO: _darknet_yolo,
    LAYERTYPE.DECONVOLUTIONAL: _darknet_not_support,
    LAYERTYPE.BATCHNORM: _darknet_not_support,
    LAYERTYPE.DETECTION: _darknet_not_support,
    LAYERTYPE.CROP: _darknet_not_support,
    LAYERTYPE.COST: _darknet_not_support,
    LAYERTYPE.NORMALIZATION: _darknet_not_support,
    LAYERTYPE.LOCAL: _darknet_not_support,
    LAYERTYPE.ACTIVE: _darknet_not_support,
    LAYERTYPE.RNN: _darknet_not_support,
    LAYERTYPE.GRU: _darknet_not_support,
    LAYERTYPE.LSTM: _darknet_not_support,
    LAYERTYPE.CRNN: _darknet_not_support,
    LAYERTYPE.NETWORK: _darknet_not_support,
    LAYERTYPE.XNOR: _darknet_not_support,
    LAYERTYPE.BLANK: _darknet_not_support,
}


def _darknet_convert_symbol(op_name, inputs, params, attrs, params_prefix):
    """Convert from darknet op to relay op.
    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, Connected, etc
    inputs : list of relay.Function
        List of input symbols.
    attrs : dict
        Dict of operator attributes
    params_prefix: str
        Params name for this operation

    Returns
    -------
    out_name : converted out name of operation
    sym : tvm.relay.Function
        Converted relay function
    """

    if op_name in _DARKNET_CONVERT_MAP:
        sym = _DARKNET_CONVERT_MAP[op_name](inputs, params, attrs, params_prefix)
    else:
        _darknet_not_support("Operator type " + str(op_name))
    return sym


def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]


class GraphProto(object):
    """A helper class for handling relay functions from darknet model."""

    def __init__(self, net, shape, dtype="float32"):
        self._net = net
        self._shape = shape
        self._dtype = dtype
        self._sym_array = {}
        self._tvmparams = {}
        self._outs = []
        self._state_ctr = {}
        self._state_ctr["rnn"] = 0
        self._state_ctr["crnn"] = 0
        self._state_ctr["lstm"] = 0
        self._state_ctr["cell_state"] = 0
        self._state_ctr["gru"] = 0

    def _read_memory_buffer(self, shape, data, dtype=None):
        if dtype is None:
            dtype = self._dtype
        length = 1
        for x in shape:
            length *= x
        data_np = np.zeros(length, dtype=dtype)
        for i in range(length):
            data_np[i] = data[i]
        return data_np.reshape(shape)

    def _get_convolution_weights(self, layer, opname):
        """Get the convolution layer weights and biases."""
        if layer.nweights == 0:
            return None

        if (layer.n * layer.c // layer.groups * layer.size * layer.size) != layer.nweights:
            raise RuntimeError("layer weights size not matching with n c h w")

        params = {}
        shape = (layer.n, layer.c // layer.groups, layer.size, layer.size)
        weights = self._read_memory_buffer(shape, layer.weights)

        biases = self._read_memory_buffer((layer.n,), layer.biases)

        k = _get_params_name(opname, "weight")
        params[k] = tvm.nd.array(weights)

        if layer.batch_normalize == 1 and layer.dontloadscales != 1:
            params.update(self._get_batchnorm_weights(layer, opname, layer.n))
            k = _get_params_name(opname, "beta")
            params[k] = tvm.nd.array(biases)
        else:
            k = _get_params_name(opname, "bias")
            params[k] = tvm.nd.array(biases)
        return params

    def _get_connected_weights(self, layer, opname):
        """Parse the weights and biases for fully connected or dense layer."""
        size = layer.outputs * layer.inputs
        if size == 0:
            return None

        weights = self._read_memory_buffer((layer.outputs, layer.inputs), layer.weights)
        biases = self._read_memory_buffer((layer.outputs,), layer.biases)

        params = {}
        k = _get_params_name(opname, "weight")
        params[k] = tvm.nd.array(weights)

        if layer.batch_normalize == 1 and layer.dontloadscales != 1:
            params.update(self._get_batchnorm_weights(layer, opname, layer.outputs))
            k = _get_params_name(opname, "beta")
            params[k] = tvm.nd.array(biases)
        else:
            k = _get_params_name(opname, "bias")
            params[k] = tvm.nd.array(biases)
        return params

    def _get_region_weights(self, layer, opname):
        """Parse the biases for region layer."""
        biases = self._read_memory_buffer((layer.n * 2,), layer.biases)
        attributes = np.array(
            [
                layer.n,
                layer.out_c,
                layer.out_h,
                layer.out_w,
                layer.classes,
                layer.coords,
                layer.background,
            ],
            dtype=np.int32,
        )
        params = {}
        k = _get_params_name(opname, "bias")
        params[k] = tvm.nd.array(biases)
        k = _get_params_name(opname, "attr")
        params[k] = tvm.nd.array(attributes)
        return params

    def _get_yolo_weights(self, layer, opname):
        """Parse the biases and mask for yolo layer."""
        biases = self._read_memory_buffer((layer.total * 2,), layer.biases)
        mask = self._read_memory_buffer((layer.n,), layer.mask, dtype="int32")
        attributes = np.array(
            [layer.n, layer.out_c, layer.out_h, layer.out_w, layer.classes, layer.total],
            dtype=np.int32,
        )
        params = {}
        k = _get_params_name(opname, "bias")
        params[k] = tvm.nd.array(biases)
        k = _get_params_name(opname, "mask")
        params[k] = tvm.nd.array(mask)
        k = _get_params_name(opname, "attr")
        params[k] = tvm.nd.array(attributes)
        return params

    def _get_batchnorm_weights(self, layer, opname, size):
        """Parse the weights for batchnorm, which includes, scales, moving mean
        and moving variances."""
        scales = self._read_memory_buffer((size,), layer.scales)
        rolling_mean = self._read_memory_buffer((size,), layer.rolling_mean)
        rolling_variance = self._read_memory_buffer((size,), layer.rolling_variance)

        params = {}
        k = _get_params_name(opname, "moving_mean")
        params[k] = tvm.nd.array(rolling_mean)
        k = _get_params_name(opname, "moving_var")
        params[k] = tvm.nd.array(rolling_variance)
        k = _get_params_name(opname, "gamma")
        params[k] = tvm.nd.array(scales)
        return params

    def _get_darknet_attrs(self, layer, layer_num):
        """Parse attributes of each layer and return."""
        attr = {}
        use_flatten = True
        layer_type = LAYERTYPE(layer.type)
        if LAYERTYPE.CONVOLUTIONAL == layer_type:
            attr.update({"pad": layer.pad})
            attr.update({"num_group": layer.groups})
            attr.update({"num_filter": layer.n})
            attr.update({"stride": layer.stride})
            attr.update({"kernel": layer.size})
            attr.update({"activation": (layer.activation)})

            if layer.nbiases == 0:
                attr.update({"use_bias": False})
            else:
                attr.update({"use_bias": True})

            if layer.batch_normalize == 1 and layer.dontloadscales != 1:
                attr.update({"use_batchNorm": True})
                attr.update({"use_scales": True})

        elif LAYERTYPE.CONNECTED == layer_type:
            attr.update({"num_hidden": layer.outputs})
            attr.update({"activation": (layer.activation)})
            if layer_num != 0:
                layer_prev = self._net.layers[layer_num - 1]
                if (
                    layer_prev.out_h == layer.h
                    and layer_prev.out_w == layer.w
                    and layer_prev.out_c == layer.c
                ):
                    use_flatten = False
            attr.update({"use_flatten": use_flatten})
            attr.update({"use_bias": True})
            if layer.batch_normalize == 1 and layer.dontloadscales != 1:
                attr.update({"use_batchNorm": True})
                attr.update({"use_scales": True})
                attr.update({"use_bias": False})

        elif LAYERTYPE.MAXPOOL == layer_type:
            attr.update({"pad": layer.pad})
            attr.update({"stride": layer.stride})
            attr.update({"kernel": layer.size})
            max_output = (layer.w - layer.size + 2 * layer.pad) / float(layer.stride) + 1
            if max_output < layer.out_w:
                extra_pad = (layer.out_w - max_output) * layer.stride
                attr.update({"extra_pad_size": int(extra_pad)})
        elif LAYERTYPE.AVGPOOL == layer_type:
            attr.update({"pad": layer.pad})
            if layer.stride == 0:
                attr.update({"stride": 1})
            else:
                attr.update({"stride": layer.stride})
            if layer.size == 0 and layer.h == layer.w:
                attr.update({"kernel": layer.h})
            else:
                attr.update({"kernel": layer.size})

        elif LAYERTYPE.DROPOUT == layer_type:
            attr.update({"p": layer.probability})

        elif LAYERTYPE.SOFTMAX == layer_type:
            attr.update({"axis": 1})
            attr.update({"use_flatten": True})
            if layer.temperature:
                attr.update({"temperature": str(layer.temperature)})

        elif LAYERTYPE.SHORTCUT == layer_type:
            add_layer = self._net.layers[layer.index]
            attr.update({"activation": layer.activation})
            attr.update({"out_channel": layer.out_c})
            attr.update({"out_size": layer.out_h})
            attr.update({"add_out_channel": add_layer.out_c})
            attr.update({"add_out_size": add_layer.out_h})

        elif LAYERTYPE.ROUTE == layer_type:
            pass

        elif LAYERTYPE.COST == layer_type:
            pass

        elif LAYERTYPE.REORG == layer_type:
            attr.update({"stride": layer.stride})

        elif LAYERTYPE.REGION == layer_type:
            attr.update({"n": layer.n})
            attr.update({"classes": layer.classes})
            attr.update({"coords": layer.coords})
            attr.update({"background": layer.background})
            attr.update({"softmax": layer.softmax})
            attr.update({"shape": (-1, layer.c, layer.h, layer.w)})

        elif LAYERTYPE.YOLO == layer_type:
            attr.update({"n": layer.n})
            attr.update({"classes": layer.classes})
            attr.update({"shape": (-1, layer.c, layer.h, layer.w)})

        elif LAYERTYPE.UPSAMPLE == layer_type:
            attr.update({"scale": layer.stride})

        elif LAYERTYPE.L2NORM == layer_type:
            pass

        else:
            err = f"Darknet layer type {layer_type} is not supported in relay."
            raise NotImplementedError(err)

        return attr

    def _get_darknet_params(self, layer, opname):
        """To parse and get the darknet params."""
        layer_type = LAYERTYPE(layer.type)
        params = None
        if LAYERTYPE.CONVOLUTIONAL == layer_type:
            params = self._get_convolution_weights(layer, opname)
        elif LAYERTYPE.CONNECTED == layer_type:
            params = self._get_connected_weights(layer, opname)
        elif LAYERTYPE.REGION == layer_type:
            params = self._get_region_weights(layer, opname)
        elif LAYERTYPE.YOLO == layer_type:
            params = self._get_yolo_weights(layer, opname)
        return params

    def _preproc_layer(self, layer, layer_num):
        """To preprocess each darknet layer, some layer doesnt need processing."""
        if layer_num == 0:
            name = "data"
            sym = new_var(name, shape=self._shape, dtype=self._dtype)
        else:
            sym = self._sym_array[layer_num - 1]
        skip_layer = False
        layer_type = LAYERTYPE(layer.type)
        if LAYERTYPE.ROUTE == layer_type:
            sym = []
            for j in range(layer.n):
                sym.append(self._sym_array[layer.input_layers[j]])
            if layer.n == 1:
                skip_layer = True

        elif LAYERTYPE.COST == layer_type:
            skip_layer = True

        elif LAYERTYPE.SHORTCUT == layer_type:
            sym = [sym, self._sym_array[layer.index]]

        elif LAYERTYPE.BLANK == layer_type:
            skip_layer = True

        if skip_layer is True:
            self._sym_array[layer_num] = sym

        return skip_layer, sym

    def _get_opname(self, layer):
        """Returs the layer name."""
        return LAYERTYPE(layer.type)

    def _new_rnn_state_var(self, state=None, name="rnn"):
        """Returs a symbol for state"""
        sym_name = name + f"{self._state_ctr[name]}_state"
        self._state_ctr[name] += 1
        return new_var(sym_name, shape=state.shape, dtype=str(state.dtype))

    def _get_rnn_state_buffer(self, layer, name):
        """Get the state buffer for rnn."""
        buffer = np.zeros((1, layer.outputs), self._dtype)
        return self._new_rnn_state_var(buffer, name)

    def _get_darknet_rnn_attrs(self, layer, name, sym):
        """Get the rnn converted symbol from attributes."""
        attr = self._get_darknet_attrs(layer, 0)
        op_name = self._get_opname(layer)
        prefix = _get_params_prefix(op_name, name)
        params = self._get_darknet_params(layer, prefix)
        sym = _darknet_convert_symbol(op_name, _as_list(sym), params, attr, prefix)
        if params:
            self._tvmparams.update(params)
        return sym

    def _handle_darknet_rnn_layers(self, layer_num, sym):
        """Parse attributes and handle the rnn layers."""
        attr = {}
        layer = self._net.layers[layer_num]
        processed = False

        layer_type = LAYERTYPE(layer.type)
        if LAYERTYPE.RNN == layer_type:
            attr.update({"n": layer.n})
            attr.update({"batch": layer.batch})
            attr.update({"num_hidden": str(layer.outputs)})
            state = self._get_rnn_state_buffer(layer, "rnn")
            for _ in range(layer.steps):
                input_layer = layer.input_layer
                prefix = "_input_" + str(layer_num)
                sym = self._get_darknet_rnn_attrs(input_layer, prefix, sym)

                self_layer = layer.self_layer
                prefix = "_self_" + str(layer_num)
                state = self._get_darknet_rnn_attrs(self_layer, prefix, state)

                state = sym + state
                self._outs.append(state)

                output_layer = layer.output_layer
                prefix = "_output_" + str(layer_num)
                sym = self._get_darknet_rnn_attrs(output_layer, prefix, state)

            self._sym_array[layer_num] = sym
            processed = True
        return processed, sym

    def _make_outlist(self, sym, op_name, layer, layer_num):
        layer_type = LAYERTYPE(layer.type)
        if layer_type == LAYERTYPE.REGION:
            # Add attributes
            k = _get_params_name(op_name, "attr")
            dshape = self._tvmparams[k].shape
            dtype = self._tvmparams[k].dtype
            self._outs.insert(0, new_var(k, shape=dshape, dtype=dtype))

            # Add bias
            k = _get_params_name(op_name, "bias")
            dshape = self._tvmparams[k].shape
            dtype = self._tvmparams[k].dtype
            self._outs.insert(0, new_var(k, shape=dshape, dtype=dtype))
            if layer_num != self._net.n - 1:
                self._outs.insert(0, sym)

        elif layer_type == LAYERTYPE.YOLO:
            # Add attributes
            k = _get_params_name(op_name, "attr")
            dshape = self._tvmparams[k].shape
            dtype = self._tvmparams[k].dtype
            self._outs.insert(0, new_var(k, shape=dshape, dtype=dtype))

            # Add bias
            k = _get_params_name(op_name, "bias")
            dshape = self._tvmparams[k].shape
            dtype = self._tvmparams[k].dtype
            self._outs.insert(0, new_var(k, shape=dshape, dtype=dtype))

            # Add mask
            k = _get_params_name(op_name, "mask")
            dshape = self._tvmparams[k].shape
            dtype = self._tvmparams[k].dtype
            self._outs.insert(0, new_var(k, shape=dshape, dtype=dtype))

            if layer_num != self._net.n - 1:
                self._outs.insert(0, sym)

    def from_darknet(self):
        """To convert the darknet symbol to relay functions."""
        for i in range(self._net.n):
            layer = self._net.layers[i]
            need_skip, sym = self._preproc_layer(layer, i)
            if need_skip:
                continue

            processed, sym = self._handle_darknet_rnn_layers(i, sym)
            if processed:
                continue

            attr = self._get_darknet_attrs(layer, i)
            op_name = self._get_opname(layer)
            prefix = _get_params_prefix(op_name, i)
            params = self._get_darknet_params(self._net.layers[i], prefix)
            sym = _darknet_convert_symbol(op_name, _as_list(sym), params, attr, prefix)

            if params:
                self._tvmparams.update(params)
            self._sym_array[i] = sym
            self._make_outlist(sym, prefix, layer, i)

        outputs = _as_list(sym) + self._outs
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        sym = _function.Function(analysis.free_vars(outputs), outputs)
        return IRModule.from_expr(sym), self._tvmparams


def from_darknet(net, shape=None, dtype="float32"):
    """Convert from Darknet's model into compatible relay Function.

    Parameters
    ----------
    net : Darknet net parameter
        Darknet net structure.
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """

    return GraphProto(net, shape, dtype).from_darknet()
