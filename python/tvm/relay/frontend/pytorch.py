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
# pylint: disable=import-self, too-many-lines, len-as-condition, no-else-return, unused-variable, too-many-nested-blocks
# pylint: disable=consider-iterating-dictionary, invalid-name, unused-argument, unused-variable, broad-except
# pylint: disable=import-outside-toplevel, simplifiable-if-expression, unnecessary-comprehension
"""PT: PyTorch frontend."""
import itertools
import logging
import sys

import numpy as np

import tvm

from .. import analysis as _analysis
from .. import expr as _expr
from .. import op as _op
from ..ty import TupleType, TensorType, Any
from ..loops import while_loop
from .common import get_relay_op
from .common import infer_shape as _infer_shape
from .common import infer_value as _infer_value
from .common import infer_type as _infer_type
from ..prelude import Prelude, StaticTensorArrayOps

from . import qnn_torch

__all__ = ["from_pytorch"]


# List ADT utilities
def _infer_type_with_prelude(val, prelude):
    body = _infer_type(val, prelude.mod)
    return body.checked_type


def _convert_to_list_adt(py_lst, prelude):
    elem_tys = [_infer_type_with_prelude(elem, prelude) for elem in py_lst]
    msg = "List elements should have identical types"
    assert all(map(lambda ty: ty == elem_tys[0], elem_tys)), msg

    adt_lst = prelude.nil()
    for elem in reversed(py_lst):
        adt_lst = prelude.cons(elem, adt_lst)
    return adt_lst


def _map_tensor_array_constructor(adt_lst, prelude, shape):
    static_tensor_array_ops = StaticTensorArrayOps(prelude, "float32", shape)
    static_tensor_array_ops.register()
    tensor_create = prelude.get_var_static('tensor_constructor', "float32", shape)
    return prelude.map(tensor_create, adt_lst)


def _convert_to_tensor_array(adt_lst, prelude):
    if prelude.length(adt_lst) == 0:
        return prelude.nil()

    checked_type = _infer_type_with_prelude(prelude.hd(adt_lst), prelude)
    shape = checked_type.shape
    tensor_array = _map_tensor_array_constructor(adt_lst, prelude, shape)
    return tensor_array, tuple(shape)


def _should_construct_dynamic_list(list_construct_node):
    # if this list is element-accessed or modified at runtime, generate List ADT
    def is_used_by_list_add(uses):
        for use in uses:
            op_name = use.user.kind()
            output_type = _get_node_type(use.user)
            if op_name in ["aten::add", "aten::add_"] and output_type == "ListType":
                return True
        return False

    def inplace_add_to_add(op_name):
        if op_name == "aten::add_":
            return "aten::add"
        else:
            return op_name

    uses = _get_uses(list_construct_node)

    for loop_use in filter(lambda use: use.user.kind() == "prim::Loop", uses):
        block_input_index = loop_use.offset - 1
        block = list(loop_use.user.blocks())[0]
        list_loop_var = list(block.inputs())[block_input_index]
        uses += _get_uses(list_loop_var.node())

    op_names = map(inplace_add_to_add, set(use.user.kind() for use in uses))

    list_ops = set(["aten::add", "aten::__getitem__", "aten::stack"])
    intersect = list_ops.intersection(op_names)

    if len(intersect) > 0 and intersect != set(["aten::add"]):
        return True

    if is_used_by_list_add(filter(lambda use: use.user.kind() != "prim::Loop", uses)):
        return True

    return False


# operator implementation
def _elemwise(name):
    def _impl(inputs, input_types):
        # TODO: Figure out a better way to get typing to work for tensor + scalar
        type0 = input_types[0]
        if isinstance(inputs[1], _expr.Expr):
            type0 = input_types[1]

        type1 = input_types[1]
        if isinstance(inputs[0], _expr.Expr):
            type1 = input_types[0]

        data0 = _convert_elemwise_input(inputs[0], type0)
        data1 = _convert_elemwise_input(inputs[1], type1)

        return get_relay_op(name)(data0, data1)
    return _impl


def _unary(name):
    def _impl(inputs, input_types):
        input_type = input_types[0]
        data = _convert_elemwise_input(inputs[0], input_type)

        return get_relay_op(name)(data)
    return _impl


def _arange():
    def _impl(inputs, input_types):
        if len(inputs) == 5:
            dtype = "float" if "float" in input_types[0:1] else _convert_dtype_value(inputs[1])
            start = _create_typed_const(0, dtype)
            stop = _create_typed_const(inputs[0], dtype)
            step = _create_typed_const(1, dtype)
        elif len(inputs) == 7:
            dtype = "float" if "float" in input_types[0:3] else _convert_dtype_value(inputs[3])
            start = _create_typed_const(inputs[0], dtype)
            stop = _create_typed_const(inputs[1], dtype)
            step = _create_typed_const(inputs[2], dtype)
        else:
            msg = "Unknown number of arguments (%d) to parse." % (len(inputs))
            raise AssertionError(msg)
        return _op.transform.arange(start=start,
                                    stop=stop,
                                    step=step,
                                    dtype=_convert_data_type(dtype))
    return _impl

def _squeeze():
    def _impl(inputs, input_types):
        data = inputs[0]
        if len(inputs) == 1:
            axis = None
        else:
            axis = [int(inputs[1])]

        return _op.transform.squeeze(data, axis)
    return _impl

def _unsqueeze():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        return _op.transform.expand_dims(data, int(axis), 1)
    return _impl


def _concatenate(prelude):
    def tensor_array_concat(lst, axis):
        assert axis == 0, "Tensor array concat supported only for axis 0"
        tensor_array, shape = _convert_to_tensor_array(lst, prelude)
        concat_shape = (Any(),) + shape[1:]
        static_tensor_array_ops = StaticTensorArrayOps(prelude, "float32", shape)
        static_tensor_array_ops.define_tensor_get_data(concat_shape)

        concat = prelude.get_var_static('tensor_array_concat', "float32", shape)
        concatenated = concat(tensor_array)
        get_tensor = prelude.get_var_static('tensor_get_data', "float32", shape)
        return get_tensor(concatenated)

    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        if not isinstance(data, list):
            return tensor_array_concat(data, axis)

        if isinstance(data, _expr.Expr):
            data = [data]

        return _op.tensor.concatenate(data, int(axis))
    return _impl

def _slice():
    def _impl(inputs, input_types):
        data = inputs[0]
        strides = []

        if isinstance(data, _expr.Expr):
            inferred_shape = _infer_shape(data)
            end = []
            for infer in inferred_shape:
                end.append(int(infer))
            if isinstance(data, _expr.Var):
                end = inferred_shape
                end = list(end)
        else:
            end = data.shape

        begin = [0] * len(end)
        dim = int(inputs[1])
        begin[dim] = int(inputs[2])

        if isinstance(inputs[3], str) and inputs[3].isdigit():
            end[dim] = min(end[dim], int(inputs[3]))
        else:
            end[dim] = inputs[3]

        strides.append(int(inputs[4]))
        return _op.transform.strided_slice(data, begin, end, strides)
    return _impl

def _split():
    def _impl(inputs, input_types):
        data = inputs[0]
        split_size = int(inputs[1])
        dim = int(inputs[2])

        split_index = split_size
        indices = []
        while split_index < _infer_shape(data)[dim]:
            indices.append(split_index)
            split_index += split_size

        return _op.split(data, indices, dim)
    return _impl

def _split_with_sizes():
    def _impl(inputs, inputs_types):
        data = inputs[0]
        dim = int(inputs[2])

        split_index = 0
        indices = []
        sections = _infer_shape(inputs[1])
        for i in range(len(sections) - 1):
            split_index += sections[i]
            indices.append(split_index)

        return _op.split(data, indices, dim)
    return _impl

def _select():
    def _impl(inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        index = _wrap_const(inputs[2])
        return _op.transform.take(data, index, axis=dim)
    return _impl

def _take():
    def _impl(inputs, input_types):
        data = inputs[0]
        import torch

        if isinstance(inputs[1], _expr.Var):
            indices = _op.cast(inputs[1], "int32")
        elif isinstance(inputs[1], torch.Tensor):
            indices = _wrap_const(inputs[1].numpy())
        else:
            msg = "Data type %s could not be parsed in take operator." % (type(inputs[1]))
            raise AssertionError(msg)

        return _op.transform.take(data, indices=indices)
    return _impl

def _topk():
    def _impl(inputs, input_types):
        data = inputs[0]
        k = int(inputs[1])
        axis = int(inputs[2])
        is_ascend = not bool(inputs[3])
        sort = bool(inputs[4])

        if not sort:
            msg = "Currently supports only sorted output for topk operator."
            raise AssertionError(msg)

        outs = _op.topk(data, k=k, axis=axis, is_ascend=is_ascend, ret_type="both")

        return outs[0], outs[1]
    return _impl

def _reciprocal():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _expr.const(1.0) / data
    return _impl

def _repeat():
    def _impl(inputs, input_types):
        data = inputs[0]
        reps = _get_dims(inputs[1])
        return _op.transform.tile(data, reps=reps)
    return _impl

def _repeat_interleave():
    def _impl(inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], int):
            repeats = inputs[1]
            axis = inputs[2]
        else:
            msg = "Only repeat with one value as repeat is currently supported."
            raise AssertionError(msg)
        if axis is None: # Flatten the data if no axis is given from torch
            data = _op.transform.reshape(data, [-1])
            axis = 0
        return _op.transform.repeat(data, repeats=repeats, axis=axis)
    return _impl

def _ones():
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch
        if isinstance(data, _expr.Expr):
            shape = _infer_shape(data)
        elif isinstance(data, list):
            shape = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            shape = data.shape
        else:
            msg = "Data type %s could not be parsed in ones op" % (type(data))
            raise AssertionError(msg)

        dtype = _convert_data_type(_convert_dtype_value(inputs[1]))

        return _op.full(_expr.const(1), shape, dtype=dtype)
    return _impl

def _ones_like():
    def _impl(inputs, input_types):
        data = inputs[0]
        out = _op.ones_like(data)

        # If the input and the output datatype is different, do a cast
        dtype = _convert_data_type(_convert_dtype_value(inputs[1]))
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out
    return _impl


def _zeros():
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch
        if isinstance(data, _expr.Expr):
            shape = _infer_shape(data)
        elif isinstance(data, list):
            shape = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            shape = data.shape
        else:
            msg = "Data type %s could not be parsed in zeros op" % (type(data))
            raise AssertionError(msg)

        dtype = _convert_data_type(_convert_dtype_value(inputs[1]))

        return _op.full(_expr.const(0), shape, dtype=dtype)
    return _impl


def _zeros_like():
    def _impl(inputs, input_types):
        data = inputs[0]
        out = _op.zeros_like(data)

        # If the input and the output datatype is different, do a cast
        dtype = _convert_data_type(_convert_dtype_value(inputs[1]))
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out
    return _impl


def _full():
    def _impl(inputs, input_types):
        data = inputs[0]

        fill_value = inputs[1]
        import torch
        if isinstance(data, _expr.Expr):
            shape = _infer_shape(data)
        elif isinstance(data, list):
            shape = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            shape = data.shape
        else:
            msg = "Data type %s could not be parsed in zeros op" % (type(data))
            raise AssertionError(msg)

        dtype = _convert_data_type(_convert_dtype_value(inputs[2]))

        return _op.full(_expr.const(fill_value), shape, dtype=dtype)
    return _impl

def _full_like():
    def _impl(inputs, input_types):
        data = inputs[0]
        fill_value = inputs[1]

        out = _op.full_like(data, _expr.const(fill_value))

        # If the input and the output datatype is different, do a cast
        dtype = _convert_data_type(_convert_dtype_value(inputs[2]))
        if input_types[0] not in dtype:
            out = _op.cast(out, dtype)

        return out
    return _impl


def _linspace():
    def _impl(inputs, input_types):
        start = inputs[0]
        stop = inputs[1]
        step = inputs[2]

        # Find the spacing between values as step
        if step != 1:
            step = (stop - start) / (step - 1)
            stop = stop + step
        else:
            stop = start + step

        dtype = "float" if "float" in input_types[0:3] else _convert_dtype_value(inputs[3])
        start = _create_typed_const(start, dtype)
        stop = _create_typed_const(stop, dtype)
        step = _create_typed_const(step, dtype)

        return _op.transform.arange(start=start,
                                    stop=stop,
                                    step=step,
                                    dtype=_convert_data_type(dtype))
    return _impl


def _relu():
    def _impl(inputs, input_types):
        data = inputs[0]
        if input_types[0] == "quint8":
            assert len(inputs) == 3, "Input quant param not found in op inputs"
            input_zero_point = _expr.const(inputs[2], dtype="int32")
            return qnn_torch.quantized_relu(data, input_zero_point)
        return _op.nn.relu(data)
    return _impl

def _prelu():
    def _impl(inputs, input_types):
        data = inputs[0]
        alpha = inputs[1]
        return _op.nn.prelu(data, alpha)
    return _impl

def _leaky_relu():
    def _impl(inputs, input_types):
        data = inputs[0]
        alpha = float(inputs[1])
        return _op.nn.leaky_relu(data, alpha)
    return _impl

def _elu():
    def _impl(inputs, input_types):
        data = inputs[0]
        alpha = _expr.const(float(inputs[1]))
        return alpha * _op.nn.relu(_expr.const(1.0) - _op.exp(data)) + _op.nn.relu(data)
    return _impl

def _celu():
    def _impl(inputs, input_types):
        data = inputs[0]
        alpha = _expr.const(float(inputs[1]))
        return alpha * _op.nn.relu(_expr.const(1.0) - _op.exp(data / alpha)) + _op.nn.relu(data)
    return _impl

def _gelu():
    def _impl(inputs, input_types):
        import math
        data = inputs[0]

        def _pow3(x):
            return x * x * x
        return _expr.const(0.5) * data * (_expr.const(1.0) +
                                          _op.tanh(_expr.const(math.sqrt(2.0 / math.pi)) *
                                                   (data + _expr.const(0.044715) * _pow3(data))))
    return _impl

def _selu():
    def _impl(inputs, input_types):
        data = inputs[0]
        # https://pytorch.org/docs/stable/nn.html#selu
        alpha = _expr.const(-1.6732632423543772848170429916717)
        gamma = _expr.const(1.0507009873554804934193349852946)
        return gamma * (alpha * _op.nn.relu(_expr.const(1.0)
                                            - _op.exp(data)) + _op.nn.relu(data))
    return _impl

def _log_sigmoid():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.log(_op.tensor.sigmoid(data))
    return _impl

def _adaptive_avg_pool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        def func(x):
            return _op.nn.adaptive_avg_pool2d(x, output_size=output_size)

        if input_types[0] == "quint8":
            return qnn_torch.apply_with_upcast(data, func)

        return func(data)

    return _impl

def _adaptive_max_pool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        # returns dummy indices too
        return _op.nn.adaptive_max_pool2d(
            data,
            output_size=output_size), None
    return _impl

def _adaptive_max_pool_3d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])
        # returns dummy indices too
        return _op.nn.adaptive_max_pool3d(data, output_size=output_size), None

    return _impl

def _adaptive_avg_pool_3d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])
        return _op.nn.adaptive_avg_pool3d(data, output_size=output_size)

    return _impl

def _maxpool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        strides = _infer_shape(inputs[2])
        padding = _infer_shape(inputs[3])
        dilation = _infer_shape(inputs[4])
        ceil_mode = int(inputs[5])

        if dilation != (1, 1):
            msg = "MaxPool2d with dilation %s is not implemented" % (str(dilation))
            raise NotImplementedError(msg)

        return _op.nn.max_pool2d(data, pool_size, strides, padding, "NCHW", ceil_mode)
    return _impl

def _maxpool_1d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        strides = _infer_shape(inputs[2])
        padding = _infer_shape(inputs[3])
        dilation = _infer_shape(inputs[4])
        ceil_mode = int(inputs[5])

        if dilation != (1,):
            msg = "MaxPool1d with dilation %s is not implemented" % (str(dilation))
            raise NotImplementedError(msg)

        return _op.nn.max_pool1d(data, pool_size, strides, padding, "NCW", ceil_mode)
    return _impl

def _maxpool_3d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        strides = _infer_shape(inputs[2])
        padding = _infer_shape(inputs[3])
        dilation = _infer_shape(inputs[4])
        ceil_mode = int(inputs[5])
        if dilation != (1, 1, 1):
            msg = "MaxPool3d with dilation %s is not implemented" % (str(dilation))
            raise NotImplementedError(msg)

        return _op.nn.max_pool3d(data,
                                 pool_size=pool_size,
                                 strides=strides,
                                 padding=padding,
                                 ceil_mode=ceil_mode)
    return _impl

def _hardtanh():
    def _impl(inputs, input_types):
        a = inputs[0]
        tanh_min = float(inputs[1])
        tanh_max = float(inputs[2])
        return _op.tensor.clip(a, tanh_min, tanh_max)
    return _impl

def _convolution():
    def _impl(inputs, input_types):
        # Use transpose or normal
        use_transpose = True if inputs[6] == 1 else False

        data = inputs[0]
        weight = inputs[1]
        bias = inputs[2]
        strides = inputs[3]
        padding = inputs[4]
        dilation = inputs[5]

        if isinstance(weight, _expr.Expr):
            inferred_shape = _infer_shape(weight)
            weight_shape = []
            for infer in inferred_shape:
                weight_shape.append(infer)
        else:
            msg = "Data type %s could not be parsed in conv op" % (type(weight))
            raise AssertionError(msg)

        # Transposed convolutions have IOHW layout.
        if use_transpose:
            weight_shape[0], weight_shape[1] = weight_shape[1], weight_shape[0]

        channels = weight_shape[0]
        groups = int(inputs[8])

        # Check if this is depth wise convolution
        # We need to reshape weight so that Relay could recognize this is depth wise
        # weight_shape[1] is always in_channels // groups
        # For depthwise, in_channels == groups, so weight_shape[1] == 1
        # If groups > 1 but weight_shape[1] != 1, this is group convolution
        if groups > 1 and weight_shape[1] == 1:
            channel_multiplier = channels // groups
            new_weight_shape = (groups, channel_multiplier, weight_shape[2], weight_shape[3])
            weight = _op.transform.reshape(weight, new_weight_shape)

        kernel_size = weight_shape[2:]
        use_bias = isinstance(bias, _expr.Expr)

        if isinstance(strides, _expr.Expr):
            strides = _infer_shape(strides)

        if isinstance(padding, _expr.Expr):
            padding = _infer_shape(padding)

        if isinstance(dilation, _expr.Expr):
            dilation = _infer_shape(dilation)

        data_layout = "NCHW"
        kernel_layout = "OIHW"
        conv_op = _op.nn.conv2d

        if use_transpose:
            assert len(kernel_size) == 2, "ConvTranspose 3D not supported"
            conv_op = _op.nn.conv2d_transpose
        if len(kernel_size) == 3:
            conv_op = _op.nn.conv3d
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"

        conv_out = conv_op(data,
                           weight,
                           strides=strides,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           channels=channels,
                           kernel_size=kernel_size,
                           data_layout=data_layout,
                           kernel_layout=kernel_layout,
                           out_layout="",
                           out_dtype="")
        if use_bias:
            return _op.nn.bias_add(conv_out, bias)
        else:
            return conv_out
    return _impl

def _softmax():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]
        if isinstance(axis, str):
            axis = int(axis)

        return _op.nn.softmax(data, axis=axis)
    return _impl

def _threshold():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.relu(data)
    return _impl

def _contiguous():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)
    return _impl

def _batch_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]

        channels = _infer_shape(data)

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        moving_mean = inputs[3]
        moving_var = inputs[4]
        epsilon = float(inputs[7])

        return _op.nn.batch_norm(data,
                                 gamma,
                                 beta,
                                 moving_mean,
                                 moving_var,
                                 axis=1,
                                 epsilon=epsilon,
                                 center=center,
                                 scale=scale)[0]
    return _impl

def _instance_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        data_type = input_types[0]
        channels = _infer_shape(data)

        if isinstance(inputs[1], _expr.Expr) and isinstance(inputs[2], _expr.Expr):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
            gamma = weight
        else:
            scale = center = False

        if not scale:
            gamma = _create_typed_const(np.ones([int(channels[1])]), data_type)

        if not center:
            beta = _create_typed_const(np.zeros([int(channels[1])]), data_type)

        epsilon = float(inputs[7])
        return _op.nn.instance_norm(data,
                                    gamma,
                                    beta,
                                    axis=1,
                                    epsilon=epsilon,
                                    center=center,
                                    scale=scale)
    return _impl

def _get_dims(data):
    import torch
    if isinstance(data, _expr.Expr):
        dims = _infer_shape(data)
    elif isinstance(data, list):
        dims = data
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        dims = data.shape
    else:
        msg = "Data type %s could not be parsed" % type(data)
        raise AssertionError(msg)
    return dims

def _layer_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        ndims = len(_get_dims(inputs[1]))
        assert ndims == 1, "Support only normalization over last one dimension."

        return _op.nn.layer_norm(data,
                                 gamma=inputs[2],
                                 beta=inputs[3],
                                 axis=-1,
                                 epsilon=float(inputs[4]),
                                 center=True,
                                 scale=True)
    return _impl


def _group_norm():
    def _impl(inputs, input_types):
        data = inputs[0]
        gamma = inputs[2]
        beta = inputs[3]
        num_groups = inputs[1]
        epsilon = float(inputs[4])

        return _op.nn.group_norm(data,
                                 gamma=gamma,
                                 beta=beta,
                                 num_groups=num_groups,
                                 axis=1,
                                 epsilon=epsilon,
                                 center=True,
                                 scale=True)
    return _impl


def _transpose(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch
        if isinstance(data, _expr.Expr):
            ndims = len(_infer_shape(data, prelude.mod))
        elif isinstance(data, list):
            ndims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            ndims = data.shape
        else:
            msg = "Data type %s could not be parsed in transpose op" % (type(data))
            raise AssertionError(msg)

        if isinstance(data, tvm.runtime.NDArray):
            ndims = len(data.shape)
        axes = list(range(ndims))

        num_inputs = len(inputs)

        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
            if not isinstance(data, _expr.Expr):
                data = _expr.const(data)

        elif num_inputs == 3:
            parse = lambda i: ndims * (i < 0) + i
            src, dst = [parse(int(inputs[i])) for i in [1, 2]]
            axes[src] = dst
            axes[dst] = src
        else:
            axes = inputs[1]
        return _op.transform.transpose(data, axes)
    return _impl

def _flatten():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.batch_flatten(data)
    return _impl

def _dense():
    def _impl(inputs, input_types):
        use_bias = isinstance(inputs[0], _expr.Expr)

        data = inputs[1]
        data_type = input_types[1]
        weight = inputs[2]

        beta = inputs[3]
        alpha = inputs[4]

        if not isinstance(alpha, _expr.Expr):
            alpha = _create_typed_const(alpha, data_type)
            data *= alpha

        if not isinstance(beta, _expr.Expr):
            beta = _create_typed_const(beta, data_type)
            weight *= beta

        weight_out = _op.transform.transpose(weight, axes=[1, 0])

        units = _infer_shape(weight_out)[0]
        dense_out = _op.nn.dense(data, weight_out, units=units)

        if use_bias:
            bias = inputs[0]
            return _op.nn.bias_add(dense_out, bias)
        else:
            return dense_out
    return _impl


def _size(prelude):
    def _impl_dynamic(inp, axis):
        shape_dynamic = _op.shape_of(inp)
        if axis is not None:
            return _op.take(shape_dynamic, _expr.const(axis), 0)
        return shape_dynamic

    def _impl(inputs, input_types):
        shape = _infer_shape(inputs[0], prelude.mod)
        axis = None
        if len(inputs) > 1:
            axis = int(inputs[1])

        if any(map(lambda s: isinstance(s, tvm.tir.expr.Any), shape)):
            if axis is None or isinstance(shape[axis], tvm.tir.expr.Any):
                return _impl_dynamic(inputs[0], axis)

        if axis is not None:
            return shape[axis]
        return shape
    return _impl


def _numtotensor():
    def _impl(inputs, input_types):
        val = inputs[0]
        dtype = type(val)

        if isinstance(val, tvm.tir.IntImm):
            val = val.__int__()
            dtype = int

        arr = val * np.ones([]).astype(dtype)
        return arr
    return _impl

def _view():
    def _impl(inputs, input_types):
        data = inputs[0]

        if len(inputs) == 3:
            new_shape = [inputs[1], _infer_shape(inputs[2])[0]]
        else:
            if isinstance(inputs[1], list):
                new_shape = inputs[1]
            else:
                new_shape = _infer_shape(inputs[1])

        return _op.transform.reshape(data, new_shape)
    return _impl

def _reshape():
    def _impl(inputs, input_types):
        data = inputs[0]
        if isinstance(inputs[1], list):
            new_shape = inputs[1]
        else:
            new_shape = _infer_shape(inputs[1])
        return _op.transform.reshape(data, new_shape)
    return _impl

def _clone():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.copy(data)
    return _impl

def _log_softmax():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = int(inputs[1])
        return _op.nn.log_softmax(data, axis)
    return _impl

def _sigmoid():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.sigmoid(data)
    return _impl

def _softplus():
    def _impl(inputs, input_types):
        data = inputs[0]
        beta = _expr.const(float(inputs[1]))
        return _op.log(_op.exp(inputs[0] * beta) + _expr.const(1.)) / beta
    return _impl

def _avg_pool2d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        if inputs[2]:
            strides = _infer_shape(inputs[2])
        else:
            strides = pool_size
        padding = _infer_shape(inputs[3])

        ceil_mode = int(inputs[4])
        count_include_pad = int(inputs[5])

        def func(x):
            return _op.nn.avg_pool2d(x,
                                     pool_size=pool_size,
                                     strides=strides,
                                     padding=padding,
                                     ceil_mode=ceil_mode,
                                     count_include_pad=count_include_pad)

        if input_types[0] == "quint8":
            return qnn_torch.apply_with_upcast(data, func)

        return func(data)

    return _impl

def _avg_pool3d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        if inputs[2]:
            strides = _infer_shape(inputs[2])
        else:
            strides = pool_size
        padding = _infer_shape(inputs[3])

        ceil_mode = int(inputs[4])
        count_include_pad = int(inputs[5])

        return _op.nn.avg_pool3d(data,
                                 pool_size=pool_size,
                                 strides=strides,
                                 padding=padding,
                                 ceil_mode=ceil_mode,
                                 count_include_pad=count_include_pad)
    return _impl

def _dropout():
    def _impl(inputs, input_types):
        data = inputs[0]
        rate = float(inputs[1])

        return _op.nn.dropout(data, rate)
    return _impl

def _reduce(name):
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = None
        keepdims = False

        if len(inputs) > 2: # default, torch have only data, axis=None, keepdims=False
            if isinstance(inputs[1], int):
                axis = int(inputs[1])
            else:
                axis = list(_infer_shape(inputs[1]))
            keepdims = bool(inputs[2])

        return get_relay_op(name)(data, axis=axis, keepdims=keepdims)

    return _impl

def _std():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = list(_infer_shape(inputs[1]))
        keepdims = bool(inputs[3])
        unbiased = bool(inputs[2])

        if unbiased:
            msg = "Currently only supports standard-deviation calculated via the biased "\
                  "estimator. Pytorch's Bessel's correction is not supported."
            raise NotImplementedError(msg)

        return _op.reduce.std(data, axis=axis, keepdims=keepdims)

    return _impl

def _variance():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = list(_infer_shape(inputs[1]))
        keepdims = bool(inputs[3])
        unbiased = bool(inputs[2])

        if unbiased:
            msg = "Currently only supports standard-deviation calculated via the biased "\
                  "estimator. Pytorch's Bessel's correction is not supported."
            raise NotImplementedError(msg)

        return _op.reduce.variance(data, axis=axis, keepdims=keepdims)

    return _impl

def _mean():
    def _impl(inputs, input_types):
        data = inputs[0]

        if inputs[1]:
            axis = _infer_shape(inputs[1])
        else:
            axis = None
        if len(inputs) > 2 and inputs[2]:
            keepdims = int(inputs[2])
        else:
            keepdims = False
        if len(inputs) > 3 and inputs[3]:
            exclude = int(inputs[3])
        else:
            exclude = False

        def func(x):
            return _op.mean(x, axis, keepdims, exclude)

        if input_types[0] == "quint8":
            assert len(inputs) == 6, "Input quant param not found in op inputs"
            input_scale = _expr.const(inputs[4])
            input_zero_point = _expr.const(inputs[5])
            return qnn_torch.quantized_mean(data, input_scale,
                                            input_zero_point, func)

        return func(data)

    return _impl

def _chunk(prelude):
    def _impl(inputs, input_types):
        data = inputs[0]

        num_chunks = int(inputs[1])
        axis = int(inputs[2])

        if isinstance(data, _expr.Expr):
            inferred_shape = _infer_shape(data, prelude.mod)

        shape = []
        for infer in inferred_shape:
            shape.append(infer)

        dim = int(shape[axis])

        if dim % num_chunks:
            unif_size = int(dim / (num_chunks - 1))
        else:
            unif_size = int(dim / num_chunks)

        chunks = []
        for i in range(0, dim, unif_size):
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = i
            end[axis] = i + unif_size
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin, end, stride)
            chunks.append(chunk_out)

        if dim % num_chunks:
            begin = [0] * len(shape)
            end = shape[:]
            begin[axis] = unif_size * (num_chunks - 1)
            end[axis] = dim
            stride = [1] * len(shape)

            chunk_out = _op.transform.strided_slice(data, begin, end, stride)
            chunks.append(chunk_out)

        return chunks
    return _impl

def _matmul():
    def _impl(inputs, input_types):
        data0 = inputs[0]
        data1 = inputs[1]
        data1_t = _op.transpose(data1, axes=(1, 0))

        return _op.nn.dense(data0, data1_t)
    return _impl

def _expand():
    def _impl(inputs, input_types):
        data_in = inputs[0]
        if isinstance(data_in, _expr.Expr):
            shape = _infer_shape(data_in)

        ndims = len(shape)
        sizes = _infer_shape(inputs[1])
        out = inputs[0]

        for i in range(ndims):
            if sizes[i] in {-1, shape[i]}:
                continue
            data = list()
            for temp in range(sizes[i]):
                data.append(out)
            call = _op.tensor.concatenate(data, i)

        return call
    return _impl

def _int():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Expr):
            return inputs[0]
        return int(inputs[0])
    return _impl

def _identity():
    def _impl(inputs, input_types):
        return inputs[0]
    return _impl

def _none():
    def _impl(inputs, input_types):
        return None
    return _impl

def _pad():
    def _impl(inputs, input_types):
        data = inputs[0]
        padding = inputs[1]
        pad_width = list(zip(padding, padding))
        pad_value = inputs[2]
        return _op.nn.pad(data, pad_width, pad_value)
    return _impl


def _clamp():
    def _impl(inputs, input_types):
        data = inputs[0]
        amin = inputs[1] if inputs[1] else np.finfo(np.float32).min
        amax = inputs[2] if inputs[2] else np.finfo(np.float32).max
        return _op.clip(data, amin, amax)
    return _impl


def _to():
    def _impl(inputs, input_types):
        data = inputs[0]
        if inputs[3] in ["cpu", "cuda"]:
            return data
        # special handling for aten::to(data, 6, _, _, _) case
        # 6 means dtype = float
        # this happens when converting upsampling with scale factor
        cast_func = {
            6: float,
            3: int,
        }
        cast_func_expr = {
            6: lambda x: _op.cast(x, "float32"),
            3: lambda x: _op.cast(x, "int32"),
        }
        if inputs[1] in cast_func and not isinstance(data, _expr.Expr):
            return cast_func[inputs[1]](data)
        elif inputs[1] in cast_func and isinstance(data, _expr.Expr):
            return cast_func_expr[inputs[1]](data)
        return data

    return _impl

def _upsample(method):
    def _impl(inputs, input_types):
        if isinstance(inputs[1], _expr.Var):
            out_size = _infer_shape(inputs[1])
        elif isinstance(inputs[1], list):
            infer_res = [_infer_value(size, {}) for size in inputs[1]]
            out_size = [np.asscalar(res.asnumpy().astype(np.int))
                        for res in infer_res]

        data = inputs[0]

        if len(inputs) > 2:
            align_corners = inputs[2]
        else:
            align_corners = False

        if align_corners:
            coord_trans = "align_corners"
        else:
            coord_trans = "half_pixel"

        def func(x):
            return _op.image.resize(x, out_size, "NCHW", method, coord_trans)

        if input_types[0] == "quint8":
            import torch
            from packaging import version

            # Torch version > 1.4 changed upsampling API
            if version.parse(torch.__version__) > version.parse("1.4.0"):
                num_inputs = 7
            else:
                num_inputs = 5

            assert len(inputs) == num_inputs, "Input quant param not found in op inputs"

            input_scale = _expr.const(inputs[-2])
            input_zero_point = _expr.const(inputs[-1])
            return qnn_torch.quantized_upsample(data, input_scale,
                                                input_zero_point, func)
        return func(data)

    return _impl

def _expand_as():
    def _impl(inputs, input_types):
        # TODO: maybe fix this
        # This assumes expand_as can be removed because TVM has broadcast op
        msg = "aten::expand_as(...) found, assume it is part of broadcast op"
        logging.warning(msg)
        return inputs[0]
    return _impl


def _Bool():
    def _impl(inputs, input_types):
        assert len(inputs) == 1
        return inputs[0]
    return _impl

def _Float():
    def _impl(inputs, input_types):
        assert len(inputs) == 1
        return _op.cast(inputs[0], "float32")
    return _impl


def _mm():
    def _impl(inputs, input_types):
        return _op.nn.dense(inputs[0], inputs[1])
    return _impl


def _bitwise_not():
    def _impl(inputs, input_types):
        data = inputs[0]
        # The input tensor must be of integral or Boolean types.
        # For bool tensors, it computes the logical NOT
        if input_types[0] == "bool":
            out = _op.logical_not(_op.cast(data, "bool"))
        else:
            out = _op.bitwise_not(_op.cast(data, "int"))

        return out
    return _impl


def _bitwise_xor():
    def _impl(inputs, input_types):
        lhs = inputs[0]

        import torch
        if isinstance(inputs[1], _expr.Var):
            rhs = inputs[1]
        elif isinstance(inputs[1], torch.Tensor):
            rhs = _wrap_const(inputs[1].numpy())
        else:
            msg = "Data type %s could not be parsed in bitwise_xor operator." % (type(inputs[1]))
            raise AssertionError(msg)

        lhs = _op.cast(lhs, "bool") if input_types[0] == "bool" else _op.cast(lhs, "int")
        rhs = _op.cast(rhs, "bool") if input_types[1] == "bool" else _op.cast(rhs, "int")

        return _op.bitwise_xor(lhs, rhs)
    return _impl


def _logical_not():
    def _impl(inputs, input_types):
        data = inputs[0]

        return _op.logical_not(_op.cast(data, "bool"))
    return _impl


def _logical_xor():
    def _impl(inputs, input_types):
        lhs = _op.cast(inputs[0], "bool")

        import torch
        if isinstance(inputs[1], _expr.Var):
            rhs = inputs[1]
        elif isinstance(inputs[1], torch.Tensor):
            rhs = _wrap_const(inputs[1].numpy())
        else:
            msg = "Data type %s could not be parsed in logical_xor operator." % (type(inputs[1]))
            raise AssertionError(msg)

        rhs = _op.cast(rhs, "bool")

        return _op.logical_xor(lhs, rhs)
    return _impl


def _list_getitem(prelude):
    def _impl(inputs, input_types):
        return prelude.nth(inputs[0], _wrap_const(inputs[1]))
    return _impl


def _list_len(prelude):
    def _impl(inputs, input_types):
        return prelude.length(inputs[0])
    return _impl


def _add(prelude):
    # add_ is overloaded for tensor add and list concat
    def _impl(inputs, input_types):
        if input_types[0] == "ListType":
            return prelude.concat(inputs[0], inputs[1])
        return _elemwise("add")(inputs, input_types)
    return _impl


def _tensor_array_stack(prelude):
    def _impl(inputs, input_types):
        tensor_array, shape = _convert_to_tensor_array(inputs[0], prelude)
        stack = prelude.get_var_static('tensor_array_stack', "float32", shape)
        stacked = stack(tensor_array)

        stacked_shape = (Any(),) + shape
        static_tensor_array_ops = StaticTensorArrayOps(prelude, "float32", shape)
        static_tensor_array_ops.define_tensor_get_data(stacked_shape)
        # passing stacked_shape below gives "'Prelude' object has no attribute" error
        get_tensor = prelude.get_var_static('tensor_get_data', "float32", shape)
        return get_tensor(stacked)
    return _impl


# Helper functions for operator implementation
def _convert_dtype_value(val):
    convert_torch_dtype_map = {7:"torch.float64",
                               6:"torch.float32",
                               5:"torch.float16",
                               4:"torch.int64",
                               3:"torch.int32",
                               2:"torch.int16",
                               1:"torch.int8",
                               0:"torch.unit8",
                               None:"torch.int64"} # Default is torch.int64
    if val in convert_torch_dtype_map:
        return convert_torch_dtype_map[val]
    else:
        msg = "Torch data type value %d is not handled yet." % (val)
        raise NotImplementedError(msg)

def _convert_data_type(input_type):
    if input_type in ["double", "torch.float64"]:
        return "float64"
    elif input_type in ["float", "torch.float32"]:
        return "float32"
    elif input_type in ["half", "torch.float16"]:
        return "float16"
    elif input_type in ["long", "torch.int64"]:
        return "int64"
    elif input_type in ["int", "torch.int32"]:
        return "int32"
    elif input_type in ["short", "torch.int16"]:
        return "int16"
    elif input_type in ["char", "torch.int8"]:
        return "int8"
    elif input_type in ["byte", "torch.uint8"]:
        return "uint8"
    else:
        raise NotImplementedError("input_type {} is not handled yet" % (input_type))
    return "float32"

def _create_typed_const(data, data_type):
    dtype = _convert_data_type(data_type)

    if dtype == "float64":
        typed_data = _expr.const(np.float64(data), dtype=dtype)
    elif dtype == "float32":
        typed_data = _expr.const(np.float32(data), dtype=dtype)
    elif dtype == "float16":
        typed_data = _expr.const(np.float16(data), dtype=dtype)
    elif dtype == "int64":
        typed_data = _expr.const(np.int64(data), dtype=dtype)
    elif dtype == "int32":
        typed_data = _expr.const(np.int32(data), dtype=dtype)
    elif dtype == "int16":
        typed_data = _expr.const(np.int16(data), dtype=dtype)
    elif dtype == "int8":
        typed_data = _expr.const(np.int8(data), dtype=dtype)
    elif dtype == "uint8":
        typed_data = _expr.const(np.uint8(data), dtype=dtype)
    else:
        raise NotImplementedError("input_type {} is not handled yet" % (data_type))
    return typed_data

def _convert_elemwise_input(data, input_type):
    import torch
    if isinstance(data, torch.Tensor):
        return _expr.const(data.item(), dtype=_convert_data_type(input_type))
    elif not isinstance(data, _expr.Expr):
        return _expr.const(data, dtype=_convert_data_type(input_type))
    else:
        return data

def _wrap_const(c):
    if not isinstance(c, (_expr.Expr, list, tvm.tir.expr.Any)):
        return _expr.const(c)
    return c

# Operator mappings
def _get_convert_map(prelude):
    convert_map = {
        "aten::device"                          : _none(),
        "aten::sub"                             : _elemwise("subtract"),
        "aten::sub_"                            : _elemwise("subtract"),
        "aten::max"                             : _elemwise("maximum"),
        "aten::min"                             : _elemwise("minimum"),
        "aten::mul"                             : _elemwise("multiply"),
        "aten::mul_"                            : _elemwise("multiply"),
        "aten::pow"                             : _elemwise("power"),
        "aten::arange"                          : _arange(),
        "aten::div"                             : _elemwise("divide"),
        "aten::div_"                            : _elemwise("divide"),
        "aten::ones"                            : _ones(),
        "aten::ones_like"                       : _ones_like(),
        "aten::zeros"                           : _zeros(),
        "aten::zeros_like"                      : _zeros_like(),
        "aten::full"                            : _full(),
        "aten::full_like"                       : _full_like(),
        "aten::linspace"                        : _linspace(),
        "aten::reciprocal"                      : _reciprocal(),
        "aten::repeat"                          : _repeat(),
        "aten::repeat_interleave"               : _repeat_interleave(),
        "aten::to"                              : _to(),
        "aten::squeeze"                         : _squeeze(),
        "aten::unsqueeze"                       : _unsqueeze(),
        "aten::cat"                             : _concatenate(prelude),
        "aten::slice"                           : _slice(),
        "aten::split"                           : _split(),
        "aten::split_with_sizes"                : _split_with_sizes(),
        "aten::select"                          : _select(),
        "aten::take"                            : _take(),
        "aten::topk"                            : _topk(),
        "aten::relu"                            : _relu(),
        "aten::relu_"                           : _relu(),
        "aten::prelu"                           : _prelu(),
        "aten::leaky_relu"                      : _leaky_relu(),
        "aten::elu"                             : _elu(),
        "aten::celu"                            : _celu(),
        "aten::gelu"                            : _gelu(),
        "aten::selu"                            : _selu(),
        "aten::log_sigmoid"                     : _log_sigmoid(),
        "aten::adaptive_avg_pool2d"             : _adaptive_avg_pool_2d(),
        "aten::adaptive_max_pool2d"             : _adaptive_max_pool_2d(),
        "aten::max_pool2d"                      : _maxpool_2d(),
        "aten::max_pool2d_with_indices"         : _maxpool_2d(),
        "aten::max_pool1d"                      : _maxpool_1d(),
        "aten::max_pool3d"                      : _maxpool_3d(),
        "aten::hardtanh"                        : _hardtanh(),
        "aten::hardtanh_"                       : _hardtanh(),
        "aten::_convolution"                    : _convolution(),
        "aten::softmax"                         : _softmax(),
        "aten::threshold"                       : _threshold(),
        "aten::threshold_"                      : _threshold(),
        "aten::contiguous"                      : _contiguous(),
        "aten::batch_norm"                      : _batch_norm(),
        "aten::instance_norm"                   : _instance_norm(),
        "aten::layer_norm"                      : _layer_norm(),
        "aten::group_norm"                      : _group_norm(),
        "aten::transpose"                       : _transpose(prelude),
        "aten::transpose_"                      : _transpose(prelude),
        "aten::t"                               : _transpose(prelude),
        "aten::flatten"                         : _flatten(),
        "aten::addmm"                           : _dense(),
        "aten::size"                            : _size(prelude),
        "aten::view"                            : _view(),
        "aten::reshape"                         : _reshape(),
        "aten::clone"                           : _clone(),
        "aten::log_softmax"                     : _log_softmax(),
        "aten::sigmoid"                         : _sigmoid(),
        "aten::softplus"                        : _softplus(),
        "aten::avg_pool2d"                      : _avg_pool2d(),
        "aten::avg_pool3d"                      : _avg_pool3d(),
        "aten::dropout"                         : _dropout(),
        "aten::dropout_"                        : _dropout(),
        "aten::feature_dropout"                 : _dropout(),
        "aten::alpha_dropout"                   : _dropout(),
        "aten::mean"                            : _mean(),
        "aten::chunk"                           : _chunk(prelude),
        "aten::matmul"                          : _matmul(),
        "aten::expand"                          : _expand(),
        "aten::Int"                             : _int(),
        "prim::NumToTensor"                     : _numtotensor(),
        "aten::constant_pad_nd"                 : _pad(),
        "aten::permute"                         : _transpose(prelude),
        "aten::sum"                             : _reduce("sum"),
        "aten::prod"                            : _reduce("prod"),
        "aten::argmin"                          : _reduce("argmin"),
        "aten::argmax"                          : _reduce("argmax"),
        "aten::std"                             : _std(),
        "aten::var"                             : _variance(),
        "aten::abs"                             : _unary("abs"),
        "aten::neg"                             : _unary("negative"),
        "aten::cos"                             : _unary("cos"),
        "aten::sin"                             : _unary("sin"),
        "aten::tan"                             : _unary("tan"),
        "aten::tanh"                            : _unary("tanh"),
        "aten::atan"                            : _unary("atan"),
        "aten::log"                             : _unary("log"),
        "aten::exp"                             : _unary("exp"),
        "aten::erf"                             : _unary("erf"),
        "aten::trunc"                           : _unary("trunc"),
        "aten::sign"                            : _unary("sign"),
        "aten::sqrt"                            : _unary("sqrt"),
        "aten::rsqrt"                           : _unary("rsqrt"),
        "aten::ceil"                            : _unary("ceil"),
        "aten::floor"                           : _unary("floor"),
        "aten::round"                           : _unary("round"),
        "aten::isfinite"                        : _unary("isfinite"),
        "aten::isnan"                           : _unary("isnan"),
        "aten::clamp"                           : _clamp(),
        "aten::detach"                          : _identity(),
        "aten::upsample_bilinear2d"             : _upsample("bilinear"),
        "aten::upsample_nearest2d"              : _upsample("nearest_neighbor"),
        "aten::expand_as"                       : _expand_as(),
        "aten::lt"                              : _elemwise("less"),
        "aten::gt"                              : _elemwise("greater"),
        "aten::le"                              : _elemwise("less_equal"),
        "aten::ge"                              : _elemwise("greater_equal"),
        "aten::ne"                              : _elemwise("not_equal"),
        "aten::eq"                              : _elemwise("equal"),
        "aten::logical_not"                     : _logical_not(),
        "aten::logical_xor"                     : _logical_xor(),
        "aten::bitwise_not"                     : _bitwise_not(),
        "aten::bitwise_xor"                     : _bitwise_xor(),
        "aten::Bool"                            : _Bool(),
        "aten::Float"                           : _Float(),
        "aten::adaptive_avg_pool3d"             : _adaptive_avg_pool_3d(),
        "aten::adaptive_max_pool3d"             : _adaptive_max_pool_3d(),
        "aten::mm"                              : _matmul(),
        "relay::tensor_array_stack"             : _tensor_array_stack(prelude),
        "aten::add"                             : _add(prelude),
        "aten::add_"                            : _add(prelude),
        "aten::stack"                           : _tensor_array_stack(prelude),
        "aten::__getitem__"                     : _list_getitem(prelude),
        "aten::len"                             : _list_len(prelude),
    }
    return convert_map


def _run_jit_passes(graph):
    """ The inline pass is necessary to unwrap prim::CallMethod """
    import torch
    torch._C._jit_pass_inline(graph)


def _is_int_seq(seq):
    return len(seq) > 0 and all([isinstance(i, int) for i in seq])


def _get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.var(name, shape=tensor.shape)
    return tensor, var


def _get_output_name(node):
    assert node.outputsSize() == 1
    return node.output().debugName()


def _get_output_names(node):
    return [output.debugName() for output in node.outputs()]


def _get_input_names(node_or_graph):
    return [inp.debugName() for inp in node_or_graph.inputs()]


def _get_op_inputs(op_node, outputs):
    return [outputs[name] for name in _get_input_names(op_node)]


def _get_node_type(node):
    assert node.outputsSize() == 1
    return node.output().type().kind()


def _get_uses(node):
    uses = []
    for output in node.outputs():
        uses += output.uses()
    return uses


def _get_users(node):
    return [use.user for use in _get_uses(node)]


def _report_missing_conversion(op_names, convert_map):
    """ Check if all ops in an input graph are supported by TVM """
    known_ops = ["prim::Constant", "prim::GetAttr",
                 "prim::ListConstruct", "prim::ListUnpack",
                 "prim::TupleConstruct", "prim::TupleUnpack",
                 "prim::If", "prim::Loop"]
    known_ops += list(convert_map.keys())
    known_ops += list(qnn_torch.convert_map.keys())

    missing = [op_name for op_name in op_names
               if op_name not in known_ops]

    if missing:
        msg = "The following operators are not implemented: {}".format(missing)
        raise NotImplementedError(msg)


def _check_inputs(graph, input_shapes):
    """
    Check the graph inputs match the expected number of inputs
    and are in the correct format
    """
    ir_inputs = _get_graph_input_names(graph)

    if not isinstance(input_shapes, list):
        msg = "Graph inputs input_shapes should be list"
        raise RuntimeError(msg)
    missing_inputs = len(ir_inputs) - len(input_shapes)
    if missing_inputs > 0:
        msg = "Missing {} graph input(s) in input_shapes".format(missing_inputs)
        raise RuntimeError(msg)

    for num, inp in enumerate(input_shapes):
        if num < len(ir_inputs):
            if not isinstance(inp, tuple):
                msg = "Graph input {} is not a tuple".format(num)
                raise RuntimeError(msg)
            if (len(inp) != 2 or not isinstance(inp[0], str)):
                msg = "Graph input {} is not valid, expected ('name', shape)".format(inp)
                raise RuntimeError(msg)
        else:
            msg = "Unused graph input {} in input_shapes".format(inp)
            logging.warning(msg)


def _getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert len(attribute_names) == 1
    attr_name = node.s(attribute_names[0])
    return attr_name


def _getattr_full_name(getattrs):
    return ".".join([_getattr_attr_name(node) for node in getattrs])


def _get_input_types(op_node):
    """ Returns a torch type for each input nodes """
    input_list_types = []
    for input_node in op_node.inputs():
        in_ty = input_node.type()
        input_node_kind = in_ty.kind()
        if input_node_kind == 'TensorType':
            if in_ty.scalarType() is None:
                # Tensor's type can be unknown if we use torch.jit.script(...)
                # Defaults to float for now
                logging.warning("Untyped Tensor found, assume it is float")
                input_list_types.append("float")
            else:
                input_list_types.append(in_ty.scalarType().lower())

        elif input_node_kind == 'ListType':
            input_list_types.append("ListType")
        elif input_node_kind in ['IntType', 'FloatType', 'BoolType',
                                 'StringType', 'OptionalType']:
            input_list_types.append(str(in_ty).lower())
        else:
            input_list_types.append('UnsupportedType')

    if op_node.kind() in ['aten::ones', 'aten::zeros']:
        node_type = op_node.output().type()
        scalar_type = node_type.scalarType()
        if scalar_type:
            input_list_types[0] = scalar_type.lower()

    return input_list_types


def _get_constant(node):
    """ Retrieve a constant associated with this prim::Constant node """
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)

    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()

        if ty in ["IntType", "BoolType"]:
            return node.i(attr_name)
        elif ty in ["FloatType", "LongType"]:
            return node.f(attr_name)
        elif ty in ["TensorType", "CompleteTensorType"]:
            tensor = node.t(attr_name)
            if len(tensor.shape) == 0:  # tensor(0.1)
                return float(tensor)
            return tensor
        elif ty == "DeviceObjType":
            return node.s(attr_name)
        elif ty == "FunctionType":
            return None
        else:
            raise NotImplementedError("Unsupported type: %s" % ty)
    else:
        assert num_attributes == 0
        return None


def _get_operator_nodes(nodes):
    """ Returns torch IR nodes that need conversion to Relay """
    ops = []
    # Traverse nodes and add to graph
    for node in nodes:
        if node.outputsSize() > 1:
            node_name = "_".join(_get_output_names(node))
        else:
            node_name = _get_output_name(node)

        if node.kind() != "prim::GetAttr":
            ops.append((node_name, node))

    return ops


def _get_graph_input_names(graph):
    """ Get the graph input names (use after graph copy and run jit passes) """
    # Variable names could change the first time a copy is made and after
    # _run_jit_passes is called, expected that those functions already invoked
    ir_inputs = _get_input_names(graph)
    return ir_inputs[1:]  # remove self at the 0th arg


def _get_relay_input_vars(graph, input_shapes, prelude):
    """
    Return Relay vars from input shapes and create entries based on
    expected graph inputs - to allow translation
    """
    def get_relay_ty(ishape):
        if _is_int_seq(ishape) or len(ishape) == 0:
            return TensorType(ishape)
        elif isinstance(ishape, tuple):
            return TupleType([get_relay_ty(elem) for elem in ishape])
        elif isinstance(ishape, list):
            assert len(ishape) > 0
            elem_tys = [get_relay_ty(s) for s in ishape]
            msg = "List elements should have identical types"
            assert all(map(lambda ty: ty == elem_tys[0], elem_tys)), msg
            return prelude.l(elem_tys[0])
        raise NotImplementedError("unsupported input type")

    input_types = [(tup[0], get_relay_ty(tup[1])) for tup in input_shapes]
    input_vars = {}
    ir_inputs = _get_graph_input_names(graph)
    for ir_input, (name, itype) in zip(ir_inputs, input_types):
        inp = _expr.var(name, type_annotation=itype)
        # Translate from graph input to user input name
        input_vars[ir_input] = inp

    return input_vars


def _unpack_tuple(tup):
    def unpack(tup, num_fields):
        return [_expr.TupleGetItem(tup, i) for i in range(num_fields)]

    if isinstance(tup, _expr.Tuple):
        return unpack(tup, len(tup.fields))
    elif isinstance(tup.type_annotation, TupleType):
        return unpack(tup, len(tup.type_annotation.fields))
    # shouldn't happen
    assert False


def _get_free_vars_from_block(block):
    block_inp_names = _get_input_names(block)
    bound_names = block_inp_names
    free_vars = set()

    for node in block.nodes():
        inp_names = _get_input_names(node)
        list_diff = [name for name in inp_names if name not in bound_names]
        free_vars.update(list_diff)
        bound_names += _get_output_names(node)

    return free_vars


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = _get_users(current)

        if not users or terminate(users):
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in users])

    return inner(root_node, [root_node])


def get_attr_chains(root_getattr_node):
    """ Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """
    def terminate(users):
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]
        return len(next_attrs) == 0

    return get_use_chains(root_getattr_node, terminate)


def convert_params(graph, state_dict):
    """
    Return Relay vars and TVM NDArrays for input parameters
    A chain of prim::GetAttr nodes is processed one at a time
    """
    getattr_nodes = graph.findAllNodes("prim::GetAttr", recurse=True)
    params = {}
    param_tensors = {}
    packed_param_map = {}
    seen = set()

    for node in getattr_nodes:
        if _get_output_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(_get_output_name, getattrs))

            full_attr = _getattr_full_name(getattrs)
            full_attr_node_name = _get_output_name(getattrs[-1])

            if full_attr.endswith("_packed_params"):  # for quantized models
                err_msg = "parameter %s not found in state dict" % full_attr
                assert full_attr in state_dict, err_msg
                packed_param_map[full_attr_node_name] = full_attr
            elif full_attr in state_dict:
                torch_tensor = state_dict[full_attr]
                tensor, var = _get_tensor_and_var(torch_tensor,
                                                  full_attr_node_name)
                param_tensors[full_attr_node_name] = tensor
                params[full_attr_node_name] = var

    return params, param_tensors, packed_param_map


def convert_block(block, outputs, convert_map, prelude):
    """ Translate Torch "Block", used for prim::If and prim::Loop """
    ops = _get_operator_nodes(block.nodes())
    ret_names = _get_input_names(block.returnNode())
    return convert_operators(ops, outputs, ret_names, convert_map, prelude)


def convert_if(if_node, outputs, convert_map, prelude):
    """ Translate Torch prim::If to Relay If """
    cond = outputs[if_node.inputsAt(0).debugName()]
    blocks = list(if_node.blocks())
    true_branch = convert_block(blocks[0], outputs, convert_map, prelude)
    false_branch = convert_block(blocks[1], outputs, convert_map, prelude)
    assert len(true_branch) == 1 and len(false_branch) == 1
    return _expr.If(cond, true_branch[0], false_branch[0])


def convert_loop(loop_node, outputs, convert_map, prelude):
    """ Translate Torch prim::Loop to Relay while_loop """
    def get_input(index):
        ivalue = loop_node.inputsAt(index)
        inode = ivalue.node()
        if inode.kind() == "prim::Constant":
            return _expr.const(_get_constant(inode))
        var_name = ivalue.debugName()
        assert var_name in outputs
        return _wrap_const(outputs[var_name])

    # Refer to the spec for prim::Loop below
    # https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md#loops
    # The first input: %max_trip_count
    # The second input: %initial_condition
    # The rest of input: loop variables
    max_loop_count = get_input(0)
    init_cond = get_input(1)
    num_loop_var = len(list(loop_node.inputs())) - 2
    init_vals = [get_input(i + 2) for i in range(num_loop_var)]

    # while loop has always max_loop_count being int64 max
    # max_loop_count.data (tvm.runtime.NDArray) is -1, so _get_constant again
    is_while_loop = (isinstance(max_loop_count, _expr.Constant) and
                     _get_constant(loop_node.inputsAt(0).node()) == sys.maxsize)

    if is_while_loop:
        loop_iter_dtype = "bool"
        # while loop with non input dependent condition such as while i < 10:
        # init_cond is int, need to cast to bool to type check
        if isinstance(init_cond, _expr.Constant):
            init_cond = _op.cast(init_cond, "bool")
        init_loop_iter_val = init_cond
    else:
        loop_iter_dtype = "int32"
        # always count from 0
        init_loop_iter_val = _expr.const(0, dtype="int32")

    body_block = list(loop_node.blocks())[0]
    block_input_names = _get_input_names(body_block)
    num_block_inputs = len(block_input_names)
    name_val_pairs = list(zip(block_input_names,
                              [init_loop_iter_val] + init_vals))
    outputs.update(name_val_pairs)

    def get_var(name, val):
        if val:
            checked_type = _infer_type_with_prelude(val, prelude)
            return _expr.var(name, type_annotation=checked_type)
        return _expr.var(name)

    loop_iter_var = _expr.var(block_input_names[0], shape=(),
                              dtype=loop_iter_dtype)
    loop_vars = [get_var(name, val) for name, val in name_val_pairs[1:]]

    # Add non constant free variables to loop variables to prevent code blow up
    # Without this, if there are two for loops in a row, which often happens
    # if the outer loop is unrolled, the computation corresponding to the first for loop
    # is inlined inside loop body, turning O(N) + O(N) computation into O(N^2).
    # This issue was found when converting from Stacked LSTM test. Torch does not add the output
    # of the eariler loop into loop variables of the next loop.
    # So the variable corresponding to the first loop output appears free in the second loop body.
    free_vars = [var for var in _get_free_vars_from_block(body_block)
                 if var in outputs and not isinstance(outputs[var], (_expr.Constant, int, float))
                 and outputs[var]]

    prev_outputs = {}
    for name in free_vars:
        prev_output = outputs[name]
        new_loop_var = get_var(name, prev_output)
        prev_outputs[name] = prev_output
        outputs[name] = new_loop_var
        loop_vars.append(new_loop_var)
        init_vals.append(prev_output)

    def cond(*current_vals):
        i = current_vals[0]

        if is_while_loop:
            return _op.equal(i, _expr.const(True, 'bool'))

        return _op.less(i, max_loop_count)

    def body(*current_vals):
        # Update loop variables using the prev iteration outputs
        assert len(current_vals) == num_block_inputs + len(free_vars)

        for (i, val) in enumerate(current_vals):
            if i < num_block_inputs:
                outputs[block_input_names[i]] = val
            else:
                outputs[free_vars[i-num_block_inputs]] = val

        block_outputs = convert_block(body_block, outputs, convert_map, prelude)
        block_outputs += [outputs[name] for name in free_vars]

        if not is_while_loop:
            # iter var increment implicit in torch, so do it manually
            # for while loop, block_outputs[0] is already a boolean,
            # the result of termination check
            incr = _expr.const(1, dtype="int32")
            block_outputs[0] = current_vals[0] + incr

        return block_outputs

    loop = while_loop(cond, [loop_iter_var] + loop_vars, body)
    loop_val = loop(init_loop_iter_val, *init_vals)

    # restore original output values for free vars
    outputs.update(prev_outputs)

    # The first element is a loop counter or boolean condition, ignore it
    return [_expr.TupleGetItem(loop_val, i+1) for i in range(num_loop_var)]


def convert_operators(operators, outputs, ret_names, convert_map, prelude):
    """ Convert each Torch IR operators to Relay equivalent """
    for node_name, op_node in operators:
        operator = op_node.kind()
        inputs = _get_op_inputs(op_node, outputs)

        if operator == "prim::Constant":
            outputs[node_name] = _get_constant(op_node)
        elif operator == "prim::ListConstruct" and _is_int_seq(inputs):
            outputs[node_name] = _expr.var(node_name, shape=inputs)
        elif operator == "prim::ListConstruct" and _should_construct_dynamic_list(op_node):
            outputs[node_name] = _convert_to_list_adt(inputs, prelude)
        elif operator == "prim::ListConstruct":
            # This assumes that no more elements will be appended to this list
            # In this case, we keep the Python list
            outputs[node_name] = inputs
        elif operator == "prim::TupleConstruct":
            outputs[node_name] = _expr.Tuple(inputs)
        elif operator in ["prim::ListUnpack", "prim::TupleUnpack"]:
            assert len(inputs) == 1
            if isinstance(inputs[0], (list, _expr.TupleWrapper)):
                unpacked = inputs[0]
            else:
                unpacked = _unpack_tuple(inputs[0])
            outputs.update(zip(_get_output_names(op_node), unpacked))
        elif operator == "prim::If":
            if_out = convert_if(op_node, outputs, convert_map, prelude)
            outputs[node_name] = if_out
        elif operator == "prim::Loop":
            loop_out = convert_loop(op_node, outputs, convert_map, prelude)
            unpacked_names = _get_output_names(op_node)
            assert len(loop_out) == len(unpacked_names)
            outputs.update(zip(unpacked_names, loop_out))
        else:
            relay_op = convert_map[operator]
            relay_out = relay_op(inputs, _get_input_types(op_node))

            if isinstance(relay_out, tuple):
                # This is for torch operators that return multiple outputs
                # See _adaptive_max_2d above for example
                out_names = _get_output_names(op_node)
                outputs.update(zip(out_names, relay_out))
            else:
                outputs[node_name] = relay_out

    return [_wrap_const(outputs[ret_name])
            for ret_name in ret_names]


def get_all_op_names(graph):
    """ Return all operator names in the input graph """
    nodes = list(graph.nodes())
    prim_with_blocks = ["prim::If", "prim::Loop"]
    for prim in prim_with_blocks:
        prim_nodes = graph.findAllNodes(prim, recurse=True)
        for prim_node in prim_nodes:
            for block in prim_node.blocks():
                nodes += block.nodes()
    return set(node.kind() for node in nodes)


def from_pytorch(script_module, input_shapes, custom_convert_map=None):
    """ Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    input_shapes : List of tuples of input name and input dimensions
        Graph level input shape list
        The same input names need to be used for deployment, so choose easy to
        remember names (such as: input0, input1)

    custom_convert_map: Dictionary of str to Relay op
        A custom op conversion map in the same format as _convert_map above

    Returns
    -------
    mod : tvm.relay.Module
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    mod = tvm.IRModule()
    prelude = Prelude(mod)

    convert_map = _get_convert_map(prelude)

    graph = script_module.graph.copy()
    _run_jit_passes(graph)

    if custom_convert_map:
        convert_map.update(custom_convert_map)

    op_names = get_all_op_names(graph)
    _report_missing_conversion(op_names, convert_map)
    _check_inputs(graph, input_shapes)

    params = script_module.state_dict()
    outputs = _get_relay_input_vars(graph, input_shapes, prelude)
    param_vars, tensors, packed_param_map = convert_params(graph, params)
    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    outputs.update(param_vars)
    ret_name = _get_input_names(graph.return_node())

    # For quantized models
    if "aten::quantize_per_tensor" in op_names:
        weight_quant_params = qnn_torch.get_weight_quant_params(script_module)
        qnn_torch.add_input_quant_params_to_op_inputs(graph)
        qnn_torch.add_quant_params_to_outputs(outputs,
                                              packed_param_map,
                                              weight_quant_params)
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        convert_map.update(qnn_torch.convert_map)

    ret = convert_operators(_get_operator_nodes(graph.nodes()),
                            outputs, ret_name, convert_map, prelude)

    mod["main"] = tvm.relay.Function(_analysis.free_vars(ret[0]), ret[0])

    return mod, tvm_params
