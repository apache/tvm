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

import numpy as np

import tvm
from tvm.ir import module as _module

from .. import analysis as _analysis
from .. import expr as _expr
from .. import op as _op
from .common import get_relay_op
from .common import infer_shape as _infer_shape
from .common import infer_value as _infer_value

from . import qnn_torch

__all__ = ["from_pytorch"]

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

def _unsqueeze():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

        return _op.transform.expand_dims(data, int(axis), 1)
    return _impl

def _concatenate():
    def _impl(inputs, input_types):
        data = inputs[0]
        axis = inputs[1]

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

        begin = [0]*len(end)
        dim = int(inputs[1])
        begin[dim] = int(inputs[2])

        if isinstance(inputs[3], str) and inputs[3].isdigit():
            end[dim] = min(end[dim], int(inputs[3]))
        else:
            end[dim] = inputs[3]

        strides.append(int(inputs[4]))
        return _op.transform.strided_slice(data, begin, end, strides)
    return _impl

def _select():
    def _impl(inputs, input_types):
        data = inputs[0]
        dim = int(inputs[1])
        index = int(inputs[2])

        return _op.transform.take(data, _expr.const(index, dtype="int32"), axis=dim)
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
            assert "data type {} could not be parsed in ones op" % (type(data))

        return _op.full(_expr.const(1), shape, dtype=_convert_data_type(input_types[0]))
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
            assert "data type {} could not be parsed in zeros op" % (type(data))

        return _op.full(_expr.const(0), shape, dtype=_convert_data_type(input_types[0]))
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

def _adaptive_avg_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        def func(x):
            return _op.nn.adaptive_avg_pool2d(x, output_size=output_size)

        if input_types[0] == "quint8":
            return qnn_torch.quantized_adaptive_avg_2d(data, func)

        return func(data)

    return _impl

def _adaptive_max_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        return _op.nn.adaptive_max_pool2d(
            data,
            output_size=output_size)
    return _impl

def _maxpool_2d():
    def _impl(inputs, input_types):
        data = inputs[0]

        pool_size = _infer_shape(inputs[1])
        strides = _infer_shape(inputs[2])
        padding = _infer_shape(inputs[3])

        ceil_mode = int(inputs[5])

        return _op.nn.max_pool2d(data, pool_size, strides, padding, "NCHW", ceil_mode)
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
        use_transpose = True if inputs[6] == "1" else False

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
            assert "data type {} could not be parsed in conv op" % (type(weight))

        # TODO: Add reshape when channel multiplier > 1. Pending PR #4644
        channels = weight_shape[0]
        groups = int(inputs[8])

        if groups > 1:
            # in torch, groups == in_channels for depth wise conv
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

        if use_transpose:
            conv_out = _op.nn.conv2d_transpose(data,
                                               weight,
                                               strides=strides,
                                               padding=padding,
                                               dilation=dilation,
                                               groups=groups,
                                               channels=channels,
                                               kernel_size=kernel_size,
                                               data_layout="NCHW",
                                               kernel_layout="OIHW",
                                               out_layout="",
                                               out_dtype="")
        else:
            conv_out = _op.nn.conv2d(data,
                                     weight,
                                     strides=strides,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     channels=channels,
                                     kernel_size=kernel_size,
                                     data_layout="NCHW",
                                     kernel_layout="OIHW",
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

def _transpose():
    def _impl(inputs, input_types):
        data = inputs[0]

        import torch
        if isinstance(data, _expr.Expr):
            ndims = len(_infer_shape(data))
        elif isinstance(data, list):
            ndims = data
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            ndims = data.shape
        else:
            assert "data type {} could not be parsed in transpose op" % (type(data))

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

def _size():
    def _impl(inputs, input_types):
        shape = _infer_shape(inputs[0])
        if len(inputs) > 1:
            axis = int(inputs[1])
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

        return _op.nn.avg_pool2d(data,
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
    def _impl(inputs, attrs, params):
        data = inputs[0]
        return get_relay_op(name)(data)
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

def _chunk():
    def _impl(inputs, input_types):
        data = inputs[0]

        num_chunks = int(inputs[1])
        axis = int(inputs[2])

        if isinstance(data, _expr.Expr):
            inferred_shape = _infer_shape(data)

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

def _sqrt():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.tensor.sqrt(data)
    return _impl

def _floor():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.floor(data)
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


# Helper functions for operator implementation

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

# Operator mappings

_convert_map = {
    "aten::device"                          : _none(),
    "aten::add"                             : _elemwise("add"),
    "aten::add_"                            : _elemwise("add"),
    "aten::sub"                             : _elemwise("subtract"),
    "aten::sub_"                            : _elemwise("subtract"),
    "aten::max"                             : _elemwise("maximum"),
    "aten::min"                             : _elemwise("minimum"),
    "aten::mul"                             : _elemwise("multiply"),
    "aten::mul_"                            : _elemwise("multiply"),
    "aten::pow"                             : _elemwise("power"),
    "aten::div"                             : _elemwise("divide"),
    "aten::div_"                            : _elemwise("divide"),
    "aten::ones"                            : _ones(),
    "aten::zeros"                           : _zeros(),
    "aten::to"                              : _to(),
    "aten::unsqueeze"                       : _unsqueeze(),
    "aten::cat"                             : _concatenate(),
    "aten::slice"                           : _slice(),
    "aten::select"                          : _select(),
    "aten::relu"                            : _relu(),
    "aten::relu_"                           : _relu(),
    "aten::adaptive_avg_pool2d"             : _adaptive_avg_2d(),
    "aten::adaptive_max_pool2d"             : _adaptive_max_2d(),
    "aten::max_pool2d"                      : _maxpool_2d(),
    "aten::max_pool2d_with_indices"         : _maxpool_2d(),
    "aten::hardtanh"                        : _hardtanh(),
    "aten::hardtanh_"                       : _hardtanh(),
    "aten::_convolution"                    : _convolution(),
    "aten::softmax"                         : _softmax(),
    "aten::threshold"                       : _threshold(),
    "aten::threshold_"                      : _threshold(),
    "aten::contiguous"                      : _contiguous(),
    "aten::batch_norm"                      : _batch_norm(),
    "aten::transpose"                       : _transpose(),
    "aten::transpose_"                      : _transpose(),
    "aten::t"                               : _transpose(),
    "aten::flatten"                         : _flatten(),
    "aten::addmm"                           : _dense(),
    "aten::size"                            : _size(),
    "aten::view"                            : _view(),
    "aten::clone"                           : _clone(),
    "aten::log_softmax"                     : _log_softmax(),
    "aten::sigmoid"                         : _sigmoid(),
    "aten::avg_pool2d"                      : _avg_pool2d(),
    "aten::dropout"                         : _dropout(),
    "aten::dropout_"                        : _dropout(),
    "aten::mean"                            : _mean(),
    "aten::chunk"                           : _chunk(),
    "aten::matmul"                          : _matmul(),
    "aten::expand"                          : _expand(),
    "aten::Int"                             : _int(),
    "prim::NumToTensor"                     : _numtotensor(),
    "prim::ListUnpack"                      : _identity(),
    "aten::constant_pad_nd"                 : _pad(),
    "aten::permute"                         : _transpose(),
    "aten::sum"                             : _reduce("sum"),
    "aten::prod"                            : _reduce("prod"),
    "aten::sqrt"                            : _sqrt(),
    'aten::floor'                           : _floor(),
    "aten::detach"                          : _identity(),
    "aten::upsample_bilinear2d"             : _upsample("bilinear"),
    "aten::upsample_nearest2d"              : _upsample("nearest_neighbor"),
    "aten::expand_as"                       : _expand_as()
}


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


def _get_op_inputs(op_node, outputs, output_index_map):
    input_names = [output_index_map[name]
                   for name in _get_input_names(op_node)]
    return [outputs[name] for name in input_names]


def _update_outputs_from_pairs(name_output_pairs, outputs, output_index_map):
    for output_name, output in name_output_pairs:
        output_index_map[output_name] = len(outputs)
        outputs.append(output)


def _report_missing_conversion(op_names):
    """ Check if all ops in an input graph are supported by TVM """
    known_ops = ["prim::Constant", "prim::GetAttr",
                 "prim::ListConstruct", "prim::ListUnpack",
                 "prim::TupleConstruct", "prim::TupleUnpack"]
    known_ops += list(_convert_map.keys())
    known_ops += list(qnn_torch.convert_map.keys())

    missing = [op_name for op_name in op_names
               if op_name not in known_ops]

    if missing:
        msg = "The following operators are not implemented: {}".format(missing)
        raise NotImplementedError(msg)

def _check_input_names(script_module, input_shapes):
    """ Check the graph inputs match the inputs """
    ir_inputs = get_graph_input_names(script_module)

    for ir_input in ir_inputs:
        if ir_input not in input_shapes:
            msg = "Missing graph input {} in input_shapes".format(ir_input)
            raise RuntimeError(msg)

    for input_name in input_shapes:
        if input_name not in ir_inputs:
            msg = "Unused graph input {} in input_shapes".format(input_name)
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
                input_list_types.append(None)
            else:
                input_list_types.append(in_ty.scalarType().lower())
        elif input_node_kind == 'ListType':
            input_list_types.append(str(in_ty.getElementType()).lower())
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


def parse_inputs(graph_inputs, input_shapes):
    """ Return Relay vars from torch input vars """
    ir_inputs = list(graph_inputs)
    input_vars = {}

    for input_name, ir_input in zip(input_shapes, ir_inputs[1:]):
        input_vars[input_name] = _expr.var(input_name,
                                           shape=input_shapes[input_name])
    return input_vars


def get_use_chains(root_node, terminate=lambda _: False):
    """
    Track a chain of users of this node forward, returning a list of chains
    See get_attr_chains below for its usage
    """
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = []
        for output in current.outputs():
            users += [use.user for use in output.uses()]

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


def parse_params(graph, state_dict):
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


def parse_operators(operators, outputs, output_index_map, ret_name):
    """ Convert each Torch IR operators to Relay equivalent """
    for node_name, op_node in operators:
        operator = op_node.kind()
        inputs = _get_op_inputs(op_node, outputs, output_index_map)

        if operator == "prim::Constant":
            output_index_map[node_name] = len(outputs)
            outputs.append(_get_constant(op_node))
        elif operator == 'prim::ListConstruct' and _is_int_seq(inputs):
            output_index_map[node_name] = len(outputs)
            outputs.append(_expr.var(node_name, shape=inputs))
        elif operator in ['prim::ListConstruct', 'prim::TupleConstruct']:
            output_index_map[node_name] = len(outputs)
            outputs.append(inputs)
        elif operator in ["prim::ListUnpack", 'prim::TupleUnpack']:
            assert len(inputs) == 1
            unpacked_names = _get_output_names(op_node)
            _update_outputs_from_pairs(zip(unpacked_names, inputs[0]),
                                       outputs, output_index_map)
        else:
            output_index_map[node_name] = len(outputs)
            relay_op = _convert_map[operator]
            outputs.append(relay_op(inputs, _get_input_types(op_node)))

    return outputs[output_index_map[ret_name]]


def get_all_op_names(graph):
    """ Return all operator names in the input graph """
    return set(node.kind() for node in graph.nodes())


def get_graph_input_names(script_module):
    """ Use this function to set the keys for input_shapes"""
    # It seems variable names could change the first time a copy is made
    # Use the copy of the graph here to prevent troubles later
    ir_inputs = _get_input_names(script_module.graph.copy())
    return ir_inputs[1:]  # remove self at the 0th arg


def from_pytorch(script_module, input_shapes, custom_convert_map=None):
    """ Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    input_shapes : Dictionary of input dimensions
        Graph level input shape dictionary
        The keys should be the same one returned by get_graph_input_names(...) above

    custom_convert_map: Dictionary of str to Relay op
        A custom op conversion map in the same format as _convert_map above

    Returns
    -------
    mod : tvm.relay.Module
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime.NDArray
        Dict of converted parameters stored in tvm.runtime.ndarray format
    """
    graph = script_module.graph.copy()
    _run_jit_passes(graph)

    if custom_convert_map:
        _convert_map.update(custom_convert_map)

    op_names = get_all_op_names(graph)
    _report_missing_conversion(op_names)
    _check_input_names(script_module, input_shapes)

    params = script_module.state_dict()
    input_vars = parse_inputs(graph.inputs(), input_shapes)
    param_vars, tensors, packed_param_map = parse_params(graph, params)
    tvm_params = {k: tvm.nd.array(v) for k, v in tensors.items()}

    input_vars.update(param_vars)
    outputs = list(input_vars.values())
    output_index_map = dict(zip(input_vars.keys(), range(len(outputs))))
    ret_name = _get_input_names(graph.return_node())[0]

    # For quantized models
    if "aten::quantize_per_tensor" in op_names:
        weight_quant_params = qnn_torch.get_weight_quant_params(script_module)
        qnn_torch.add_input_quant_params_to_op_inputs(graph)
        qnn_torch.add_quant_params_to_outputs(outputs, output_index_map,
                                              packed_param_map,
                                              weight_quant_params)
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        _convert_map.update(qnn_torch.convert_map)

    body = parse_operators(_get_operator_nodes(graph.nodes()), outputs,
                           output_index_map, ret_name)
    func = tvm.relay.Function(_analysis.free_vars(body), body)

    return _module.IRModule.from_expr(func), tvm_params
