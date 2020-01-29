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
"""PT: PyTorch frontend."""
import numpy as np

import torch
import tvm

from .. import analysis as _analysis
from .. import expr as _expr
from .. import module as _module
from .. import op as _op
from .common import get_relay_op
from .common import infer_shape as _infer_shape

__all__ = ['from_pytorch']

# operator implementation
def _elemwise(name):
    def _impl(inputs, input_types):
        data0 = _convert_elemwise_input(inputs[0])
        data1 = _convert_elemwise_input(inputs[1])

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

        if isinstance(data, (_expr.Call, _expr.TupleGetItem, _expr.Var)):
            data = [data]

        return _op.tensor.concatenate(data, int(axis))
    return _impl

def _slice():
    def _impl(inputs, input_types):
        data = inputs[0]
        strides = []

        inferred_shape = _infer_shape(data)
        end = []
        for infer in inferred_shape:
            end.append(int(infer))
        if isinstance(data, _expr.Var):
            end = _infer_shape(data)
            end = list(end)

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
        inferred_shape = _infer_shape(data)
        end = []

        for infer in inferred_shape:
            end.append(int(infer))

        begin = [0]*len(end)
        dim = int(inputs[1])
        index = int(inputs[2])

        end[dim] = index+1
        begin[dim] = index

        strides = [1]*len(end)

        sym = _op.transform.strided_slice(data, begin, end, strides)
        axis = [dim]

        return _op.transform.squeeze(sym, axis)
    return _impl

def _convert_data_type(input_type):
    if input_type == 'double' or input_type == 'torch.float64':
        return 'float64'
    elif input_type == 'float' or input_type == 'torch.float32':
        return 'float32'
    elif input_type == 'half' or input_type == 'torch.float16':
        return 'float16'
    elif input_type == 'long' or input_type == 'torch.int64':
        return 'int64'
    elif input_type == 'int' or input_type == 'torch.int32':
        return 'int32'
    elif input_type == 'short' or input_type == 'torch.int16':
        return 'int16'
    elif input_type == 'char' or input_type == 'torch.int8':
        return 'int8'
    elif input_type == 'byte' or input_type == 'torch.uint8':
        return 'uint8'
    else:
        return input_type

def _ones():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Var):
            shape = _infer_shape(inputs[0])
        elif isinstance(inputs[0], (_expr.Call, _expr.TupleGetItem)):
            shape = _infer_shape(inputs[0])
        else:
            shape = inputs[0].shape

        return _op.full(_expr.const(1), shape, dtype=_convert_data_type(input_types[0]))
    return _impl

def _zeros():
    def _impl(inputs, input_types):
        if isinstance(inputs[0], _expr.Var):
            shape = _infer_shape(inputs[0])
        elif isinstance(inputs[0], (_expr.Call, _expr.TupleGetItem)):
            shape = _infer_shape(inputs[0])
        else:
            shape = inputs[0].shape

        return _op.full(_expr.const(0), shape, dtype=_convert_data_type(input_types[0]))
    return _impl

def _relu():
    def _impl(inputs, input_types):
        data = inputs[0]
        return _op.nn.relu(data)
    return _impl

def _adaptive_avg_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        return _op.contrib.contrib.adaptive_avg_pool2d(
            data,
            output_size=output_size)
    return _impl

def _adaptive_max_2d():
    def _impl(inputs, input_types):
        data = inputs[0]
        output_size = _infer_shape(inputs[1])

        return _op.contrib.contrib.adaptive_max_pool2d(
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
        use_transpose = False
        if inputs[6] == '1':
            use_transpose = True

        use_bias = False
        if isinstance(inputs[2], _expr.Var):
            use_bias = True

            data = inputs[0]
            weight = inputs[1]
            bias = inputs[2]

            if isinstance(weight, (_expr.Call, _expr.Var, _expr.TupleGetItem)):
                inferred_shape = _infer_shape(weight)
                weight_shape = []
                for infer in inferred_shape:
                    weight_shape.append(infer)
            else:
                weight_shape = weight.shape
            channels = weight_shape[0]

            strides = inputs[3]
            padding = inputs[4]
            dilation = inputs[5]

            kernel_size = weight_shape[2:]

        else:
            data = inputs[0]
            weight = inputs[1]
            bias = inputs[2]

            if isinstance(weight, (_expr.Call, _expr.Var, _expr.TupleGetItem)):
                inferred_shape = _infer_shape(weight)
                weight_shape = []
                for infer in inferred_shape:
                    weight_shape.append(infer)
            else:
                weight_shape = weight.shape
            channels = weight_shape[0]

            strides = inputs[3]
            padding = inputs[4]
            dilation = inputs[5]

            kernel_size = weight_shape[2:]

        if isinstance(strides, _expr.Var):
            strides = _infer_shape(strides)

        if isinstance(padding, _expr.Var):
            padding = _infer_shape(padding)

        if isinstance(dilation, _expr.Var):
            dilation = _infer_shape(dilation)

        groups = int(inputs[8])

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

        if isinstance(inputs[1], _expr.Var) and isinstance(inputs[2], _expr.Var):
            scale = center = True
            weight = inputs[1]
            beta = inputs[2]
        else:
            scale = center = False

        if scale:
            gamma = weight
        else:
            if data_type == 'double':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float64'))
            elif data_type == 'float':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float32'))
            elif data_type == 'half':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float16'))
            elif data_type == 'long':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int64'))
            elif data_type == 'int':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int32'))
            elif data_type == 'short':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int16'))
            elif data_type == 'char':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('int8'))
            elif data_type == 'byte':
                gamma = _expr.const(np.ones([int(channels[1])]).astype('uint8'))
            else:
                gamma = _expr.const(np.ones([int(channels[1])]).astype('float32'))

        if center:
            beta = beta
        else:
            if data_type == 'double':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float64'))
            elif data_type == 'float':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float32'))
            elif data_type == 'half':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float16'))
            elif data_type == 'long':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int64'))
            elif data_type == 'int':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int32'))
            elif data_type == 'short':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int16'))
            elif data_type == 'char':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('int8'))
            elif data_type == 'byte':
                beta = _expr.const(np.zeros([int(channels[1])]).astype('uint8'))
            else:
                beta = _expr.const(np.zeros([int(channels[1])]).astype('float32'))

        moving_mean = inputs[3]
        moving_var = inputs[4]
        epsilon = float(inputs[7])

        center = center
        scale = scale

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

        if isinstance(data, _expr.Var):
            ndims = len(_infer_shape(data))
        elif isinstance(data, (_expr.Call, _expr.TupleGetItem)):
            ndims = _infer_shape(data)
        else:
            ndims = data.shape

        if isinstance(data, tvm.ndarray.NDArray):
            ndims = len(data.shape)
        axes = list(range(ndims))

        num_inputs = len(inputs)

        if num_inputs == 1:
            if ndims >= 2:
                axes[-1] = ndims - 2
                axes[-2] = ndims - 1
            if not isinstance(data, _expr.Var):
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
        use_bias = False

        if isinstance(inputs[0], _expr.Var):
            use_bias = True

        data = inputs[1]
        data_type = input_types[1]
        weight = inputs[2]

        beta = inputs[3]
        alpha = inputs[4]

        if not isinstance(alpha, (_expr.Var, _expr.Call, _expr.TupleGetItem)):
            if data_type == 'double':
                alpha = _expr.const(np.float64(alpha), dtype='float64')
            elif data_type == 'float':
                alpha = _expr.const(np.float32(alpha), dtype='float32')
            elif data_type == 'half':
                alpha = _expr.const(np.float16(alpha), dtype='float16')
            elif data_type == 'long':
                alpha = _expr.const(np.int64(alpha), dtype='int64')
            elif data_type == 'int':
                alpha = _expr.const(np.int32(alpha), dtype='int32')
            elif data_type == 'short':
                alpha = _expr.const(np.int16(alpha), dtype='int16')
            elif data_type == 'char':
                alpha = _expr.const(np.int8(alpha), dtype='int8')
            elif data_type == 'byte':
                alpha = _expr.const(np.uint8(alpha), dtype='uint8')
            else:
                alpha = _expr.const(np.float32(alpha), dtype='float32')
            data *= alpha

        if not isinstance(beta, (_expr.Var, _expr.Call, _expr.TupleGetItem)):
            if data_type == 'double':
                beta = _expr.const(np.float64(beta), dtype='float64')
            elif data_type == 'float':
                beta = _expr.const(np.float32(beta), dtype='float32')
            elif data_type == 'half':
                beta = _expr.const(np.float16(beta), dtype='float16')
            elif data_type == 'long':
                beta = _expr.const(np.int64(beta), dtype='int64')
            elif data_type == 'int':
                beta = _expr.const(np.int32(beta), dtype='int32')
            elif data_type == 'short':
                beta = _expr.const(np.int16(beta), dtype='int16')
            elif data_type == 'char':
                beta = _expr.const(np.int8(beta), dtype='int8')
            elif data_type == 'byte':
                beta = _expr.const(np.uint8(beta), dtype='uint8')
            else:
                beta = _expr.const(np.float32(beta), dtype='float32')
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
        axis = int(inputs[1])
        shape = _infer_shape(inputs[0])
        return shape[axis]
    return _impl

def _numtotensor():
    def _impl(inputs, input_types):
        val = inputs[0]
        dtype = type(val)

        if isinstance(val, tvm.expr.IntImm):
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
        strides = _infer_shape(inputs[2])
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
        axis = _infer_shape(inputs[1])

        keepdims = int(inputs[2])
        exclude = int(inputs[3])

        return _op.mean(data, axis, keepdims, exclude)
    return _impl

def _chunk():
    def _impl(inputs, input_types):
        data = inputs[0]

        num_chunks = int(inputs[1])
        axis = int(inputs[2])

        if isinstance(data, _expr.Var):
            inferred_shape = _infer_shape(data)
        elif isinstance(data, (_expr.Call, _expr.TupleGetItem)):
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
        if isinstance(data_in, _expr.Var):
            shape = _infer_shape(data_in)
        elif isinstance(data_in, (_expr.Call, _expr.TupleGetItem)):
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
        if isinstance(inputs[0], _expr.Call):
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

# Helper functions for operator implementation

# TODO: Fix typing
def _convert_elemwise_input(data):
    if isinstance(data, torch.Tensor):
        return _expr.const(data.item(), dtype='float32')
    elif not isinstance(data, (_expr.Call, _expr.TupleGetItem, _expr.Var)):
        return _expr.const(int(data), dtype='float32')
    else:
        return data

# Operator mappings

_convert_map = {
    'aten::device'                          : _none(),
    'aten::add'                             : _elemwise('add'),
    'aten::add_'                            : _elemwise('add'),
    'aten::sub'                             : _elemwise('subtract'),
    'aten::sub_'                            : _elemwise('subtract'),
    'aten::max'                             : _elemwise('maximum'),
    'aten::min'                             : _elemwise('minimum'),
    'aten::mul'                             : _elemwise('multiply'),
    'aten::mul_'                            : _elemwise('multiply'),
    'aten::pow'                             : _elemwise('power'),
    'aten::div'                             : _elemwise('divide'),
    'aten::div_'                            : _elemwise('divide'),
    'aten::ones'                            : _ones(),
    'aten::zeros'                           : _zeros(),
    'aten::to'                              : _identity(),
    'aten::unsqueeze'                       : _unsqueeze(),
    'aten::cat'                             : _concatenate(),
    'aten::slice'                           : _slice(),
    'aten::select'                          : _select(),
    'aten::relu'                            : _relu(),
    'aten::relu_'                           : _relu(),
    'aten::adaptive_avg_pool2d'             : _adaptive_avg_2d(),
    'aten::adaptive_max_pool2d'             : _adaptive_max_2d(),
    'aten::max_pool2d'                      : _maxpool_2d(),
    'aten::max_pool2d_with_indices'         : _maxpool_2d(),
    'aten::hardtanh'                        : _hardtanh(),
    'aten::hardtanh_'                       : _hardtanh(),
    'aten::_convolution'                    : _convolution(),
    'aten::softmax'                         : _softmax(),
    'aten::threshold'                       : _threshold(),
    'aten::threshold_'                      : _threshold(),
    'aten::contiguous'                      : _contiguous(),
    'aten::batch_norm'                      : _batch_norm(),
    'aten::transpose'                       : _transpose(),
    'aten::transpose_'                      : _transpose(),
    'aten::t'                               : _transpose(),
    'aten::flatten'                         : _flatten(),
    'aten::addmm'                           : _dense(),
    'aten::size'                            : _size(),
    'aten::view'                            : _view(),
    'aten::clone'                           : _clone(),
    'aten::log_softmax'                     : _log_softmax(),
    'aten::sigmoid'                         : _sigmoid(),
    'aten::avg_pool2d'                      : _avg_pool2d(),
    'aten::dropout'                         : _dropout(),
    'aten::dropout_'                        : _dropout(),
    'aten::mean'                            : _mean(),
    'aten::chunk'                           : _chunk(),
    'aten::matmul'                          : _matmul(),
    'aten::expand'                          : _expand(),
    'aten::Int'                             : _int(),
    'prim::NumToTensor'                     : _numtotensor(),
    'prim::ListUnpack'                      : _identity(),
    'aten::constant_pad_nd'                 : _pad(),
    'aten::permute'                         : _transpose(),
    'aten::sum'                             : _reduce('sum'),
    'aten::prod'                            : _reduce('prod'),
    'aten::sqrt'                            : _sqrt()
}

# Internal graph for parsing

class Graph(object):
    """ A helper class for parsing PyTorch model to Relay graph."""

    def __init__(self, script_module, input_shapes):

        self._script_module = script_module
        self._graph = script_module.graph.copy()

        # TODO: Temporary fix to remove prim::CallMethod node introduced in PT 1.4
        torch._C._jit_pass_inline(self._graph)

        self._inputs_r = {}
        self._params = {}
        self._param_tensors = {}
        self._consts = {}
        self._ops = {}
        self._op_inputs_r = {}
        self._op_inputs_types = {}
        self._input_shapes = input_shapes if input_shapes else {}
        self._nid_to_node_name = {}

    def from_pytorch(self):
        """ Construct relay nodes from PyTorch graph

        Currently only supports traced PyTorch format which means no control flow.
        User must perform torch.jit.trace on a model and pass this in.
        Future support should include support scripted models (torch.jit.script) which
        preserves control flow.

        Returns
        -------
        mod : tvm.relay.Module
            The module that optimizations will be performed on.

        params : dict of str to tvm.ndarray
            Dict of converted parameters stored in tvm.ndarray format
        """
        # Check for missing ops
        missing_operators = self._parse_import_prerequisites()

        if missing_operators:
            raise NotImplementedError( \
                "The following operators are not implemented: {}".format(missing_operators))

        # Translate PyTorch graph to by decorating Graph with state dict and inputs into each op
        self._parse_inputs()
        self._parse_params()
        self._parse_ops()

        outputs = []
        nid = 0

        for op_name, op_node in self._ops.items():
            if op_node.kind() == 'prim::ListConstruct':
                if any(inp.debugName() in self._nid_to_node_name.keys() \
                       for inp in op_node.inputs()):
                    listconstr = []
                    for i in op_node.inputs():
                        if i.debugName() in self._nid_to_node_name.keys():
                            listconstr.append( \
                                outputs[self._nid_to_node_name[i.debugName()]])
                        elif i.node().kind() == 'prim::Constant':
                            listconstr.append(int(self._consts[i.debugName()]))
                        elif i.debugName() in self._inputs_r.keys():
                            listconstr.append(int(self._inputs_r[i.debugName()]))

                    # Unwrap for tensors
                    if len(listconstr) == 1:
                        listconstr = listconstr[0]

                    outputs.append(listconstr)
                    self._nid_to_node_name[op_name] = nid
                    nid = nid+1
            elif op_node.kind() != "prim::Constant":
                for i in op_node.inputs():
                    if i.debugName() in self._nid_to_node_name.keys():
                        for cnt in range(0, len(self._op_inputs_r[op_name])):
                            if isinstance(self._op_inputs_r[op_name][cnt], str):
                                if "call/var" in self._op_inputs_r[op_name][cnt]:
                                    self._op_inputs_r[op_name][cnt] = \
                                        outputs[self._nid_to_node_name[i.debugName()]]
                                    break

                call = _convert_map[op_node.kind()](self._op_inputs_r[op_name],
                                              self._op_inputs_types[op_name])

                outputs.append(call)
                self._nid_to_node_name[op_name] = nid
                nid = nid+1

        func = tvm.relay.Function(_analysis.free_vars(outputs[-1]), outputs[-1])

        param = {k: tvm.nd.array(v) for k, v in self._param_tensors.items()}

        return  _module.Module.from_expr(func), param

    def _parse_inputs(self):
        """ Map inputs to parser and inputs to graph. """
        # Get names and objects of inputs for IR
        ir_inputs = [i for i in self._graph.inputs()]

        # Create corresponding shape and add to input
        for input_name, ir_input in zip(self._input_shapes, ir_inputs[1:]):
            input_shape = self._input_shapes[input_name]
            ir_input.setDebugName(input_name)

            ir_dtype = _convert_data_type(ir_input.type().scalarType().lower())
            self._inputs_r[input_name] = _expr.var(input_name,
                                                   shape=self._input_shapes[input_name],
                                                   dtype=ir_dtype)

        # Add self (first input of a PyTorch graph) to inputs
        input_shape = [3]
        tensor = tvm.nd.array(np.zeros(input_shape).astype(np.float32))
        input_name = ir_inputs[0].debugName()
        self._inputs_r[input_name] = tensor

    def _parse_params(self):
        """ Map state dictionary values to corresponding prim::GetAttr op node. """
        # Grab weights, biases, etc. from graph
        state_dict = self._script_module.state_dict()
        param_names = []
        for key, value in state_dict.items():
            param_str = str(key)
            param_name = param_str.split('.')[-1]
            param_names.append(param_name)

        # Get names of all inputs
        input_names = [i for i in self._inputs_r.keys()]

        # Iterate through graph for getAttr nodes and match full state_dict name to nodes
        node_weight_map = {}
        for node in self._graph.nodes():
            if node.kind() == "prim::GetAttr":

                attribute_names = node.attributeNames()
                assert(len(attribute_names) == 1)
                node_getattr_name = node.s(attribute_names[0])
                node_arg = node.input().debugName()

                if node.outputsSize() == 1:
                    node_name = node.output().debugName()
                else:
                    node_name = [output.debugName() for output in node.outputs()][0]

                if node_arg in input_names:
                    node_weight_map[node_name] = node_getattr_name
                else:
                    previous_map = node_weight_map[node_arg[:]]
                    node_weight_map[node_name] = previous_map+"."+node_getattr_name

                if node_getattr_name in param_names:

                    value = state_dict[node_weight_map[node_name]]
                    tensor = tvm.nd.array(value.cpu().numpy())
                    shape = tensor.shape
                    self._param_tensors[node_name] = tensor

                    self._params[node_name] = _expr.var(node_name,
                                                        shape=shape,
                                                        dtype=_convert_data_type(str(value.dtype)))

    def _parse_ops(self):
        """ Iterate through nodes and decorate graph with constants, operators,
        and the inputs to each operator. """
        # Traverse nodes and add to graph
        for node in self._graph.nodes():

            if node.outputsSize() == 1:
                node_name = node.output().debugName()
            else:
                node_name = [output.debugName() for output in node.outputs()][0]

            if node.kind() == "prim::Constant":
                if node.hasAttributes():
                    attribute_names = node.attributeNames()
                    attr_name = attribute_names[0]
                    ty = node.output().type().kind()

                    if ty == "IntType" or ty == "BoolType":
                        self._consts[node_name] = node.i(attr_name)
                    elif ty == "FloatType" or ty == "LongType":
                        self._consts[node_name] = node.f(attr_name)
                    elif ty == "TensorType":
                        self._consts[node_name] = node.output().toIValue()
                    else:
                        self._consts[node_name] = '0'
                else:
                    self._consts[node_name] = '0'
            elif node.kind() == "prim::ListConstruct":
                list_shape = []
                for input_node in node.inputs():
                    if input_node.debugName() in self._inputs_r.keys():
                        c = self._inputs_r[input_node.debugName()]
                        assert(isinstance(c, int))
                        list_shape.append(c)
                    elif input_node.debugName() in self._consts.keys():
                        c = self._consts[input_node.debugName()]
                        assert(isinstance(c, int))
                        list_shape.append(c)
                self._inputs_r[node_name] = _expr.var(node_name, shape=list_shape)

            if node.kind() != "prim::GetAttr":
                self._add_op(node_name, node)

    # Graph Helper Functions

    def _add_op(self, node_id, op_node):
        """ Add an operator and its operators inputs to the graph and insert placeholders
            where an input is a call node.

        Parameters
        ----------
        node_id : string
            The ID of the op node

        op_node : PyTorch Node object
            The full Node object for the op node

        """
        self._ops[(node_id)] = op_node
        input_list_r = []
        input_list_types = []
        for input_node in op_node.inputs():
            if input_node.debugName() in self._inputs_r.keys():
                input_list_r.append(self._inputs_r[input_node.debugName()])
            elif input_node.debugName() in self._params.keys():
                input_list_r.append(self._params[input_node.debugName()])
            elif input_node.node().kind() == "prim::Constant":
                input_list_r.append(self._consts[input_node.debugName()])
            else:
                input_list_r.append("call/var."+input_node.debugName())

                # If the inputs of a ListConstruct op is a call or var, remove it from inputs
                if op_node.kind() == 'prim::ListConstruct':
                    if node_id in self._inputs_r.keys():
                        self._inputs_r.pop(node_id)

            try:
                input_node_kind = input_node.type().kind()
                if input_node_kind == 'TensorType':
                    if input_node.type().scalarType() is None:
                        input_list_types.append('float')
                    else:
                        input_list_types.append(input_node.type().scalarType().lower())
                elif input_node_kind == 'ListType':
                    input_list_types.append(str(input_node.type().getElementType()).lower())
                elif input_node_kind == 'IntType' or input_node_kind == 'FloatType' or \
                        input_node_kind == 'BoolType' or input_node_kind == 'StringType' or \
                        input_node_kind == 'OptionalType':
                    input_list_types.append(str(input_node.type()).lower())
                else:
                    input_list_types.append('UnsupportedType')
                    print('UnsupportedType '++str(input_node.type())+' and '+str(input_node_kind))
            except Exception as e:
                print('Internal PyTorch error. Failed to grab type.')

        node_str = str(op_node)
        node_assign = (node_str.split(' = ')[0]).split(' : ')
        node_type = node_assign[1]

        if op_node.kind() == 'aten::ones':
            node_type = node_type.split('(')[0]
            input_list_types[0] = node_type.lower()

        if op_node.kind() == 'aten::zeros':
            node_type = node_type.split('(')[0]
            input_list_types[0] = node_type.lower()

        self._op_inputs_r[node_id] = input_list_r
        self._op_inputs_types[node_id] = input_list_types


    def _parse_import_prerequisites(self):
        """ Calculate the named preconditions from PyTorch graph.

        Returns
        -------
        missing_operators : set object
            Set of operator names which don't have their mapping in TVM,
            i.e. which are not supported

        """
        missing_operators = set()
        for node in self._graph.nodes():
            if node.kind() == "prim::Constant" or node.kind() == 'prim::ListConstruct' or \
                    node.kind() == 'prim::GetAttr':
                pass
            else:
                if any([node.kind() in _convert_map]):
                    pass
                else:
                    missing_operators.add(node.kind())

        return missing_operators

def from_pytorch(script_module, input_shapes):
    """ Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input)

    shape : Dictionary of input dimensions
        Graph level input shape dictionary

    Returns
    -------
    mod : tvm.relay.Module
        The module that optimizations will be performed on.

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = Graph(script_module, input_shapes)
    mod, params = g.from_pytorch()
    return mod, params
