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
import numpy as np

import tvm
from tvm.ir import module as _module

from .. import analysis as _analysis
from .. import expr as _expr
from .. import op as _op
from .common import get_relay_op
from .common import infer_shape as _infer_shape

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
        return _expr.const(int(data), dtype=_convert_data_type(input_type))
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
    "aten::to"                              : _identity(),
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
    "aten::sqrt"                            : _sqrt()
}

# Internal graph for parsing

class Graph(object):
    """ A helper class for parsing PyTorch model to Relay graph."""

    def __init__(self, script_module, input_shapes):

        self._script_module = script_module
        self._graph = script_module.graph.copy()

        # TODO: Temporary fix to remove prim::CallMethod node introduced in PT 1.4
        import torch
        from packaging import version
        if version.parse(torch.__version__) >= version.parse("1.4.0"):
            torch._C._jit_pass_inline(self._graph)

        self._inputs_r = {}
        self._params = {}
        self._param_tensors = {}
        self._consts = {}
        self._ops = {}
        self._op_inputs_r = {}
        self._op_inputs_types = {}
        self._input_shapes = input_shapes if input_shapes else {}
        self._parsed_node_names = {}

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

        params : dict of str to tvm.runtime
            Dict of converted parameters stored in tvm.runtime format
        """
        # Check for missing ops
        missing_operators = self._parse_import_prerequisites()

        if missing_operators:
            raise tvm.error.OpNotImplemented( \
                "The following operators are not implemented: {}".format(missing_operators))

        # Translate PyTorch graph to by decorating Graph with state dict and inputs into each op
        self._parse_inputs()
        self._parse_params()
        self._parse_ops()

        outputs = []
        nid = 0

        for op_name, op_node in self._ops.items():
            if op_node.kind() == "prim::ListConstruct":
                if any(inp.debugName() in self._parsed_node_names.keys() \
                       for inp in op_node.inputs()):
                    list_constr = []
                    for i in op_node.inputs():
                        if i.debugName() in self._parsed_node_names.keys():
                            list_constr.append( \
                                outputs[self._parsed_node_names[i.debugName()]])
                        elif i.node().kind() == "prim::Constant":
                            list_constr.append(int(self._consts[i.debugName()]))
                        elif i.debugName() in self._inputs_r.keys():
                            list_constr.append(int(self._inputs_r[i.debugName()]))

                    # Unwrap for tensors
                    if len(list_constr) == 1:
                        list_constr = list_constr[0]

                    outputs.append(list_constr)
                    self._parsed_node_names[op_name] = nid
                    nid = nid+1
            elif op_node.kind() != "prim::Constant":
                for i in op_node.inputs():
                    if i.debugName() in self._parsed_node_names.keys():
                        for cnt in range(0, len(self._op_inputs_r[op_name])):
                            if isinstance(self._op_inputs_r[op_name][cnt], str):
                                if "call/var" in self._op_inputs_r[op_name][cnt]:
                                    self._op_inputs_r[op_name][cnt] = \
                                        outputs[self._parsed_node_names[i.debugName()]]
                                    break

                call = _convert_map[op_node.kind()](self._op_inputs_r[op_name],
                                                    self._op_inputs_types[op_name])

                outputs.append(call)
                self._parsed_node_names[op_name] = nid
                nid = nid+1

        func = tvm.relay.Function(_analysis.free_vars(outputs[-1]), outputs[-1])

        param = {k: tvm.nd.array(v) for k, v in self._param_tensors.items()}

        return  _module.IRModule.from_expr(func), param

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

        # Add self (first input of a PyTorch graph) to inputs, the value doesn't matter here
        input_name = ir_inputs[0].debugName()
        self._inputs_r[input_name] = "self"

    def _parse_params(self):
        """ Map state dictionary values to corresponding prim::GetAttr op node. """
        # Grab weights, biases, etc. from graph
        state_dict = self._script_module.state_dict()
        param_names = []
        for key, value in state_dict.items():
            param_str = str(key)
            param_name = param_str.split(".")[-1]
            param_names.append(param_name)

        # Get names of all inputs
        input_names = [i for i in self._inputs_r.keys()]

        # Iterate through graph for getAttr nodes and match full state_dict name to nodes
        node_weight_map = {}
        for node in self._graph.nodes():
            if node.kind() == "prim::GetAttr":

                attribute_names = node.attributeNames()
                assert len(attribute_names) == 1
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

                    if ty in ["IntType", "BoolType"]:
                        self._consts[node_name] = node.i(attr_name)
                    elif ty in ["FloatType", "LongType"]:
                        self._consts[node_name] = node.f(attr_name)
                    elif ty in ["TensorType", "CompleteTensorType"]:
                        self._consts[node_name] = node.output().toIValue()
                    else:
                        self._consts[node_name] = "0"
                else:
                    self._consts[node_name] = "0"
            elif node.kind() == "prim::ListConstruct":
                list_shape = []
                for input_node in node.inputs():
                    if input_node.debugName() in self._inputs_r.keys():
                        c = self._inputs_r[input_node.debugName()]
                        assert isinstance(c, int)
                        list_shape.append(c)
                    elif input_node.debugName() in self._consts.keys():
                        c = self._consts[input_node.debugName()]
                        assert isinstance(c, int)
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
        for input_value in op_node.inputs():

            inode_id = input_value.debugName()
            inode = input_value.node()

            if inode_id in self._inputs_r.keys():
                input_list_r.append(self._inputs_r[inode_id])
            elif inode_id in self._params.keys():
                input_list_r.append(self._params[inode_id])
            elif inode.kind() == "prim::Constant":
                input_list_r.append(self._consts[inode_id])
            else:
                input_list_r.append("call/var."+inode_id)

                # If the inputs of a ListConstruct op is a call or var, remove it from inputs
                if op_node.kind() == "prim::ListConstruct":
                    if node_id in self._inputs_r.keys():
                        self._inputs_r.pop(node_id)

            try:
                input_value_kind = input_value.type().kind()
                if input_value_kind in ["TensorType", "CompleteTensorType"]:
                    if input_value.type().scalarType() is None:
                        input_list_types.append("float")
                    else:
                        input_list_types.append(input_value.type().scalarType().lower())
                elif input_value_kind == "ListType":
                    input_list_types.append(str(input_value.type().getElementType()).lower())
                elif input_value_kind in ["IntType", "FloatType", "BoolType", "StringType",
                                          "OptionalType"]:
                    input_list_types.append(str(input_value.type()).lower())
                else:
                    input_list_types.append("UnsupportedType")
                    print("UnsupportedType "+str(input_value.type())+" and "+str(input_value_kind))
            except Exception as e:
                print("Internal PyTorch error. Failed to grab type.")

        if op_node.kind() in ["aten::ones", "aten::zeros"]:
            node_type = op_node.output().type().scalarType()
            input_list_types[0] = node_type.lower()

        self._op_inputs_r[node_id] = input_list_r
        self._op_inputs_types[node_id] = input_list_types

    def _parse_import_prerequisites(self):
        """ Calculate the named preconditions from PyTorch graph.

        Returns
        -------
        missing_operators : set object
            Set of operator names which don't have their mapping in TVM
            i.e. which are not supported

        """
        missing_operators = set()
        for node in self._graph.nodes():
            if not node.kind() in ["prim::Constant", "prim::ListConstruct", "prim::GetAttr"] \
                    and not node.kind() in _convert_map:
                missing_operators.add(node.kind())

        return missing_operators

def from_pytorch(script_module, input_shapes):
    """ Load PyTorch model in the form of a scripted PyTorch model and convert into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    script_module : TopLevelTracedModule object
        TorchScripted PyTorch graph
        Note: We currently only support traces (ie: torch.jit.trace(model, input))

    shape : Dictionary of input dimensions
        Graph level input shape dictionary

    Returns
    -------
    mod : tvm.relay.Module
        The module that optimizations will be performed on.

    params : dict of str to tvm.runtime
        Dict of converted parameters stored in tvm.runtime format
    """
    g = Graph(script_module, input_shapes)
    mod, params = g.from_pytorch()
    return mod, params
