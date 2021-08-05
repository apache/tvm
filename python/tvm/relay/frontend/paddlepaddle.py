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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""Paddle: PArallel Distributed Deep LEarning."""
import copy
import warnings
import six

import numpy as np

import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import ty as _ty
from .. import op as _op
from .common import (
    fold_constant,
    infer_shape,
    infer_type,
    infer_value,
    new_var,
)

__all__ = ["from_paddle"]


def shape_of(x, dtype='int32'):
    ttype = infer_type(x).checked_type
    if not _ty.is_dynamic(ttype):
        shape = list(ttype.shape)
        return _expr.const(shape, dtype)
    return _op.shape_of(x, dtype)


def convert_arg_max(g, op, block):
    """Operator converter for arg_max."""

    axis = op.attr('axis')
    keepdims = op.attr('keepdims')
    flatten = op.attr('flatten')
    assert not flatten, "Only flatten==True is supported for PaddlePaddle's arg_max"

    x = g.get_node(x.input('X')[0])
    out = _op.argmax(x, axis=axis, keepdims=keepdims)
    g.add_node(op.output('Out')[0], out)


def convert_assign(g, op, block):
    """Operator converter for assign."""

    out = _op.copy(g.get_node(op.input('X')[0]))
    g.add_node(op.output('Out')[0], out)


def convert_batch_norm(g, op, block):
    """Operator converter for batch_norm."""

    ipt_name = op.input('X')[0]
    scale_name = op.input('Scale')[0]
    bias_name = op.input('Bias')[0]
    mean_name = op.input('Mean')[0]
    variance_name = op.input('Variance')[0]
    epsilon = op.attr('epsilon')
    momentum = op.attr('momentum')
    out = _op.nn.batch_norm(g.get_node(ipt_name),
                            g.get_node(scale_name),
                            g.get_node(bias_name),
                            g.get_node(mean_name),
                            g.get_node(variance_name),
                            epsilon=epsilon)
    g.add_node(op.output('Y')[0], out[0])


def convert_cast(g, op, block):
    """Operator converter for cast."""

    dtype = block.var(op.output('Out')[0]).dtype
    dtype = str(dtype).strip().split('.')[1]
    x = g.get_node(op.input('X')[0])
    out = _op.cast(x, dtype=dtype)
    g.add_node(op.output('Out')[0], out)


def convert_concat(g, op, block):
    """Operator converter for concat."""

    inputs = [g.get_node(op.input('X')[i]) for i in range(len(op.input('X')))]
    axis = op.attr('axis')
    out = _op.concatenate(inputs, axis=axis)
    g.add_node(op.output('Out')[0], out)


def convert_conv2d(g, op, block):
    """Operator converter for conv2d."""
    def get_pad_size(in_size, dilated_kernel_size, stride_size):
        if stride_size == 1 or in_size & stride_size == 0:
            pad = max(dilated_kernel_size - stride_size, 0)
        else:
            pad = max(dilated_kernel_size - (in_size % stride_size), 0)
        return [pad // 2, pad - pad // 2]

    dilations = op.attr('dilations')
    groups = op.attr('groups')
    paddings = op.attr('paddings')
    padding_algorithm = op.attr('padding_algorithm')
    strides = op.attr('strides')

    kernel = g.get_node(op.input('Filter')[0])
    input = g.get_node(op.input('Input')[0])
    out_channels, _, k_h, k_w = infer_shape(kernel)
    in_h, in_w = infer_shape(input)[2:]
    out = _op.nn.conv2d(input,
                        kernel,
                        strides=strides,
                        padding=paddings,
                        dilation=dilations,
                        groups=groups,
                        channels=out_channels,
                        kernel_size=[k_h, k_w])
    g.add_node(op.output('Output')[0], out)


def convert_cumsum(g, op, block):
    """Operator converter for cumsum."""

    axis = op.attr('axis')
    exclusive = op.attr('exclusive')
    flatten = op.attr('flatten')
    reverse = op.attr('reverse')

    assert not flatten, "Only flatten==False is supported for PaddlePaddle's cumsum"

    x = g.get_node(op.input('X')[0])
    if reverse:
        x = _op.reverse(x, axis=axis)
        out = _op.cumsum(x, axis=axis, exclusive=exclusive)
        out = _op.reverse(out, axis=axis)
    else:
        out = _op.cumsum(x, axis=axis, exclusive=exclusive)
    g.add_node(op.output('Out')[0], out)


def convert_dropout(g, op, block):
    """Operator converter for dropout."""

    x = g.get_node(op.input('X')[0])
    out = _op.copy(x)
    g.add_node(op.output('Out')[0], out)


def convert_elementwise_op(g, op, block):
    """Operator converter for all the elementwise operators."""

    op_map = {
        'elementwise_div': lambda x, y: x / y,
        'elementwise_add': lambda x, y: x + y,
        'elementwise_mul': lambda x, y: x * y,
        'elementwise_sub': lambda x, y: x - y,
        'elementwise_mod': lambda x, y: x % y,
    }
    op_func = op_map[op.type]
    ipt0 = g.get_node(op.input('X')[0])
    ipt1 = g.get_node(op.input('Y')[0])
    ipt0_shape = block.var(op.input('X')[0]).shape
    ipt1_shape = block.var(op.input('Y')[0]).shape
    axis = op.attr('axis')
    if len(ipt0_shape) != len(ipt1_shape):
        if axis < 0:
            axis = axis + len(ipt0_shape)
        if axis != len(ipt0_shape) - 1:
            ipt1 = _op.expand_dims(ipt1,
                                   axis=axis,
                                   num_newaxis=(len(ipt0_shape) - axis - 1))
    out = op_func(ipt0, ipt1)
    g.add_node(op.output('Out')[0], out)


def convert_equal(g, op, block):
    """Operator converter for equal."""

    x = g.get_node(op.input('X')[0])
    y = g.get_node(op.input('Y')[0])
    out = _op.equal(x, y)
    g.add_node(op.output('Out')[0], out)


def convert_activation(g, op, block):
    """Operator converter for all the activation."""

    op_map = {
        'exp': _op.exp,
        'relu': _op.nn.relu,
        'tanh': _op.tanh,
        'sqrt': _op.sqrt,
        'erf': _op.erf,
        'abs': _op.abs,
    }
    act_func = op_map[op.type]
    out = act_func(g.get_node(op.input('X')[0]))
    g.add_node(op.output('Out')[0], out)


def convert_feed(g, op, block):
    """Converter for model input node."""

    if block is not None:
        ipt_name = op.output('Out')[0]
        ipt_shape = block.var(ipt_name).shape
        ipt_dtype = block.var(ipt_name).dtype
        ipt_dtype = str(ipt_dtype).strip().split('.')[1]
    else:
        ipt_shape = op.shape
        ipt_dtype = str(op.dtype).strip().split('.')[1]
        ipt_name = op.name
    if g.shape_dict is not None:
        ipt_shape = g.shape_dict[ipt_name]
    out = new_var(ipt_name, shape=ipt_shape, dtype=ipt_dtype)
    g.add_node(ipt_name, out)


def convert_fill_any_like(g, op, block):
    """Operator converter for fill_any_like."""

    out_name = op.output('Out')[0]
    out_dtype = block.var(out_name).dtype
    out_dtype = str(out_dtype).strip().split('.')[1]
    x = g.get_node(op.input('X')[0])
    ipt_type = infer_type(x).checked_type
    value = op.attr('value')
    if not _ty.is_dynamic(ipt_type):
        shape = infer_shape(x)
        const = np.ones(shape) * value
        out = _expr.const(const.astype(out_dtype))
    else:
        out = _op.transform.full_like(x, value).astype(out_dtype)
    g.add_node(op.output('Out')[0], out)


def convert_fill_constant(g, op, block):
    """Operator converter for fill_constant."""

    value = op.attr('value')
    shape = block.var(op.output('Out')[0]).shape
    dtype = block.var(op.output('Out')[0]).dtype
    dtype = str(dtype).strip().split('.')[1]
    value = np.full(shape, value, dtype)
    out = _expr.const(value.astype(dtype)).astype(dtype)
    g.add_node(op.output('Out')[0], out)


def convert_gelu(g, op, block):
    """Operator converter for gelu."""

    x = g.get_node(op.input('X')[0])
    out = x * (_expr.const(0.5, dtype='float32') +
               _op.erf(x * _expr.const(0.5**0.5, dtype='float32')) *
               _expr.const(0.5, dtype='float32'))
    g.add_node(op.output('Out')[0], out)


def convert_hard_sigmoid(g, op, block):
    """Operator converter for hard_sigmoid."""

    slope = op.attr('slope')
    offset = op.attr('offset')
    x = g.get_node(op.input('X')[0])
    out = x * _expr.const(slope) + _expr.const(0.5)
    out = _op.clip(out, 0, 1)
    g.add_node(op.output('Out')[0], out)


def convert_hard_swish(g, op, block):
    """Operator converter for hard_swish."""

    offset = op.attr('offset')
    scale = op.attr('scale')
    threshold = op.attr('threshold')
    assert np.isclose(
        offset, 3.0), "Only support offset==3.0 for PaddlePaddle's hard_swish"
    assert np.isclose(
        scale, 6.0), "Only support scale==6.0 for PaddlePaddle's hard_swish"
    assert np.isclose(
        threshold,
        6.0), "Only support threshold==6.0 for PaddlePaddle's hard_swish"
    x = g.get_node(op.input('X')[0])
    out = _op.clip(x, -1 * offset, offset)
    out = out / _expr.const(threshold) + _expr.const(0.5)
    out = x * out
    g.add_node(op.output('Out')[0], out)


def convert_layer_norm(g, op, block):
    """Operator converter for layer_norm."""

    begin_norm_axis = op.attr('begin_norm_axis')
    epsilon = op.attr('epsilon')
    x = g.get_node(op.input('X')[0])
    bias = g.get_node(op.input('Bias')[0])
    scale = g.get_node(op.input('Scale')[0])
    out = _op.nn.layer_norm(x,
                            gamma=scale,
                            beta=bias,
                            axis=begin_norm_axis,
                            epsilon=epsilon,
                            center=True,
                            scale=True)
    g.add_node(op.output('Y')[0], out)


def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr('alpha')
    x = g.get_node(op.input('X')[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output('Out')[0])


def convert_lookup_table(g, op, block):
    """Operator converter for lookup_table_v2."""

    indices = g.get_node(op.input('Ids')[0])
    padding_idx = op.attr('padding_idx')
    is_sparse = op.attr('is_sparse')
    height_sections = op.attr('height_sections')
    if padding_idx != -1:
        g.get_params[op.input('W')[0]][padding_idx] = 0.0
        g.add_node(op.input('W')[0], _expr.const(g.params[op.input('W')[0]]))
    weights = g.get_node(op.input('W')[0])
    out = _op.take(weights, indices.astype('int32'), axis=0)
    g.add_node(op.output('Out')[0], out)


def convert_matmul(g, op, block):
    """Operator converter for matmul."""

    inputs = [g.get_node(op.input('X')[0]), g.get_node(op.input('Y')[0])]
    a_shape = infer_shape(inputs[0])
    b_shape = infer_shape(inputs[1])
    try:
        # for matmul_v2
        trans_x = op.attr('trans_x')
        trans_y = op.attr('trans_y')
    except:
        # for matmul
        trans_x = op.attr('transpose_X')
        trans_y = op.attr('transpose_Y')
    if trans_x:
        perm = list(range(len(a_shape)))
        perm[-2] = len(a_shape) - 1
        perm[-1] = len(a_shape) - 2
        inputs[0] = _op.transpose(inputs[0], axes=perm)
    if trans_y:
        perm = list(range(len(b_shape)))
        perm[-2] = len(b_shape) - 1
        perm[-1] = len(b_shape) - 2
        inputs[1] = _op.transpose(inputs[1], axes=perm)

    # This implemention almost keeps same with ONNX
    # Need to check input shape as batch matmul must be supported.
    a_shape = shape_of(inputs[0])
    a_rank = infer_shape(a_shape)[0]
    b_shape = shape_of(inputs[1])
    b_rank = infer_shape(b_shape)[0]
    # When performing a batch matmul, we need to properly handle N-dim shapes.
    if a_rank > 2 or b_rank > 2:

        def flatten_to_nd(x, x_shape, nd=3):
            ndims = infer_shape(x_shape)[0]
            if ndims == nd:
                return x
            newshape = _op.concatenate(
                [
                    _expr.const([-1],
                                dtype=infer_type(x_shape).checked_type.dtype),
                    _op.strided_slice(x_shape, [ndims - nd + 1], [ndims]),
                ],
                0,
            )
            out = _op.reshape(x, fold_constant(newshape))
            return out

        b_type = infer_type(inputs[1])
        # Convert to dense if the second matrix is 2d and non-dynamic
        if b_rank == 2 and not _ty.is_dynamic(b_type.checked_type):
            a = flatten_to_nd(inputs[0], a_shape, 2)
            b = _op.transpose(inputs[1])
            output = _op.nn.dense(a, b)
        else:
            # Convert a and b into 3 dimensional tensors.
            a = flatten_to_nd(inputs[0], a_shape, 3)
            b = flatten_to_nd(inputs[1], b_shape, 3)
            # Transpose matrix dimensions of b.
            b = _op.transpose(b, [0, 2, 1])
            # Perform a batch matmul.
            output = _op.nn.batch_matmul(a, b)
        # Determine the output batch dimension.
        if a_rank > b_rank:
            out_batch = _op.strided_slice(a_shape, [0], [a_rank - 2])
        elif a_rank < b_rank:
            out_batch = _op.strided_slice(b_shape, [0], [b_rank - 2])
        # If its unclear how broadcasting should be applied, the output
        # shape is determined by choosing the maximum value from each input.
        else:
            out_batch = _op.concatenate(
                [
                    _op.maximum(
                        _op.strided_slice(a_shape, [i], [i + 1]),
                        _op.strided_slice(b_shape, [i], [i + 1]),
                    ) for i in range(a_rank - 2)
                ],
                0,
            )
        # Reshape output to original dimensions.
        final_shape = _op.concatenate(
            [
                out_batch,
                _op.strided_slice(a_shape, [infer_shape(a_shape)[0] - 2],
                                  [infer_shape(a_shape)[0] - 1]),
                _op.strided_slice(b_shape, [infer_shape(b_shape)[0] - 1],
                                  [infer_shape(b_shape)[0]]),
            ],
            0,
        )
        out = _op.reshape(output, fold_constant(final_shape))
    else:
        # Otherwise a simple dense op will get the job done.
        input_1_t = _op.transpose(inputs[1], axes=(1, 0))
        out = _op.nn.dense(inputs[0], input_1_t)
    try:
        alpha = op.attr('alpha')
        if not np.isclose(alpha, 1.0):
            out = out * _expr.const(alpha).astype('float32')
    except:
        pass
    g.add_node(op.output('Out')[0], out)


def convert_mul(g, op, block):
    """Operator converter for mul."""

    x = g.get_node(op.input('X')[0])
    y = g.get_node(op.input('Y')[0])
    x_num_col_dims = op.attr('x_num_col_dims')
    y_num_col_dims = op.attr('y_num_col_dims')
    x_shape = shape_of(x)
    y_shape = shape_of(y)
    x_dim = infer_shape(x_shape)[0]
    y_dim = infer_shape(y_shape)[0]
    if x_num_col_dims < 0:
        x_num_col_dims += x_dim
    if y_num_col_dims < 0:
        y_num_col_dims += y_dim
    if x_num_col_dims == 1:
        x = _op.nn.batch_flatten(x)
    else:
        pre_shape = _op.prod(_op.strided_slice(x_shape, [0], [x_num_col_dims],
                                               [1]),
                             keepdims=True)
        post_shape = _op.prod(_op.strided_slice(x_shape, [x_num_col_dims],
                                                [x_dim], [1]),
                              keepdims=True)
        new_shape = _op.concatenate([pre_shape, post_shape], axis=0)
        new_shape = fold_constant(new_shape)
        x = _op.reshape(x, new_shape)
    if y_num_col_dims == 1:
        y = _op.nn.batch_flatten(y)
    else:
        pre_shape = _op.prod(_op.strided_slice(y_shape, [0], [y_num_col_dims],
                                               [1]),
                             keepdims=True)
        post_shape = _op.prod(_op.strided_slice(y_shape, [y_num_col_dims],
                                                [y_dim], [1]),
                              keepdims=True)
        new_shape = _op.concatenate([pre_shape, post_shape], axis=0)
        new_shape = fold_constant(new_shape)
        y = _op.reshape(y, new_shape)
    y = _op.transpose(y)
    out = _op.nn.dense(x, y)
    out_pre_shape = _op.strided_slice(x_shape, [0], [x_num_col_dims], [1])
    out_post_shape = _op.strided_slice(y_shape, [y_num_col_dims], [y_dim], [1])
    out_shape = _op.concatenate([out_pre_shape, out_post_shape], axis=0)
    out_shape = fold_constant(out_shape)
    out = _op.reshape(out, out_shape)
    g.add_node(op.output('Out')[0], out)


def convert_pool2d(g, op, block):
    """Operator converter for pool2d."""

    adaptive = op.attr('adaptive')
    ceil_mode = op.attr('ceil_mode')
    global_pooling = op.attr('global_pooling')
    ksize = op.attr('ksize')
    paddings = op.attr('paddings')
    padding_algorithm = op.attr('padding_algorithm')
    pooling_type = op.attr('pooling_type')
    if global_pooling:
        adaptive = True
        ksize = [1, 1]

    op_map = {
        'avg': 'avg_pool2d',
        'max': 'max_pool2d',
    }
    strides = op.attr('strides')
    if isinstance(strides, int):
        strides = [strides, strides]
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(paddings, int):
        paddings = [paddings] * 2

    x = g.get_node(op.input('X')[0])
    if not adaptive:
        out = getattr(_op.nn, op_map[pooling_type])(x,
                                                    pool_size=ksize,
                                                    strides=strides,
                                                    padding=paddings,
                                                    ceil_mode=ceil_mode)
    else:
        out = getattr(_op.nn,
                      "adaptive_" + op_map[pooling_type])(x, output_size=ksize)
    g.add_node(op.output('Out')[0], out)


def convert_reshape(g, op, block):
    """Operator converter for reshape."""

    shape = op.attr('shape')
    out = _op.reshape(g.get_node(op.input('X')[0]), shape)
    g.add_node(op.output('Out')[0], out)


def convert_scale(g, op, block):
    """Operator converter for scale."""

    scale = op.attr('scale')
    bias = op.attr('bias')
    bias_after_scale = op.attr('bias_after_scale')
    x = g.get_node(op.input('X')[0])
    if np.isclose(scale, 1.0) and np.isclose(bias, 0.0):
        out = _op.copy(x)
    else:
        if np.isclose(bias, 0.0):
            out = x * _expr.const(np.array(scale).astype('float32'))
        elif np.isclose(scale, 1.0):
            out = x + _expr.const(np.array(bias).astype('float32'))
        else:
            if bias_after_scale:
                out = x * _expr.const(
                    np.array(scale).astype('float32')) + _expr.const(
                        np.array(bias).astype('float32'))
            else:
                out = (x + _expr.const(np.array(bias).astype('float32'))
                       ) * _expr.const(np.array(scale).astype('float32'))
    g.add_node(op.output('Out')[0], out)


def convert_shape(g, op, block):
    """Operator converter for shape."""

    x = g.get_node(op.input('Input')[0])
    out = shape_of(x)
    g.add_node(op.output('Out')[0], out)


def convert_slice(g, op, block):
    """Operator converter for slice."""
    def parameter_process(starts, ends, axes):
        new_axes = []
        new_starts = []
        new_ends = []
        pop_index = 0
        for i in range(max(axes) + 1):
            new_axes.append(i)
            if i in axes:
                new_starts.append(starts[pop_index])
                new_ends.append(ends[pop_index])
                pop_index += 1
            else:
                new_starts.append(0)
                new_ends.append(np.iinfo(np.int32).max)
        return new_starts, new_ends, new_axes

    starts = op.attr('starts')
    ends = op.attr('ends')
    axes = op.attr('axes')
    if isinstance(starts, int):
        starts = [starts]
    if isinstance(ends, int):
        ends = [ends]
    if isinstance(axes, int):
        axes = [axes]
    starts, ends, axes = parameter_process(starts, ends, axes)
    out = _op.strided_slice(g.get_node(op.input('Input')[0]),
                            begin=starts,
                            end=ends)
    g.add_node(op.output('Out')[0], out)


def convert_softmax(g, op, block):
    """Operator converter for softmax."""

    axis = op.attr('axis')
    input_shape = block.var(op.input('X')[0]).shape
    if axis < 0:
        axis = len(input_shape) + axis
    x = g.get_node(op.input('X')[0])
    m = _op.max(x, axis, keepdims=True)
    e = _op.exp(x - m)
    out = e / _op.sum(e, axis, keepdims=True)
    g.add_node(op.output('Out')[0], out)


def convert_transpose(g, op, block):
    """Operator converter for transpose."""

    perm = op.attr('axis')
    out = _op.transpose(g.get_node(op.input('X')[0]), axes=perm)
    g.add_node(op.output('Out')[0], out)


def convert_unsqueeze(g, op, block):
    """Operator converter for unsqueeze."""

    x = g.get_node(op.input('X')[0])
    axes = sorted(op.attr('axes'))
    for axis in axes:
        x = _op.expand_dims(x, axis=axis, num_newaxis=1)
    g.add_node(op.output('Out')[0], x)


_convert_map = {
    'arg_max': convert_arg_max,
    'assign': convert_assign,
    'batch_norm': convert_batch_norm,
    'cast': convert_cast,
    'concat': convert_concat,
    'conv2d': convert_conv2d,
    'cumsum': convert_cumsum,
    'depthwise_conv2d': convert_conv2d,
    'dropout': convert_dropout,
    'elementwise_add': convert_elementwise_op,
    'elementwise_div': convert_elementwise_op,
    'elementwise_mul': convert_elementwise_op,
    'elementwise_sub': convert_elementwise_op,
    'equal': convert_equal,
    'exp': convert_activation,
    'feed': convert_feed,
    'fill_any_like': convert_fill_any_like,
    'fill_constant': convert_fill_constant,
    'gelu': convert_gelu,
    'hard_sigmoid': convert_hard_sigmoid,
    'hard_swish': convert_hard_swish,
    'layer_norm': convert_layer_norm,
    'leaky_relu': convert_leaky_relu,
    'lookup_table_v2': convert_lookup_table,
    'matmul': convert_matmul,
    'matmul_v2': convert_matmul,
    'mul': convert_mul,
    'pool2d': convert_pool2d,
    'relu': convert_activation,
    'reshape2': convert_reshape,
    'scale': convert_scale,
    'shape': convert_shape,
    'slice': convert_slice,
    'softmax': convert_softmax,
    'tanh': convert_activation,
    'transpose2': convert_transpose,
    'unsqueeze2': convert_unsqueeze,
}


class GraphProto(object):
    """ A helper class for handling relay functions from PaddlePaddle model."""
    def __init__(self):
        self.nodes = {}
        self.params = {}
        self.shape_dict = None

    def get_node(self, name):
        assert name in self.nodes
        return self.nodes[name]

    def add_node(self, name, node):
        self.nodes[name] = fold_constant(node)

    def get_params(self, name):
        assert name in self.params
        return self.params[name]

    def extract_parameters(self, program, scope=None):
        """ Extract all the weights from PaddlePaddle program."""

        self.params = {}
        variables = program.global_block().vars
        for name in variables:
            var = program.global_block().var(name)
            if name.endswith('feed') or name.endswith('fetch'):
                continue
            if not var.persistable:
                continue
            if isinstance(scope, dict):
                self.params[name] = scope[name]
            else:
                self.params[name] = np.array(scope.var(name).get_tensor())
            self.nodes[name] = _expr.const(self.params[name])

    def check_input_shape(self, op, block):
        """ Check the shape information of model's inputs, fixed shape is recommended."""

        ipt_name = op.input(op.input_names[0])
        ipt_shape = block.var(ipt_name).shape
        for i in ipt_shape:
            if i < 0:
                warning_msg = (
                    "Input {}(shape={}) has unkown dimension shapes. Specifying static values may improve performance"
                    .format(ipt_name, ipt_shape))
                warings.warn(warning_msg)

    def check_unsupported_ops(self, program):
        """ Check whether all the operators are supported."""

        unsupported_ops = set()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == 'fetch':
                    continue
                if op.type not in _convert_map:
                    unsupported_ops.add(op.type)
        if len(unsupported_ops) > 0:
            msg = "The following operators are not supported for frontend Paddle: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def ops_to_relay(self, program, input_specs=None):
        """ Convert PaddlePaddle operators to TVM relay functions."""

        if input_specs is not None:
            for input_spec in input_specs:
                convert_feed(self, input_spec, None)
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == 'fetch':
                    continue
                convert_func = _convert_map[op.type]
                convert_func(self, op, block)

    def from_program(self, program, shape_dict, scope):
        """ Construct the TVM relay expression from PaddlePaddle program."""

        self.shape_dict = shape_dict
        if scope is None:
            import paddle
            scope = paddle.fluid.global_scope()
        self.check_unsupported_ops(program)
        self.extract_parameters(program, scope)
        self.ops_to_relay(program)

        output_names = list()
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == "fetch":
                    output_names.append(op.input('X')[0])

        outputs = [self.nodes[name] for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        return mod, self.params

    def from_translated_layer(self, layer, shape_dict):
        """ Construct the TVM relay expression from PaddlePaddle TranslatedLayer."""

        self.shape_dict = shape_dict
        program = layer.program()
        parameters = dict()
        for param in layer.parameters():
            parameters[param.name] = np.array(param.value().get_tensor())
        self.check_unsupported_ops(program)
        self.extract_parameters(program, parameters)

        input_specs = layer._input_spec()
        self.ops_to_relay(program, input_specs)

        output_names = [x.name for x in layer._output_spec()]

        outputs = [self.nodes[name] for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        return mod, self.params


def from_paddle(program_or_layer, shape_dict=None, scope=None):
    """ Convert a PaddlePaddle model into an equivalent Relay Function.

    PaddlePaddle Program/TranslatedLayer represent the computation graph of PaddlePaddle model, 
    and PaddlePaddle scope stores all the weights of PaddlePaddle model. 
    """

    import paddle
    g = GraphProto()
    if isinstance(program_or_layer, paddle.fluid.dygraph.TranslatedLayer):
        # model is loaded by `paddle.jit.load`
        mod, params = g.from_translated_layer(program_or_layer, shape_dict)
    elif isinstance(program_or_layer, paddle.fluid.framework.Program):
        # model is loaded by `paddle.static.load_inference_model`
        mod, params = g.from_program(program_or_layer, shape_dict, scope)
    else:
        raise Exception(
            "Only PaddlePaddle's Program and TranslatedLayer are supported.")
    return mod, params
