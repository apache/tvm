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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, unused-argument
"""Convert an NNVM graph to Relay."""
import numpy

from tvm import relay, nd
from tvm.relay import op, expr, var
from tvm.relay.frontend.common import StrAttrsDict
from tvm.relay.frontend.nnvm_common import _rename, _binop_scalar, _rbinop_scalar, \
     _elemwise_sum, _softmax_op, _compare, _reduce
from .symbol import Symbol
from .compiler import graph_attr
from .graph import create as graph_create

def _nn_batch_flatten(children, attrs, odtype='float32'):
    assert len(children) == 1
    return op.nn.batch_flatten(children[0])


def _dense(children, attrs, odtype='float32'):
    use_bias = attrs.get_bool('use_bias', True)
    units = attrs.get_int('units')
    dense = op.nn.dense(children[0], children[1], units=units)
    if use_bias:
        return op.nn.bias_add(dense, children[2])
    else:
        return dense

def _conv2d(children, attrs, odtype='float32'):
    use_bias = attrs.get_bool('use_bias', True)

    if use_bias:
        data, weight, bias = children
    else:
        data, weight = children

    kernel_size = attrs.get_int_tuple('kernel_size')
    channels = attrs.get_int('channels')
    strides = attrs.get_int_tuple('strides', (1, 1))
    padding = attrs.get_int_tuple('padding', (0, 0))
    dilation = attrs.get_int_tuple('dilation', (1, 1))
    groups = attrs.get_int('groups', 1)
    data_layout = attrs.get_str('layout', 'NCHW')
    kernel_layout = attrs.get_str('kernel_layout', 'OIHW')
    out_layout = ''
    out_dtype = attrs.get_str('out_dtype', '')

    conv_out = op.nn.conv2d(
        data,
        weight,
        kernel_size=kernel_size,
        channels=channels,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=out_layout,
        out_dtype=out_dtype)

    if use_bias:
        return op.nn.bias_add(conv_out, bias)
    else:
        return conv_out


def _conv2d_transpose(children, attrs, odtype='float32'):
    use_bias = attrs.get_bool('use_bias', False)

    if use_bias:
        data, weight, bias = children
    else:
        data, weight = children

    strides = attrs.get_int_tuple('strides', (1, 1))
    padding = attrs.get_int_tuple('padding', (0, 0))
    dilation = attrs.get_int_tuple('dilation', (1, 1))
    groups = attrs.get_int('groups', 1)
    data_layout = attrs.get_str('layout', 'NCHW')
    kernel_layout = attrs.get_str('kernel_layout', 'OIHW')
    out_dtype = attrs.get_str('out_dtype', '')

    out_conv2d = op.nn.conv2d_transpose(
        data,
        weight,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_dtype=out_dtype)

    if use_bias:
        return op.nn.bias_add(out_conv2d, bias)
    else:
        return out_conv2d


def _batch_norm(children, attrs, odtype='float32'):
    data, gamma, beta, moving_mean, moving_view = children
    axis = attrs.get_int('axis', 1)
    epsilon = attrs.get_float('epsilon', 1e-05)
    center = attrs.get_bool('center', True)
    scale = attrs.get_bool('scale', True)

    return op.nn.batch_norm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_view,
        axis=axis,
        epsilon=epsilon,
        center=center,
        scale=scale)[0]


def _max_pool2d(children, attrs, odtype='float32'):
    assert len(children) == 1
    data = children[0]
    pool_size = attrs.get_int_tuple('pool_size', (1, 1))
    strides = attrs.get_int_tuple('strides', (1, 1))
    padding = attrs.get_int_tuple('padding', (0, 0))
    layout = attrs.get_str('layout', 'NCHW')
    ceil_mode = attrs.get_bool('ceil_mode', False)

    return op.nn.max_pool2d(
        data,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        layout=layout,
        ceil_mode=ceil_mode)


def _reshape(children, attrs, odtype='float32'):
    data = children[0]
    shape = attrs.get_int_list('shape')
    return op.reshape(data, shape)


def _transpose(children, attrs, odtype='float32'):
    axes = attrs.get_int_list('axes', None)
    return op.transpose(children[0], axes=axes)


def _clip(children, attrs, odtype='float32'):
    a_min = attrs.get_float('a_min')
    a_max = attrs.get_float('a_max')
    return op.clip(children[0], a_min, a_max)


def _cast(children, attrs, odtype='float32'):
    data = children[0]
    dtype = attrs.get_str('dtype')
    return data.astype(dtype)


def _expand_dims(children, attrs, odtype='float32'):
    data = children[0]
    axis = attrs.get_int('axis')
    num_newaxis = attrs.get_int('num_newaxis', 1)
    return op.transform.expand_dims(data, axis, num_newaxis=num_newaxis)


def broadcast_to(children, attrs, odtype='float32'):
    # TODO(@jroesch) export broadcast to?
    data = children[0]
    shape = attrs.get_int_tuple('shape')
    array = numpy.zeros(shape).astype(odtype)
    rconst = relay.Constant(nd.array(array))
    return op.broadcast_to_like(data, rconst)


def _global_avg_pool2d(children, attrs, odtype='float32'):
    data = children[0]
    layout = attrs.get_str('layout', "NCHW")
    return op.nn.global_avg_pool2d(data, layout)


def _avg_pool2d(children, attrs, odtype='float32'):
    data = children[0]
    pool_size = attrs.get_int_tuple('pool_size', (1, 1))
    strides = attrs.get_int_tuple('strides', (1, 1))
    padding = attrs.get_int_tuple('padding', (0, 0))
    layout = attrs.get_str('layout', "NCHW")
    ceil_mode = attrs.get_bool('ceil_mode', False)
    count_include_pad = attrs.get_bool('layout', False)
    return op.nn.avg_pool2d(
        data,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        layout=layout,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad)


def _upsampling(children, attrs, odtype='float32'):
    scale = attrs.get_int('scale')
    layout = attrs.get_str('layout', 'NCHW')
    method = attrs.get_str('method', 'NEAREST_NEIGHBOR')
    return op.nn.upsampling(
        children[0],
        scale=scale,
        layout=layout,
        method=method)


def _pad(children, attrs, odtype='float32'):
    pad_value = attrs.get_float('pad_value', 0.0)
    pad_width = attrs.get_tuple_tuple_int('pad_width')
    return op.nn.pad(children[0], pad_width, pad_value=pad_value)

def _leaky_relu(children, attrs, odtype='float32'):
    alpha = attrs.get_float('alpha')
    return op.nn.leaky_relu(children[0], alpha)


def _full_like(children, attrs, odtype='float32'):
    fill_value = relay.const(attrs.get_float('fill_value'), dtype='float32')
    return op.full_like(children[0], fill_value)


def _strided_slice(children, attrs, odtype='float32'):
    begin = attrs.get_int_list('begin')
    end = attrs.get_int_list('end')
    strides = attrs.get_int_list('stride', None)
    return op.strided_slice(children[0], begin, end, strides=strides)


def _split(children, attrs, odtype='float32'):
    indices_or_sections = None
    try:
        indices_or_sections = attrs.get_int('indices_or_sections', None)
    except ValueError:
        indices_or_sections = indices_or_sections or attrs.get_int_tuple(
            'indices_or_sections')

    axis = attrs.get_int('axis', 0)

    return op.split(children[0], indices_or_sections, axis)

def _squeeze(children, attrs, odtype='float32'):
    axis = attrs.get_int_tuple('axis', None)
    axis = [axis] if isinstance(axis, int) else axis

    return op.squeeze(children[0], axis)

def _concatenate(children, attrs, odtype='float32'):
    axis = attrs.get_int('axis', 1)
    return op.concatenate(children, axis)

def _dropout(children, attrs, odtype='float32'):
    rate = attrs.get_float('rate', 0.5)
    return op.nn.dropout(children[0], rate)

def _mean(children, attrs, odtype='float32'):
    axis = attrs.get_int_tuple('axis', None)
    keepdims = attrs.get_bool('keepdims')

    return op.mean(children[0], axis, keepdims)


def _prelu(children, attrs, odtype='float32'):
    axis = attrs.get_int('axis', 1)
    return op.nn.prelu(children[0], children[1], axis)


def _lrn(children, attrs, odtype='float32'):
    size = attrs.get_int("size", 5)
    axis = attrs.get_int("axis", 1)
    bias = attrs.get_float("bias", 2)
    alpha = attrs.get_float("alpha", 1e-05)
    beta = attrs.get_float("beta", 0.75)
    return op.nn.lrn(children[0], size, axis, bias, alpha, beta)


def _l2_nomalize(children, attrs, odtype='float32'):
    eps = attrs.get_float('eps')
    axis = attrs.get_int_tuple('axis', None)
    return op.nn.l2_normalize(children[0], eps, axis)


def _take(children, attrs, odtype='float32'):
    axis = attrs.get_int('axis', None)
    return op.take(children[0], children[1], axis)


def _matmul(children, attrs, odtype='float32'):
    input_1_t = op.transpose(children[1], axes=(1, 0))
    return op.nn.dense(children[0], input_1_t)


def _collapse_sum(children, attrs, odtype='float32'):
    for key in ["axis", "keepdims", "exclude"]:
        if key in attrs.attrs:
            raise NotImplementedError("Parameter '" + key + "' is not supported.")
    return op.collapse_sum_like(children[0], children[1])


def _not_implemented(new_op):
    def _impl(children, attrs, odtype='float32'):
        raise NotImplementedError(str(new_op) + " is not implemented.")
    return _impl


NNVM_OP_2_RELAY_OP = {
    'flatten': _nn_batch_flatten,
    'dense': _dense,
    'softmax': _softmax_op(op.nn.softmax),
    'log_softmax': _softmax_op(op.nn.log_softmax),
    'conv2d': _conv2d,
    'batch_norm': _batch_norm,
    'max_pool2d': _max_pool2d,
    'reshape': _reshape,
    'transpose': _transpose,
    'dropout': _dropout,
    'mean': _mean,
    # Addition
    '__add_scalar__': _binop_scalar(op.add),
    'broadcast_add' : _rename(op.add),
    'elemwise_add'  : _rename(op.add),
    # Subtraction
    '__sub_scalar__' : _binop_scalar(op.subtract),
    '__rsub_scalar__': _rbinop_scalar(op.subtract),
    'broadcast_sub'  : _rename(op.subtract),
    'elemwise_sub'   : _rename(op.subtract),
    # Multiply
    '__mul_scalar__': _binop_scalar(op.multiply),
    'broadcast_mul' : _rename(op.multiply),
    'elemwise_mul'  : _rename(op.multiply),
    # Division
    '__div_scalar__': _binop_scalar(op.divide),
    'broadcast_div' : _rename(op.divide),
    'elemwise_div'  : _rename(op.divide),
    'broadcast_mod' : _rename(op.mod),
    # Negative
    'negative': _rename("negative"),
    # Power
    '__pow_scalar__': _binop_scalar(op.power),
    '__rpow_scalar__': _rbinop_scalar(op.power),
    'broadcast_pow': _rename(op.power),
    # Sum
    'sum': _reduce(op.sum),
    'elemwise_sum': _elemwise_sum,
    'collapse_sum': _collapse_sum,
    'broadcast_max': _rename(op.maximum),
    'broadcast_min': _rename(op.minimum),

    # Comparsion
    'greater': _compare(op.greater),
    'broadcast_greater': _compare(op.greater),
    'greater_equal': _compare(op.greater_equal),
    'broadcast_greater_equal': _compare(op.greater_equal),
    'less': _compare(op.less),
    'broadcast_less': _compare(op.less),
    'less_equal': _compare(op.less_equal),
    'broadcast_less_equal': _compare(op.less_equal),
    'broadcast_equal': _compare(op.equal),
    'broadcast_not_equal': _compare(op.not_equal),

    # Activations
    'sigmoid': _rename('sigmoid'),
    'relu': _rename('nn.relu'),
    'exp': _rename('exp'),
    'log': _rename('log'),
    'tanh': _rename('tanh'),
    'leaky_relu': _leaky_relu,
    'prelu': _prelu,
    'clip': _clip,
    'round': _rename('round'),
    'cast': _cast,
    'expand_dims': _expand_dims,
    'broadcast_to': broadcast_to,
    '__lshift_scalar__': _binop_scalar(op.left_shift),
    '__rshift_scalar__': _binop_scalar(op.right_shift),
    'broadcast_left_shift': _rename(op.left_shift),
    'broadcast_right_shift': _rename(op.right_shift),
    'copy': _rename(op.copy),
    'global_avg_pool2d': _global_avg_pool2d,
    'avg_pool2d': _avg_pool2d,
    'conv2d_transpose': _conv2d_transpose,
    'upsampling': _upsampling,
    'pad': _pad,
    'full_like': _full_like,
    'strided_slice': _strided_slice,
    'split': _split,
    'squeeze': _squeeze,
    'concatenate': _concatenate,
    'abs': _rename(op.abs),
    'ceil': _rename(op.ceil),
    'floor': _rename(op.floor),
    'trunc': _rename(op.trunc),
    'take': _take,
    'lrn': _lrn,
    'l2_normalize': _l2_nomalize,
    'matmul': _matmul,
    'zeros_like': _rename(op.zeros_like),
    'reshape_like': _rename(op.reshape_like),
    'ones_like': _rename(op.ones_like),

    'expand_like': _not_implemented("expand_like"),
    'gather_nd': _not_implemented("gather_nd"),
    'block_grad': _not_implemented("block_grad"),
}


def to_relay(graph, shape_dict, dtype_dict, params):
    """Convert an NNVM graph into the corresponding Relay expression.

    Parameters
    ----------
    graph : Graph
       The input graph.

    shape_dict : dict of str to shape
       The input shape.

    dtype_dict : dict of str to str/dtype
       The input shape.

    params : dict of str to array
        The parameters.

    Returns
    -------
    (expr, params) : Tuple[relay.Expr, dict of str to array]
        The corresponding Relay expression and parameters.
    """
    if isinstance(graph, Symbol):
        graph = graph_create(graph)

    param_shapes = dict((k, params[k].shape) for k in params)
    shape_dict = shape_dict.copy()
    shape_dict.update(param_shapes)
    graph = graph_attr.set_shape_inputs(graph, shape_dict)
    graph = graph_attr.set_dtype_inputs(graph, dtype_dict)
    graph = graph.apply(["InferShape", "InferType"])
    shape = graph.json_attr("shape")
    dtype = [graph_attr.TCODE_TO_DTYPE[di] for di in graph.json_attr("dtype")]

    gidx = graph.index
    relay_map = {}
    fn_params = []

    for nid, node in enumerate(gidx.nodes):
        children = []
        for i in node['inputs']:
            child = relay_map[i[0]]
            if isinstance(child, expr.TupleWrapper):
                children.append(child[i[1]])
            else:
                children.append(child)

        oshape = shape[gidx.entry_id(nid, 0)]
        odtype = dtype[gidx.entry_id(nid, 0)]
        attrs = node.get("attrs", {})
        node_name = node["name"]
        op_name = node["op"]

        if op_name == "null":
            v = var(node_name, shape=oshape, dtype=odtype)
            fn_params.append(v)
            relay_map[nid] = v
        else:
            if op_name in NNVM_OP_2_RELAY_OP:
                str_attrs = StrAttrsDict(attrs)
                call = NNVM_OP_2_RELAY_OP[op_name](children, str_attrs, odtype)
                relay_map[nid] = call
            else:
                raise Exception(
                    "nnvm.to_relay: unsupported operator: {0}".format(op_name))

    outputs = []
    for nid, idx, _ in gidx.output_entries:
        output = relay_map[nid]
        if isinstance(output, expr.TupleWrapper):
            outputs.append(output[idx])
        else:
            outputs.append(output)

    if len(outputs) == 1:
        body = outputs[0]
    else:
        body = expr.Tuple(outputs)

    func = relay.Function(fn_params, body)
    return func, params
