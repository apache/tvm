# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, unused-argument
"""Convert an NNVM graph to Relay."""
import json
import numpy

from tvm import relay, nd
from tvm.relay import op, expr, var
from tvm.relay.frontend.common import StrAttrsDict
from tvm.relay.frontend.nnvm_common import _rename
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

def _nn_softmax(children, attrs, odtype='float32'):
    assert len(children) == 1
    axis = attrs.get_int('axis', 1)
    return op.nn.softmax(children[0], axis)

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


def _add(children, attrs, odtype='float32'):
    if len(children) == 1:
        left = children[0]
        scalar = attrs.get_float('scalar')
        right = relay.const(scalar, dtype=odtype)
    else:
        assert len(children) == 2
        left = children[0]
        right = children[1]

    return op.add(left, right)


def _subtract(children, attrs, odtype='float32'):
    if len(children) == 1:
        left = children[0]
        scalar = attrs.get_float('scalar')
        right = relay.const(scalar, dtype=odtype)
    else:
        assert len(children) == 2
        left = children[0]
        right = children[1]

    return op.subtract(left, right)


def _rsubtract(children, attrs, odtype='float32'):
    if len(children) == 1:
        left = children[0]
        scalar = attrs.get_float('scalar')
        right = relay.const(scalar, dtype=odtype)
    else:
        assert len(children) == 2
        left = children[0]
        right = children[1]

    return op.subtract(right, left)


def _multiply(children, attrs, odtype='float32'):
    if len(children) == 1:
        left = children[0]
        scalar = attrs.get_float('scalar')
        right = relay.const(scalar, dtype=odtype)
    else:
        assert len(children) == 2
        left = children[0]
        right = children[1]

    return op.multiply(left, right)


def _divide(children, attrs, odtype='float32'):
    if len(children) == 1:
        left = children[0]
        scalar = attrs.get_float('scalar')
        right = relay.const(scalar, dtype=odtype)
    else:
        assert len(children) == 2
        left = children[0]
        right = children[1]

    return op.divide(left, right)


def _rshift(children, attrs, odtype='float32'):
    if len(children) == 1:
        left = children[0]
        scalar = attrs.get_float('scalar')
        right = relay.const(scalar, dtype='int32')
    else:
        assert len(children) == 2
        left = children[0]
        right = children[1]

    return op.right_shift(left, right)


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

def _copy(children, attrs, odtype='float32'):
    return op.copy(children[0])


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


def _greater(children, attrs, odtype='float32'):
    out_type = attrs.get_str('out_type')
    if out_type:
        return op.greater(children[0], children[1]).astype(out_type)
    else:
        return op.greater(children[0], children[1])


def _greater_equal(children, attrs, odtype='float32'):
    out_type = attrs.get_str('out_type', None)
    if out_type:
        return op.greater_equal(children[0], children[1]).astype(out_type)
    else:
        return op.greater_equal(children[0], children[1])


def _less(children, attrs, odtype='float32'):
    out_type = attrs.get_str('out_type', None)
    if out_type:
        return op.less(children[0], children[1]).astype(out_type)
    else:
        return op.less(children[0], children[1])


def _less_equal(children, attrs, odtype='float32'):
    out_type = attrs.get_str('out_type', None)
    if out_type:
        return op.less_equal(children[0], children[1]).astype(out_type)
    else:
        return op.less_equal(children[0], children[1])


def _strided_slice(children, attrs, odtype='float32'):
    begin = attrs.get_int_list('begin')
    end = attrs.get_int_list('end')
    strides = attrs.get_int_list('strides', None)
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
    axis = None
    try:
        axis = [attrs.get_int('axis', None)]
    except ValueError:
        axis = axis or attrs.get_int_tuple('axis', None)

    return op.squeeze(children[0], axis)

def _concatenate(children, attrs, odtype='float32'):
    axis = attrs.get_int('axis', 1)
    return op.concatenate(children, axis)

def _dropout(children, attrs, odtype='float32'):
    rate = attrs.get_float('rate', 0.5)
    return op.nn.dropout(children[0], rate)

def _mean(children, attrs, odtype='float32'):
    axis = None
    try:
        axis = [attrs.get_int('axis', None)]
    except ValueError:
        axis = axis or attrs.get_int_tuple('axis', None)
    keepdims = attrs.get_bool('keepdims')

    return op.mean(children[0], axis, keepdims)


NNVM_OP_2_RELAY_OP = {
    'flatten': _nn_batch_flatten,
    'dense': _dense,
    'softmax': _nn_softmax,
    'conv2d': _conv2d,
    'batch_norm': _batch_norm,
    'max_pool2d': _max_pool2d,
    'reshape': _reshape,
    'transpose': _transpose,
    'dropout': _dropout,
    'mean': _mean,
    # Addition
    '__add_scalar__': _add,
    'broadcast_add': _add,
    'elemwise_add': _add,
    # Subtraction
    '__sub_scalar__': _subtract,
    '__rsub_scalar__': _rsubtract,
    'broadcast_sub': _subtract,
    'elemwise_sub': _subtract,
    # Multiply
    '__mul_scalar__': _multiply,
    'broadcast_mul': _multiply,
    'elemwise_mul': _multiply,
    # Division
    '__div_scalar__': _divide,
    'broadcast_div': _divide,
    'elemwise_div': _divide,
    # Negative
    'negative': _rename("negative"),

    # Comparsion
    'greater': _greater,
    'greater_equal': _greater_equal,
    'less': _less,
    'less_equal': _less_equal,

    # Activations
    'sigmoid': _rename('sigmoid'),
    'relu': _rename('nn.relu'),
    'exp': _rename('exp'),
    'log': _rename('log'),
    'tanh': _rename('tanh'),
    'leaky_relu': _leaky_relu,
    'clip': _clip,
    'round': _rename('round'),
    'cast': _cast,
    'expand_dims': _expand_dims,
    'broadcast_to': broadcast_to,
    '__rshift_scalar__': _rshift,
    'copy': _copy,
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
    heads = [x[0] for x in json.loads(graph.json())['heads']]

    gidx = graph.index
    relay_map = {}
    fn_params = []
    output_ids = []

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
            if nid in heads:
                output_ids.append(nid)

            if op_name in NNVM_OP_2_RELAY_OP:
                str_attrs = StrAttrsDict(attrs)
                call = NNVM_OP_2_RELAY_OP[op_name](children, str_attrs, odtype)
                relay_map[nid] = call
            else:
                raise Exception(
                    "nnvm.to_relay: unsupported operator: {0}".format(op_name))

    outputs = [relay_map[nid] for nid in output_ids]
    if len(outputs) == 1:
        body = outputs[0]
    else:
        body = expr.Tuple(outputs)

    func = relay.Function(fn_params, body)
    return func, params
