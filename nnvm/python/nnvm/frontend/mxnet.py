# pylint: disable=invalid-name
"""MXNet symbol frontend."""
from __future__ import absolute_import as _abs
import json
from .. import symbol as _sym

__all__ = ['from_mxnet']

def _required_attr(attr, key):
    assert isinstance(attr, dict)
    if key not in attr:
        raise AttributeError("Required attribute {} not found.".format(key))
    return attr[key]

def _raise_not_supported(attr, op='nnvm'):
    err = "{} is not supported in {}.".format(attr, op)
    raise NotImplementedError(err)

def _warn_not_used(attr, op='nnvm'):
    import warnings
    err = "{} is ignored in {}.".format(attr, op)
    warnings.warn(err)

def _parse_tshape(tshape):
    """Parse tshape in string."""
    return [int(x.strip()) for x in tshape.strip('()').split(',')]

def _parse_bool_str(attr, key, default='False'):
    """Parse bool string to boolean."""
    return attr.get(key, default).strip().lower() in ['true', '1', 't', 'y', 'yes']

def _rename(new_name):
    def impl(attr):
        return new_name, attr
    return impl

def _variable(attrs):
    return "Variable", attrs

def _pooling(attrs):
    kernel = _parse_tshape(_required_attr(attrs, 'kernel'))
    if len(kernel) != 2:
        _raise_not_supported('non-2d kernel', 'pool_2d')
    global_pool = 'global' if _parse_bool_str(attrs, 'global_pool') else ''
    pool_type = _required_attr(attrs, 'pool_type')
    if pool_type not in ['avg', 'max']:
        _raise_not_supported('non-avg/max', 'pool2d')
    op_name, new_attrs = '_'.join([global_pool, pool_type, 'pool2d']).strip('_'), {}
    # new_attrs['layout'] = 'NCHW'
    if not global_pool:
        new_attrs['pool_size'] = kernel
        new_attrs['strides'] = attrs.get('stride', (1, 1))
        new_attrs['padding'] = attrs.get('pad', (0, 0))
        new_attrs['ceil_mode'] = (attrs.get('pooling_convention', 'valid') == 'full')
    return op_name, new_attrs

def _batch_norm(attrs):
    if _parse_bool_str(attrs, 'output_mean_var'):
        _raise_not_supported('output_mean_var', 'batch_norm')
    if _parse_bool_str(attrs, 'fix_gamma'):
        _warn_not_used('fix_gamma', 'batch_norm')
    if _parse_bool_str(attrs, 'use_global_stats'):
        _warn_not_used('use_global_stats', 'batch_norm')
    if _parse_bool_str(attrs, 'momentum'):
        _warn_not_used('momentum', 'batch_norm')
    op_name, new_attrs = 'batch_norm', {}
    new_attrs['axis'] = attrs.get('axis', 1)
    new_attrs['epsilon'] = attrs.get('eps', 0.001)
    new_attrs['center'] = True
    new_attrs['scale'] = True
    return op_name, new_attrs

def _concat(attrs):
    op_name = 'concatenate'
    new_attrs = {'axis': attrs.get('dim', 1)}
    return op_name, new_attrs

def _conv2d(attrs):
    kernel = _parse_tshape(_required_attr(attrs, 'kernel'))
    if len(kernel) != 2:
        _raise_not_supported('non 2d kernel', 'conv2d')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        _raise_not_supported('layout: ' + layout, 'conv2d')
    op_name, new_attrs = 'conv2d', {}
    new_attrs['channels'] = _required_attr(attrs, 'num_filter')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['use_bias'] = attrs.get('no_bias', 'False').strip() == 'False'
    return op_name, new_attrs

def _conv2d_transpose(attrs):
    if 'target_shape' in attrs:
        _raise_not_supported('target_shape', 'conv2d_transpose')
    kernel = _parse_tshape(_required_attr(attrs, 'kernel'))
    if len(kernel) != 2:
        _raise_not_supported('non-2d kernel', 'conv2d_transpose')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        _raise_not_supported('layout: ' + layout, 'conv2d_transpose')
    op_name, new_attrs = 'conv2d_transpose', {}
    new_attrs['channels'] = _required_attr(attrs, 'num_filter')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['output_padding'] = attrs.get('adj', (0, 0))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['use_bias'] = not _parse_bool_str(attrs, 'no_bias')
    return op_name, new_attrs

def _dense(attrs):
    op_name, new_attrs = 'dense', {}
    new_attrs['units'] = _required_attr(attrs, 'num_hidden')
    new_attrs['use_bias'] = not _parse_bool_str(attrs, 'no_bias')
    return op_name, new_attrs

def _dropout(attrs):
    op_name, new_attrs = 'dropout', {}
    new_attrs['rate'] = attrs.get('p', 0.5)
    return op_name, new_attrs

def _leaky_relu(attrs):
    act_type = _required_attr(attrs, 'act_type')
    if act_type not in ['leaky']:
        _raise_not_supported('act_type: ' + act_type)
    op_name, new_attrs = 'leaky_relu', {}
    new_attrs['alpha'] = attrs.get('slope', 0.25)
    return op_name, new_attrs

def _activations(attrs):
    act_type = _required_attr(attrs, 'act_type')
    if act_type not in ['relu', 'sigmoid', 'tanh']:
        _raise_not_supported('act_type: ' + act_type)
    op_name, new_attrs = act_type, {}
    return op_name, new_attrs

def _reshape(attrs):
    if _parse_bool_str(attrs, 'reverse'):
        _raise_not_supported('reverse', 'reshape')
    op_name, new_attrs = 'reshape', {}
    new_attrs['shape'] = _required_attr(attrs, 'shape')
    return op_name, new_attrs

def _split(attrs):
    if _parse_bool_str(attrs, 'squeeze_axis'):
        _raise_not_supported('squeeze_axis', 'split')
    op_name, new_attrs = 'split', {}
    new_attrs['indices_or_sections'] = _required_attr(attrs, 'num_outputs')
    new_attrs['axis'] = attrs.get('axis', 1)
    return op_name, new_attrs

_identity_list = ['__add_scalar__', '__add_symbol__', '__div_scalar__',
                  '__div_symbol__', '__mul_scalar__', '__mul_symbol__',
                  '__pow_scalar__', '__rdiv_scalar__', '__rpow_scalar__',
                  '__rsub_scalar__', '__sub_scalar__', '__sub_symbol__',
                  'broadcast_add', 'broadcast_div', 'broadcast_mul',
                  'broadcast_sub', 'broadcast_to', 'cast', 'elemwise_add',
                  'elemwise_div', 'elemwise_mul', 'elemwise_sub', 'exp',
                  'flatten', 'log', 'log_softmax', 'max', 'min', 'negative',
                  'relu', 'sigmoid', 'softmax', 'sum', 'tanh', 'transpose']

_convert_map = {
    'null'          : _variable,
    'Activation'    : _activations,
    'BatchNorm'     : _batch_norm,
    'BatchNorm_v1'  : _batch_norm,
    'Cast'          : _rename('cast'),
    'Concat'        : _concat,
    'Convolution'   : _conv2d,
    'Convolution_v1': _conv2d,
    'Deconvolution' : _conv2d_transpose,
    'Dropout'       : _dropout,
    'Flatten'       : _rename('flatten'),
    'FullyConnected': _dense,
    'LeakyReLU'     : _leaky_relu,
    'Pooling'       : _pooling,
    'Pooling_v1'    : _pooling,
    'Reshape'       : _reshape,
    'Softmax'       : _rename('softmax'),
    'concat'        : _concat,
    'max_axis'      : _rename('max'),
    'min_axis'      : _rename('min'),
    'reshape'       : _reshape,
    'sum_axis'      : _rename('sum'),
}

def _convert_symbol(op_name, attrs,
                    identity_list=None,
                    convert_map=None):
    """Convert from mxnet op to nnvm op.
    The converter must specify some conversions explicitly to
    support gluon format ops such as conv2d...

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, FullyConnected
    attrs : dict
        Dict of operator attributes
    identity_list : list
        List of operators that don't require conversion
    convert_map : dict
        Dict of name : callable, where name is the op's name that
        require conversion to nnvm, callable are functions which
        take attrs and return (new_op_name, new_attrs)

    Returns
    -------
    (op_name, attrs)
        Converted (op_name, attrs) for nnvm.
    """
    identity_list = identity_list if identity_list else _identity_list
    convert_map = convert_map if convert_map else _convert_map
    if op_name in identity_list:
        pass
    elif op_name in convert_map:
        op_name, attrs = convert_map[op_name](attrs)
    else:
        _raise_not_supported('Operator: ' + op_name)
    op = getattr(_sym, op_name, None)
    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op, attrs

def _is_mxnet_group_symbol(symbol):
    """Internal check for mxnet group symbol."""
    return len(symbol.list_outputs()) > 1

def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]

def _from_mxnet_impl(symbol, graph):
    """Convert mxnet symbol to nnvm implementation.
    Reconstruct a nnvm symbol by traversing the mxnet symbol.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol
        Incompatible symbol from mxnet, sharing similar graph structure.
        The op_name and attrs inside are not always compatible.
    graph : dict
        Reusable nodes are stored in graph.

    Returns:
    -------
    nnvm.sym.Symbol
        Converted symbol
    """
    try:
        from mxnet import sym as mx_sym  # pylint: disable=import-self
    except ImportError as e:
        raise ImportError('{}. MXNet is required to parse symbols.'.format(e))

    if not isinstance(symbol, mx_sym.Symbol):
        raise ValueError("Provided {}, while MXNet symbol is expected", type(symbol))

    if _is_mxnet_group_symbol(symbol):
        return [_from_mxnet_impl(s, graph) for s in symbol]

    name = symbol.attr('name')
    node = graph.get(name, None)
    if node:
        return node
    # op_name = symbol.attr('op_name')
    if symbol.get_children():
        op_name = symbol.attr('op_name')
    else:
        op_name = json.loads(symbol.tojson())['nodes'][0]['op']
    attr = symbol.list_attr()
    new_op, new_attr = _convert_symbol(op_name, attr)
    if new_op == _sym.Variable:
        node = new_op(name=name, **new_attr)
    else:
        childs = symbol.get_children()
        childs = [_from_mxnet_impl(c, graph) for c in _as_list(childs)]
        childs = [x for y in childs for x in _as_list(y)]  # expand group symbol
        node = new_op(name=name, *childs, **new_attr)
    graph[name] = node
    return node


def from_mxnet(symbol):
    """Convert from mxnet.Symbol to compatible nnvm.Symbol

    Parameters
    ----------
    symbol : mxnet.Symbol
        MXNet symbol

    Returns
    -------
    nnvm.Symbol
        Compatible nnvm symbol
    """
    return _from_mxnet_impl(symbol, {})
