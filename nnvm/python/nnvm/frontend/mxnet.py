# pylint: disable=invalid-name, import-self
"""MXNet symbol frontend."""
from __future__ import absolute_import as _abs
import json
import tvm
from .. import symbol as _sym

__all__ = ['from_mxnet']

def _get_nnvm_op(op_name):
    op = getattr(_sym, op_name)
    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op

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
    def impl(inputs, attrs):
        return _get_nnvm_op(new_name)(*inputs, **attrs)
    return impl

def _pooling(inputs, attrs):
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
        if pool_type == 'avg':
            new_attrs['count_include_pad'] = attrs.get('count_include_pad', True)
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _batch_norm(inputs, attrs):
    if _parse_bool_str(attrs, 'output_mean_var'):
        _raise_not_supported('output_mean_var', 'batch_norm')
    # if _parse_bool_str(attrs, 'fix_gamma'):
    #     _warn_not_used('fix_gamma', 'batch_norm')
    if _parse_bool_str(attrs, 'use_global_stats'):
        _warn_not_used('use_global_stats', 'batch_norm')
    # if _parse_bool_str(attrs, 'momentum'):
    #     _warn_not_used('momentum', 'batch_norm')
    op_name, new_attrs = 'batch_norm', {}
    new_attrs['axis'] = attrs.get('axis', 1)
    new_attrs['epsilon'] = attrs.get('eps', 0.001)
    new_attrs['center'] = True
    new_attrs['scale'] = not _parse_bool_str(attrs, 'fix_gamma', default="False")
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _concat(inputs, attrs):
    op_name = 'concatenate'
    new_attrs = {'axis': attrs.get('dim', 1)}
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _conv2d(inputs, attrs):
    kernel = _parse_tshape(_required_attr(attrs, 'kernel'))
    if len(kernel) != 2:
        _raise_not_supported('non 2d kernel', 'conv2d')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        _raise_not_supported('layout: ' + layout, 'conv2d')
    if 'kernel_layout' in attrs:
        kernel_layout = attrs['kernel_layout']
    else:
        kernel_layout = 'HWIO' if layout == 'NHWC' else 'OIHW'
    op_name, new_attrs = 'conv2d', {}
    new_attrs['channels'] = _required_attr(attrs, 'num_filter')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['kernel_layout'] = kernel_layout
    new_attrs['use_bias'] = attrs.get('no_bias', 'False').strip() == 'False'
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _conv2d_transpose(inputs, attrs):
    if 'target_shape' in attrs:
        _raise_not_supported('target_shape', 'conv2d_transpose')
    kernel = _parse_tshape(_required_attr(attrs, 'kernel'))
    if len(kernel) != 2:
        _raise_not_supported('non-2d kernel', 'conv2d_transpose')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        _raise_not_supported('layout: ' + layout, 'conv2d_transpose')
    if 'kernel_layout' in attrs:
        kernel_layout = attrs['kernel_layout']
    else:
        kernel_layout = 'HWIO' if layout == 'NHWC' else 'OIHW'
    op_name, new_attrs = 'conv2d_transpose', {}
    new_attrs['channels'] = _required_attr(attrs, 'num_filter')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['output_padding'] = attrs.get('adj', (0, 0))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['kernel_layout'] = kernel_layout
    new_attrs['use_bias'] = not _parse_bool_str(attrs, 'no_bias')
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _dense(inputs, attrs):
    import mxnet as mx
    op_name, new_attrs = 'dense', {}
    new_attrs['units'] = _required_attr(attrs, 'num_hidden')
    new_attrs['use_bias'] = not _parse_bool_str(attrs, 'no_bias')
    try:
        _ = mx.sym.FullyConnected(mx.sym.var('x'), num_hidden=1, flatten=True)
        has_flatten = True
    except mx.base.MXNetError:
        # no flatten attribute in old mxnet
        has_flatten = False
    use_flatten = _parse_bool_str(attrs, 'flatten', 'True')
    if has_flatten and use_flatten:
        inputs[0] = _sym.flatten(inputs[0])
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _dropout(inputs, attrs):
    op_name, new_attrs = 'dropout', {}
    new_attrs['rate'] = attrs.get('p', 0.5)
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _leaky_relu(inputs, attrs):
    act_type = _required_attr(attrs, 'act_type')
    if act_type in ['leaky', 'prelu']:
        op_name, new_attrs = act_type, {}
        if act_type == 'leaky':
            new_attrs['alpha'] = attrs.get('slope', 0.25)
        sym = _get_nnvm_op(op_name)(*inputs, **new_attrs)
    elif act_type == 'elu':
        slope = attrs.get('slope', 0.25)
        sym = -slope * _sym.relu(1 - _sym.exp(*inputs)) + _sym.relu(*inputs)
    elif act_type == 'rrelu':
        lower_bound = float(_required_attr(attrs, 'lower_bound'))
        upper_bound = float(_required_attr(attrs, 'upper_bound'))
        slope = (lower_bound + upper_bound) / 2.0
        op_name, new_attrs = 'leaky_relu', {'alpha': str(slope)}
        sym = _get_nnvm_op(op_name)(*inputs, **new_attrs)
    else:
        _raise_not_supported('act_type: ' + act_type)
    return sym

def _activations(inputs, attrs):
    act_type = _required_attr(attrs, 'act_type')
    if act_type in ['relu', 'sigmoid', 'tanh']:
        op_name, new_attrs = act_type, {}
        sym = _get_nnvm_op(op_name)(*inputs, **new_attrs)
    elif act_type == 'softrelu':
        sym = _sym.log((1 + _sym.exp(*inputs)))
    else:
        _raise_not_supported('act_type: ' + act_type)
    return sym

def _reshape(inputs, attrs):
    if _parse_bool_str(attrs, 'reverse'):
        _raise_not_supported('reverse', 'reshape')
    op_name, new_attrs = 'reshape', {}
    new_attrs['shape'] = _required_attr(attrs, 'shape')
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _split(inputs, attrs):
    op_name, new_attrs = 'split', {}
    axis = attrs.get('axis', 1)
    new_attrs['indices_or_sections'] = _required_attr(attrs, 'num_outputs')
    new_attrs['axis'] = axis
    outputs = _get_nnvm_op(op_name)(*inputs, **new_attrs)
    if _parse_bool_str(attrs, 'squeeze_axis'):
        squeeze_attrs = {'axis': axis}
        outputs = _sym.Group([_get_nnvm_op('squeeze')(o, **squeeze_attrs) for o in outputs])
    return outputs

def _softmax_activation(inputs, attrs):
    op_name, new_attrs = 'softmax', {}
    mode = attrs.get('mode', 'instance')
    new_attrs['axis'] = 0 if mode == 'instance' else 1
    return _get_nnvm_op(op_name)(inputs[0], **new_attrs)

def _softmax_output(inputs, attrs):
    op_name, new_attrs = 'softmax', {}
    if _parse_bool_str(attrs, 'multi_output'):
        new_attrs['axis'] = 1
    return _get_nnvm_op(op_name)(inputs[0], **new_attrs)

def _upsampling(inputs, attrs):
    scale = attrs.get('scale')
    new_attrs = {'scale':int(scale)}
    return _get_nnvm_op('upsampling')(inputs[0], **new_attrs)

def _clip(inputs, attrs):
    op_name, new_attrs = "clip", {}
    new_attrs['a_min'] = _required_attr(attrs, 'a_min')
    new_attrs['a_max'] = _required_attr(attrs, 'a_max')
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _contrib_multibox_detection(inputs, attrs):
    clip = _parse_bool_str(attrs, 'clip', default='True')
    threshold = attrs.get('threshold') or 0.01
    nms_threshold = attrs.get('nms_threshold') or 0.5
    force_suppress = _parse_bool_str(attrs, 'force_suppress', default='False')
    variances = tuple([float(x.strip()) for x in attrs.get('variances').strip('()').split(',')]) \
        if attrs.get('variances') is not None else (0.1, 0.1, 0.2, 0.2)
    nms_topk = attrs.get('nms_topk') or -1
    new_attrs0 = {'clip': clip, 'threshold': float(threshold), 'variances': variances}
    new_attrs1 = {'nms_threshold': float(nms_threshold), 'force_suppress': force_suppress,
                  'nms_topk': int(nms_topk)}
    data, valid_count = _get_nnvm_op('multibox_transform_loc')(inputs[0], inputs[1],
                                                               inputs[2], **new_attrs0)
    return _get_nnvm_op('nms')(data, valid_count, **new_attrs1)

def _elemwise_sum(inputs, _):
    new_attrs = {'num_args':len(inputs)}
    return _get_nnvm_op('elemwise_sum')(*inputs, **new_attrs)

def _crop_like(inputs, attrs):
    new_attrs = {}
    offsets = \
        tuple([float(x.strip()) for x in attrs.get('offsets').strip('()').split(',')]) \
            if attrs.get('offsets') is not None else (0, 0)
    if offsets != (0, 0):
        raise RuntimeError("Currently only supports offsets to be zero.")
    center_crop = _parse_bool_str(attrs, 'center_crop', default="False")
    if center_crop:
        raise RuntimeError("center crop is not supported.")
    if len(inputs) < 2:
        raise RuntimeError("Only support crop_like pattern.")
    new_attrs["axis"] = [2, 3]
    return _get_nnvm_op('slice_like')(inputs[0], inputs[1], **new_attrs)


def _expand_dims(inputs, attrs):
    op_name, new_attrs = "expand_dims", {}
    new_attrs['axis'] = _required_attr(attrs, 'axis')
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

def _lrn(inputs, attrs):
    op_name, new_attrs = "lrn", {}
    new_attrs['alpha'] = attrs.get('alpha', 0.0001)
    new_attrs['beta'] = attrs.get('beta', 0.75)
    new_attrs['bias'] = attrs.get('knorm', 2)
    # NCHW format and normalization along channel axis
    new_attrs['axis'] = 1
    new_attrs['size'] = _required_attr(attrs, 'nsize')
    return _get_nnvm_op(op_name)(*inputs, **new_attrs)

_identity_list = ['__add_scalar__', '__add_symbol__', '__div_scalar__',
                  '__div_symbol__', '__mul_scalar__', '__mul_symbol__',
                  '__pow_scalar__', '__rdiv_scalar__', '__rpow_scalar__',
                  '__rsub_scalar__', '__sub_scalar__', '__sub_symbol__',
                  'broadcast_add', 'broadcast_div', 'broadcast_mul',
                  'broadcast_sub', 'broadcast_to', 'cast', 'elemwise_add',
                  'elemwise_div', 'elemwise_mul', 'elemwise_sub', 'exp',
                  'flatten', 'log', 'log_softmax', 'max', 'min', 'negative',
                  'relu', 'sigmoid', 'slice_like', 'softmax', 'sum', 'tanh',
                  'transpose']

_convert_map = {
    '_copy'         : _rename('copy'),
    '_div_scalar'   : _rename('__div_scalar__'),
    '_minus_scalar' : _rename('__sub_scalar__'),
    '_mul_scalar'   : _rename('__mul_scalar__'),
    '_plus_scalar'  : _rename('__add_scalar__'),
    '_rdiv_scalar'  : _rename('__rdiv_scalar__'),
    '_rminus_scalar': _rename('__rsub_scalar__'),
    '_contrib_MultiBoxPrior' : _rename('multibox_prior'),
    '_contrib_MultiBoxDetection' : _contrib_multibox_detection,
    'Activation'    : _activations,
    'BatchNorm'     : _batch_norm,
    'BatchNorm_v1'  : _batch_norm,
    'Cast'          : _rename('cast'),
    'Concat'        : _concat,
    'Convolution'   : _conv2d,
    'Convolution_v1': _conv2d,
    'Crop'          : _crop_like,
    'Deconvolution' : _conv2d_transpose,
    'Dropout'       : _dropout,
    'Flatten'       : _rename('flatten'),
    'FullyConnected': _dense,
    'LeakyReLU'     : _leaky_relu,
    'Pooling'       : _pooling,
    'Pooling_v1'    : _pooling,
    'Reshape'       : _reshape,
    'SliceChannel'  : _split,
    'split'         : _split,
    'Softmax'       : _rename('softmax'),
    'SoftmaxActivation' : _softmax_activation,
    'SoftmaxOutput' : _softmax_output,
    'add_n'         : _elemwise_sum,
    'concat'        : _concat,
    'max_axis'      : _rename('max'),
    'min_axis'      : _rename('min'),
    'reshape'       : _reshape,
    'sum_axis'      : _rename('sum'),
    'UpSampling'    : _upsampling,
    'clip'          : _clip,
    'expand_dims'   : _expand_dims,
    'LRN'           : _lrn
}

def _convert_symbol(op_name, inputs, attrs,
                    identity_list=None,
                    convert_map=None):
    """Convert from mxnet op to nnvm op.
    The converter must specify some conversions explicitly to
    support gluon format ops such as conv2d...

    Parameters
    ----------
    op_name : str
        Operator name, such as Convolution, FullyConnected
    inputs : list of nnvm.Symbol
        List of input symbols.
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
    sym : nnvm.Symbol
        Converted nnvm Symbol
    """
    identity_list = identity_list if identity_list else _identity_list
    convert_map = convert_map if convert_map else _convert_map
    if op_name in identity_list:
        op = _get_nnvm_op(op_name)
        sym = op(*inputs, **attrs)
    elif op_name in convert_map:
        sym = convert_map[op_name](inputs, attrs)
    else:
        _raise_not_supported('Operator: ' + op_name)
    return sym

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
    if len(symbol.list_outputs()) > 1:
        return [_from_mxnet_impl(s, graph) for s in symbol]

    name = symbol.attr('name')
    output_index = json.loads(symbol.tojson())['heads'][0][1]
    node = graph.get(name, None)
    if node:
        return node[output_index]
    attr = symbol.list_attr()
    # op_name = symbol.attr('op_name')
    childs = symbol.get_children()
    if childs is not None:
        op_name = symbol.attr('op_name')
        childs = [_from_mxnet_impl(childs[i], graph) for i in range(len(childs.list_outputs()))]
        childs = [x for y in childs for x in _as_list(y)]  # expand group symbol
        node = _convert_symbol(op_name, childs, attr)
    else:
        op_name = json.loads(symbol.tojson())['nodes'][0]['op']
        node = _sym.Variable(name=name, **attr)
    graph[name] = node
    return node[output_index]

def from_mxnet(symbol, arg_params=None, aux_params=None):
    """Convert from MXNet's model into compatible NNVM format.

    Parameters
    ----------
    symbol : mxnet.Symbol or mxnet.gluon.HybridBlock
        MXNet symbol

    arg_params : dict of str to mx.NDArray
        The argument parameters in mxnet

    aux_params : dict of str to mx.NDArray
        The auxiliary parameters in mxnet

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import mxnet as mx
    except ImportError as e:
        raise ImportError('{}. MXNet is required to parse symbols.'.format(e))

    if isinstance(symbol, mx.sym.Symbol):
        sym = _from_mxnet_impl(symbol, {})
        params = {}
        arg_params = arg_params if arg_params else {}
        aux_params = aux_params if aux_params else {}
        for k, v in arg_params.items():
            params[k] = tvm.nd.array(v.asnumpy())
        for k, v in aux_params.items():
            params[k] = tvm.nd.array(v.asnumpy())
    elif isinstance(symbol, mx.gluon.HybridBlock):
        data = mx.sym.Variable('data')
        sym = symbol(data)
        sym = _from_mxnet_impl(sym, {})
        params = {}
        for k, v in symbol.collect_params().items():
            params[k] = tvm.nd.array(v.data().asnumpy())
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    if isinstance(sym, list):
        sym = _sym.Group(sym)
    return sym, params
