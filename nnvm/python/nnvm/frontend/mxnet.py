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
# pylint: disable=invalid-name, import-self
"""MXNet symbol frontend."""
from __future__ import absolute_import as _abs
import json
import tvm
from .. import symbol as _sym
from .common import get_nnvm_op, required_attr, parse_tshape, parse_bool_str

__all__ = ['from_mxnet']

def _rename(new_name):
    def impl(inputs, attrs):
        return get_nnvm_op(new_name)(*inputs, **attrs)
    return impl

def _pooling(inputs, attrs):
    kernel = parse_tshape(required_attr(attrs, 'kernel', 'pooling'))
    if len(kernel) != 2:
        raise tvm.error.OpAttributeUnimplemented(
            'Non-2D kernels are not supported for Pool2D.')
    global_pool = 'global' if parse_bool_str(attrs, 'global_pool') else ''
    pool_type = required_attr(attrs, 'pool_type', 'pooling')
    if pool_type not in ['avg', 'max']:
        raise tvm.error.OpNotImplemented(
            'Only max and average pooling are supported in frontend MXNet.')
    op_name, new_attrs = '_'.join([global_pool, pool_type, 'pool2d']).strip('_'), {}
    # new_attrs['layout'] = 'NCHW'
    if not global_pool:
        new_attrs['pool_size'] = kernel
        new_attrs['strides'] = attrs.get('stride', (1, 1))
        new_attrs['padding'] = attrs.get('pad', (0, 0))
        new_attrs['ceil_mode'] = (attrs.get('pooling_convention', 'valid') == 'full')
        if pool_type == 'avg':
            new_attrs['count_include_pad'] = attrs.get('count_include_pad', True)
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _batch_norm(inputs, attrs):
    if parse_bool_str(attrs, 'output_mean_var'):
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "output_mean_var" is not supported in operator batch_norm.')
    # if parse_bool_str(attrs, 'fix_gamma'):
    #     _warn_not_used('fix_gamma', 'batch_norm')
    if parse_bool_str(attrs, 'use_global_stats'):
        from warnings import warn
        warn(
            'Attribute "use_global_stats" is ignored in operator batch_norm.')
    # if parse_bool_str(attrs, 'momentum'):
    #     _warn_not_used('momentum', 'batch_norm')
    op_name, new_attrs = 'batch_norm', {}
    new_attrs['axis'] = attrs.get('axis', 1)
    new_attrs['epsilon'] = attrs.get('eps', 0.001)
    new_attrs['center'] = True
    new_attrs['scale'] = not parse_bool_str(attrs, 'fix_gamma', default="False")
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _concat(inputs, attrs):
    op_name = 'concatenate'
    new_attrs = {'axis': attrs.get('dim', 1)}
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _conv2d(inputs, attrs):
    kernel = parse_tshape(required_attr(attrs, 'kernel', 'conv2d'))
    if len(kernel) != 2:
        raise tvm.error.OpAttributeUnimplemented(
            'Non-2D kernels are not supported for operator Conv2D.')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        raise tvm.error.OpAttributeUnimplemented(
            'Layout {} is not supported in operator Conv2D.'.format(layout))
    if 'kernel_layout' in attrs:
        kernel_layout = attrs['kernel_layout']
    else:
        kernel_layout = 'HWIO' if layout == 'NHWC' else 'OIHW'
    op_name, new_attrs = 'conv2d', {}
    new_attrs['channels'] = required_attr(attrs, 'num_filter', 'conv2d')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['kernel_layout'] = kernel_layout
    new_attrs['use_bias'] = attrs.get('no_bias', 'False').strip() == 'False'
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _conv2d_transpose(inputs, attrs):
    if 'target_shape' in attrs:
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "target_shape" is not supported in operator Conv2D-transpose.')
    kernel = parse_tshape(required_attr(attrs, 'kernel', 'conv2d_transpose'))
    if len(kernel) != 2:
        raise tvm.error.OpAttributeInvalid(
            'Non-2D kernels are not supported in Conv2D-transpose.')
    layout = attrs.get('layout', 'NCHW')
    if layout not in ['NCHW', 'NHWC']:
        raise tvm.error.OpAttributeUnimplemented(
            'Layout {} is not supported in operator Conv2D-transpose.')
    if 'kernel_layout' in attrs:
        kernel_layout = attrs['kernel_layout']
    else:
        kernel_layout = 'HWIO' if layout == 'NHWC' else 'OIHW'
    op_name, new_attrs = 'conv2d_transpose', {}
    new_attrs['channels'] = required_attr(attrs, 'num_filter', 'conv2d_transpose')
    new_attrs['kernel_size'] = kernel
    new_attrs['strides'] = attrs.get('stride', (1, 1))
    new_attrs['output_padding'] = attrs.get('adj', (0, 0))
    new_attrs['padding'] = attrs.get('pad', (0, 0))
    new_attrs['dilation'] = attrs.get('dilate', (1, 1))
    new_attrs['groups'] = attrs.get('num_group', 1)
    new_attrs['layout'] = layout
    new_attrs['kernel_layout'] = kernel_layout
    new_attrs['use_bias'] = not parse_bool_str(attrs, 'no_bias')
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _dense(inputs, attrs):
    import mxnet as mx
    op_name, new_attrs = 'dense', {}
    new_attrs['units'] = required_attr(attrs, 'num_hidden', 'dense')
    new_attrs['use_bias'] = not parse_bool_str(attrs, 'no_bias')
    try:
        _ = mx.sym.FullyConnected(mx.sym.var('x'), num_hidden=1, flatten=True)
        has_flatten = True
    except mx.base.MXNetError:
        # no flatten attribute in old mxnet
        has_flatten = False
    use_flatten = parse_bool_str(attrs, 'flatten', 'True')
    if has_flatten and use_flatten:
        inputs[0] = _sym.flatten(inputs[0])
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _dropout(inputs, attrs):
    op_name, new_attrs = 'dropout', {}
    new_attrs['rate'] = attrs.get('p', 0.5)
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _leaky_relu(inputs, attrs):
    act_type = required_attr(attrs, 'act_type', 'leaky_relu')
    if act_type in ['leaky', 'prelu']:
        op_name, new_attrs = act_type, {}
        if act_type == 'leaky':
            new_attrs['alpha'] = attrs.get('slope', 0.25)
        sym = get_nnvm_op(op_name)(*inputs, **new_attrs)
    elif act_type == 'elu':
        slope = attrs.get('slope', 0.25)
        sym = -slope * _sym.relu(1 - _sym.exp(*inputs)) + _sym.relu(*inputs)
    elif act_type == 'rrelu':
        lower_bound = float(required_attr(attrs, 'lower_bound', 'leaky_relu'))
        upper_bound = float(required_attr(attrs, 'upper_bound', 'leaky_relu'))
        slope = (lower_bound + upper_bound) / 2.0
        op_name, new_attrs = 'leaky_relu', {'alpha': str(slope)}
        sym = get_nnvm_op(op_name)(*inputs, **new_attrs)
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported in frontend MXNet.'.format(act_type))
    return sym

def _activations(inputs, attrs):
    act_type = required_attr(attrs, 'act_type', 'activations')
    if act_type in ['relu', 'sigmoid', 'tanh']:
        op_name, new_attrs = act_type, {}
        sym = get_nnvm_op(op_name)(*inputs, **new_attrs)
    elif act_type == 'softrelu':
        sym = _sym.log((1 + _sym.exp(*inputs)))
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported in frontend MXNet.'.format(act_type))
    return sym

def _reshape(inputs, attrs):
    if parse_bool_str(attrs, 'reverse'):
        raise tvm.error.OpAttributeUnimplemented(
            'Attribute "reverse" is not supported in operator Reshape.')
    op_name, new_attrs = 'reshape', {}
    new_attrs['shape'] = required_attr(attrs, 'shape', 'reshape')
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _slice(inputs, attrs):
    begin = attrs.get('begin', None)
    end = attrs.get('end', None)
    stride = attrs.get('step', None)
    if begin is None or end is None:
        raise RuntimeError('begin and end are required params')
    if 'None' in begin or 'None' in end:
        raise RuntimeError('None in begin or end not supported yet...')
    new_attrs = {'begin': begin, 'end': end}
    if stride is not None:
        new_attrs['stride'] = stride
    return get_nnvm_op('strided_slice')(inputs[0], **new_attrs)

def _split(inputs, attrs):
    op_name, new_attrs = 'split', {}
    axis = attrs.get('axis', 1)
    new_attrs['indices_or_sections'] = required_attr(attrs, 'num_outputs', 'split')
    new_attrs['axis'] = axis
    outputs = get_nnvm_op(op_name)(*inputs, **new_attrs)
    if parse_bool_str(attrs, 'squeeze_axis'):
        squeeze_attrs = {'axis': axis}
        outputs = _sym.Group([get_nnvm_op('squeeze')(o, **squeeze_attrs) for o in outputs])
    return outputs

def _softmax_activation(inputs, attrs):
    op_name, new_attrs = 'softmax', {}
    mode = attrs.get('mode', 'instance')
    new_attrs['axis'] = 0 if mode == 'instance' else 1
    return get_nnvm_op(op_name)(inputs[0], **new_attrs)

def _softmax_output(inputs, attrs):
    op_name, new_attrs = 'softmax', {}
    if parse_bool_str(attrs, 'multi_output'):
        new_attrs['axis'] = 1
    return get_nnvm_op(op_name)(inputs[0], **new_attrs)

def _upsampling(inputs, attrs):
    scale = attrs.get('scale')
    new_attrs = {'scale':int(scale)}
    return get_nnvm_op('upsampling')(inputs[0], **new_attrs)

def _clip(inputs, attrs):
    op_name, new_attrs = "clip", {}
    new_attrs['a_min'] = required_attr(attrs, 'a_min', 'clip')
    new_attrs['a_max'] = required_attr(attrs, 'a_max', 'clip')
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _contrib_multibox_detection(inputs, attrs):
    clip = parse_bool_str(attrs, 'clip', default='True')
    threshold = attrs.get('threshold') or 0.01
    nms_threshold = attrs.get('nms_threshold') or 0.5
    force_suppress = parse_bool_str(attrs, 'force_suppress', default='False')
    variances = tuple([float(x.strip()) for x in attrs.get('variances').strip('()').split(',')]) \
        if attrs.get('variances') is not None else (0.1, 0.1, 0.2, 0.2)
    nms_topk = attrs.get('nms_topk') or -1
    new_attrs0 = {'clip': clip, 'threshold': float(threshold), 'variances': variances}
    new_attrs1 = {'return_indices': False, 'iou_threshold': float(nms_threshold),
                  'force_suppress': force_suppress, 'top_k': int(nms_topk)}
    data, valid_count = get_nnvm_op('multibox_transform_loc')(inputs[0], inputs[1],
                                                              inputs[2], **new_attrs0)
    return get_nnvm_op('non_max_suppression')(data, valid_count, **new_attrs1)

def _elemwise_sum(inputs, _):
    new_attrs = {'num_args':len(inputs)}
    return get_nnvm_op('elemwise_sum')(*inputs, **new_attrs)

def _crop_like(inputs, attrs):
    new_attrs = {}
    offsets = \
        tuple([float(x.strip()) for x in attrs.get('offsets').strip('()').split(',')]) \
            if attrs.get('offsets') is not None else (0, 0)
    if offsets != (0, 0):
        raise tvm.error.OpAttributeInvalid(
            'crop_like offsets must equal (0,0).')
    center_crop = parse_bool_str(attrs, 'center_crop', default="False")
    if center_crop:
        raise tvm.error.OpAttributeUnimplemented(
            'Center crop is not supported in operator crop_like.')
    if len(inputs) < 2:
        raise tvm.error.OpAttributeUnimplemented("Only support crop_like pattern.")
    new_attrs["axis"] = [2, 3]
    return get_nnvm_op('slice_like')(inputs[0], inputs[1], **new_attrs)


def _expand_dims(inputs, attrs):
    op_name, new_attrs = 'expand_dims', {}
    new_attrs['axis'] = required_attr(attrs, 'axis', 'expand_dims')
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _lrn(inputs, attrs):
    op_name, new_attrs = 'lrn', {}
    new_attrs['alpha'] = attrs.get('alpha', 0.0001)
    new_attrs['beta'] = attrs.get('beta', 0.75)
    new_attrs['bias'] = attrs.get('knorm', 2)
    # NCHW format and normalization along channel axis
    new_attrs['axis'] = 1
    new_attrs['size'] = required_attr(attrs, 'nsize', 'lrn')
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _minimum(inputs, attrs):
    return get_nnvm_op('broadcast_min')(*inputs, **attrs)

def _maximum(inputs, attrs):
    return get_nnvm_op('broadcast_max')(*inputs, **attrs)

def _ones(_, attrs):
    op_name = 'ones'
    return get_nnvm_op(op_name)(**attrs)

def _zeros(_, attrs):
    op_name = 'zeros'
    return get_nnvm_op(op_name)(**attrs)

def _argmax(inputs, attrs):
    op_name, new_attrs = 'argmax', {}
    new_attrs['dtype'] = 'float32'
    new_attrs['axis'] = attrs.get('axis', 0)
    new_attrs['keepdims'] = parse_bool_str(attrs, 'keepdims', default="False")
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

def _argmin(inputs, attrs):
    op_name, new_attrs = 'argmin', {}
    new_attrs['dtype'] = 'float32'
    new_attrs['axis'] = attrs.get('axis', 0)
    new_attrs['keepdims'] = parse_bool_str(attrs, 'keepdims', default="False")
    return get_nnvm_op(op_name)(*inputs, **new_attrs)

_identity_list = ['__add_scalar__', '__add_symbol__', '__div_scalar__',
                  '__div_symbol__', '__mul_scalar__', '__mul_symbol__',
                  '__pow_scalar__', '__rdiv_scalar__', '__rpow_scalar__',
                  '__rsub_scalar__', '__sub_scalar__', '__sub_symbol__',
                  'broadcast_add', 'broadcast_div', 'broadcast_mul',
                  'broadcast_sub', 'broadcast_to', 'cast', 'elemwise_add',
                  'elemwise_div', 'elemwise_mul', 'elemwise_sub', 'exp',
                  'flatten', 'log', 'log_softmax', 'max', 'min', 'negative',
                  'ones_like', 'relu', 'sigmoid', 'slice_like', 'softmax',
                  'sum', 'tanh', 'transpose', 'zeros_like', 'gather_nd',
                  'reshape_like', 'where']

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
    '_minimum'      : _minimum,
    '_maximum'      : _maximum,
    '_ones'         : _ones,
    '_zeros'        : _zeros,
    'argmax'        : _argmax,
    'argmin'        : _argmin,
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
    'slice'         : _slice,
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
        op = get_nnvm_op(op_name)
        sym = op(*inputs, **attrs)
    elif op_name in convert_map:
        sym = convert_map[op_name](inputs, attrs)
    else:
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported in frontend MXNet.'.format(op_name))
    return sym

def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]

def _topo_sort(symbol):
    """Sort all symbols in the mxnet graph in topological order.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol

    Returns:
    -------
    list
        List of mxnet symbol
    """
    queue = []
    symbol_map = {}
    deps = {}
    dep_cnts = {}
    for s in symbol:
        symbol_map[s.attr('name')] = s
        queue.append(s)
    while queue:
        sym = queue.pop(0)
        name = sym.attr('name')
        childs = sym.get_children()
        if childs is None:
            dep_cnts[name] = 0
        else:
            dep_cnts[name] = len({c.attr('name') for c in childs})
            for child in childs:
                child_name = child.attr('name')
                if child_name not in deps:
                    deps[child_name] = set()
                deps[child_name].add(name)
                if child_name not in symbol_map:
                    symbol_map[child_name] = child
                    queue.append(child)
    order = []
    while dep_cnts:
        remove = []
        for name in dep_cnts:
            if dep_cnts[name] == 0:
                order.append(symbol_map[name])
                remove.append(name)
                if name in deps:
                    for other in deps[name]:
                        dep_cnts[other] -= 1
        for name in remove:
            del dep_cnts[name]
    return order

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
    def get_node(sym):
        name = sym.attr('name')
        if name not in graph:
            return None
        output_index = json.loads(sym.tojson())['heads'][0][1]
        return graph[name][output_index]

    assert symbol is not None
    # Traverse all symbols in topological order
    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()
        if childs is not None:
            childs = [get_node(child) for child in childs]
            childs = [x for y in childs for x in _as_list(y)]
            node = _convert_symbol(op_name, childs, attr)
        elif op_name != 'null':
            node = _convert_symbol(op_name, [], attr)
        else:
            node = _sym.Variable(name=name, **attr)
        graph[name] = node
    nodes = []
    for sym in symbol:
        node = get_node(sym)
        assert node is not None
        nodes.append(node)
    if len(nodes) > 1:
        return _sym.Group(nodes)
    return nodes[0]

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
