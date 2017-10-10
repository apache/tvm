# pylint: disable=import-self, invalid-name
"""ONNX: Open Neural Network Exchange frontend."""
from __future__ import absolute_import as _abs
import tvm
from .. import symbol as _sym
from .. import graph as _graph
from .. compiler import graph_util
from .common import Renamer, AttrConverter as AttrCvt

__all__ = ['from_onnx']

def _revert_caffe2_pad(attr):
    """Caffe2 require two times the normal padding."""
    if len(attr) == 4:
        attr = attr[:2]
    elif len(attr) == 2:
        pass
    else:
        raise ValueError("Invalid caffe2 type padding: {}".format(attr))
    return attr

def _math_name_picker(surfix):
    def _impl(attr):
        if attr.get('broadcast', 0):
            return 'broadcast_' + surfix
        return 'elemwise_' + surfix
    return _impl

def _broadcast_constraint():
    def _broadcast_check(attrs):
        if attrs.get('axis', None):
            return False
        return True
    return _broadcast_check, "Specifying broadcast axis not allowed."

def _dimension_picker(prefix, surfix=''):
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 2:
            return prefix + '2d' + surfix
        else:
            raise NotImplementedError("Only 2d kernel supported.")
    return _impl

def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False
    return _dim_check, "Only 2d kernel supported."

def _elemwise(name):
    return AttrCvt(
        op_name=_math_name_picker(name),
        disables=['axis'],
        ignores=['broadcast'])

def _pooling(name):
    return AttrCvt(
        op_name=_dimension_picker(name),
        transforms={
            'kernel_shape': 'pool_size',
            'pads': ('padding', (0, 0), _revert_caffe2_pad)},
        # very weird attributes here in onnx, force check
        ignores=['dilations'],
        # TODO(zhreshold): make sure ceil_mode in onnx, and layout?
        extras={'ceil_mode': False},
        custom_check=_dimension_constraint())

def _conv():
    return AttrCvt(
        op_name=_dimension_picker('conv'),
        transforms={
            'kernel_shape': 'kernel_size',
            'dilations': ('dilation', (0, 0)),
            'pads': ('padding', (0, 0), _revert_caffe2_pad),
            'group': ('groups', 1)},
        custom_check=_dimension_constraint())

def _conv_transpose():
    return AttrCvt(
        op_name=_dimension_picker('conv', '_transpose'),
        transforms={
            'kernel_shape': 'kernel_size',
            'dilations': ('dilation', (0, 0)),
            'pads': ('padding', (0, 0), _revert_caffe2_pad)},
        disables=['output_shape'],
        custom_check=_dimension_constraint())

def _batch_norm():
    # TODO(zhreshold): 'spatial' is not properly handled here.
    return AttrCvt(
        op_name='batch_norm',
        disables=['momentum'],
        ignores=['spatial', 'is_test', 'consumed_inputs'])


# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
_convert_map = {
    # defs/experimental
    'FC'            : AttrCvt('dense', ignores=['axis', 'axis_w']),
    'SpatialBN'     : _batch_norm(),

    # defs/generator
    # 'Constant'
    # 'RandomUniform'
    # 'RandomNormal'
    # 'RandomUniformLike'
    # 'RandomNormalLike'

    # defs/logical

    # defs/math
    'Add'           : _elemwise('add'),
    'Sub'           : _elemwise('sub'),
    'Mul'           : _elemwise('mul'),
    'Div'           : _elemwise('div'),
    'Neg'           : Renamer('negative'),
    # 'Abs'
    # 'Reciprocal'
    # 'Floor'
    # 'Ceil'
    # 'Sqrt'
    'Relu'          : Renamer('relu'),
    'LeakyRelu'     : Renamer('leaky_relu'),
    # 'Selu'
    # 'Elu'
    'Exp'           : Renamer('exp'),
    'Log'           : Renamer('log'),
    'Tanh'          : Renamer('tanh'),
    # 'Pow'
    # 'Dot'
    # 'PRelu'
    'Sigmoid'       : Renamer('sigmoid'),
    # 'Max' : this is the elemwise maximum
    # 'Min' : this is the elemwise minimum
    # 'Sum' : elemwise sum
    # softmax default axis is different in onnx
    'Softmax'       : AttrCvt('softmax', {'axis': ('axis', 1)}),

    # defs/nn
    'AveragePool'   : _pooling('avg_pool'),
    'MaxPool'       : _pooling('max_pool'),
    'Conv'          : _conv(),
    'ConvTranspose' : _conv_transpose(),
    'GlobalAveragePool': Renamer('global_avg_pool2d'),
    'GlobalMaxPool' : Renamer('global_max_pool2d'),
    'BatchNormalization': _batch_norm(),
    'Dropout'       : AttrCvt('dropout', {'ratio': 'rate'}, ignores=['is_test']),
    'Flatten'       : Renamer('flatten'),

    # defs/reduction
    'ReduceMax'     : AttrCvt('max', {'axes', 'axis'}),
    'ReduceMin'     : AttrCvt('min', {'axes', 'axis'}),
    'ReduceSum'     : AttrCvt('sum', {'axes', 'axis'}),
    # 'ReduceMean'
    # 'ReduceProd'
    # 'ReduceLogSumExp'
    # 'ArgMax'
    # 'ArgMin'

    # defs/tensor
    'Cast'          : AttrCvt('cast', {'to': 'dtype'}),
    'Reshape'       : Renamer('reshape'),
    'Concat'        : Renamer('concatenate'),
    'Split'         : AttrCvt('split', {'split': 'indices_or_sections'}),
    # 'Slice'
    'Transpose'     : AttrCvt('transpose', {'perm': 'axes'}),
    # 'Gather'
    # 'Squeeze'
}

def _convert_operator(op_name, attrs, identity_list=None, convert_map=None):
    """Convert from onnx operator to nnvm operator.
    The converter must specify conversions explicity for incompatible name, and
    apply handlers to operator attributes.

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
        raise NotImplementedError("Operator {} not implemented.".format(op_name))
    op = getattr(_sym, op_name, None)
    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op, attrs


class GraphProto(object):
    """A helper class for handling nnvm graph copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0

    def from_onnx(self, graph):
        """Construct nnvm nodes from onnx graph.
        The inputs from onnx graph is vague, only providing "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # parse network inputs to nnvm, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = self._parse_array(init_tensor)
        for i in graph.input:
            if i in self._params:
                # i is a param instead of input
                name_param = 'param_{}'.format(self._num_param)
                self._num_param += 1
                self._params[name_param] = self._params.pop(i)
                self._nodes[name_param] = _sym.Variable(name=name_param)
                self._renames[i] = name_param
            else:
                name_input = 'input_{}'.format(self._num_input)
                self._num_input += 1
                self._nodes[name_input] = _sym.Variable(name=name_input)
                self._renames[i] = name_input
        # construct nodes, nodes are stored as directed acyclic graph
        for idx, node in enumerate(graph.node):
            op_name = node.op_type
            node_name = node.name.strip()
            node_name = node_name if node_name else None
            attr = self._parse_attr(node.attribute)
            new_op, new_attr = _convert_operator(op_name, attr)
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]
            # some hacks for onnx problem
            new_attr = self._fix_bias(new_op, new_attr, len(inputs))
            new_attr = self._fix_channels(new_op, new_attr, list(node.input))
            self._fix_bias_shape(node.op_type, graph.node[idx-1].op_type, node.input)
            op = new_op(name=node_name, *inputs, **new_attr)
            node_output = self._fix_outputs(op_name, node.output)
            assert len(node_output) == len(op.list_output_names()), (
                "Number of output mismatch {} vs {} in {}.".format(
                    len(node_output), len(op.list_output_names()), op_name))
            for k, i in zip(list(node_output), range(len(node_output))):
                self._nodes[k] = op[i]
        # now return the outputs
        out = [self._nodes[i] for i in graph.output]
        if len(out) > 1:
            out = _sym.Group(out)
        else:
            out = out[0]
        return out, self._params

    def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return tvm.nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['floats', 'ints', 'strings']:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ['t', 'g']:
                if a.HasField(f):
                    raise NotImplementedError("Filed {} is not supported in nnvm.".format(f))
            for f in ['tensors', 'graphs']:
                if list(getattr(a, f)):
                    raise NotImplementedError("Filed {} is not supported in nnvm.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _fix_outputs(self, op, outputs):
        """A hack to handle dropout or similar operator that have more than one out
        in ONNX.
        """
        if op == 'Dropout':
            assert len(outputs) == 2, "ONNX have two outputs for dropout layer."
            outputs = outputs[:-1]
        return outputs

    def _fix_bias(self, op, attrs, num_inputs):
        """A hack for 'use_bias' attribute since onnx don't provide this attribute,
        we have to check the number of inputs to decide it."""
        if op not in [_sym.conv2d, _sym.conv2d_transpose, _sym.dense]:
            return attrs
        if num_inputs == 3:
            attrs['use_bias'] = True
        elif num_inputs == 2:
            attrs['use_bias'] = False
        else:
            raise ValueError("Unexpected number of inputs for: {}".format(op))
        return attrs

    def _fix_bias_shape(self, op_name, last_op_name, inputs):
        """A hack to reshape bias term to (1, num_channel)."""
        if op_name == 'Add' and last_op_name == 'Conv':
            assert len(list(inputs)) == 2
            bias_name = self._renames.get(inputs[1], inputs[1])
            bias = self._params[bias_name]
            assert len(bias.shape) == 1
            # reshape to (1, n)
            bias = tvm.nd.array(bias.asnumpy().reshape((1, -1, 1, 1)))
            self._params[bias_name] = bias

    def _fix_channels(self, op, attrs, inputs):
        """A hack for getting 'channles' or 'units' since onnx don't provide
        these attributes. We check the shape of weights provided to get the number.
        """
        if op not in [_sym.conv2d, _sym.conv2d_transpose, _sym.dense]:
            return attrs
        if inputs[1] not in self._renames:
            assert inputs[1] in self._nodes
            g = _graph.create(self._nodes[inputs[1]])
            shape_dict = {k: v.shape for k, v in self._params.items()}
            _, out_shapes = graph_util.infer_shape(g, **shape_dict)
            channels = out_shapes[0][0]
        else:
            weight_name = self._renames[inputs[1]]
            if not weight_name in self._params:
                raise ValueError("Unable to get channels/units attr from onnx graph.")
            else:
                wshape = self._params[weight_name].shape
                assert len(wshape) >= 2, "Weights shape is invalid: {}".format(wshape)
                channels = wshape[0]
        if op in [_sym.dense]:
            attrs['units'] = channels
        else:
            attrs['channels'] = channels
        return attrs

def from_onnx(graph):
    """Load onnx graph which is a python protobuf object in to nnvm graph.
    The companion parameters will be handled automatically.
    The inputs from onnx graph is vague, only providing "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    Parameters
    ----------
    graph : protobuf object
        ONNX graph

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    sym, params = g.from_onnx(graph)
    return sym, params
