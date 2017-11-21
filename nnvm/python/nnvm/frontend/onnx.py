# pylint: disable=import-self, invalid-name, unused-argument
"""ONNX: Open Neural Network Exchange frontend."""
from __future__ import absolute_import as _abs
import tvm
from .. import symbol as _sym
from .. import graph as _graph
from .. compiler import graph_util
from .common import get_nnvm_op, Renamer, AttrConverter as AttrCvt

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

def _infer_channels(inputs, params, transpose=False):
    """A hack for getting 'channles' or 'units' since onnx don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels

def _elemwise(name):
    def _impl(inputs, attr, *args):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(len(inputs))
        op_name = _math_name_picker(name)(attr)
        axis = int(attr.get('axis', 0))
        conv_ops = ["conv2d", "conv2d_transpose"]
        if op_name == 'broadcast_add' and inputs[0].attr('op_name') in conv_ops:
            # TODO(zhreshold): remove hard coded infershape
            inputs[1] = _sym.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_nnvm_op(op_name)(*inputs)
    return _impl

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
    def _impl(inputs, attr, params):
        # get number of channels
        channels = _infer_channels(inputs[1], params)
        attr['channels'] = channels
        return AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'dilations': ('dilation', (0, 0)),
                'pads': ('padding', (0, 0), _revert_caffe2_pad),
                'group': ('groups', 1)},
            extras={'use_bias': False},
            custom_check=_dimension_constraint())(inputs, attr)
    return _impl

def _conv_transpose():
    def _impl(inputs, attr, params):
        # get number of channels
        channels = _infer_channels(inputs[1], params, True)
        attr['channels'] = channels
        groups = attr.pop('group')
        attr['groups'] = groups
        return AttrCvt(
            op_name=_dimension_picker('conv', '_transpose'),
            transforms={
                'kernel_shape': 'kernel_size',
                'dilations': ('dilation', (0, 0)),
                'pads': ('padding', (0, 0), _revert_caffe2_pad)},
            disables=['output_shape'],
            extras={'use_bias': False},
            custom_check=_dimension_constraint())(inputs, attr)
    return _impl

def _fully_connected():
    def _impl(inputs, attr, params):
        # get number of channels
        channels = _infer_channels(inputs[1], params)
        attr['units'] = channels
        return AttrCvt('dense', ignores=['axis', 'axis_w'])(inputs, attr)
    return _impl

def _batch_norm():
    # TODO(zhreshold): 'spatial' is not properly handled here.
    return AttrCvt(
        op_name='batch_norm',
        disables=['momentum'],
        ignores=['spatial', 'is_test', 'consumed_inputs'])


def _gemm():
    def _impl(inputs, attr, params):
        assert len(inputs) == 3, "Gemm op take 3 inputs, {} given".format(len(inputs))
        # Y = alpha * A * B + beta * C
        alpha = float(attr.get('alpha', 1.0))
        beta = float(attr.get('beta', 1.0))
        transA = int(attr.get('transA', 0))
        transB = int(attr.get('transB', 0))
        # get number of channels
        channels = _infer_channels(inputs[1], params, not transB)
        if transA:
            inputs[0] = _sym.transpose(inputs[0], axes=(1, 0))
        if not transB:
            inputs[1] = _sym.transpose(inputs[1], axes=(1, 0))
        return _sym.dense(alpha * inputs[0], inputs[1], beta * inputs[2], units=channels)
    return _impl


# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
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
    'Gemm'          : _gemm(),

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
    # 'LRN'

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
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name = self._parse_value_proto(i)
            if i_name in self._params:
                # i is a param instead of input
                name_param = 'param_{}'.format(self._num_param)
                self._num_param += 1
                self._params[name_param] = self._params.pop(i_name)
                self._nodes[name_param] = _sym.Variable(
                    name=name_param, shape=self._params[name_param].shape)
                self._renames[i_name] = name_param
            else:
                name_input = 'input_{}'.format(self._num_input)
                self._num_input += 1
                self._nodes[name_input] = _sym.Variable(name=name_input)
                self._renames[i_name] = name_input
        # construct nodes, nodes are stored as directed acyclic graph
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]
            op = self._convert_operator(op_name, inputs, attr)
            node_output = self._fix_outputs(op_name, node.output)
            assert len(node_output) == len(op.list_output_names()), (
                "Number of output mismatch {} vs {} in {}.".format(
                    len(node_output), len(op.list_output_names()), op_name))
            for k, i in zip(list(node_output), range(len(node_output))):
                self._nodes[k] = op[i]
        # now return the outputs
        out = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        if len(out) > 1:
            out = _sym.Group(out)
        else:
            out = out[0]
        return out, self._params

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

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

    def _convert_operator(self, op_name, inputs, attrs, identity_list=None, convert_map=None):
        """Convert from onnx operator to nnvm operator.
        The converter must specify conversions explicity for incompatible name, and
        apply handlers to operator attributes.

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
            sym = get_nnvm_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym

    def _fix_outputs(self, op_name, outputs):
        """A hack to handle dropout or similar operator that have more than one out
        in ONNX.
        """
        if op_name == 'Dropout':
            if len(outputs) == 1:
                return outputs
            # TODO(zhreshold): support dropout mask?
            outputs = outputs[:-1]
        return outputs


def from_onnx(graph):
    """Load onnx graph which is a python protobuf object in to nnvm graph.
    The companion parameters will be handled automatically.
    The inputs from onnx graph is vague, only providing "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    Parameters
    ----------
    graph : protobuf object
        ONNX GraphProto, or ONNX ModelProto after ONNX v0.2

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    if hasattr(graph, 'graph'):
        # it's a ModelProto wrapper
        graph = graph.graph
    sym, params = g.from_onnx(graph)
    return sym, params
