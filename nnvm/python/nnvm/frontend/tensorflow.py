# pylint: disable=import-self, invalid-name, unused-argument
"""TF: Tensorflow frontend."""
from __future__ import absolute_import as _abs
from __future__ import print_function

# Numpy support
import numpy as np

import tvm
from .. import symbol as _sym
from .. import graph as _graph
from .. compiler import graph_util
from .common import get_nnvm_op, AttrConverter as AttrConvert

__all__ = ['from_tensorflow']

class AttrCvt(object):
    """A Wrapper to handle some common jobs:
    """
    def __init__(self, op_name, transforms=None,
                 excludes=None, disables=None, ignores=None,
                 extras=None, custom_check=None):
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, attrs, *args):
        self._ignores.append('_output_shapes')
        self._ignores.append('_input_shapes')
        self._ignores.append('T')
        self._ignores.append('use_cudnn_on_gpu')
        return AttrConvert(self._op_name, self._transforms, self._excludes,
                           self._disables, self._ignores, self._extras,
                           self._custom_check)(inputs, attrs, *args)

def _get_pad_pair(input1d, kernel1d, stride1d):
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]

def _math_name_picker(surfix):
    def _impl(attr):
        return 'broadcast_' + surfix
    return _impl

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
    """A hack for getting 'channles' or 'units' since tensorflow don't provide
    these attributes. We check the shape of weights provided to get the number.
    """
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels

def _rsqrt():
    def _impl(inputs, attr, *args):
        return AttrCvt(op_name="__pow_scalar__", extras={'scalar': -0.5})(inputs, attr)
    return _impl

def _elemwise(name):
    def _impl(inputs, attr, *args):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(len(inputs))
        op_name = _math_name_picker(name)(attr)
        axis = int(attr.get('axis', 0))
        conv_ops = ["conv2d", "conv2d_transpose"]
        if op_name == 'broadcast_add' and inputs[0].attr('op_name') in conv_ops:
            # TODO: remove hard coded infershape
            inputs[1] = _sym.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_nnvm_op(op_name)(*inputs)
    return _impl

def _pooling(name):
    def _impl(inputs, attr, params):

        attr['data_format'] = attr['data_format'].decode("utf-8")

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (attr['ksize'][1], attr['ksize'][2])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (attr['ksize'][2], attr['ksize'][3])
        else:
            raise TypeError("Unsupported data_format type : {}".format(attr['data_format']))

        # Fix strides
        attr['strides'] = (attr['strides'][1], attr['strides'][2])

        # Fix padding
        input_shapes = attr['_input_shapes'][inputs[0]]
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shapes[0][1]
                in_w = input_shapes[0][2]
            else:
                in_h = input_shapes[0][2]
                in_w = input_shapes[0][3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            if attr['data_format'] == 'NHWC':
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1]),
                                                (0, 0)))
            else:
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1])))
            attr['padding'] = [0, 0]
        else:
            raise TypeError("Unsupported padding type : {}".format(attr['padding']))

        return AttrCvt(
            op_name=_dimension_picker(name),
            transforms={
                'kernel_shape':'pool_size',
                'data_format':'layout'},
            ignores=['ksize'],
            extras={'ceil_mode': False},
            custom_check=_dimension_constraint())(inputs, attr)
    return _impl

def _conv():
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")

        # Extract kernel shape from params
        conv_param_weights = params[inputs[1].list_output_names()[0]]

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (conv_param_weights.shape[0], conv_param_weights.shape[1])
            attr['channels'] = conv_param_weights.shape[3]
            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][0], attr['dilations'][1])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (conv_param_weights.shape[2], conv_param_weights.shape[3])
            attr['channels'] = conv_param_weights.shape[1]
            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][2], attr['dilations'][3])
        else:
            raise TypeError("Unsupported data format type : {}".format(attr['data_format']))

        # Fix strides
        attr['strides'] = (attr['strides'][1], attr['strides'][2])

        # Fix padding
        input_shapes = attr['_input_shapes'][inputs[0]]
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shapes[0][1]
                in_w = input_shapes[0][2]
            else:
                in_h = input_shapes[0][2]
                in_w = input_shapes[0][3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            if attr['data_format'] == 'NHWC':
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1]),
                                                (0, 0)))
            else:
                inputs[0] = _sym.pad(data=inputs[0],
                                     pad_width=((0, 0),
                                                (0, 0),
                                                (pad_v[0], pad_v[1]),
                                                (pad_h[0], pad_h[1])))

            attr['padding'] = [0, 0]

        else:
            raise TypeError("Unsupported padding type : {}".format(attr['padding']))

        if 'kernel_layout' not in attr:
            attr['kernel_layout'] = 'HWIO' if attr['data_format'] == 'NHWC' else 'OIHW'

        return AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            extras={'use_bias': len(inputs) == 3},
            custom_check=_dimension_constraint())(inputs, attr)
    return _impl

def _decode_image():
    def _impl(inputs, attr, params):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        print("DecodeJpeg: It's a pass through, please handle preprocessing before input")
        return inputs[0]
    return _impl

def _cast():
    def _impl(inputs, attr, params):
        # Convert from tensorflow Dtype to str
        attr['DstT'] = attr['DstT'].name
        return AttrCvt(op_name='cast', transforms={'DstT': 'dtype'}, ignores=['SrcT'])(inputs, attr)
    return _impl

def _expand_dims():
    def _impl(inputs, attr, params):
        dim_input = inputs.pop(1)
        axis = params[dim_input.list_output_names()[0]]
        params.pop(dim_input.list_output_names()[0])
        return AttrCvt(op_name="expand_dims", ignores=['Tdim'],
                       extras={'axis': axis.asnumpy()[0]})(inputs, attr)
    return _impl

def _resize_bilinear():
    def _impl(inputs, attr, params):
        # Change this when we have corresponding resize bilinear operation.
        print("ResizeBilinear:Only NN (nearest neighbor) supported in symetric mode of dimensions")
        print("Change this when we have corresponding resize bilinear operation")

        # NHWC
        input_shape = attr['_input_shapes'][inputs[0]][0]
        in_hw = (input_shape[1], input_shape[2])
        out_hw = params[inputs[1].list_output_names()[0]]
        inputs.pop(1)
        attr['layout'] = 'NHWC'

        if in_hw[0] < 0 or in_hw[1] < 0:
            scale = 1
        else:
            # Considering height alone for scale
            scale = out_hw[0] / in_hw[0]

        return AttrCvt(op_name="upsampling",
                       ignores=['Tdim', 'align_corners'],
                       extras={'scale': scale})(inputs, attr)
    return _impl

def _check_numerics():
    def _impl(inputs, attr, params):
        # Making a copy node assuming no need to verify
        return AttrCvt(op_name="copy", ignores=['message'])(inputs, attr)
    return _impl


def _matmul():
    def _impl(inputs, attr, params):
        channels = _infer_channels(inputs[1], params, not attr['transpose_b'])
        if attr['transpose_a']:
            inputs[0] = _sym.transpose(inputs[0], axis(1, 0))
        if not attr['transpose_b']:
            inputs[1] = _sym.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(op_name="dense",
                       extras={'use_bias': False, 'units': channels},
                       ignores=['transpose_a', 'transpose_b', 'T'])(inputs, attr)

    return _impl

def _identity():
    def _impl(inputs, attr, params):
        return inputs[0]
    return _impl

def _concatV2():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(len(inputs)-1)
        axis = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(
            op_name="concatenate", ignores=['T', 'N', 'Tidx'],
            extras={'axis': axis.asnumpy()[0]})(inputs, attr)
    return _impl

def _concat():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(0)
        axis = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(
            op_name="concatenate", ignores=['N'],
            extras={'axis': axis.asnumpy()[0]})(inputs, attr)
    return _impl

def _reshape():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(1)
        shape_arg = params[pop_node.list_output_names()[0]]
        params.pop(pop_node.list_output_names()[0])
        return AttrCvt(
            op_name="reshape",
            extras={'shape':tuple(shape_arg.asnumpy())},
            ignores=['Tshape'])(inputs, attr)
    return _impl

def _bias_add():
    def _impl(inputs, attr, params):
        return _sym.broadcast_add(inputs[0], inputs[1])
    return _impl

def _squeeze():
    def _impl(inputs, attr, params):
        return AttrCvt(
            op_name="squeeze",
            transforms={'squeeze_dims':'axis'},
            ignores=['T'])(inputs, attr)
    return _impl

def _batch_norm():
    def _impl(inputs, attr, params):
        # Rearrange inputs from
        # (data, moving_mean, moving_variance, beta, gamma)
        #     to
        # (data, gamma, beta, moving_mean, moving_var)
        new_inputs = [inputs[0], inputs[4], inputs[3], inputs[1], inputs[2]]

        return AttrCvt(
            op_name='batch_norm',
            transforms={'scale_after_normalization':'scale', 'variance_epsilon':'epsilon'},
            extras={'axis': 3}, # Fix axis
            disables=['momentum'])(new_inputs, attr)
    return _impl

# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    'AvgPool'                           : _pooling('avg_pool'),
    'BatchNormWithGlobalNormalization'  : _batch_norm(),
    'BiasAdd'                           : _bias_add(),
    'Cast'                              : _cast(),
    'CheckNumerics'                     : _check_numerics(),
    'Concat'                            : _concat(),
    'ConcatV2'                          : _concatV2(),
    'Conv2D'                            : _conv(),
    'DecodeJpeg'                        : _decode_image(),
    'ExpandDims'                        : _expand_dims(),
    'Identity'                          : _identity(),
    'MatMul'                            : _matmul(),
    'MaxPool'                           : _pooling('max_pool'),
    'Mul'                               : _elemwise('mul'),
    'Relu'                              : AttrCvt('relu'),
    'Reshape'                           : _reshape(),
    'ResizeBilinear'                    : _resize_bilinear(),
    'Softmax'                           : AttrCvt('softmax', {'axis': ('axis', 1)}),
    'Sub'                               : _elemwise('sub'),
    'Add'                               : _elemwise('add'),
    'Rsqrt'                             : _rsqrt(),
    'Squeeze'                           : _squeeze(),
}


class GraphProto(object):
    """ A helper class for handling nnvm graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._replacements = {}
        self._output_shapes = {}
        self._num_input = 0
        self._num_param = 0
        self._input_node = ''

    def from_tensorflow(self, graph):
        """Construct nnvm nodes from tensorflow  graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to NNVM.
        Some of the assumptions listed below.

            -> First Const or Placeholder node will be considered as graph input.
            -> Rest all Const nodes are params.
            -> Last node is assumed as graph output.
            -> _output_shapes : Attribute should present in the tenserflow forzen graph.
            -> DecodeJpeg, ResizeBilinear: These are dummy operators.
                                           Hence user should handle preprocessing outside.
            -> CheckNumerics: No implementation as of now for this.
                              Just copies input to output.


        Parameters
        ----------
        graph : tensorflow graph definition object
            The loaded tensorflow GraphDef

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # Parse throught all nodes and start extracting
        # params aka Const nodes
        # input nodes  : First const node
        # normal nodes : other normal nodes

        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        for node in graph.node:
            # Tensorflow doesn't have seperate list for params extraction.
            # Operator name 'Const' is treated as a parameter to build NNVM params dict.
            if node.op == "Placeholder":
                # Assuming only one input graph with type 'Placeholder'
                self._input_node = node.name
                self._num_input += 1
                self._nodes[node.name] = _sym.Variable(name=node.name)

                self._output_shapes[node.name] = \
                     [tensor_util.TensorShapeProtoToList(shape) \
                     for shape in self._parse_attr(node.attr)['_output_shapes']]
            elif node.op == "Const":
                # Assuming first Const node as Graph Input node
                if self._input_node == '':
                    self._input_node = node.name
                    self._num_input += 1
                    self._nodes[node.name] = _sym.Variable(name=node.name)
                else:
                    # Rest all nodes are Param nodes, lets parse
                    self._num_param += 1
                    for key, value in node.attr.items():
                        self._parse_param(key, value, node.name)
                    if node.name not in self._nodes:
                        raise NotImplementedError( \
                            "Const {} couldn't be converted to Param.".format(node.name))

                self._output_shapes[node.name] = \
                     [tensor_util.TensorShapeProtoToList(shape) \
                     for shape in self._parse_attr(node.attr)['_output_shapes']]
            else:
                attr = self._parse_attr(node.attr)
                self._output_shapes[node.name] = \
                     [tensor_util.TensorShapeProtoToList(shape) for shape in attr['_output_shapes']]

                # Pass the parsed shapes instead
                attr["_output_shapes"] = self._output_shapes[node.name]

                try:
                    inputs = [self._nodes[i] for i in node.input]
                    input_shapes = {}
                    for i in node.input:
                        if i not in self._params:
                            input_shapes[self._nodes[i]] = self._output_shapes[i]
                    attr['_input_shapes'] = input_shapes
                except KeyError:
                    # TODO: Need to find clean way to handle '^CheckNumerics'
                    print("Some Exception while inputs list:", node.input, " ignoring...")

                inputs = self._fix_extranodes(node.op, attr, inputs)

                op = self._convert_operator(node.op, inputs, attr)
                # Assuming only one output.
                self._nodes[node.name] = op
                node_output = op
        # Assume the final node is the output node
        out = node_output
        return out, self._params

    def _parse_param(self, key, value, name):
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        if key == 'value':
            np_array = tensor_util.MakeNdarray(value.tensor)
            array_ndim = len(np_array.shape)
            if array_ndim == 0:
                new_array = np.empty([1], dtype=np_array.dtype)
                new_array[0] = np_array
                self._params[name] = tvm.nd.array(new_array)
            else:
                self._params[name] = tvm.nd.array(np_array)
            self._nodes[name] = _sym.Variable(name=name,
                                              shape=self._params[name].shape)
        else:
            if key != 'dtype' and key != '_output_shapes':
                raise NotImplementedError \
                    ("Other attributes for a Const(param) Node {} ? .".format(key))

    def _get_attr(self, buf):
        """Returns the value of the attr of this buf with the given `name`.

        Args:
          buf: attrvalue protobuf.

        Returns:
          The value of the attr, as a Python object.

        Raises:
          ValueError: If this op does not have an attr with the given `name`.
        """
        fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]

        x = buf

        ret = []

        try:
            from tensorflow.python.framework import dtypes
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        # Treat an empty oneof value as an empty list.
        if not x.WhichOneof("value"):
            return ret
        if x.HasField("list"):
            for f in fields:
                if getattr(x.list, f):
                    if f == "type":
                        ret = [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                    else:
                        ret = list(getattr(x.list, f))
        else:
            for f in fields:
                if x.HasField(f):
                    if f == "type":
                        ret = dtypes.as_dtype(getattr(x, f))
                    else:
                        ret = getattr(x, f)
        return ret

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for key, value in attr_proto.items():
            attrs[key] = self._get_attr(value)

        return attrs

    def _convert_operator(self, op_name, inputs, attrs, identity_list=None, convert_map=None):
        """Convert from Tensorflow operator to nnvm operator.
        The converter must specify conversions explicity for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
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

    def _fix_extranodes(self, op_name, attr, inputs):
        if op_name == "Softmax":
            # Require some times flatten of data before it goes to softmax
            # Need to relook into this with latest softmax axis support.
            op = AttrCvt(op_name='flatten')(inputs, {})
            node_output = op.list_output_names()
            for k, i in zip(list(node_output), range(len(node_output))):
                self._nodes[k] = op[i]
            inputs = [op]

        return inputs

def from_tensorflow(graph):
    """  Load tensorflow graph which is a python tensorflow graph object into nnvm graph.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    sym, params = g.from_tensorflow(graph)
    return sym, params
