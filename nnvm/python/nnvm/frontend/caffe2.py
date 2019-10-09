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
# pylint: disable=import-self, invalid-name, line-too-long, unused-argument
"""Caffe2 frontend"""
from __future__ import absolute_import as _abs
import tvm
from nnvm import symbol as _sym
from .common import get_nnvm_op, Renamer, AttrConverter as AttrCvt
from .onnx_caffe2_utils import dimension_picker, dimension_constraint, infer_channels, revert_caffe2_pad
from . import onnx

__all__ = ['from_caffe2']


def _clean_up_pool_args(args):
    """ A helper function to clean up common arguments in conv and pooling ops.
    """
    assert isinstance(args, dict)

    if 'stride_h' in args and 'stride_w' in args:
        assert 'stride' not in args and 'strides' not in args
        args['strides'] = [args['stride_h'], args['stride_w']]
        args.pop('stride_h')
        args.pop('stride_w')
    elif 'stride' in args:
        args['strides'] = [args['stride'], args['stride']]
        args.pop('stride')

    # rename 'kernel', 'kernels', to 'kernel_shape'
    if 'kernel_h' in args and 'kernel_w' in args:
        assert 'kernel' not in args and 'kernels' not in args
        args['kernel_shape'] = [args['kernel_h'], args['kernel_w']]
        args.pop('kernel_h')
        args.pop('kernel_w')
    elif 'kernel' in args:
        args['kernel_shape'] = [args['kernel'], args['kernel']]
        args.pop('kernel')
    elif 'kernels' in args:
        args['kernel_shape'] = args['kernels']
        args.pop('kernels')

    if 'pad_t' in args and 'pad_l' in args and 'pad_b' in args and 'pad_r' in args:
        assert 'pad' not in args and 'pads' not in args
        args['pads'] = [
            args['pad_t'], args['pad_l'], args['pad_b'], args['pad_r']
        ]
        for pad in ['pad_t', 'pad_l', 'pad_b', 'pad_r']:
            args.pop(pad)
    elif 'pad' in args:
        args['pads'] = [args['pad'], args['pad']]
        args.pop('pad')

    if 'dilation_h' in args and 'dilation_w' in args:
        assert 'dilation' not in args and 'dilations' not in args
        args['dilations'] = [args['dilation_h'], args['dilation_w']]
        args.pop('dilation_h')
        args.pop('dilation_w')
    elif 'dilation' in args:
        args['dilations'] = [args['dilation'], args['dilation']]
        args.pop('dilation')

    return args


class Caffe2OpConverter(object):
    """ A helper class for holding Caffe2 op converters.
    """

    @classmethod
    def get_converter(cls):
        """ Get converter.

        :return: converter, which should be `_impl`.
        """

        if hasattr(cls, '_impl'):
            return getattr(cls, '_impl')
        raise tvm.error.OpNotImplemented(
            'Operator {} is not implemented in frontend Caffe2.'.format(cls.__name__))


_caffe2_internal_args = {
    # nnpack args
    'algo',
    'convolution_transform_strategy',
    'float16_compute',
    'shared_buffer',

    # training args
    'init_params',
    'cudnn_exhaustive_search',
    'exhaustive_search',

    # training args
    'adj',
    'hwgq',

    # args that we don't care
    'legacy_pad',
}


class Pool(Caffe2OpConverter):
    """ A helper class for pool op converters.
    """

    name = ''

    @classmethod
    def _impl(cls, inputs, args, params):
        _clean_up_pool_args(args)
        if 'global_pooling' in args and args['global_pooling'] == 1:
            op_name = dimension_picker('global_' + cls.name)
            return get_nnvm_op(op_name(args))(*inputs)

        return AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                'kernel_shape': 'pool_size',
                'pads': ('padding', (0, 0), revert_caffe2_pad),
                'strides': 'strides',
            },
            excludes={
                # TVM poolop does not support dilation
                'dilations',
            },
            ignores=_caffe2_internal_args | {'global_pooling', 'order'},
            custom_check=dimension_constraint())(inputs, args, params)


class AveragePool(Pool):
    name = 'avg_pool'


class MaxPool(Pool):
    name = 'max_pool'


class Conv(Caffe2OpConverter):
    """ Operator converter for Conv.
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        # get number of channels
        channels = infer_channels(inputs[1], params)
        args['channels'] = channels
        _clean_up_pool_args(args)
        return AttrCvt(
            op_name=dimension_picker('conv'),
            transforms={
                'group': ('groups', 1),
                'kernel_shape':
                'kernel_size',
                'pads': ('padding', (0, 0), revert_caffe2_pad),
                'strides':
                'strides',
                'dilations': ('dilation', (1, 1)),
                'order':
                ('layout', ("NCHW"),
                 lambda x: x if isinstance(x, str) else x.decode('UTF-8')),
            },
            excludes={},
            ignores=_caffe2_internal_args,
            extras={'use_bias': len(inputs) == 3},
            custom_check=dimension_constraint())(inputs, args, params)


class Concat(Caffe2OpConverter):
    """ Operator converter for Concat.
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        def _get_axis_from_order_str(order):
            order = order if isinstance(order, str) else order.decode('UTF-8')
            if order == 'NCHW':
                return 1
            if order == 'NHWC':
                return 3
            raise tvm.error.OpAttributeInvalid('Value {} in attribute {} of operator {} is not valid.'.format(order, 'order', 'Concat'))

        return AttrCvt(
            op_name='concatenate',
            transforms={
                'order': ('axis', (1), _get_axis_from_order_str),
            },
            excludes={
                'add_axis',
            })(inputs, args, params)


class NormalizePlanarYUV(Caffe2OpConverter):
    """ Operator converter for NormalizePlanarYUV.
    caffe2 definition: https://github.com/pytorch/pytorch/blob/master/caffe2/operators/norm_planar_yuv_op.cc
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        assert len(inputs) == 3
        mean = _sym.expand_dims(inputs[1], axis=2, num_newaxis=2)
        std = _sym.expand_dims(inputs[2], axis=2, num_newaxis=2)

        return _sym.broadcast_div(_sym.broadcast_sub(inputs[0], mean), std)


class ResizeNearest(Caffe2OpConverter):
    """ Operator converter for Upsample (nearest mode).
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        width_scale = args['width_scale'] if 'width_scale' in args else 1
        height_scale = args['height_scale'] if 'height_scale' in args else 1
        assert width_scale == height_scale

        return _sym.upsampling(
            inputs[0], scale=int(width_scale), method="NEAREST_NEIGHBOR")


class FC(Caffe2OpConverter):
    """ Operator converter for FC.
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        inputs[0] = _sym.flatten(inputs[0])
        args['units'] = infer_channels(inputs[1], params)
        return AttrCvt(
            'dense',
            ignores=['axis', 'axis_w'],
            extras={'use_bias': len(inputs) == 3},
        )(inputs, args, params)


class SpatialBN(Caffe2OpConverter):
    """ Operator converter for SpatialBN.
    """

    @classmethod
    def _impl(cls, inputs, args, params):
        return AttrCvt(
            op_name='batch_norm',
            disables=['momentum'],
            ignores=[
                'order', 'spatial', 'is_test', 'consumed_inputs', 'num_batches'
            ])(inputs, args, params)


# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)

# Minimal set of ops for squeezenet and resnet50
def _get_convert_map():
    return {
        # caffe2/onnx common operators
        'Add': onnx.Add.get_converter(opset=1),
        'Sum': onnx.Sum.get_converter(opset=1),
        'Softmax': onnx.Softmax.get_converter(opset=1),

        # nn
        'AveragePool': AveragePool.get_converter(),
        'MaxPool': MaxPool.get_converter(),
        'Conv': Conv.get_converter(),
        'Concat': Concat.get_converter(),
        'FC': FC.get_converter(),
        'SpatialBN': SpatialBN.get_converter(),
        'ResizeNearest': ResizeNearest.get_converter(),
        'Relu': AttrCvt('relu', {}, ignores=['order']),
        'Sigmoid': Renamer('sigmoid'),
        'Dropout': AttrCvt('dropout', {'ratio': 'rate'}, ignores=['is_test']),

        # c2 image preprocessing ops
        'NormalizePlanarYUV': NormalizePlanarYUV.get_converter(),
    }


class Caffe2NetDef(object):
    """A helper class for handling nnvm graph copying from pb2.GraphProto.
    Definition: https://github.com/pytorch/pytorch/blob/master/caffe2/proto/caffe2.proto
    """

    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._visited_nodes = set()
        self._ops = {}

    def from_caffe2(self, init_net, predict_net):
        """Construct nnvm nodes from caffe2 graph.

        Parameters
        ----------
        workspace : Caffe2 workspace
        predict_net : protobuf object

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        from caffe2.python import workspace
        workspace.RunNetOnce(init_net)

        # Input
        input_name = predict_net.op[0].input[0]

        # Params
        self._params = {}
        used_blobs = set()
        for c2_op in predict_net.op:
            for i in c2_op.input:
                used_blobs.add(i)
        for blob in workspace.Blobs():
            if blob in used_blobs and blob != input_name:
                self._params[blob] = tvm.nd.array(workspace.FetchBlob(blob))

        # Variables
        self._nodes = {}
        for blob in predict_net.external_input:
            self._nodes[blob] = _sym.Variable(name=blob)

        # Ops
        for c2_op in predict_net.op:
            for blob in c2_op.output:
                self._ops[blob] = c2_op
        for c2_op in predict_net.op:
            self._process_op(c2_op)

        # Outputs
        out = []
        for blob in predict_net.external_output:
            out.append(self._nodes[blob])

        if len(out) > 1:
            sym = _sym.Group(out)
        else:
            sym = out[0]

        return sym, self._params

    def _get_node(self, blob):
        """Get the nnvm Symbol of blob and detect cyclic dependency in the graph."""
        if blob in self._nodes:
            return self._nodes[blob]

        assert blob not in self._visited_nodes, 'Cyclic dependency in the graph (in {})'.format(
            blob)
        self._visited_nodes.add(blob)

        self._process_op(self._ops[blob])
        return self._nodes[blob]

    def _process_op(self, c2_op):
        op_type = c2_op.type
        args = self._parse_arg(c2_op.arg)
        inputs = [self._get_node(i) for i in c2_op.input]
        tvm_op = self._convert_operator(op_type, inputs, args)
        # Ignore all outputs except the first one
        self._nodes[c2_op.output[0]] = tvm_op[0]

    def _parse_arg(self, arg):
        """Convert a list of Argument to a dict, with names as keys."""
        args = {}
        for a in arg:
            for f in ['f', 'i', 's']:
                if a.HasField(f):
                    args[a.name] = getattr(a, f)
            for f in ['floats', 'ints', 'strings']:
                if list(getattr(a, f)):
                    assert a.name not in args, "Only one type of attr is allowed"
                    args[a.name] = tuple(getattr(a, f))
            for f in ['n']:
                if a.HasField(f):
                    raise NotImplementedError(
                        "Field {} is not supported in nnvm.".format(f))
            for f in ['nets']:
                if list(getattr(a, f)):
                    raise NotImplementedError(
                        "Field {} is not supported in nnvm.".format(f))
            if a.name not in args:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return args

    def _convert_operator(self,
                          op_type,
                          inputs,
                          args,
                          identity_list=None,
                          convert_map=None):
        """Convert from Caffe2 operator to nnvm operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_type : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of nnvm.Symbol
            List of input symbols.
        args : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to nnvm, callable are functions which
            take args and return (new_op_type, new_args)

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _get_convert_map()
        if op_type in identity_list:
            sym = get_nnvm_op(op_type)(*inputs, **args)
        elif op_type in convert_map:
            # Add a sanitizing step to convert all byte strings in args to strings
            sym = convert_map[op_type](inputs, args, self._params)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend Caffe2.'.format(op_type))
        return sym


def from_caffe2(init_net, predict_net):
    """Load caffe2 graph which contains init_net and predict_net into nnvm graph.

    Parameters
    ----------
    init_net : protobuf object
        Caffe2 NetDef containing the weights

    predict_net : protobuf object
        Caffe2 NetDef containing the graph

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """

    caffe2 = Caffe2NetDef()
    return caffe2.from_caffe2(init_net, predict_net)
