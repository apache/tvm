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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines
"""ONNX: Open Neural Network Exchange frontend."""
from __future__ import absolute_import as _abs
import numpy as np
import tvm
from .. import symbol as _sym
from .common import get_nnvm_op, Renamer, SymbolTable, AttrConverter as AttrCvt
from .onnx_caffe2_utils import dimension_picker, dimension_constraint, \
    infer_channels, revert_caffe2_pad

__all__ = ['from_onnx']


def onnx_storage_order2layout(storage_order):
    if storage_order not in (0, 1):
        raise tvm.error.OpAttributeInvalid('Mode of storage_order must be either 0 or 1')

    return 'NCHW' if storage_order == 0 else 'NHWC'


class OnnxOpConverter(object):
    """ A helper class for holding onnx op converters.
    """

    @classmethod
    def get_converter(cls, opset):
        """ Get converter matches given opset.

        :param opset: opset from model.
        :return: converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        versions = [
            int(d.replace('_impl_v', '')) for d in dir(cls) if '_impl_v' in d
        ]
        versions = sorted(versions + [opset])
        version = versions[
            max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, '_impl_v{}'.format(version)):
            return getattr(cls, '_impl_v{}'.format(version))
        raise NotImplementedError(
            'opset version {} of {} not implemented'.format(
                version, cls.__name__))


class Elemwise(OnnxOpConverter):
    """ A helper class for elemwise op converters.
    """

    name = ''

    @classmethod
    def _math_name_picker(cls, suffix):

        def _impl(attr):
            if attr.get('broadcast', 0):
                return 'broadcast_' + suffix
            return 'elemwise_' + suffix

        return _impl

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(
            len(inputs))
        op_name = cls._math_name_picker(cls.name)(attr)
        axis = int(attr.get('axis', 0))
        conv_ops = ["conv2d", "conv2d_transpose"]
        if op_name == 'broadcast_add' and inputs[0].attr('op_name') in conv_ops:
            # TODO(zhreshold): remove hard coded infershape
            inputs[1] = _sym.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_nnvm_op(op_name)(*inputs)


class Pool(OnnxOpConverter):
    """ A helper class for pool op converters.
    """

    name = ''

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                'kernel_shape': 'pool_size',
                'pads': ('padding', (0, 0), revert_caffe2_pad)
            },
            # very weird attributes here in onnx, force check
            ignores=['dilations'],
            # TODO(zhreshold): make sure ceil_mode in onnx, and layout?
            extras={'ceil_mode': False},
            custom_check=dimension_constraint())(inputs, attr, params)


class Absolute(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _sym.relu(inputs[0]) + _sym.relu(_sym.negative(inputs[0]))


class Add(Elemwise):
    name = 'add'


class AveragePool(Pool):
    name = 'avg_pool'


class BatchNorm(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # TODO(zhreshold): 'spatial' is not properly handled here.
        return AttrCvt(
            op_name='batch_norm',
            disables=['momentum'],
            ignores=['spatial', 'is_test', 'consumed_inputs'])(inputs, attr,
                                                               params)


class Conv(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # get number of channels
        channels = infer_channels(inputs[1], params)
        attr['channels'] = channels
        return AttrCvt(
            op_name=dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'dilations': ('dilation', (0, 0)),
                'pads': ('padding', (0, 0), revert_caffe2_pad),
                'group': ('groups', 1)
            },
            extras={'use_bias': len(inputs) == 3},
            custom_check=dimension_constraint())(inputs, attr, params)


class ConvTranspose(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # get number of channels
        channels = infer_channels(inputs[1], params, True)
        attr['channels'] = channels
        groups = attr.pop('group')
        attr['groups'] = groups
        return AttrCvt(
            op_name=dimension_picker('conv', '_transpose'),
            transforms={
                'kernel_shape': 'kernel_size',
                'dilations': ('dilation', (0, 0)),
                'pads': ('padding', (0, 0), revert_caffe2_pad)
            },
            disables=['output_shape'],
            extras={'use_bias': len(inputs) == 3},
            custom_check=dimension_constraint())(inputs, attr, params)


class Div(Elemwise):
    name = 'div'


class Elu(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
        return -alpha * _sym.relu(1 - _sym.exp(inputs[0])) + _sym.relu(
            inputs[0])


class Gemm(OnnxOpConverter):
    """ Operator converter for Gemm.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 3, "Gemm op take 3 inputs, {} given".format(
            len(inputs))
        # Y = alpha * A * B + beta * C
        alpha = float(attr.get('alpha', 1.0))
        beta = float(attr.get('beta', 1.0))
        transA = int(attr.get('transA', 0))
        transB = int(attr.get('transB', 0))
        # get number of channels
        channels = infer_channels(inputs[1], params, not transB)
        if transA:
            inputs[0] = _sym.transpose(inputs[0], axes=(1, 0))
        if not transB:
            inputs[1] = _sym.transpose(inputs[1], axes=(1, 0))
        inputs[0] = _sym.flatten(inputs[0])
        return _sym.dense(
            alpha * inputs[0], inputs[1], beta * inputs[2], units=channels)


class MaxPool(Pool):
    """ Operator converter for MaxPool
    """
    name = 'max_pool'

    @classmethod
    def _impl_v8(cls, inputs, attr, params):
        return AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                'kernel_shape': 'pool_size',
                'pads': ('padding', (0, 0), revert_caffe2_pad),
                'storage_order': ('layout', 'NCHW', onnx_storage_order2layout),
            },
            # very weird attributes here in onnx, force check
            ignores=['dilations', 'auto_pad'],
            # TODO(higumachan): make sure ceil_mode in onnx, and layout?
            extras={'ceil_mode': False},
            custom_check=dimension_constraint())(inputs, attr, params)

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        return AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                'kernel_shape': 'pool_size',
                'pads': ('padding', (0, 0), revert_caffe2_pad),
                'storage_order': ('layout', 'NCHW', onnx_storage_order2layout),
                'ceil_mode': 'ceil_mode'
            },
            # very weird attributes here in onnx, force check
            ignores=['dilations', 'auto_pad'],
            custom_check=dimension_constraint())(inputs, attr, params)

class Mul(Elemwise):
    name = 'mul'


class Pad(OnnxOpConverter):
    """ Operator converter for Pad.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        pad_width = []
        pads = attr.pop('paddings')
        dims = int(len(pads) / 2)
        for i in range(dims):
            pad_width.append((pads[i], pads[i+dims]))
        attr['pad_width'] = pad_width

        return AttrCvt(
            op_name='pad',
            transforms={
                'value': 'pad_value',
            },
            ignores=['mode'],
            custom_check=(lambda attrs: attrs.get('mode', 'constant').decode("utf-8") == 'constant',
                          'split mode != constant'))(inputs, attr, params)

    @classmethod
    def _impl_v2(cls, inputs, attr, params):
        pad_width = []
        pads = attr.pop('pads')
        dims = int(len(pads) / 2)
        for i in range(dims):
            pad_width.append((pads[i], pads[i+dims]))
        attr['pad_width'] = pad_width

        return AttrCvt(
            op_name='pad',
            transforms={
                'value': 'pad_value',
            },
            ignores=['mode'],
            custom_check=(lambda attrs: attrs.get('mode', 'constant').decode("utf-8") == 'constant',
                          'split mode != constant'))(inputs, attr, params)


class ParametricSoftPlus(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
        beta = float(attr.get('beta', 1.0))
        return _sym.log(_sym.exp(beta * inputs[0]) + 1) * alpha


class Prelu(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Prelu need 2 inputs, {} given".format(
            len(inputs))
        channels = infer_channels(inputs[1], params, False)
        if channels == 1:
            return inputs[0] * inputs[1]
        return _sym.broadcast_mul(inputs[0], inputs[1])


class Reciprocal(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return 1.0 / inputs[0]


class Reshape(OnnxOpConverter):
    """ Operator converter for Reshape.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _sym.reshape(inputs[0], shape=attr['shape'])

    @classmethod
    def _impl_v5(cls, inputs, attr, params):
        if inputs[1].list_output_names()[0] in params:
            shape = tuple(params[inputs[1].list_output_names()[0]].asnumpy())
            out = _sym.reshape(inputs[0], shape=shape)
        else:
            out = _sym.reshape_like(inputs[0], inputs[1])

        return out

class Scale(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        scale = float(attr.get('scale', 1.0))
        return inputs[0] * scale


class Selu(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.6732))
        gamma = float(attr.get('gamma', 1.0507))
        return gamma * (
            -alpha * _sym.relu(1 - _sym.exp(inputs[0])) + _sym.relu(inputs[0]))


class ScaledTanh(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
        beta = float(attr.get('beta', 1.0))
        return _sym.tanh(beta * inputs[0]) * alpha


class SoftPlus(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _sym.log(_sym.exp(inputs[0]) + 1)


class Softsign(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return inputs[0] / (1 + Absolute.get_converter(1)(inputs, attr, params))


class Sub(Elemwise):
    name = 'sub'


class Sum(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Onnx Sum Operator
        for in_index in range(len(inputs) - 1):
            inputs[in_index + 1] = _sym.broadcast_add(inputs[in_index],
                                                      inputs[in_index + 1])

        return inputs[len(inputs) - 1]


class ThresholdedRelu(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
        alpha_tensor = _sym.full_like(inputs[0], fill_value=float(alpha))
        return _sym.elemwise_mul(inputs[0], _sym.greater(inputs[0], alpha_tensor))

class ImageScaler(OnnxOpConverter):

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        channelScale = attr['scale']
        bias_attr = attr['bias']
        bias = SymbolTable().new_const(np.array(bias_attr).reshape([3, 1, 1]))
        scaledChannel = _sym.__mul_scalar__(inputs[0], scalar=channelScale)
        ret = _sym.broadcast_add(scaledChannel, bias)
        return ret


def _broadcast_constraint():

    def _broadcast_check(attrs):
        if attrs.get('axis', None):
            return False
        return True

    return _broadcast_check, "Specifying broadcast axis not allowed."


def _fully_connected(opset):

    def _impl(inputs, attr, params):
        # get number of channels
        channels = infer_channels(inputs[1], params)
        attr['units'] = channels
        return AttrCvt('dense', ignores=['axis', 'axis_w'])(inputs, attr)

    return _impl


class Upsample(OnnxOpConverter):
    """ Operator converter for Upsample (nearest mode).
    """

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        scales = attr.get('scales')
        if not scales:
            #Here we are going to higher OPSET version.
            assert len(inputs) == 2, "Upsample op take 2 inputs, {} given".format(len(inputs))
            input_name = inputs[1].list_input_names()[0]
            scales = params[input_name].asnumpy()
            inputs = inputs[:1]
        assert len(scales) == 4 and scales[0] == 1.0 and scales[1] == 1.0 and scales[2] == scales[3]
        mode = attr.get('mode')
        if mode == b'nearest':
            method = "NEAREST_NEIGHBOR"
        elif mode == b'linear':
            method = "BILINEAR"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Upsample is not valid.'.format(mode))
        return _sym.upsampling(inputs[0], scale=int(scales[-1]), method=method, layout='NCHW')


class Shape(OnnxOpConverter):
    """ Operator converter for Shape.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Result of this operator is prominently used by reshape operator.
        # Just pass the input as it is so that reshape_like can be used there.
        print("Shape: Differently implemented in NNVM as a bypass (dummy operator)")
        return inputs[0]

class Cast(OnnxOpConverter):
    """ Operator converter for Cast.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return AttrCvt(op_name='cast', transforms={'to': 'dtype'})(inputs, attr)

    @classmethod
    def _impl_v5(cls, inputs, attr, params):
        try:
            from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
            attr['to'] = TENSOR_TYPE_TO_NP_TYPE[attr['to']]
        except ImportError as e:
            raise ImportError(
                "Unable to import onnx.mapping which is required {}".format(e))
        return AttrCvt(op_name='cast', transforms={'to': 'dtype'})(inputs, attr)


class Unsqueeze(OnnxOpConverter):
    """ Operator converter for Unsqueeze.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        for axes in attr['axes']:
            inputs[0] = _sym.expand_dims(inputs[0], axis=axes, num_newaxis=1)
        return inputs[0]


class Split(OnnxOpConverter):
    """ Operator converter for Split.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        attr['indices_or_sections'] = []
        index = 0
        for i in attr['split'][:-1]:
            index += i
            attr['indices_or_sections'].append(index)
        return AttrCvt(
            op_name='split',
            ignores=['split'])(inputs, attr, params)


class Slice(OnnxOpConverter):
    """ Operator converter for Slice.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if isinstance(attr['starts'], int):
            attr['starts'] = (attr['starts'],)
            attr['ends'] = (attr['ends'],)

        try:
            # Update the starts and ends according to axes if required.
            if isinstance(attr['axes'], int):
                attr['axes'] = (attr['axes'],)

            if (max(attr['axes']) + 1) != len(attr['axes']):
                new_axes = []
                new_starts = []
                new_ends = []
                pop_index = 0
                for i in range(max(attr['axes']) + 1):
                    if i in attr['axes']:
                        new_axes.append(i)
                        new_starts.append(attr['starts'][pop_index])
                        new_ends.append(attr['ends'][pop_index])
                        pop_index += 1
                    else:
                        new_axes.append(i)
                        new_starts.append(0)
                        new_ends.append(np.iinfo(np.int32).max)
                attr['axes'] = new_axes
                attr['starts'] = new_starts
                attr['ends'] = new_ends
        except KeyError:
            pass

        return AttrCvt(op_name='strided_slice',
                       transforms={'starts': 'begin',
                                   'ends': 'end'},
                       ignores=['axes'])(inputs, attr)

class Gather(OnnxOpConverter):
    """ Operator converter for Gather.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        return AttrCvt(op_name='take',
                       extras={'axis':axis})(inputs, attr)

class LRN(OnnxOpConverter):
    """ Operator converter for Local Response Normalization.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        """LRN support only NCHW format
        https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN
        """
        axis = 1
        alpha = attr.get('alpha', 0.0001)
        beta = attr.get('beta', 0.75)
        bias = attr.get('bias', 1.0)
        nsize = attr.get('size')
        return _sym.lrn(inputs[0], size=nsize, axis=axis,
                        alpha=alpha, beta=beta, bias=bias)

class Maximum(OnnxOpConverter):
    """ Operator converter for Maximum.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Expect minimum 2 inputs")
        _max = inputs[0]
        for i in range(1, len(inputs)):
            _max = AttrCvt(op_name='broadcast_max')([_max, inputs[i]], {})
        return _max

class Minimum(OnnxOpConverter):
    """ Operator converter for Minimum.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Expect minimum 2 inputs")
        _min = inputs[0]
        for i in range(1, len(inputs)):
            _min = AttrCvt(op_name='broadcast_min')([_min, inputs[i]], {})
        return _min

class Mean(OnnxOpConverter):
    """ Operator converter for Mean.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Expect minimum 2 inputs")
        count = len(inputs)
        _sum = inputs[0]
        for i in range(1, count):
            _sum = AttrCvt(op_name='broadcast_add')([_sum, inputs[i]], {})
        return _sum / count

class HardSigmoid(OnnxOpConverter):
    """ Operator converter for HardSigmoid.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = attr.get('alpha', 0.2)
        beta = attr.get('beta', 0.5)
        transformX = (inputs[0] * alpha) + beta
        attr = {'a_min':0, 'a_max':1}
        return AttrCvt(op_name='clip')([transformX], attr)

class ArgMax(OnnxOpConverter):
    """ Operator converter for ArgMax.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        keepdims = attr.get('keepdims', True)
        attr = {'axis':axis, 'keepdims':keepdims}
        return AttrCvt(op_name='argmax')(inputs, attr)

class ArgMin(OnnxOpConverter):
    """ Operator converter for ArgMin.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        keepdims = attr.get('keepdims', True)
        attr = {'axis':axis, 'keepdims':keepdims}
        return AttrCvt(op_name='argmin')(inputs, attr)

class Softmax(OnnxOpConverter):
    """ Operator converter for Softmax.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # set default value when axis is not set in the model
        if 'axis' not in attr:
            attr['axis'] = 1
        return AttrCvt(
            op_name='softmax',
            transforms={
                'axis': ('axis', 1),
            })(inputs, attr, params)

class ConstantFill(OnnxOpConverter):
    """ Operator converter for ConstantFill.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        is_full = True
        num_inputs = len(inputs)
        if 'shape' in attr:
            if num_inputs > 0:
                raise ImportError(
                    "Can't set shape and input tensor at a time")
            shape = attr.pop('shape')
        else:
            if num_inputs == 0:
                raise ImportError(
                    "Either shape attribute or input should be set")
            if 'input_as_shape' in attr and attr['input_as_shape']:
                shape = params[inputs[0].list_output_names()[0]].asnumpy()
            else:
                is_full = False

        if not is_full:
            if 'extra_shape' in attr:
                raise ImportError(
                    "Extra Shape not supported with fill_like")

            out = AttrCvt(
                op_name='full_like',
                transforms={'value': 'fill_value'},
                ignores=['dtype'])(inputs, attr)
            return _sym.cast(out, dtype=attr['dtype'].decode("utf-8"))
        if 'extra_shape' in attr:
            shape = shape + attr.pop('extra_shape')

        return AttrCvt(
            op_name='full',
            transforms={'value': 'fill_value'},
            extras={'shape':shape})(inputs, attr)

# compatible operators that do NOT require any conversion.
_identity_list = []


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
def _get_convert_map(opset):
    return {
        # defs/experimental
        'Identity': Renamer('copy'),
        # 'Affine'
        'ThresholdedRelu': ThresholdedRelu.get_converter(opset),
        'ScaledTanh': ScaledTanh.get_converter(opset),
        'ParametricSoftplus': ParametricSoftPlus.get_converter(opset),
        'ConstantFill': ConstantFill.get_converter(opset),
        # 'GivenTensorFill'
        'FC': AttrCvt('dense', ignores=['axis', 'axis_w']),
        'Scale': Scale.get_converter(opset),
        # 'GRUUnit'
        # 'ATen'
        'ImageScaler': ImageScaler.get_converter(opset),
        # 'MeanVarianceNormalization'
        # 'Crop'
        # 'Embedding'
        'Upsample' : Upsample.get_converter(opset),
        'SpatialBN': BatchNorm.get_converter(opset),

        # defs/generator
        # 'Constant' # Implemented
        # 'RandomUniform'
        # 'RandomNormal'
        # 'RandomUniformLike'
        # 'RandomNormalLike'

        # defs/logical

        # defs/math
        'Add': Add.get_converter(opset),
        'Sub': Sub.get_converter(opset),
        'Mul': Mul.get_converter(opset),
        'Div': Div.get_converter(opset),
        'Neg': Renamer('negative'),
        'Abs': Absolute.get_converter(opset),
        'Reciprocal': Reciprocal.get_converter(opset),
        'Floor': Renamer('floor'),
        'Ceil': Renamer('ceil'),
        'Sqrt': Renamer('sqrt'),
        'Relu': Renamer('relu'),
        'LeakyRelu': Renamer('leaky_relu'),
        'Selu': Selu.get_converter(opset),
        'Elu': Elu.get_converter(opset),
        'Exp': Renamer('exp'),
        'Log': Renamer('log'),
        'Tanh': Renamer('tanh'),
        'Pow': Renamer('broadcast_pow'),
        'PRelu': Prelu.get_converter(opset),
        'Sigmoid': Renamer('sigmoid'),
        'HardSigmoid': HardSigmoid.get_converter(opset),
        'Max': Maximum.get_converter(opset),
        'Min': Minimum.get_converter(opset),
        'Sum': Sum.get_converter(opset),
        'Mean': Mean.get_converter(opset),
        'Clip': AttrCvt('clip', transforms={'min': 'a_min', 'max': 'a_max'}),
        # softmax default axis is different in onnx
        'Softmax': Softmax.get_converter(opset),
        'LogSoftmax': AttrCvt('log_softmax', {'axis': ('axis', 1)}),
        # 'Hardmax'
        'Softsign': Softsign.get_converter(opset),
        'SoftPlus': SoftPlus.get_converter(opset),
        'Gemm': Gemm.get_converter(opset),
        'MatMul': Renamer('matmul'),

        # defs/nn
        'AveragePool': AveragePool.get_converter(opset),
        'MaxPool': MaxPool.get_converter(opset),
        'Conv': Conv.get_converter(opset),
        'ConvTranspose': ConvTranspose.get_converter(opset),
        'GlobalAveragePool': Renamer('global_avg_pool2d'),
        'GlobalMaxPool': Renamer('global_max_pool2d'),
        'BatchNormalization': BatchNorm.get_converter(opset),
        # 'InstanceNormalization'
        # 'LpNormalization'
        'Dropout': AttrCvt('dropout', {'ratio': 'rate'}, ignores=['is_test']),
        'Flatten': Renamer('flatten'),
        'LRN': LRN.get_converter(opset),

        # defs/reduction
        'ReduceMax': AttrCvt('max', {'axes': 'axis'}),
        'ReduceMin': AttrCvt('min', {'axes': 'axis'}),
        'ReduceSum': AttrCvt('sum', {'axes': 'axis'}),
        'ReduceMean': AttrCvt('mean', {'axes': 'axis'}),
        # 'ReduceProd'
        # 'ReduceLogSumExp'
        'ArgMax': ArgMax.get_converter(opset),
        'ArgMin': ArgMin.get_converter(opset),

        # defs/tensor
        'Cast': Cast.get_converter(opset),
        'Reshape': Reshape.get_converter(opset),
        'Concat': Renamer('concatenate'),
        'Split': Split.get_converter(opset),
        'Slice': Slice.get_converter(opset),
        'Transpose': AttrCvt('transpose', {'perm': 'axes'}),
        'Gather': Gather.get_converter(opset),
        'Squeeze': AttrCvt('squeeze', {'axes': 'axis'}),
        'Unsqueeze': Unsqueeze.get_converter(opset),
        'Pad': Pad.get_converter(opset),
        'Shape': Shape.get_converter(opset),
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

    def from_onnx(self, graph, opset):
        """Construct nnvm nodes from onnx graph.
        The inputs from onnx graph is vague, only providing "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph
        opset : opset version

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
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = _sym.Variable(
                    name=i_name, shape=self._params[i_name].shape)
            else:
                self._num_input += 1
                self._nodes[i_name] = _sym.Variable(name=i_name)
        # get list of unsupported ops
        convert_map = _get_convert_map(opset)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if op_name not in convert_map and \
               op_name != 'Constant' and \
               op_name not in _identity_list:
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = 'The following operators are not supported for frontend ONNX: '
            msg += ', '.join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)
        # construct nodes, nodes are stored as directed acyclic graph
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]
            if op_name == "Constant":
                t_proto = self._parse_attr(node.attribute)["value"]
                self._num_param += 1
                self._params[node.output[0]] = self._parse_array(t_proto)
                self._nodes[node.output[0]] = _sym.Variable(name=node.output[0],
                                                            shape=list(t_proto.dims))
            else:
                op = self._convert_operator(op_name, inputs, attr, opset)
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
            raise ImportError(
                "Unable to import onnx which is required {}".format(e))
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
            for f in ['t']:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ['tensors']:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ['g']:
                if a.HasField(f):
                    raise NotImplementedError(
                        "Filed {} is not supported in nnvm.".format(f))
            for f in ['graphs']:
                if list(getattr(a, f)):
                    raise NotImplementedError(
                        "Filed {} is not supported in nnvm.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _convert_operator(self,
                          op_name,
                          inputs,
                          attrs,
                          opset,
                          identity_list=None,
                          convert_map=None):
        """Convert from onnx operator to nnvm operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of nnvm.Symbol
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version
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
        convert_map = convert_map if convert_map else _get_convert_map(opset)
        if op_name in identity_list:
            sym = get_nnvm_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend ONNX.')
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


def from_onnx(model):
    """Load onnx graph which is a python protobuf object into nnvm graph.
    The companion parameters will be handled automatically.
    The inputs from onnx graph is vague, only providing "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    graph = model.graph
    try:
        opset = model.opset_import[0].version if model.opset_import else 1
    except AttributeError:
        opset = 1
    sym, params = g.from_onnx(graph, opset)
    return sym, params
