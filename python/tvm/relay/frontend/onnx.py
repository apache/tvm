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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
"""ONNX: Open Neural Network Exchange frontend for Relay."""
from __future__ import absolute_import as _abs

import numpy as np
import tvm
from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import module as _module
from .. import op as _op
from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, infer_value, infer_value_simulated, get_name

__all__ = ['from_onnx']


def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError(
            "Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)


def dimension_picker(prefix, surfix=''):
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 2:
            return prefix + '2d' + surfix
        msg = 'Only 2D kernels are supported for operator {}.'
        op_name = prefix + '2d'
        raise tvm.error.OpAttributeInvalid(msg.format(op_name))

    return _impl


def revert_caffe2_pad(pads):
    """Caffe2 requires two times the normal padding."""
    if len(pads) == 4:
        pads = pads[:2]
    elif len(pads) == 2:
        pass
    else:
        raise tvm.error.OpAttributeInvalid(
            'Number of pads must be either 2 or 4.')
    return pads


def onnx_storage_order2layout(storage_order):
    """converter of onnx storage order parameter to tvm storage order format"""
    if storage_order not in (0, 1):
        raise tvm.error.OpAttributeInvalid('Mode of storage_order must be either 0 or 1')

    return 'NCHW' if storage_order == 0 else 'NHWC'


def dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) == 2:
            return True
        return False

    return _dim_check, "Only 2d kernel supported."


class OnnxOpConverter(object):
    """ A helper class for holding onnx op converters.
    """

    @classmethod
    def get_converter(cls, opset):
        """ Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
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
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(
            len(inputs))
        op_name = cls.name
        conv_ops = ["conv2d", "conv2d_transpose"]
        if attr.get('broadcast', 0) and any(x in str(inputs[0]) for x in conv_ops):
            # TODO(zhreshold): remove hard coded infershape
            axis = int(attr.get('axis', 0))
            inputs[1] = _op.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_relay_op(op_name)(*inputs)


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
            ignores=['dilations', 'auto_pad'],
            # TODO(zhreshold): make sure ceil_mode in onnx, and layout?
            extras={'ceil_mode': False},
            custom_check=dimension_constraint())(inputs, attr, params)


class Absolute(OnnxOpConverter):
    """ Operator converter for Absolute.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.nn.relu(inputs[0]) + _op.nn.relu(_op.negative(inputs[0]))


class Add(Elemwise):
    """ Operator converter for Add.
    """
    name = 'add'


class AveragePool(Pool):
    """ Operator converter for AveragePool.
    """
    name = 'avg_pool'


class BatchNorm(OnnxOpConverter):
    """ Operator converter for BatchNorm.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # TODO(zhreshold): 'spatial' is not properly handled here.
        out = AttrCvt(
            op_name='batch_norm',
            ignores=['spatial', 'is_test', 'consumed_inputs', 'momentum'])(inputs, attr,
                                                                           params)
        return out[0]


class InstanceNorm(OnnxOpConverter):
    """ Operator converter for BatchNorm.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return AttrCvt(op_name='instance_norm')(inputs, attr, params)


class Conv(OnnxOpConverter):
    """ Operator converter for Conv.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        out = AttrCvt(op_name=dimension_picker('conv'),
                      transforms={
                          'kernel_shape': 'kernel_size',
                          'dilations': ('dilation', (0, 0)),
                          'pads': ('padding', (0, 0), revert_caffe2_pad),
                          'group': ('groups', 1)},
                      ignores=['auto_pad'],
                      custom_check=dimension_constraint())(inputs[:2], attr, params)
        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class ConvTranspose(OnnxOpConverter):
    """ Operator converter for ConvTranspose.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # get number of channels
        channels = infer_channels(inputs[1], True)
        attr['channels'] = channels
        groups = attr.pop('group')
        attr['groups'] = groups
        out = AttrCvt(
            op_name=dimension_picker('conv', '_transpose'),
            transforms={
                'kernel_shape': 'kernel_size',
                'dilations': ('dilation', (0, 0)),
                'pads': ('padding', (0, 0), revert_caffe2_pad)
            },
            disables=['output_shape'],
            custom_check=dimension_constraint())(inputs[:2], attr, params)
        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class Div(Elemwise):
    """ Operator converter for Divide.
    """
    name = 'divide'


class Elu(OnnxOpConverter):
    """ Operator converter for Elu.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
        return _expr.const(-alpha) * _op.nn.relu(_expr.const(1.) - _op.exp(inputs[0])) + \
                                     _op.nn.relu(inputs[0])


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
        channels = infer_channels(inputs[1], not transB)
        if transA:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not transB:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        inputs[0] = _op.nn.batch_flatten(inputs[0])
        out = _op.nn.dense(_expr.const(alpha) * inputs[0],
                           inputs[1], units=channels)
        return _op.nn.bias_add(out, _expr.const(beta) * inputs[2])


class MatMul(OnnxOpConverter):
    """ Operator converter for MatMul.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "MatMul op take 2 inputs, {} given".format(len(inputs))
        # Need to check input shape as batch matmul must be supported.
        a_shape = infer_shape(inputs[0])
        # When performing a batch matmul, we need to properly handle N-dim shapes.
        if len(a_shape) > 2:
            b_shape = infer_shape(inputs[1])
            # Convert a and b into 3 dimensional tensors.
            a = _op.reshape(inputs[0], [-1, a_shape[-2], a_shape[-1]])
            b = _op.reshape(inputs[1], [-1, b_shape[-2], b_shape[-1]])
            # Broadcast b to match batch size of a
            new_b_shape = list(infer_shape(b))
            new_a_shape = infer_shape(a)
            if new_a_shape[0] > new_b_shape[0]:
                new_b_shape[0] = new_a_shape[0]
                b = _op.broadcast_to(b, new_b_shape)
            # Transpose matrix dimensions of b.
            b = _op.transpose(b, [0, 2, 1])
            # Perform a batch matmul.
            output = _op.nn.batch_matmul(a, b)
            # Reshape output to original dimensions.
            return _op.reshape(output, [*a_shape[:-2], a_shape[-2], b_shape[-1]])
        # Otherwise a simple dense op will get the job done.
        input_1_t = _op.transpose(inputs[1], axes=(1, 0))
        return _op.nn.dense(inputs[0], input_1_t)


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
    """ Operator converter for Multiply.
    """
    name = 'multiply'


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
        pad_mode = attr.get('mode', 'constant').decode('utf-8')
        if pad_mode in ['constant', 'edge', 'reflect']:
            attr['pad_mode'] = pad_mode
            attr.pop('mode', None)
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value ' + pad_mode + ' in attribute "mode" is invalid for operator Pad.')

        return AttrCvt(
            _op.nn.pad,
            transforms={
                'value': 'pad_value',
            },
            )(inputs, attr, params)

    @classmethod
    def _impl_v2(cls, inputs, attr, params):
        pad_width = []
        pads = attr.pop('pads')
        dims = int(len(pads) / 2)
        for i in range(dims):
            pad_width.append((pads[i], pads[i+dims]))
        attr['pad_width'] = pad_width
        pad_mode = attr.get('mode', 'constant').decode('utf-8')
        if pad_mode in ['constant', 'edge', 'reflect']:
            attr['pad_mode'] = pad_mode
            attr.pop('mode', None)
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value ' + pad_mode + ' in attribute "mode" is invalid for operator Pad.')

        return AttrCvt(
            'pad',
            transforms={
                'value': 'pad_value',
            },
            )(inputs, attr, params)


class ParametricSoftPlus(OnnxOpConverter):
    """ Operator converter for ParametricSoftPlus.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = _expr.const(float(attr.get('alpha', 1.0)))
        beta = _expr.const(float(attr.get('beta', 1.0)))
        return _op.log(_op.exp(beta * inputs[0]) + _expr.const(1.)) * alpha


class Prelu(OnnxOpConverter):
    """ Operator converter for Prelu.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Prelu need 2 inputs, {} given".format(len(inputs))
        return _op.nn.prelu(inputs[0], inputs[1])


class Reciprocal(OnnxOpConverter):
    """ Operator converter for Reciprocal.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _expr.const(1.0) / inputs[0]


class Flatten(OnnxOpConverter):
    """ Operator converter for Flatten.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 1)
        if axis == 1:
            out = _op.nn.batch_flatten(inputs[0])
        else:
            newshape = [0] * (axis + 1)
            newshape[axis] = -1
            out = _op.reshape(inputs[0], list(newshape))
        return out


class Reshape(OnnxOpConverter):
    """ Operator converter for Reshape.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.reshape(inputs[0], attr['shape'])

    @classmethod
    def _impl_v5(cls, inputs, attr, params):
        if get_name(inputs[1]) in params:
            shape = tuple(params[inputs[1].name_hint].asnumpy())
            out = _op.reshape(inputs[0], shape)
        else:
            data, shape = inputs
            static_shape = infer_value_simulated(shape, params)
            out = _op.reshape(data, newshape=tuple(
                static_shape.asnumpy().astype('int32')))
        return out


class DepthToSpace(OnnxOpConverter):
    """ Operator converter for DepthToSpace.
    """

    @classmethod
    def _impl_v11(cls, inputs, attr, params):

        block_size = int(attr['blocksize'])
        mode = attr.get("mode", "DCR")

        # handle NCHW layout
        indata = infer_value_simulated(inputs[0], params)
        in_n, in_c, in_h, in_w = indata.shape

        # reshape to proper output
        new_c = int(in_c / (block_size * block_size))
        new_h = in_h * block_size
        new_w = in_w * block_size
        newshape = (in_n, new_c, new_h, new_w)

        if mode == "DCR":
            # expand input to larger dimension.
            expanded = _op.reshape(inputs[0],
                                   newshape=(in_n, block_size, block_size, new_c, in_h, in_w))
            # reorder to expand spatial blocks.
            transposed = _op.transpose(expanded, axes=(0, 3, 4, 1, 5, 2))

        else:  # CRD mode
            # expand input to larger dimension.
            expanded = _op.reshape(inputs[0],
                                   newshape=(in_n, new_c, block_size, block_size, in_h, in_w))
            # reorder to expand spatial blocks.
            transposed = _op.transpose(expanded, axes=(0, 1, 4, 2, 5, 3))

        return AttrCvt(op_name="reshape",
                       extras={'newshape': newshape},
                       ignores=['mode', 'blocksize'])([transposed], attr)


class SpaceToDepth(OnnxOpConverter):
    """ Operator converter for SpaceToDepth.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):

        block_size = int(attr['blocksize'])

        # handle NCHW layout
        indata = infer_value_simulated(inputs[0], params)
        in_n, in_c, in_h, in_w = indata.shape

        # reshape to proper output
        new_c = in_c * (block_size * block_size)
        new_h = int(in_h / block_size)
        new_w = int(in_w / block_size)
        newshape = (in_n, new_c, new_h, new_w)

        # expand input to larger dimension.
        expanded = _op.reshape(inputs[0],
                               newshape=(in_n, in_c, new_h, block_size, new_w, block_size))
        # reorder to expand spatial blocks.
        transposed = _op.transpose(expanded, axes=(0, 3, 5, 1, 2, 4))

        return AttrCvt(op_name="reshape",
                       extras={'newshape': newshape},
                       ignores=['blocksize'])([transposed], attr)


class Concat(OnnxOpConverter):
    """ Operator converter for Concat.
    """

    @classmethod
    def _impl_v1(cls, inputs, args, params):
        return AttrCvt(op_name='concatenate')((inputs,), args)

class Scale(OnnxOpConverter):
    """ Operator converter for Scale.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        scale = float(attr.get('scale', 1.0))
        return inputs[0] * _expr.const(scale)


class Selu(OnnxOpConverter):
    """ Operator converter for Selu.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.6732))
        gamma = float(attr.get('gamma', 1.0507))
        return _expr.const(gamma) * (_expr.const(-alpha) *
                                     _op.nn.relu(_expr.const(1.) - _op.exp(inputs[0])) +
                                     _op.nn.relu(inputs[0]))


class ScaledTanh(OnnxOpConverter):
    """ Operator converter for ScaledTanh.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
        beta = float(attr.get('beta', 1.0))
        return _op.tanh(_expr.const(beta) * inputs[0]) * _expr.const(alpha)


class SoftPlus(OnnxOpConverter):
    """ Operator converter for SoftPlus.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.log(_op.exp(inputs[0]) + _expr.const(1.))


class Softsign(OnnxOpConverter):
    """ Operator converter for Softsign.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return inputs[0] / (_expr.const(1.) + Absolute.get_converter(1)(inputs, attr, params))


class Sub(Elemwise):
    """ Operator converter for Subtract.
    """
    name = 'subtract'


class Sum(OnnxOpConverter):
    """ Operator converter for Sum.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Onnx Sum Operator
        for in_index in range(len(inputs) - 1):
            inputs[in_index + 1] = _op.add(inputs[in_index], inputs[in_index + 1])

        return inputs[len(inputs) - 1]


class ThresholdedRelu(OnnxOpConverter):
    """ Operator converter for ThresholdedRelu.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 0.0))
        alpha_tensor = _op.full_like(inputs[0], fill_value=_expr.const(alpha))
        mask = _op.greater(inputs[0], alpha_tensor).astype("float32")
        return inputs[0] * mask


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
            scales = params[inputs[1].name_hint].asnumpy()
            inputs = inputs[:1]
        assert len(scales) == 4 and scales[0] == 1.0 and scales[1] == 1.0
        mode = attr.get('mode')
        if mode == b'nearest':
            method = "nearest_neighbor"
        elif mode == b'linear':
            method = "bilinear"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Upsample is not valid.'.format(mode))
        attr = {'scale_h':scales[-2], 'scale_w':scales[-1], 'method':method,
                'layout':'NCHW', 'align_corners':True}
        return AttrCvt('upsampling')(inputs, attr)


class Shape(OnnxOpConverter):
    """ Operator converter for Shape.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # TODO(@jroesch): use shape_of once it has been fixed)
        return _op.shape_of(inputs[0])

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
            attr['to'] = str(TENSOR_TYPE_TO_NP_TYPE[attr['to']])
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
            inputs[0] = _op.expand_dims(inputs[0], axis=axes, num_newaxis=1)
        return inputs[0]


class Split(OnnxOpConverter):
    """ Operator converter for Split.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        splits = attr.get('split', False)
        if splits:
            attr['indices_or_sections'] = []
            index = 0
            for i in splits[:-1]:
                index += i
                attr['indices_or_sections'].append(index)
        # When splits isnt specified divide evenly over axis.
        else:
            in_shape = infer_shape(inputs[0])
            attr['indices_or_sections'] = in_shape[attr['axis']]
        return AttrCvt(
            'split',
            ignores=['split'])(inputs, attr, params)


class Slice(OnnxOpConverter):
    """ Operator converter for Slice.
    """

    @classmethod
    def _common(cls, starts, ends, axes):
        new_axes = []
        new_starts = []
        new_ends = []
        pop_index = 0
        for i in range(max(axes) + 1):
            if i in axes:
                new_axes.append(i)
                new_starts.append(starts[pop_index])
                new_ends.append(ends[pop_index])
                pop_index += 1
            else:
                new_axes.append(i)
                new_starts.append(0)
                new_ends.append(np.iinfo(np.int32).max)
        return new_starts, new_ends, new_axes

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
                new_starts, new_ends, new_axes = cls._common(
                    attr['starts'], attr['ends'], attr['axes'])
                attr['axes'] = new_axes
                attr['starts'] = new_starts
                attr['ends'] = new_ends
        except KeyError:
            pass

        return AttrCvt('strided_slice',
                       transforms={'starts': 'begin',
                                   'ends': 'end'},
                       ignores=['axes'])(inputs, attr)

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        starts = params[get_name(inputs[1])].asnumpy()
        ends = params[get_name(inputs[2])].asnumpy()

        # Update the starts and ends according to axes if required.
        if len(inputs) >= 4:
            axes = params[get_name(inputs[3])].asnumpy()

            if max(axes + 1) != len(axes):
                new_starts, new_ends, _ = cls._common(
                    starts, ends, axes)
                starts = new_starts
                ends = new_ends
        return _op.strided_slice(inputs[0], begin=starts, end=ends)


class Gather(OnnxOpConverter):
    """ Operator converter for Gather.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        return AttrCvt('take',
                       extras={'axis':axis})(inputs, {})


class Greater(OnnxOpConverter):
    """ Operator logical greater.
    """
    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        return _op.greater(inputs[0], inputs[1])


class Less(OnnxOpConverter):
    """ Operator logical less than.
    """
    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        return _op.less(inputs[0], inputs[1])


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
        attr = {'size':nsize, 'axis':axis, 'alpha':alpha, 'beta':beta, 'bias':bias}
        return AttrCvt('lrn')(inputs, attr)

class Maximum(OnnxOpConverter):
    """ Operator converter for Maximum.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Expect minimum 2 inputs")
        _max = inputs[0]
        for i in range(1, len(inputs)):
            _max = AttrCvt('maximum')([_max, inputs[i]], {})
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
            _min = AttrCvt('minimum')([_min, inputs[i]], {})
        return _min

class Mean(OnnxOpConverter):
    """ Operator converter for Mean.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not isinstance(inputs, list) or len(inputs) < 2:
            raise ValueError("Expect minimum 2 inputs")
        # avoid overflow
        concat = _op.concatenate([_op.expand_dims(x, axis=0) for x in inputs], axis=0)
        return _op.mean(concat, axis=0, keepdims=False)

class HardSigmoid(OnnxOpConverter):
    """ Operator converter for HardSigmoid.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = attr.get('alpha', 0.2)
        beta = attr.get('beta', 0.5)
        transformX = (inputs[0] * _expr.const(alpha)) + _expr.const(beta)
        attr = {'a_min':0, 'a_max':1}
        return AttrCvt('clip')([transformX], attr)

class Reduce(OnnxOpConverter):
    """ Operator converter for reduce ops.
    """
    name = ''
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if 'axes' in attr:
            axis = attr.get('axes', 0)
        else:
            axis_len = len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
        attr = {'axis':axis, 'keepdims':attr.get('keepdims', True)}
        return AttrCvt(cls.name)(inputs, attr)

class ReduceMax(Reduce):
    """ Operator converter for ReduceMax.
    """
    name = 'max'

class ReduceMin(Reduce):
    """ Operator converter for ReduceMin.
    """
    name = 'min'

class ReduceSum(Reduce):
    """ Operator converter for ReduceSum.
    """
    name = 'sum'

class ReduceMean(Reduce):
    """ Operator converter for ReduceMean.
    """
    name = 'mean'

class ReduceProd(Reduce):
    """ Operator converter for ReduceProd.
    """
    name = 'prod'

class ArgMax(OnnxOpConverter):
    """ Operator converter for ArgMax.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        keepdims = attr.get('keepdims', True)
        attr = {'axis':axis, 'keepdims':keepdims}
        return AttrCvt('argmax')(inputs, attr)

class ArgMin(OnnxOpConverter):
    """ Operator converter for ArgMin.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        keepdims = attr.get('keepdims', True)
        attr = {'axis':axis, 'keepdims':keepdims}
        return AttrCvt('argmin')(inputs, attr)

class Softmax(OnnxOpConverter):
    """ Operator converter for Softmax.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # set default value when axis is not set in the model
        if 'axis' not in attr:
            attr['axis'] = 1
        return AttrCvt('softmax', transforms={'axis': ('axis', 1)})(inputs, attr, params)


class OneHot(OnnxOpConverter):
    """ Operator converter for OneHot.
    """
    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        # Extract relay one_hot inputs.
        indices, depth, values = inputs
        # Split onnx on off values into two separate expressions.
        off_value, on_value = _op.take(
            values, _op.const(0)), _op.take(values, _op.const(1))
        # Extract the datatype of the output from on_value.
        dtype = infer_type(on_value).checked_type.dtype
        # Convert depth into an integer.
        depth = int(infer_value(depth, params).asnumpy()[0])
        # set default value when axis is not set in the model
        if 'axis' not in attr:
            attr['axis'] = -1
        return _op.one_hot(indices,
                           on_value,
                           off_value,
                           depth,
                           int(attr['axis']),
                           dtype=dtype)


class ConstantOfShape(OnnxOpConverter):
    """ Operator converter for ConstantOfShape.
    """
    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if 'value' in attr:
            np_value = get_numpy(attr.pop('value'))[0]
            value = _expr.const(np_value)
            dtype = np_value.dtype.name
        else:
            value = _expr.const(0)
            dtype = 'float32'
        static_shape = infer_value_simulated(inputs[0], params)
        output = _op.full(
            value, shape=tuple(static_shape.asnumpy().astype('int32')), dtype=dtype)
        return output


class Sign(OnnxOpConverter):
    """ Operator converter for Sign.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.sign(inputs[0])

class Equal(Elemwise):
    """ Operator converter for Equal.
    """
    name = 'equal'


class Not(Elemwise):
    """ Operator converter for Not.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.logical_not(inputs[0])


class And(Elemwise):
    """ Operator converter for And.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.logical_and(inputs[0], inputs[1])


class Tile(Elemwise):
    """Operator converter for Tile
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if 'repeats' not in attr:
            raise tvm.error.OpAttributeInvalid('Attribute "repeats" should be set '
                                               'for operator Tile.')
        reps = attr.pop('repeats')  # The number of times repeating the tensor data.
        return _op.tile(inputs[0], reps)

    @classmethod
    def _impl_v6(cls, inputs, attr, params):
        reps = tuple(infer_value_simulated(
            inputs[1], params).asnumpy().astype('int32'))
        return _op.tile(inputs[0], reps)

class Erf(OnnxOpConverter):
    """Operator converter for Erf
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.erf(inputs[0])

class Where(OnnxOpConverter):
    """Operator converter for Where
    """
    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        # x and y can be broadcasted
        condition_shape = infer_shape(inputs[0])
        x_shape = infer_shape(inputs[1])
        y_shape = infer_shape(inputs[2])
        if len(condition_shape) > len(x_shape):
            inputs[1] = _op.broadcast_to(inputs[1], condition_shape)
        if len(condition_shape) > len(y_shape):
            inputs[2] = _op.broadcast_to(inputs[2], condition_shape)
        return _op.where(inputs[0], inputs[1], inputs[2])

class Or(Elemwise):
    """ Operator converter for Or.
    """
    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        return _op.logical_or(inputs[0], inputs[1])


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
        'ConstantOfShape': ConstantOfShape.get_converter(opset),
        # 'GivenTensorFill'
        'FC': AttrCvt('dense', ignores=['axis', 'axis_w']),
        'Scale': Scale.get_converter(opset),
        # 'GRUUnit'
        # 'ATen'
        # 'ImageScaler'
        # 'MeanVarianceNormalization'
        # 'Crop'
        # 'Embedding'
        'Upsample': Upsample.get_converter(opset),
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
        'Greater': Greater.get_converter(opset),
        'Less': Less.get_converter(opset),
        'Log': Renamer('log'),
        'Tanh': Renamer('tanh'),
        'Pow': Renamer('power'),
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
        'OneHot': OneHot.get_converter(opset),
        # 'Hardmax'
        'Softsign': Softsign.get_converter(opset),
        'SoftPlus': SoftPlus.get_converter(opset),
        'Gemm': Gemm.get_converter(opset),
        'MatMul': MatMul.get_converter(opset),

        # defs/nn
        'AveragePool': AveragePool.get_converter(opset),
        'MaxPool': MaxPool.get_converter(opset),
        'Conv': Conv.get_converter(opset),
        'ConvTranspose': ConvTranspose.get_converter(opset),
        'GlobalAveragePool': Renamer('global_avg_pool2d'),
        'GlobalMaxPool': Renamer('global_max_pool2d'),
        'BatchNormalization': BatchNorm.get_converter(opset),
        'InstanceNormalization': InstanceNorm.get_converter(opset),
        # 'LpNormalization'
        'Dropout': AttrCvt('dropout', {'ratio': 'rate'}, ignores=['is_test']),
        'Flatten': Flatten.get_converter(opset),
        'LRN': LRN.get_converter(opset),

        # defs/reduction
        'ReduceMax': ReduceMax.get_converter(opset),
        'ReduceMin': ReduceMin.get_converter(opset),
        'ReduceSum': ReduceSum.get_converter(opset),
        'ReduceMean': ReduceMean.get_converter(opset),
        'ReduceProd': ReduceProd.get_converter(opset),
        # 'ReduceProd'
        # 'ReduceLogSumExp'
        'ArgMax': ArgMax.get_converter(opset),
        'ArgMin': ArgMin.get_converter(opset),

        # defs/tensor
        'Cast': Cast.get_converter(opset),
        'Reshape': Reshape.get_converter(opset),
        'Concat': Concat.get_converter(opset),
        'Split': Split.get_converter(opset),
        'Slice': Slice.get_converter(opset),
        'Transpose': AttrCvt('transpose', {'perm': 'axes'}),
        'DepthToSpace': DepthToSpace.get_converter(opset),
        'SpaceToDepth': SpaceToDepth.get_converter(opset),
        'Gather': Gather.get_converter(opset),
        'Squeeze': AttrCvt('squeeze', {'axes': 'axis'}),
        'Unsqueeze': Unsqueeze.get_converter(opset),
        'Pad': Pad.get_converter(opset),
        'Shape': Shape.get_converter(opset),
        'Sign': Sign.get_converter(opset),
        'Equal': Equal.get_converter(opset),
        'Not': Not.get_converter(opset),
        'And': And.get_converter(opset),
        'Tile': Tile.get_converter(opset),
        'Erf': Erf.get_converter(opset),
        'Where': Where.get_converter(opset),
        'Or': Or.get_converter(opset)
    }


class GraphProto(object):
    """A helper class for handling Relay expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph
    """

    def __init__(self, shape, dtype):
        self._nodes = {}
        self._params = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape if shape else {}
        self._dtype = dtype

    def from_onnx(self, graph, opset):
        """Construct Relay expression from ONNX graph.

        Onnx graph is a python protobuf object.
        The companion parameters will be handled automatically.
        However, the input names from onnx graph is vague, mixing inputs and
        network weights/bias such as "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        opset : opset version

        Returns
        -------
        mod : tvm.relay.Module
            The returned relay module

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # parse network inputs to relay, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = self._parse_array(init_tensor)
            self._nodes[init_tensor.name] = new_var(init_tensor.name,
                                                    shape=self._params[init_tensor.name].shape,
                                                    dtype=self._params[init_tensor.name].dtype)
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name = self._parse_value_proto(i)
            d_type = self._parse_dtype(i, 'float32')
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = new_var(i_name,
                                              shape=self._params[i_name].shape,
                                              dtype=self._params[i_name].dtype)
            else:
                self._num_input += 1
                if i_name in self._shape:
                    tshape = self._shape[i_name]
                else:
                    raise ValueError("Must provide an input shape for `{0}`.".format(i_name))
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=tshape, dtype=dtype)
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
                # We should convert scalar integers to int32, to normalize.
                array = self._parse_array(t_proto)
                if len(array.shape) == 0 and array.dtype == 'int64':
                    array = _nd.array(array.asnumpy().astype('int32'))
                self._params[node.output[0]] = array
                self._nodes[node.output[0]] = new_var(
                    node.output[0],
                    shape=list(t_proto.dims),
                    dtype=array.dtype)
            else:
                i_name = self._parse_value_proto(node)
                attr['tvm_custom'] = {}
                attr['tvm_custom']['name'] = i_name

                op = self._convert_operator(op_name, inputs, attr, opset)
                node_output = self._fix_outputs(op_name, node.output)
                if not isinstance(op, _expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(op)
                assert len(node_output) == outputs_num, (
                    "Number of output mismatch {} vs {} in {}.".format(
                        len(node_output), outputs_num, op_name))
                if outputs_num == 1:
                    self._nodes[node_output[0]] = op
                else:
                    for k, i in zip(list(node_output), range(len(node_output))):
                        self._nodes[k] = op[i]

        # now return the outputs
        outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        func = _expr.Function(analysis.free_vars(outputs), outputs)
        return _module.Module.from_expr(func), self._params

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_dtype(self, value_proto, dtype):
        """Parse dtype."""
        try:
            from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
            return TENSOR_TYPE_TO_NP_TYPE[value_proto.type.tensor_type.elem_type].name
        except AttributeError:
            return dtype

    def _parse_array(self, tensor_proto):
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return _nd.array(np_array)

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
                        "Filed {} is not supported in relay.".format(f))
            for f in ['graphs']:
                if list(getattr(a, f)):
                    raise NotImplementedError(
                        "Filed {} is not supported in relay.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _convert_operator(self,
                          op_name,
                          inputs,
                          attrs,
                          opset):
        """Convert ONNX operator into a Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.expr.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version

        Returns
        -------
        sym : tvm.relay.expr.Function
            Converted relay function
        """
        convert_map = _get_convert_map(opset)
        if op_name in _identity_list:
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise NotImplementedError(
                "Operator {} not implemented.".format(op_name))
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

def from_onnx(model,
              shape=None,
              dtype="float32",
              opset=None):
    """Convert a ONNX model into an equivalent Relay Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    Returns
    -------
    mod : tvm.relay.Module
        The relay module for compilation

    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    """
    try:
        import onnx
        if hasattr(onnx.checker, 'check_model'):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except onnx.onnx_cpp2py_export.checker.ValidationError as e:
                import warnings
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    g = GraphProto(shape, dtype)
    graph = model.graph
    if opset is None:
        try:
            opset = model.opset_import[0].version if model.opset_import else 1
        except AttributeError:
            opset = 1
    mod, params = g.from_onnx(graph, opset)
    return mod, params
