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
# pylint: disable=import-outside-toplevel
"""ONNX: Open Neural Network Exchange frontend for Relay."""
import numpy as np
import tvm
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .common import AttrCvt, Renamer
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, infer_value, infer_value_simulated, get_name

__all__ = ['from_onnx']


class onnx_input():
    """ Dual purpose list or dictionary access object."""

    def __init__(self):
        self.input_keys = []
        self.input_dict = {}

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.input_dict[self.input_keys[item]]
        if isinstance(item, str):
            if item not in self.input_keys:
                return None
            return self.input_dict[item]
        if isinstance(item, slice):
            keys = self.input_keys[item]
            return [self.input_dict[key] for key in keys]

        raise ValueError("Only integer, string, and slice accesses allowed.")

    def __setitem__(self, item, value):
        if isinstance(item, int):
            self.input_dict[self.input_keys[item]] = value
        elif isinstance(item, str):
            self.input_keys.append(item)
            self.input_dict[item] = value
        else:
            raise ValueError("Only integer and string indexed writes allowed.")

    def keys(self):
        return self.input_keys

    def __len__(self):
        return len(self.input_keys)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.input_keys):
            output = self.input_dict[self.input_keys[self.n]]
            self.n += 1
            return output

        raise StopIteration


def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError(
            "Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)


def dimension_picker(prefix, suffix=''):
    """Check that dimensions are supported."""
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 1:
            return prefix + '1d' + suffix
        if len(kernel) == 2:
            return prefix + '2d' + suffix
        if len(kernel) == 3:
            return prefix + '3d' + suffix
        msg = 'Only 1D, 2D, and 3D kernels are supported for operator {}.'
        op_name = prefix + '1d/2d/3d'
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


def get_pad_pair(input1d, kernel1d, stride1d):
    """infer pad size"""
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return [pad_before, pad_after]


def onnx_default_layout(dims):
    if dims == 1:
        return 'NCW'
    if dims == 2:
        return 'NCHW'
    if dims == 3:
        return 'NCDHW'

    msg = "Only 1D, 2D and 3D layouts are currently supported"
    raise tvm.error.OpAttributeInvalid(msg.format(op_name))


def onnx_storage_order2layout(storage_order, dims=2):
    """converter of onnx storage order parameter to tvm storage order format"""
    if storage_order not in (0, 1):
        raise tvm.error.OpAttributeInvalid('Mode of storage_order must be either 0 or 1')

    if dims == 1:
        return 'NCW' if storage_order == 0 else 'NWC'
    if dims == 2:
        return 'NCHW' if storage_order == 0 else 'NHWC'
    if dims == 3:
        return 'NCDHW' if storage_order == 0 else 'NDHWC'

    msg = "Only 1D, 2D and 3D layouts are currently supported"
    raise tvm.error.OpAttributeInvalid(msg.format(op_name))


def dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) in [1, 2, 3]:
            return True
        return False

    return _dim_check, "Only 1d, 2d and 3d kernel supported."


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


class Unary(OnnxOpConverter):
    """ A helper class for unary op converters.
    """
    name = ''

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 1, "Unary math op {} takes 1 input, {} given".format(
            cls.name, len(inputs))
        op_name = cls.name
        return get_relay_op(op_name)(*inputs)


class Elemwise(OnnxOpConverter):
    """ A helper class for elemwise op converters.
    """
    name = ''

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Math op {} take 2 inputs, {} given".format(
            cls.name, len(inputs))
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
        input_shape = infer_shape(inputs[0])
        if 'auto_pad' in attr:
            attr['auto_pad'] = attr['auto_pad'].decode('utf-8')
            if attr['auto_pad'] in ('SAME_UPPER', 'SAME_LOWER'):
                pad_tuple = []
                for axis in range(len(input_shape) - 2):
                    axis_shape = input_shape[2 + axis]
                    stride = attr['strides'][axis]
                    kernel = attr['kernel_shape'][axis]
                    pad = get_pad_pair(axis_shape, kernel, stride)
                    pad_tuple.append(pad)
                pad_tuple = tuple([val for pair in zip(*pad_tuple) for val in pair])
                attr['pads'] = pad_tuple
            elif attr['auto_pad'] == 'VALID':
                attr['pads'] = 0
            elif attr['auto_pad'] == 'NOTSET':
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator {} is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr['auto_pad'], cls.name))
            attr.pop("auto_pad")

        if 'storage_order' in attr:
            attr['layout'] = onnx_storage_order2layout(attr['storage_order'],
                                                       dims=(len(input_shape) - 2))
        else:
            attr['layout'] = onnx_default_layout(dims=(len(input_shape) - 2))

        return AttrCvt(
            op_name=dimension_picker(cls.name),
            transforms={
                'kernel_shape': 'pool_size',
                'pads': ('padding', 0)
            },
            ignores=['dilations'],
            custom_check=dimension_constraint())(inputs, attr, params)


class Absolute(Unary):
    """ Operator converter for Absolute.
    """
    name = 'abs'


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
        # Use shape of input to determine convolution type.
        input_shape = infer_shape(inputs[0])

        if 'auto_pad' in attr:
            attr['auto_pad'] = attr['auto_pad'].decode('utf-8')
            if attr['auto_pad'] in ('SAME_UPPER', 'SAME_LOWER'):
                pad_tuple = []
                for axis in range(len(input_shape) - 2):
                    axis_shape = input_shape[2 + axis]
                    stride = attr['strides'][axis]
                    kernel = attr['kernel_shape'][axis]
                    dilation = attr['dilations'][axis]
                    dilated_kernel = (kernel - 1) * dilation + 1
                    pad = get_pad_pair(axis_shape, dilated_kernel, stride)
                    pad_tuple.append(pad)
                pad_tuple = tuple([val for pair in zip(*pad_tuple) for val in pair])
                attr['pads'] = pad_tuple
            elif attr['auto_pad'] == 'VALID':
                attr['pads'] = tuple([0 for i in range(len(input_shape) - 2)])
            elif attr['auto_pad'] == 'NOTSET':
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr['auto_pad']))
            attr.pop('auto_pad')
        elif len(attr['kernel_shape']) == 2:
            sym_pad = True
            padding = attr['pads']
            for i in range(0, len(padding), 2):
                sym_pad = sym_pad and padding[i] == padding[i + 1]

            if sym_pad:
                attr['pads'] = padding[0::2]

        out = AttrCvt(
            op_name=dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'dilations': ('dilation', 1),
                'pads': ('padding', 0),
                'group': ('groups', 1)
            },
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
        # infer pads for auto_pad
        if 'auto_pad' in attr:
            attr['auto_pad'] = attr['auto_pad'].decode('utf-8')
            if attr['auto_pad'] in ('SAME_UPPER', 'SAME_LOWER'):
                input_shape = infer_shape(inputs[0])
                in_h, in_w = input_shape[2], input_shape[3]
                stride_h, stride_w = attr['strides']
                kernel_h, kernel_w = attr['kernel_shape']
                dilation_h, dilation_w = attr['dilations']
                dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
                dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
                pad_v = get_pad_pair(in_h, dilated_kernel_h, stride_h)
                pad_h = get_pad_pair(in_w, dilated_kernel_w, stride_w)
                attr['pads'] = (pad_v[0], pad_h[0], pad_v[1], pad_h[1])
            elif attr['auto_pad'] == 'VALID':
                attr['pads'] = (0, 0)
            elif attr['auto_pad'] == 'NOTSET':
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr['auto_pad']))
            attr.pop('auto_pad')

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
        pad_mode = attr.get('mode', b'constant').decode('utf-8')
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
        pad_mode = attr.get('mode', b'constant').decode('utf-8')
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
            # pop shape out of parameters since it wont be needed later.
            shape = tuple(params.pop(inputs[1].name_hint).asnumpy())
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
        mode = attr.get('mode', b'DCR').decode('utf-8')
        return _op.nn.depth_to_space(inputs[0], block_size, mode=mode)


class SpaceToDepth(OnnxOpConverter):
    """ Operator converter for SpaceToDepth.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):

        block_size = int(attr['blocksize'])
        return _op.nn.space_to_depth(inputs[0], block_size)


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


class Affine(OnnxOpConverter):
    """ Operator converter for Affine transformation.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = _expr.const(attr.get('alpha', 1.0))
        beta = _expr.const(attr.get('beta', 0.0))
        return (alpha * inputs[0]) + beta


class ThresholdedRelu(OnnxOpConverter):
    """ Operator converter for ThresholdedRelu.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get('alpha', 1.0))
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
        assert scales[0] == 1.0 and scales[1] == 1.0
        input_shape = infer_shape(inputs[0])
        dims = len(input_shape)
        mode = attr.get('mode')
        if mode == b'nearest':
            method = "nearest_neighbor"
        elif mode == b'linear':
            method = "trilinear" if dims == 5 else "bilinear"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Upsample is not valid.'.format(mode))
        attr = {'scale_h': scales[-2],
                'scale_w': scales[-1],
                'method': method}
        if dims == 5:
            assert len(scales) == 5
            attr['scale_d'] = scales[-3]
            attr['layout'] = 'NCDHW'
            op_name = 'upsampling3d'
        else:
            assert len(scales) == 4
            attr['layout'] = 'NCHW'
            attr['align_corners'] = True
            op_name = 'upsampling'
        return AttrCvt(op_name)(inputs, attr)

class Shape(OnnxOpConverter):
    """ Operator converter for Shape.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.shape_of(inputs[0], "int64")

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
                       extras={'axis': axis})(inputs, {})


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
        attr = {'size': nsize, 'axis': axis, 'alpha': alpha, 'beta': beta, 'bias': bias}
        return AttrCvt('lrn')(inputs, attr)

class Maximum(OnnxOpConverter):
    """ Operator converter for Maximum.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not isinstance(inputs, (list, onnx_input)) or len(inputs) < 2:
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
        if not isinstance(inputs, (list, onnx_input)) or len(inputs) < 2:
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
        if not isinstance(inputs, (list, onnx_input)) or len(inputs) < 2:
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
        attr = {'a_min': 0, 'a_max': 1}
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
        attr = {'axis': axis, 'keepdims': attr.get('keepdims', True)}
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
        attr = {'axis': axis, 'keepdims': keepdims}
        return AttrCvt('argmax')(inputs, attr)

class ArgMin(OnnxOpConverter):
    """ Operator converter for ArgMin.
    """
    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get('axis', 0)
        keepdims = attr.get('keepdims', True)
        attr = {'axis': axis, 'keepdims': keepdims}
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
        condition_shape = infer_shape(inputs[0])
        x_shape = infer_shape(inputs[1])
        y_shape = infer_shape(inputs[2])

        # condition, x, and y can all be broadcasted.
        # broadcast each of them to the longest shape.
        # if two shapes have the same number of dimensions,
        # try to choose the one that doesn't have "1" as
        # a dimension.
        shapes = [condition_shape, x_shape, y_shape]
        shape_lens = [len(shape) for shape in shapes]
        max_size = max(shape_lens)
        max_size_idxs = [i for i, x in enumerate(shape_lens) if x == max_size]
        broadcast_idx = max_size_idxs[0]
        if len(max_size_idxs) > 1:
            for idx in max_size_idxs:
                if 1 not in shapes[idx]:
                    broadcast_idx = idx

        broadcast_shape = shapes[broadcast_idx]

        if condition_shape != broadcast_shape:
            inputs[0] = _op.broadcast_to(inputs[0], broadcast_shape)
        if x_shape != broadcast_shape:
            inputs[1] = _op.broadcast_to(inputs[1], broadcast_shape)
        if y_shape != broadcast_shape:
            inputs[2] = _op.broadcast_to(inputs[2], broadcast_shape)
        return _op.where(inputs[0], inputs[1], inputs[2])

class Or(Elemwise):
    """ Operator converter for Or.
    """
    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        return _op.logical_or(inputs[0], inputs[1])


class Expand(OnnxOpConverter):
    """ Operator converter for Expand.
    """
    @classmethod
    def _impl_v8(cls, inputs, attr, params):
        in_shape = np.array(infer_shape(inputs[0])).astype('int32')
        if get_name(inputs[1]) in params:
            shape = params[inputs[1].name_hint].asnumpy().astype('int32')
        else:
            shape = infer_value_simulated(inputs[1], params).asnumpy().astype('int32')

        # Currently 'op.broadcast_to' expect the rank of the given 'shape'
        # (the 2nd input) is always higher than that of the given 'input' (the 1st input)
        # However, ONNX Expand supports multi-directional broadcasting, which allows
        # above pattern and also some extent of 'shape' can be smaller than the corresponding
        # extent of 'input'. In this case, the extent of 'shape' must be 1.
        # https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
        # In above cases, we cannot directorly apply 'op.broadcast_to' instead of 'expand'
        # so, here we solved this problem by expanding the given 'shape' itself.
        def expand_shape(in_shape, shape):
            """ A function expands the shape when the rank is lower than that of the given
            intput. Also it replaces the extent of the shape with the corresponding extent
            of the intput when it is 1.
            """

            # here we flip the shapes because this can be more simply written
            # when the innermost dimension is located at the index 0.
            in_shape = np.flip(in_shape, axis=0)
            shape = np.flip(shape, axis=0)

            if in_shape.size < shape.size:
                for i in range(shape.size):
                    if i < in_shape.size and in_shape[i] > shape[i]:
                        shape[i] = in_shape[i]
            else:
                for i in range(in_shape.size):
                    if i >= shape.size:
                        np.append(shape, in_shape[i])
                    elif shape[i] == 1:
                        shape[i] = in_shape[i]

            new_shape = np.flip(shape, axis=0)
            return new_shape

        shape = expand_shape(in_shape, shape)
        return _op.broadcast_to(inputs[0], shape=tuple(shape))


class LSTM(OnnxOpConverter):
    """ Operator converter for LSTM.
    """

    @classmethod
    def _activation_helper(cls, activation, alpha, beta):
        convert_map = _get_convert_map(1)
        attrs = {}
        if alpha is not None:
            attrs['alpha'] = alpha
        if beta is not None:
            attrs['beta'] = beta
        return lambda x: convert_map[activation.decode("utf-8")]([x], attrs, {})

    @classmethod
    def _activation_needs_alpha(cls, activation):
        needs_alpha = [
            "Affine",
            "LeakyRelu",
            "ThresholdedRelu",
            "ScaledTanh",
            "HardSigmoid",
            "Elu",
        ]
        return activation.decode("utf-8") in needs_alpha

    @classmethod
    def _activation_needs_beta(cls, activation):
        needs_beta = [
            "Affine",
            "ScaledTanh",
            "HardSigmoid",
        ]
        return activation.decode("utf-8") in needs_beta

    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        # Unpack inputs, note that if optional and not provided then value will be None.
        X = inputs[0]
        W = inputs[1]
        R = inputs[2]
        B = inputs['B']
        # Sequence length currently unused as it can be inferred from shapes.
        #sequence_lens = inputs['sequence_lens']
        h_0 = inputs['initial_h']
        c_0 = inputs['initial_c']
        P = inputs['P']

        num_directions = infer_shape(W)[0]
        W_dtype = infer_type(W).type_annotation.dtype

        if num_directions != 1:
            raise NotImplementedError("Bidirectional LSTMs not yet supported.")
        # Remove num_directions axis from weights.
        W = _op.squeeze(W, axis=[0])
        R = _op.squeeze(R, axis=[0])
        if B is not None:
            B = _op.squeeze(B, axis=[0])

        X_shape = infer_shape(X)
        hidden_size = infer_shape(R)[-1]
        batch_size = X_shape[1]

        # Initialize state if not provided.
        # Otherwise remove bidirectional axis.
        if h_0 is None:
            h_0 = _op.zeros((batch_size, hidden_size), W_dtype)
        else:
            h_0 = _op.squeeze(h_0, axis=[0])
        if c_0 is None:
            c_0 = _op.zeros((batch_size, hidden_size), W_dtype)
        else:
            c_0 = _op.squeeze(c_0, axis=[0])

        if P is not None:
            P = _op.squeeze(P, axis=[0])
            p_i, p_o, p_f = _op.split(P, 3)
        H_t = h_0
        C_t = c_0
        h_list = []

        if 'activations' in attr:
            activations = attr['activations']
            if len(activations) != 3:
                raise NotImplementedError("LSTM assumes 3 activation functions are provided")
            alpha_loc = 0
            alphas = attr.get('activation_alpha', [])
            if isinstance(alphas, float):
                alphas = [alphas]
            beta_loc = 0
            betas = attr.get('activation_beta', [])
            if isinstance(betas, float):
                betas = [betas]
            acts = []
            for i in range(3):
                alpha = None
                beta = None
                activation = activations[i]
                if cls._activation_needs_alpha(activation) and len(alphas) > alpha_loc:
                    alpha = alphas[alpha_loc]
                    alpha_loc += 1
                if cls._activation_needs_beta(activation) and len(betas) > beta_loc:
                    beta = betas[beta_loc]
                    beta_loc += 1
                acts.append(cls._activation_helper(activation, alpha, beta))
            f_act, g_act, h_act = acts
        else:
            f_act = _op.sigmoid
            g_act = _op.tanh
            h_act = _op.tanh

        X_steps = _op.split(X, indices_or_sections=X_shape[0], axis=0)
        for step in X_steps:
            step = _op.squeeze(step, axis=[0])
            gates = _op.nn.dense(step, W) + _op.nn.dense(H_t, R)
            if B is not None:
                WB, RB = _op.split(B, 2)
                gates += WB + RB
            i, o, f, c = _op.split(gates, 4, axis=-1)
            if P is not None:
                i = f_act(i + p_i * C_t)
                f = f_act(f + p_f * C_t)

            else:
                i = f_act(i)
                f = f_act(f)
            c = g_act(c)
            C = f * C_t + i * c
            if P is not None:
                o = f_act(o + p_o * C)
            else:
                o = f_act(o)
            H = o * h_act(C)
            H_t = H
            C_t = C
            h_list.append(_op.expand_dims(H, axis=0))
        # Concatenate outputs and add back in direction axis.
        concatenated = _op.concatenate(h_list, 0)
        output = _op.expand_dims(concatenated, axis=1)
        H_t = _op.expand_dims(H_t, axis=0)
        C_t = _op.expand_dims(C_t, axis=0)

        return _expr.TupleWrapper(_expr.Tuple((output, H_t, C_t)), 3)


class Resize(OnnxOpConverter):
    """Operator converter for Resize
    """
    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        mode = attr.get('mode')
        if mode == b'nearest':
            method = "nearest_neighbor"
        elif mode == b'linear':
            method = "bilinear"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Resize is not valid.'.format(mode))

        in_size = np.array(infer_shape(inputs[0]))
        scale = infer_value_simulated(inputs[2], params).asnumpy()
        if len(inputs) == 4:
            assert len(scale) == 0, "One of scale or size should be passed, not both."
            size = infer_value_simulated(inputs[3], params).asnumpy().astype(np.int32)
        else:
            assert len(scale) != 0, "One of scale or size should be passed."
            size = (in_size * scale).astype(np.int32)

        coord_trans = attr.get('coordinate_transformation_mode')
        if coord_trans in [b'pytorch_half_pixel', b'half_pixel']:
            coord_trans = "half_pixel"
        elif coord_trans == b'align_corners':
            coord_trans = "align_corners"
        elif coord_trans == b'asymmetric' or method == "nearest_neighbor":
            coord_trans = "asymmetric"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Unsupported coordinate_transformation_mode: {}'.format(coord_trans))
        layout = "NCHW"  # ONNX assumes NCHW layout
        out_size = (size[2], size[3])
        return _op.image.resize(inputs[0], out_size, layout, method, coord_trans)


class NonZero(OnnxOpConverter):
    """Operator converter for NonZero
    """
    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if len(inputs) > 1:
            raise ValueError("Expect 1 input only")

        output = AttrCvt(op_name='argwhere')(inputs, attr, params)
        return _op.transpose(output, axes=(1, 0))


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
        'Affine': Affine.get_converter(opset),
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
        # Recurrent Layers
        'LSTM': LSTM.get_converter(opset),

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
        'Expand': Expand.get_converter(opset),
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
        'Or': Or.get_converter(opset),
        'Resize': Resize.get_converter(opset),
        'NonZero': NonZero.get_converter(opset),
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
        mod : tvm.IRModule
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
            # Create and populate onnx input object.
            inputs = onnx_input()
            for i in node.input:
                if i != '':
                    inputs[i] = self._nodes[self._renames.get(i, i)]
            if op_name == "Constant":
                t_proto = self._parse_attr(node.attribute)["value"]
                self._num_param += 1
                # We should convert scalar integers to int32, to normalize.
                array = self._parse_array(t_proto)
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
        func = _function.Function(analysis.free_vars(outputs), outputs)
        return IRModule.from_expr(func), self._params

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
        inputs : list of tvm.relay.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version

        Returns
        -------
        sym : tvm.relay.function.Function
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
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
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
