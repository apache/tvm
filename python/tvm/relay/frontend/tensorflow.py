
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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel
"""TF: Tensorflow frontend."""
from __future__ import absolute_import as _abs
from __future__ import print_function

import warnings
from collections import defaultdict

# Numpy support
import numpy as np

import tvm

from tvm.ir import IRModule
from tvm.relay.prelude import Prelude

from .. import analysis
from .. import expr as _expr
from .. import op as _op
from ..expr_functor import ExprMutator
from .common import AttrCvt, get_relay_op
from .common import infer_type as _infer_type
from .common import infer_shape as _infer_shape
from .common import infer_channels as _infer_channels
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated

__all__ = ['from_tensorflow']


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
        if len(kernel) == 3:
            return prefix + '3d' + surfix
        raise tvm.error.OpAttributeInvalid(
            'Only 2D or 3D kernels are supported for operator {}'.format(prefix + '2d or 3d'))
    return _impl

def _dimension_constraint():
    def _dim_check(attrs):
        if len(attrs['kernel_shape']) in (2, 3):
            return True
        return False
    return _dim_check, "Only 2d or 3d kernel supported."

def _get_param(params, input_node):
    if isinstance(input_node, _expr.Constant):
        return np.atleast_1d(input_node.data.asnumpy())
    return params.pop(input_node.name_hint).asnumpy()

def _get_num_param(params, input_node):
    return _get_param(params, input_node).item()

def _get_list_param(params, input_node):
    return _get_param(params, input_node).tolist()

def _get_tuple_param(params, input_node):
    return tuple(_get_param(params, input_node))

def _need_module_for_shape_inference(op):
    return op in ['StridedSlice']

def _need_prelude_for_shape_inference(op):
    return "TensorArray" in op

def _rsqrt():
    def _impl(inputs, attr, params):
        inputs.append(tvm.relay.const(-0.5, attr['T'].name))
        return AttrCvt(op_name="power")(inputs, attr)
    return _impl

def _argx(func, func_name):
    """ A common wrapper for argmin and argmax operations """
    def _impl(inputs, attr, params):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_value = [_get_num_param(params, inputs[1])]
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for `{}` : `axis` should be a constant".format(func_name))
        return func(inputs[0], axis=axis_input_value, keepdims=False)
    return _impl

def _elemwise(name):
    def _impl(inputs, attr, params):
        assert len(inputs) == 2, "{} take 2 inputs, {} given".format(name, len(inputs))
        return get_relay_op(name)(*inputs)
    return _impl

def _pool3d(name):
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        input_shape = attr['_input_shapes'][inputs[0]]

        if attr['data_format'] == 'NDHWC':
            attr['kernel_shape'] = (attr['ksize'][1], attr['ksize'][2], attr['ksize'][3])
            attr['strides'] = (attr['strides'][1], attr['strides'][2], attr['strides'][3])
        elif attr['data_format'] == 'NCDHW':
            attr['kernel_shape'] = (attr['ksize'][2], attr['ksize'][3], attr['ksize'][4])
            attr['strides'] = (attr['strides'][2], attr['strides'][3], attr['strides'][4])
        else:
            msg = 'Value {} of attribute "data_format" of operator Pooling ' \
                  'is not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['data_format']))
        if attr['data_format'] == "NDHWC":
            input_shape = [attr['_input_shapes'][inputs[0]][i] for i in (0, 4, 1, 2, 3)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 4, 1, 2, 3))
            attr['data_format'] = "NCDHW"
            attr['_input_shapes'][inputs[0]] = input_shape
            flip_layout = True

        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0, 0, 0, 0, 0]
        elif attr['padding'] == 'SAME':
            stride_d, stride_h, stride_w = attr['strides']
            kernel_d, kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NDHWC':
                in_d = input_shape[1]
                in_h = input_shape[2]
                in_w = input_shape[3]
            else:
                in_d = input_shape[2]
                in_h = input_shape[3]
                in_w = input_shape[4]
            pad_d = _get_pad_pair(in_d, kernel_d, stride_d)
            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            attr['padding'] = [pad_d[0], pad_v[0], pad_h[0], pad_d[1], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Pooling is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if name == "avg_pool":
            attr['count_include_pad'] = False
        attr['ceil_mode'] = False
        out = AttrCvt(
            op_name=name,
            transforms={
                'kernel_shape': 'pool_size',
                'data_format': 'layout'},
            ignores=['ksize'])(inputs, attr)
        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 4, 1))
        return out

    return _impl

def _pooling(name):
    def _impl(inputs, attr, params):

        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        input_shape = attr['_input_shapes'][inputs[0]]

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (attr['ksize'][1], attr['ksize'][2])
            attr['strides'] = (attr['strides'][1], attr['strides'][2])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (attr['ksize'][2], attr['ksize'][3])
            attr['strides'] = (attr['strides'][2], attr['strides'][3])
        else:
            msg = 'Value {} of attribute "data_format" of operator Pooling ' \
                  'is not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['data_format']))

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            tmp_shape = attr['_input_shapes'][inputs[0]]
            input_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            attr['data_format'] = "NCHW"
            flip_layout = True

        # Fix padding
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            attr['padding'] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Pooling is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if name == "avg_pool":
            attr['count_include_pad'] = False

        out = AttrCvt(
            op_name=_dimension_picker(name),
            transforms={
                'kernel_shape':'pool_size',
                'data_format':'layout'},
            ignores=['ksize'],
            extras={'ceil_mode': False},
            custom_check=_dimension_constraint())(inputs, attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _conv(opname):
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        if opname == 'conv_transpose' and attr['data_format'] == 'NHWC':
            # transform to NCHW for TVM backend compatible and set 'flip_layout'
            # to have output flip back to NHWC
            tmp_shape = attr['_input_shapes'][inputs[2]]
            tmp_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[2] = _op.transpose(inputs[2], axes=(0, 3, 1, 2))
            attr['_input_shapes'][inputs[2]] = tmp_shape
            attr['strides'][1], attr['strides'][2], attr['strides'][3] = \
                attr['strides'][3], attr['strides'][1], attr['strides'][2]
            attr['data_format'] = 'NCHW'

            if opname == 'conv_transpose' and len(attr['_output_shapes']) > 0:
                tmp_shape = attr['_output_shapes'][0]
                tmp_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
                attr['_output_shapes'][0] = tmp_shape

            flip_layout = True

        inputs_data = inputs[0] if opname != 'conv_transpose' else inputs[2]

        # NCHW Layout require weights transpose
        if attr['data_format'] == 'NCHW':
            tmp_shape = attr['_input_shapes'][inputs[1]]
            if opname in ['conv', 'conv_transpose']:
                tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                tmp_shape = [tmp_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))
            attr['_input_shapes'][inputs[1]] = tmp_shape

        input_shape = attr['_input_shapes'][inputs_data]
        weights_shape = attr['_input_shapes'][inputs[1]]

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs_data = _op.transpose(inputs_data, axes=(0, 3, 1, 2))
            if opname in ['conv', 'conv_transpose']:
                weights_shape = [weights_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                weights_shape = [weights_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))

            attr['data_format'] = "NCHW"
            attr['strides'] = [attr['strides'][ii] for ii in (0, 3, 1, 2)]
            flip_layout = True

        if attr['data_format'] == 'NHWC':
            kernel_h, kernel_w, _, depth_mult = weights_shape
            attr['kernel_shape'] = (weights_shape[0], weights_shape[1])
            if opname == 'conv':
                attr['channels'] = weights_shape[3]
            elif opname == 'conv_transpose':
                attr['channels'] = weights_shape[2]
            else:
                attr['channels'] = input_shape[3] * depth_mult

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][1], attr['dilations'][2])
            attr['strides'] = (attr['strides'][1], attr['strides'][2])
        elif attr['data_format'] == 'NCHW':
            _, depth_mult, kernel_h, kernel_w = weights_shape
            attr['kernel_shape'] = (weights_shape[2], weights_shape[3])
            if opname == 'conv':
                attr['channels'] = weights_shape[0]
            elif opname == 'conv_transpose':
                attr['channels'] = weights_shape[1]
            else:
                attr['channels'] = input_shape[1] * depth_mult
                if attr['channels'] < 0:
                    attr['channels'] *= -1

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][2], attr['dilations'][3])
            attr['strides'] = (attr['strides'][2], attr['strides'][3])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['data_format']))

        if opname == 'depthwise':
            attr['groups'] = attr['channels']

        # Fix padding
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']

            pdata_shape = input_shape
            if opname == 'conv_transpose' and len(attr['_output_shapes']) > 0:
                pdata_shape = attr['_output_shapes'][0]

            if attr['data_format'] == 'NHWC':
                in_h = pdata_shape[1]
                in_w = pdata_shape[2]
            else:
                in_h = pdata_shape[2]
                in_w = pdata_shape[3]

            dilation_h = attr['dilations'][0]
            dilation_w = attr['dilations'][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

            attr['padding'] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Conv is not ' \
                  'valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if 'kernel_layout' not in attr:
            if opname in ['conv', 'conv_transpose']:
                attr['kernel_layout'] = 'HWIO' if attr['data_format'] == 'NHWC' else 'OIHW'
            else:
                attr['kernel_layout'] = 'HWOI' if attr['data_format'] == 'NHWC' else 'OIHW'

        use_bias = len(inputs) == (3 if opname != 'conv_transpose' else 4)
        channel_axis = 1 if attr['data_format'] == "NCHW" else 3

        # Ignore the new attributes from TF2.0, for now.
        out = AttrCvt(
            op_name=_dimension_picker('conv', \
                surfix="_transpose" if opname == 'conv_transpose' else ""),
            ignores=['explicit_paddings'],
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'data_layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            custom_check=_dimension_constraint())([inputs_data, inputs[1]], attr)

        if use_bias:
            out = _op.nn.bias_add(out,
                                  inputs[2] if opname != 'conv_transpose' else inputs[3],
                                  axis=channel_axis)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _conv3d(opname):
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        inputs_data = inputs[0] if opname != 'conv_transpose' else inputs[2]

        # NCDHW Layout require weights transpose
        if attr['data_format'] == 'NCDHW':
            tmp_shape = attr['_input_shapes'][inputs[1]]
            tmp_shape = [tmp_shape[ii] for ii in (4, 3, 0, 1, 2)]
            inputs[1] = _op.transpose(inputs[1], axes=(4, 3, 0, 1, 2))
            attr['_input_shapes'][inputs[1]] = tmp_shape

        input_shape = attr['_input_shapes'][inputs_data]
        weights_shape = attr['_input_shapes'][inputs[1]]

        if attr['_target_layout'] == "NCDHW" and attr['data_format'] == "NDHWC":
            input_shape = [input_shape[ii] for ii in (0, 4, 1, 2, 3)]
            inputs_data = _op.transpose(inputs_data, axes=(0, 4, 1, 2, 3))
            weights_shape = [weights_shape[ii] for ii in (4, 3, 0, 1, 2)]
            inputs[1] = _op.transpose(inputs[1], axes=(4, 3, 0, 1, 2))

            attr['data_format'] = "NCDHW"
            attr['strides'] = [attr['strides'][ii] for ii in (0, 4, 1, 2, 3)]
            flip_layout = True

        if attr['data_format'] == 'NDHWC':
            kernel_d, kernel_h, kernel_w, _, _ = weights_shape
            attr['kernel_shape'] = (kernel_d, kernel_h, kernel_w)
            if opname == 'conv':
                attr['channels'] = weights_shape[4]
            elif opname == 'conv_transpose':
                attr['channels'] = weights_shape[3]

            if 'dilations' in attr:
                attr['dilations'] =\
                    (attr['dilations'][1], attr['dilations'][2], attr['dilations'][3])
            attr['strides'] = (attr['strides'][1], attr['strides'][2], attr['strides'][3])
        elif attr['data_format'] == 'NCDHW':
            _, _, kernel_d, kernel_h, kernel_w = weights_shape
            attr['kernel_shape'] = (kernel_d, kernel_h, kernel_w)
            if opname == 'conv':
                attr['channels'] = weights_shape[0]
            elif opname == 'conv_transpose':
                attr['channels'] = weights_shape[1]

            if 'dilations' in attr:
                attr['dilations'] =\
                    (attr['dilations'][2], attr['dilations'][3], attr['dilations'][4])
            attr['strides'] = (attr['strides'][2], attr['strides'][3], attr['strides'][4])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['data_format']))

        # Fix padding
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0, 0]
        elif attr['padding'] == 'SAME':
            stride_d, stride_h, stride_w = attr['strides']
            kernel_d, kernel_h, kernel_w = attr['kernel_shape']

            pdata_shape = input_shape
            if opname == 'conv_transpose' and len(attr['_output_shapes']) > 0:
                pdata_shape = attr['_output_shapes'][0]

            if attr['data_format'] == 'NDHWC':
                in_d = pdata_shape[1]
                in_h = pdata_shape[2]
                in_w = pdata_shape[3]
            else:
                in_d = pdata_shape[2]
                in_h = pdata_shape[3]
                in_w = pdata_shape[4]

            dilation_d = attr['dilations'][0]
            dilation_h = attr['dilations'][1]
            dilation_w = attr['dilations'][2]
            dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_d = _get_pad_pair(in_d, dilated_kernel_d, stride_d)
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

            attr['padding'] = [pad_d[0], pad_v[0], pad_h[0], pad_d[1], pad_v[1], pad_h[1]]

        else:
            msg = 'Value {} in attribute "padding" of operator Conv is not ' \
                  'valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if 'kernel_layout' not in attr:
            attr['kernel_layout'] = 'DHWIO' if attr['data_format'] == 'NDHWC' else 'OIDHW'

        use_bias = len(inputs) == (3 if opname != 'conv_transpose' else 4)
        channel_axis = 1 if attr['data_format'] == "NCDHW" else 4

        # Ignore the new attributes from TF2.0, for now.
        out = AttrCvt(
            op_name=_dimension_picker('conv', \
                surfix="_transpose" if opname == 'conv_transpose' else ""),
            ignores=['explicit_paddings'],
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'data_layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            custom_check=_dimension_constraint())([inputs_data, inputs[1]], attr)

        if use_bias:
            out = _op.nn.bias_add(out,
                                  inputs[2] if opname != 'conv_transpose' else inputs[3],
                                  axis=channel_axis)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 4, 1))

        return out
    return _impl

def _decode_image():
    def _impl(inputs, attr, params):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        warnings.warn("DecodeJpeg: It's a pass through, please handle preprocessing before input")
        return inputs[0]
    return _impl

def _crop_and_resize():
    def _impl(inputs, attr, params):
        # input image is a 4-D tensor of shape [batch, image_height, image_width, depth]
        # boxes is a 2-D tensor of shape [num_boxes, 4], 4 is for [y1, x1, y2, x2]
        try:
            crop_size = _get_list_param(params, inputs[3])
        except (IndexError, KeyError):
            crop_size = _infer_value(inputs[3], params).asnumpy().tolist()

        method = attr['method'].decode()
        method = 'nearest_neighbor' if method == 'nearest' else method
        if method not in ['bilinear', 'nearest_neighbor']:
            raise tvm.error.OpAttributeUnImplemented(
                'Method {} is not supported'.format(method))
        layout = attr['layout'] if 'layout' in attr else 'NHWC'
        extrapolation_value = attr['extrapolation_value']

        return get_relay_op("crop_and_resize")(inputs[0], inputs[1], inputs[2], crop_size,
                                               layout, method, extrapolation_value)
    return _impl

def _cast():
    def _impl(inputs, attr, params):
        return inputs[0].astype(attr['DstT'].name)
    return _impl

def _expand_dims():
    def _impl(inputs, attr, params):
        dim_input = inputs.pop(1)
        axis = _get_num_param(params, dim_input)
        return AttrCvt(op_name="expand_dims", ignores=['Tdim', 'N'],
                       extras={'axis': int(axis), 'num_newaxis': 1})(inputs, attr)
    return _impl

def _resize(method):
    def _impl(inputs, attr, params):
        output_shape0 = attr['_output_shapes'][0]
        # Dynamic size models might have _output_shapes attr equal to [None] here
        size = output_shape0[1:3] if output_shape0 is not None else [-1, -1]
        # Important that the size is defined. If an axis is not, we need to infer what
        # the shape should be.
        if -1 in size:
            size = _infer_value(inputs[1], params).asnumpy().reshape([-1]).tolist()
        attr['size'] = size
        inputs.pop(1)
        # NHWC
        attr['layout'] = 'NHWC'
        if attr.pop('align_corners') is True:
            attr['coordinate_transformation_mode'] = 'align_corners'
        else:
            attr['coordinate_transformation_mode'] = 'asymmetric'

        # Ignore the new attributes from TF2.0, for now.
        return AttrCvt(op_name='resize',
                       ignores=['Tdim', 'half_pixel_centers'],
                       extras={'method': method})(inputs, attr)
    return _impl

def _check_numerics():
    def _impl(inputs, attr, params):
        # Making a copy node assuming no need to verify
        return AttrCvt(op_name="copy", ignores=['message'])(inputs, attr)
    return _impl

def _assert():
    # ToDo: In general people want asserts to be gone from TensorFlow graphs
    # when they are optimizing them, so converting it to a no-op is
    # reasonable. However, it would be nice to have the option to keep them
    # once Relay gets a Halt or Assert op.
    return _no_op()

def _no_op():
    def _impl(inputs, attr, params):
        # ToDo: This should really be an op that returns nothing, which could
        # be represented as an empty tuple. It turns out that TVM
        # infrastructure doesn't like running functions that return None and
        # also don't like running functions that return an empty tuple. So it
        # doesn't work, but it should be made to work and then this could be
        # improved. In the mean time, it is hard to imagine a case where it
        # matters in any real way that a no-op is converted to a constant 0.
        return tvm.relay.const(0)
    return _impl

def _matmul():
    def _impl(inputs, attr, params):
        channels = _infer_channels(inputs[1], not attr['transpose_b'])
        if attr['transpose_a']:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not attr['transpose_b']:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(op_name="dense",
                       extras={'units': channels},
                       ignores=['transpose_a', 'transpose_b', 'T'])(inputs, attr)

    return _impl

def _batch_matmul():
    def _impl(inputs, attr, params):
        input_x = inputs[0]
        input_y = inputs[1]
        orig_shape_x = attr['_input_shapes'][input_x]
        orig_shape_y = attr['_input_shapes'][input_y]

        # reshape n-dimensional batch matmul into 3d
        if len(orig_shape_x) > 3:
            outer_dims = [orig_shape_x[i] for i in range(0, len(orig_shape_x) - 2)]
            num_outer_elts = np.prod(outer_dims)
            new_shape_x = (num_outer_elts, orig_shape_x[-2], orig_shape_x[-1])
            new_shape_y = (num_outer_elts, orig_shape_y[-2], orig_shape_y[-1])
            input_x = _op.reshape(input_x, newshape=new_shape_x)
            input_y = _op.reshape(input_y, newshape=new_shape_y)

        adj_x = attr['adj_x']
        adj_y = attr['adj_y']
        input_x = _op.transpose(input_x, axes=[0, 2, 1]) if adj_x else input_x
        input_y = _op.transpose(input_y, axes=[0, 2, 1]) if not adj_y else input_y
        ret = get_relay_op('batch_matmul')(input_x, input_y)

        # reshape result back to n-dimensional
        if len(orig_shape_x) > 3:
            final_shape = list(orig_shape_x)
            final_shape[-2] = orig_shape_x[-1] if adj_x else orig_shape_x[-2]
            final_shape[-1] = orig_shape_y[-2] if adj_y else orig_shape_y[-1]
            ret = _op.reshape(ret, newshape=final_shape)

        return ret
    return _impl

def _identity():
    def _impl(inputs, attr, params):
        return inputs[0]
    return _impl

def _concatV2():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(len(inputs)-1)
        axis = int(_get_num_param(params, pop_node))
        return AttrCvt(
            op_name="concatenate", ignores=['T', 'N', 'Tidx'],
            extras={'axis': axis})([inputs], attr)
    return _impl

def _concat():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(0)
        axis = int(_get_num_param(params, pop_node))
        return AttrCvt(
            op_name="concatenate", ignores=['N'],
            extras={'axis': axis})([inputs], attr)
    return _impl

def _pack():
    def _impl(inputs, attr, params):
        axis = int(attr["axis"])
        inputs_reshaped = [_op.expand_dims(i, axis=axis, num_newaxis=1) for i in inputs]
        return _op.concatenate(inputs_reshaped, axis)
    return _impl

def _tensor_array():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get('dtype').name
        tensor_array_constructor = prelude.get_var('tensor_array', dtype_str)
        return tensor_array_constructor(_op.take(inputs[0], tvm.relay.const(0)))
    return _impl

def _tensor_array_scatter():
    def _impl(inputs, attr, params, prelude):
        dtype_str = attr.get('T').name
        values_rank = len(inputs[2].type_annotation.shape)
        unstack_name = "tensor_array_unstack_tensor{}".format(values_rank)
        unstack_function = prelude.get_var(unstack_name, dtype_str)
        values = unstack_function(inputs[2])
        tensor_array_scatter_func = prelude.get_var('tensor_array_scatter', dtype_str)
        return tensor_array_scatter_func(inputs[0], inputs[1], values)
    return _impl

def _tensor_array_gather():
    def _impl(inputs, attr, params, prelude):
        return prelude.tensor_array_gather(inputs[2], inputs[1])
    return _impl

def _tensor_array_size():
    def _impl(inputs, attr, params, prelude):
        return prelude.length(inputs[0])
    return _impl

def _tensor_array_write():
    def _impl(inputs, attr, params, prelude):
        input_rank = len(inputs[2].type_annotation.shape)
        dtype = attr.get('T').name

        tensor_name = 'tensor{}'.format(input_rank)
        tensor_func = prelude.get_var(tensor_name, dtype)
        v = tensor_func(inputs[2])
        write_func = prelude.get_var('tensor_array_write', dtype)

        return write_func(inputs[3], _op.take(inputs[1], tvm.relay.const(0)), v)
    return _impl

def _tensor_array_read():
    def _impl(inputs, attr, params, prelude):
        read_func = prelude.get_var('tensor_array_read', attr.get('dtype').name)
        return read_func(inputs[2], _op.take(inputs[1], tvm.relay.const(0)))
    return _impl

def _tensor_array_split():
    def _impl(inputs, attr, params, prelude):
        input_rank = len(inputs[1].type_annotation.shape)
        dtype_str = attr.get('T').name
        v = prelude.get_var("tensor{}".format(input_rank), dtype_str)(inputs[1])
        lengths = _op.cast(inputs[2], 'int32')
        split_var = prelude.get_var('tensor_array_split', dtype_str)
        return split_var(inputs[0], v, lengths)
    return _impl

def _tensor_array_concat():
    def _impl(inputs, attr, params, prelude):
        concat_func = prelude.get_var('tensor_array_concat', attr['dtype'].name)
        return concat_func(inputs[1])
    return _impl

def _tile():
    def _impl(inputs, attr, params):
        reps = _get_list_param(params, inputs.pop())
        new_input = []
        new_input.append(inputs.pop(0))

        return AttrCvt(
            op_name='tile',
            extras={'reps': tuple(reps)},
            ignores=['Tmultiples'])(new_input, attr)
    return _impl

def _slice():
    def _impl(inputs, attr, params):
        try:
            begin = _get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            begin = _infer_value(inputs[1], params).asnumpy().tolist()[0]
        try:
            size = _get_list_param(params, inputs[2])
        except (IndexError, KeyError, AttributeError):
            size = _infer_value(inputs[2], params).asnumpy().tolist()[0]
        data_shape = attr['_input_shapes'][inputs[0]]
        data_dim = len(data_shape)
        end = size
        for i in range(data_dim):
            if size[i] == -1:
                end[i] = data_shape[i]
            else:
                end[i] += begin[i]
        return _op.strided_slice(inputs[0], begin=begin, end=end)
    return _impl


def _reshape():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(1)

        try:
            shape_arg = _get_tuple_param(params, pop_node)
        except AttributeError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            try:
                params_new = _infer_value(pop_node, params)
                shape_arg = tuple(params_new.asnumpy().astype('int64').flatten())
            except Exception:
                # Deal with symbolic shape case.
                # Currently only shape_of can be the direct ancestor.
                if not isinstance(pop_node, tvm.relay.expr.Call) or \
                        "shape_of" not in str(pop_node.op):
                    raise RuntimeError("If shape operator is used in reshape to "
                                       "express reshape_like, shape_of must be "
                                       "the direct ancestor of reshape when input "
                                       "shape is symbolic.")
                return _op.reshape_like(inputs[0], pop_node.args[0])
        return AttrCvt(
            op_name="reshape",
            extras={'newshape': shape_arg},
            ignores=['Tshape'])(inputs, attr)
    return _impl


def _depth_to_space():
    def _impl(inputs, attr, params):
        block_size = int(attr['block_size'])
        layout = attr['data_format'].decode("utf-8")
        return _op.nn.depth_to_space(inputs[0], block_size, layout)

    return _impl


def _space_to_depth():
    def _impl(inputs, attr, params):
        block_size = int(attr['block_size'])
        layout = attr['data_format'].decode("utf-8")
        return _op.nn.space_to_depth(inputs[0], block_size, layout)

    return _impl


def _bias_add():
    def _impl(inputs, attr, params):
        # Must expand for proper broadcasting in NCHW.
        if attr['data_format'].decode("utf-8") == 'NCHW':
            bias = _op.reshape(inputs[1], newshape=(1, -1, 1, 1))
        else:
            bias = inputs[1]
        return _op.add(inputs[0], bias)
    return _impl

def _broadcast_to():
    def _impl(inputs, attr, params):
        if isinstance(inputs[1], _expr.Var):
            shape = params[inputs[1].name_hint]
        else:
            shape = _infer_value(inputs[1], params)
        shape = list(shape.asnumpy().reshape([-1]))
        return _op.broadcast_to(inputs[0], shape)
    return _impl

def _squeeze():
    def _impl(inputs, attr, params):
        if len(attr['squeeze_dims']) == 0:
            attr['squeeze_dims'] = None
        return AttrCvt(
            op_name="squeeze",
            transforms={'squeeze_dims':'axis'},
            ignores=['T'])(inputs, attr)
    return _impl

def _fused_batch_norm():
    def _impl(inputs, attr, params):
        # Tensorflow: (data, gamma, beta, moving_mean, moving_variance)
        # Relay:       (data, gamma, beta, moving_mean, moving_varience)
        axis = 3
        need_cast = False

        if 'data_format' in attr:
            attr['data_format'] = attr['data_format'].decode("utf-8")
            if attr['data_format'] == 'NCHW':
                axis = 1
        if 'U' in attr:
            need_cast = True
            inputs[0] = _op.cast(inputs[0], dtype=attr['U'].name)

        out = AttrCvt(op_name='batch_norm',
                      transforms={'scale_after_normalization':'scale',
                                  'variance_epsilon':'epsilon'},
                      extras={'axis': axis},
                      ignores=['data_format', 'U'],
                      disables=['momentum'])(inputs, attr)

        if need_cast:
            out = _op.cast(out, dtype=attr['T'].name)
        return out
    return _impl

def _batch_norm():
    def _impl(inputs, attr, params):
        # Rearrange inputs from
        # (data, moving_mean, moving_variance, beta, gamma)
        #     to
        # (data, gamma, beta, moving_mean, moving_var)
        new_inputs = [inputs[0], inputs[4], inputs[3], inputs[1], inputs[2]]

        axis = 3
        if 'data_format' in attr:
            attr['data_format'] = attr['data_format'].decode("utf-8")
            if attr['data_format'] == 'NCHW':
                axis = 1

        return AttrCvt(
            op_name='batch_norm',
            transforms={'scale_after_normalization':'scale', 'variance_epsilon':'epsilon'},
            extras={'axis': axis},
            ignores=['data_format'],
            disables=['momentum'])(new_inputs, attr)
    return _impl

def _relu6():
    def _impl(inputs, attr, params):
        return _op.clip(inputs[0], a_min=0, a_max=6)
    return _impl

def _shape():
    def _impl(inputs, attr, params):
        is_symbolic_shape = False
        for axis in attr['_input_shapes'][inputs[0]]:
            if not isinstance(axis, (int, tvm.tir.IntImm)):
                is_symbolic_shape = True
                break

        if is_symbolic_shape:
            ret = _op.shape_of(inputs[0], dtype='int32')
        else:
            ret = np.array(attr['_input_shapes'][inputs[0]], dtype='int32')
        return ret

    return _impl

def _fill():
    def _impl(inputs, attr, params):
        output_shape = attr['_output_shapes'][0]
        # Output shape must be defined to avoid errors. If any axis is not, we must
        # try to compute its shape.
        if output_shape is None or -1 in output_shape:
            output_shape = _infer_value(inputs[0], params).asnumpy().reshape([-1]).tolist()

        fill_arg = _get_num_param(params, inputs.pop(1))
        dtype = attr['T'].name
        return _op.full(tvm.relay.const(fill_arg, dtype),
                        output_shape, dtype)
    return _impl

def _lrn():
    def _impl(inputs, attr, params):
        attr_new = {}
        depth_radius = attr.get('depth_radius', 5)
        size = (depth_radius * 2) + 1
        attr_new['axis'] = 3 # Fix axis, NHWC format
        attr_new['size'] = size
        attr_new['bias'] = attr.get('bias', 1)
        attr_new['alpha'] = attr.get('alpha', 1) * size
        attr_new['beta'] = attr.get('beta', 0.5)
        return AttrCvt(op_name='lrn')(inputs, attr_new)
    return _impl

def _sum():
    def _impl(inputs, attr, params):
        axis = _get_tuple_param(params, inputs[1])
        return AttrCvt(
            op_name='sum',
            extras={'axis': axis},
            transforms={'keep_dims':'keepdims'},
            ignores=['name', 'Tidx'])([inputs[0]], attr)
    return _impl

def _reduce(op):
    def _impl(inputs, attr, params):
        axis = _get_list_param(params, inputs[1])
        axis = tuple(axis)
        return AttrCvt(
            op_name=op,
            extras={'axis': axis},
            transforms={'keep_dims':'keepdims'},
            ignores=['name', 'Tidx'])([inputs[0]], attr)
    return _impl

def _square():
    def _impl(inputs, attr, params):
        return _op.multiply(inputs[0], inputs[0])
    return _impl

def _gather():
    "GatherV2, Gather"
    def _impl(inputs, attr, params):
        if len(inputs) > 2:
            axis = _get_num_param(params, inputs.pop(2))
        else:
            axis = 0
        if int(attr.get('batch_dims', 0)) != 0:
            raise tvm.error.OpAttributeUnImplemented(
                'Attribute batch_dims is not supported')
        new_input = inputs[0:2]
        return AttrCvt(op_name="take",
                       extras={'axis': tvm.const(axis, 'int32')},
                       ignores=['Tindices', 'Tparams', 'validate_indices',
                                'Taxis', '_class', 'batch_dims'])(new_input, attr)
    return _impl

def _gather_nd():
    """GatherNd"""
    def _impl(inputs, attr, params):
        return AttrCvt(op_name="gather_nd",
                       ignores=['Tindices', 'Tparams',\
                                'Taxis', '_class'])(inputs, attr)
    return _impl

def _stridedSlice():
    def _impl(inputs, attr, params, mod):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = _get_list_param(params, inputs[1])
        end = _get_list_param(params, inputs[2])
        stride = _get_list_param(params, inputs[3])
        begin_mask = int(attr.get('begin_mask', 0))
        end_mask = int(attr.get('end_mask', 0))
        ellipsis_mask = int(attr.get('ellipsis_mask', 0))
        new_axis_mask = int(attr.get('new_axis_mask', 0))
        shrink_axis_mask = int(attr.get('shrink_axis_mask', 0))
        data_shape = attr['_input_shapes'][inputs[0]]
        data_dim = len(data_shape)
        stride_dim = len(stride)

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            #Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                #Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= (1 << stride_dim)
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    #Identify the end index for applying ellipsis_mask
                    to_index = min(((data_dim - (stride_dim-index)) + 1 \
                                     + new_axes_after_ellipsis), data_dim)
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask &new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[final_index] \
                                                     if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = 0 if stride[index] < 0 \
                                                 else data_shape[final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        #Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = data_shape[final_index] + begin[index] \
                                                 if begin[index] < 0 else begin[index]
                        m_end[final_index] = begin[index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)
        out = _op.strided_slice(inputs[0], begin=begin, end=end, strides=stride)
        out_shape = _infer_shape(out, mod=mod)
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        #Create final output shape.
        final_output = []
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
            elif gather_index == -2:
                pass
            else:
                final_output.append(out_shape[gather_index])

        if not final_output:
            return out
        return _op.reshape(out, newshape=tuple(final_output))
    return _impl

def _pad(name):
    def _impl(inputs, attr, params):
        padlist = _get_param(params, inputs[1])
        paddings = tuple(tuple(l) for l in padlist)
        attr['pad_width'] = paddings
        attr['pad_value'] = 0
        new_inputs = [inputs[0]]
        if name == 'PadV2':
            constant_values = _get_num_param(params, inputs[2])
            attr['pad_value'] = constant_values
        return AttrCvt(
            op_name='pad',
            ignores=['Tpaddings'],)(new_inputs, attr)
    return _impl

def _mirror_pad():
    def _impl(inputs, attr, params):
        padlist = _get_param(params, inputs[1])
        paddings = tuple(tuple(l) for l in padlist)
        attr['pad_width'] = paddings
        mode = attr['mode'].decode('utf-8')
        attr['mode'] = mode
        new_inputs = [inputs[0]]
        return AttrCvt(
            op_name='mirror_pad',
            ignores=['Tpaddings'],)(new_inputs, attr)
    return _impl

def _transpose():
    def _impl(inputs, attr, params):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        try:
            axes = _get_list_param(params, inputs[1])
        except (IndexError, KeyError, AttributeError):
            axes = _infer_value_simulated(inputs[1], params).asnumpy()
        return _op.transpose(inputs[0], axes=axes)
    return _impl

def _where():
    def _impl(inputs, attr, params):
        if len(inputs) == 1:
            return AttrCvt(op_name="argwhere")(inputs, attr)
        return AttrCvt(op_name="where")(inputs, attr)
    return _impl

def _clip_by_value():
    def _impl(inputs, attr, params):
        a_min = _get_num_param(params, inputs[1])
        a_max = _get_num_param(params, inputs[2])
        return _op.clip(inputs[0], a_min=a_min, a_max=a_max)
    return _impl

def _reverse_v2():
    def _impl(inputs, attr, params):
        axis = _get_num_param(params, inputs[1])
        return AttrCvt(
            op_name="reverse",
            ignores=['Tidx'],
            extras={'axis': int(axis)})([inputs[0]], attr)
    return _impl

def _rank():
    def _impl(inputs, attr, params):
        input_shape = attr['_input_shapes'][inputs[0]]

        name = attr["_node_name"]
        params[name] = tvm.nd.array([len(input_shape)])
        return [_expr.var(name,
                          shape=params[name].shape,
                          dtype='int32')]

    return _impl


def _range():
    def _impl(inputs, attr, params):
        start = _get_param(params, inputs[0])[0]
        if hasattr(inputs[1], "name_hint") or isinstance(inputs[1], _expr.Constant):
            limit = _get_param(params, inputs[1])[0]
        else:
            if any(['Rank' in param for param in params]):
                limit = params.pop('Rank').asnumpy()[0]
            else:
                limit = _infer_value_simulated(inputs[1], params).asnumpy()[0]
        delta = _get_param(params, inputs[2])[0]
        dtype = attr['Tidx'].name if 'Tidx' in attr else str(start.dtype)
        return AttrCvt(
            op_name="arange",
            ignores=['Tidx'],
            extras={'start': _expr.const(start),
                    "stop": _expr.const(limit),
                    'step': _expr.const(delta),
                    'dtype': dtype})([], attr)
    return _impl


def _elu():
    def _impl(inputs, attr, params):
        dtype = attr['T'].name
        alpha = tvm.relay.const(-1.0, dtype)
        return alpha * _op.nn.relu(tvm.relay.const(1, dtype) \
                                   - _op.exp(inputs[0])) + _op.nn.relu(inputs[0])
    return _impl

def _selu():
    def _impl(inputs, attr, params):
        dtype = attr['T'].name
        alpha = tvm.relay.const(-1.6732632423543772848170429916717, dtype)
        gamma = tvm.relay.const(1.0507009873554804934193349852946, dtype)
        return gamma * (alpha * _op.nn.relu(tvm.relay.const(1, dtype) \
                                            - _op.exp(inputs[0])) + _op.nn.relu(inputs[0]))
    return _impl

def _mean():
    def _impl(inputs, attr, params):
        axis = _get_tuple_param(params, inputs[1])
        return AttrCvt(op_name="mean", ignores=['Tdim', 'Tidx'],
                       transforms={'keep_dims': 'keepdims'},
                       extras={'axis': axis})([inputs[0]], attr)
    return _impl

def _broadcast(name):
    def _impl(inputs, attr, params):
        return AttrCvt(
            op_name=name,
            ignores=['name', 'Tidx']
        )(inputs, attr)
    return _impl

def _split(has_size_vector):
    # TF documentation https://www.tensorflow.org/api_docs/python/tf/split
    def _impl(inputs, attr, params):
        try:
            # order and number of inputs are different:
            # if has_size_vector:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split-v
            # else:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split

            # in addition, `axis` and `num_or_size_splits` can be tensors in TensorFlow,
            # we can only support constants
            if has_size_vector:
                input_node_index = 0
                input_axis_index = 2
                size_splits = _get_param(params, inputs[1])
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr['num_split']
            input_node = inputs[input_node_index]
            axis_input_value = _get_num_param(params, inputs[input_axis_index])
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for split: `axis` and `num_or_size_splits` " \
                "should be constants")
        return _op.split(input_node,
                         indices_or_sections=indices_or_sections,
                         axis=int(axis_input_value))
    return _impl

def _unpack():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        axis = attr['axis']
        input_shape = attr['_input_shapes'][input_node]
        axis_length = input_shape[axis]
        if axis_length < 0:
            raise TypeError("Unstack with unknown axis length")
        splitted = _op.split(input_node,
                             indices_or_sections=axis_length,
                             axis=axis)
        #name=attr.get('_node_name', 'unstack'))
        if axis == 0:
            axis = None
        else:
            axis = [axis]
        return _expr.TupleWrapper(
            _expr.Tuple([_op.squeeze(split_item, axis=axis) \
            for split_item in splitted]), len(splitted))
    return _impl

def _softmax():
    def _impl(inputs, attr, params):
        return AttrCvt(op_name='softmax',
                       transforms={'axis': ('axis', 1)})([inputs[0]], attr)
    return _impl

def _softplus():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/softplus
    def _impl(inputs, attr, params):
        exp_out = AttrCvt('exp')(inputs, attr)
        inputs.append(tvm.relay.const(1, attr['T'].name))
        rh = tvm.relay.const(1, attr['T'].name)
        add_out = get_relay_op('add')(exp_out, rh)
        return get_relay_op('log')(add_out)
    return _impl

def _topk():
    def _impl(inputs, attr, params):
        k = int(_get_num_param(params, inputs.pop(1)))
        if k < 1:
            raise tvm.error.OpAttributeInvalid(
                'Attribute k must be positive in operator TopKV2')
        if attr['sorted'] is False:
            raise tvm.error.OpAttributeUnImplemented(
                'Attribute sorted=False is not supported in operator TopKV2')
        return AttrCvt(op_name='topk',
                       ignores=['sorted'],
                       extras={'k': k, 'is_ascend': False, 'dtype': 'int32'})(inputs, attr)
    return _impl

def _floordiv():
    def _impl(inputs, attr, params):
        assert len(inputs) == 2
        return AttrCvt('floor_divide')(inputs, attr)
    return _impl

def _floormod():
    def _impl(inputs, attr, params):
        assert len(inputs) == 2
        return AttrCvt('floor_mod')(inputs, attr)
    return _impl

def _logical(name):
    def _impl(inputs, attr, params):
        return AttrCvt(op_name=name)(inputs, attr)
    return _impl

def _space_to_batch_nd():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        input_shape = attr['_input_shapes'][input_node]
        block_shape = _get_list_param(params, inputs[1])
        paddings = _get_list_param(params, inputs[2])
        N = len(input_shape)
        M = len(block_shape)
        batch = input_shape[0]
        remaining_shape_length = N - M - 1
        paddings = [(0, 0)] + paddings + [(0, 0)] * remaining_shape_length
        # From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch-n-d:
        # Zero-pad the start and end of dimensions [1, ..., M] of the input according to paddings
        # to produce padded of shape padded_shape.
        padded = tvm.relay.nn.pad(input_node, pad_width=paddings)
        # Reshape padded to reshaped_padded of shape:
        # [batch] + [padded_shape[1] / block_shape[0], block_shape[0], ...,
        # padded_shape[M] / block_shape[M-1], block_shape[M-1]] + remaining_shape
        shape1 = [batch] + [item for i in range(M) for item in [-4, -1, block_shape[i]]] + [-2]
        reshaped_padded = tvm.relay.reshape(padded, newshape=shape1)
        # Permute dimensions of reshaped_padded to produce permuted_reshaped_padded of shape:
        # block_shape + [batch] + [padded_shape[1] / block_shape[0], ...,
        # padded_shape[M] / block_shape[M-1]] + remaining_shape
        axes = [2 * i + 2 for i in range(M)] + [0] + [2 * i + 1 for i in range(M)] + \
               list(range(1 + 2 * M, 1 + 2 * M + remaining_shape_length))
        permuted_reshaped_padded = tvm.relay.transpose(reshaped_padded, axes=axes)
        permuted_reshaped_padded_shape = _infer_shape(permuted_reshaped_padded)
        # Reshape permuted_reshaped_padded to flatten block_shape into the batch dimension,
        # producing an output tensor of shape:
        # [batch * prod(block_shape)] + [padded_shape[1] / block_shape[0], ...,
        # padded_shape[M] / block_shape[M-1]] + remaining_shape
        shape2 = [batch * np.prod(block_shape)] + list(permuted_reshaped_padded_shape)[M + 1:]
        reshaped_permuted_reshaped_padded = tvm.relay.reshape(permuted_reshaped_padded,
                                                              newshape=shape2)
        return reshaped_permuted_reshaped_padded

    return _impl


def _batch_to_space_nd():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        input_shape = attr['_input_shapes'][input_node]
        block_shape = _get_list_param(params, inputs[1])
        crops = _get_list_param(params, inputs[2])
        M = len(block_shape)
        batch = input_shape[0]
        # From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d:
        # Reshape input to reshaped of shape:
        # [block_shape[0], ..., block_shape[M-1], batch / prod(block_shape),
        #  input_shape[1], ..., input_shape[N-1]]
        shape1 = block_shape + [batch // np.prod(block_shape)] + input_shape[1:]
        reshaped = tvm.relay.reshape(input_node, newshape=shape1)
        # Permute dimensions of reshaped to produce permuted of shape
        # [batch / prod(block_shape), input_shape[1], block_shape[0], ...,
        # input_shape[M], block_shape[M-1], input_shape[M+1], ..., input_shape[N-1]]
        axes = [M] + [axis for i in range(M) for axis in [M + i + 1, i]] + \
            list(range(2 * M + 1, len(shape1)))
        permuted = tvm.relay.transpose(reshaped, axes=axes)
        # Reshape permuted to produce reshaped_permuted of shape
        # [batch / prod(block_shape), input_shape[1] * block_shape[0], ...,
        #  input_shape[M] * block_shape[M-1], input_shape[M+1], ..., input_shape[N-1]]
        shape2 = [0] + [-3] * M + [-2]
        reshaped_permuted = tvm.relay.reshape(permuted, newshape=shape2)
        # Crop the start and end of dimensions [1, ..., M] of reshaped_permuted according to crops
        # to produce the output of shape:
        # [batch / prod(block_shape), input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
        #  ..., input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
        #  input_shape[M+1], ..., input_shape[N-1]]
        reshaped_permuted_shape = _infer_shape(reshaped_permuted)
        cropped = reshaped_permuted
        for axis in range(1, M+1):
            crop = crops[axis - 1]
            if crop != [0, 0]:
                indices = tvm.relay.arange(
                    _expr.const(crop[0]),
                    _expr.const(reshaped_permuted_shape[axis] - crop[1]),
                    dtype='int32'
                )
                cropped = tvm.relay.take(cropped, indices=indices, axis=axis)

        return cropped

    return _impl


def _prod():
    def _impl(inputs, attr, params):
        axis = _get_num_param(params, inputs[1])
        keepdims = attr['keep_dims']
        return _op.prod(inputs[0], int(axis), keepdims=keepdims)
    return _impl

def _log1p():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/log1p
    def _impl(inputs, attr, params):
        one = tvm.relay.const(1, attr['T'].name)
        add_out = get_relay_op('add')(inputs[0], one)
        return get_relay_op('log')(add_out)
    return _impl

def _one_hot():
    def _impl(inputs, attr, params):
        depth = int(_get_num_param(params, inputs[1]))
        dtype = attr['T'].name

        on_value = _get_num_param(params, inputs[2])
        off_value = _get_num_param(params, inputs[3])
        new_inputs = [inputs[0], \
                      tvm.relay.const(on_value, dtype), \
                      tvm.relay.const(off_value, dtype)]
        return AttrCvt('one_hot',
                       ignores=['TI'],
                       extras={'depth' : depth, 'dtype' : dtype})(new_inputs, attr)
    return _impl

def _squared_difference():
    def _impl(inputs, attr, params):
        difference = _op.subtract(inputs[0], inputs[1])
        return _op.multiply(difference, difference)
    return _impl

def _size():
    def _impl(inputs, attr, params):
        new_attr = attr
        new_attr['out_type'] = attr['out_type'].name
        return AttrCvt('ndarray_size', transforms={'out_type' : 'dtype'})(inputs, new_attr)
    return _impl

def _add_n():
    def _impl(inputs, attr, params):
        if not isinstance(inputs, tuple):
            inputs = list(inputs)
        assert len(inputs) > 0, "add_n take >=1 inputs, but 0 given."
        _res = inputs[0]
        for each in inputs[1:]:
            _res = _op.add(_res, each)
        return  _res
    return _impl


# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    'Abs'                               : AttrCvt('abs'),
    'Add'                               : _elemwise('add'),
    'AddN'                              : _add_n(),
    'All'                               : _reduce('all'),
    'Any'                               : _reduce('any'),
    'ArgMax'                            : _argx(_op.argmax, 'argmax'),
    'ArgMin'                            : _argx(_op.argmin, 'argmin'),
    'Assert'                            : _assert(),
    'AvgPool'                           : _pooling('avg_pool'),
    'AvgPool3D'                         : _pool3d('avg_pool3d'),
    'BatchMatMul'                       : _batch_matmul(),
    'BatchMatMulV2'                     : _batch_matmul(),
    'BatchNormWithGlobalNormalization'  : _batch_norm(),
    'BatchToSpaceND'                    : _batch_to_space_nd(),
    'BiasAdd'                           : _bias_add(),
    'BroadcastTo'                       : _broadcast_to(),
    'Cast'                              : _cast(),
    'Ceil'                              : AttrCvt('ceil'),
    'CheckNumerics'                     : _check_numerics(),
    'ClipByValue'                       : _clip_by_value(),
    'Concat'                            : _concat(),
    'ConcatV2'                          : _concatV2(),
    'Conv2D'                            : _conv('conv'),
    'Conv3D'                            : _conv3d('conv'),
    'Conv2DBackpropInput'               : _conv('conv_transpose'),
    'CropAndResize'                     : _crop_and_resize(),
    'DecodeJpeg'                        : _decode_image(),
    'DepthwiseConv2dNative'             : _conv('depthwise'),
    'DepthToSpace'                      : _depth_to_space(),
    'Equal'                             : _broadcast('equal'),
    'Elu'                               : _elu(),
    'Erf'                               : AttrCvt('erf'),
    'Exp'                               : AttrCvt('exp'),
    'ExpandDims'                        : _expand_dims(),
    'Fill'                              : _fill(),
    'Floor'                             : AttrCvt('floor'),
    'FloorDiv'                          : _floordiv(),
    'FloorMod'                          : _floormod(),
    'FusedBatchNorm'                    : _fused_batch_norm(),
    'FusedBatchNormV2'                  : _fused_batch_norm(),
    'Gather'                            : _gather(),
    'GatherNd'                          : _gather_nd(),
    'GatherV2'                          : _gather(),
    'Greater'                           : _broadcast('greater'),
    'GreaterEqual'                      : _broadcast('greater_equal'),
    'Identity'                          : _identity(),
    'LeakyRelu'                         : AttrCvt('leaky_relu'),
    'LeftShift'                         : AttrCvt('left_shift'),
    'Less'                              : _broadcast('less'),
    'LessEqual'                         : _broadcast('less_equal'),
    'Log'                               : AttrCvt('log'),
    'Log1p'                             : _log1p(),
    'Cos'                               : AttrCvt('cos'),
    'Sin'                               : AttrCvt('sin'),
    'LogicalAnd'                        : _logical('logical_and'),
    'LogicalOr'                         : _logical('logical_or'),
    'LogicalNot'                        : _logical('logical_not'),
    'LogSoftmax'                        : AttrCvt('log_softmax'),
    'LRN'                               : _lrn(),
    'MatMul'                            : _matmul(),
    'Max'                               : _reduce('max'),
    'MaxPool'                           : _pooling('max_pool'),
    'MaxPool3D'                         : _pool3d('max_pool3d'),
    'Maximum'                           : _elemwise('maximum'),
    'Mean'                              : _mean(),
    'Min'                               : _reduce('min'),
    'Minimum'                           : _elemwise('minimum'),
    'MirrorPad'                         : _mirror_pad(),
    'Mod'                               : _elemwise('mod'),
    'Mul'                               : _elemwise('multiply'),
    'Neg'                               : AttrCvt('negative'),
    'NoOp'                              : _no_op(),
    'NotEqual'                          : _broadcast('not_equal'),
    'OneHot'                            : _one_hot(),
    'Pack'                              : _pack(),
    'TensorArrayV3'                     : _tensor_array(),
    'TensorArrayScatterV3'              : _tensor_array_scatter(),
    'TensorArrayGatherV3'               : _tensor_array_gather(),
    'TensorArraySizeV3'                 : _tensor_array_size(),
    'TensorArrayWriteV3'                : _tensor_array_write(),
    'TensorArrayReadV3'                 : _tensor_array_read(),
    'TensorArraySplitV3'                : _tensor_array_split(),
    'TensorArrayConcatV3'               : _tensor_array_concat(),
    'Pad'                               : _pad('Pad'),
    'PadV2'                             : _pad('PadV2'),
    'Pow'                               : _elemwise('power'),
    'Prod'                              : _prod(),
    'Range'                             : _range(),
    'Rank'                              : _rank(),
    'RealDiv'                           : _elemwise('divide'),
    'Relu'                              : AttrCvt('relu'),
    'Relu6'                             : _relu6(),
    'Reshape'                           : _reshape(),
    'ResizeBilinear'                    : _resize('bilinear'),
    'ResizeBicubic'                     : _resize('bilinear'),
    'ResizeNearestNeighbor'             : _resize('nearest_neighbor'),
    'ReverseV2'                         : _reverse_v2(),
    'RightShift'                        : AttrCvt('right_shift'),
    'Round'                             : AttrCvt('round'),
    'Rsqrt'                             : _rsqrt(),
    'Select'                            : _where(),
    'Selu'                              : _selu(),
    'Shape'                             : _shape(),
    'Sigmoid'                           : AttrCvt('sigmoid'),
    'Sign'                              : AttrCvt('sign'),
    'Size'                              : _size(),
    'Slice'                             : _slice(),
    'Softmax'                           : _softmax(),
    'Softplus'                          : _softplus(),
    'SpaceToBatchND'                    : _space_to_batch_nd(),
    'SpaceToDepth'                      : _space_to_depth(),
    'Split'                             : _split(False),
    'SplitV'                            : _split(True),
    'Sqrt'                              : AttrCvt('sqrt'),
    'Square'                            : _square(),
    'SquaredDifference'                 : _squared_difference(),
    'Squeeze'                           : _squeeze(),
    'StopGradient'                      : _identity(),
    'StridedSlice'                      : _stridedSlice(),
    'Sub'                               : _elemwise('subtract'),
    'Sum'                               : _sum(),
    'Tanh'                              : AttrCvt('tanh'),
    'Tile'                              : _tile(),
    'TopKV2'                            : _topk(),
    'Transpose'                         : _transpose(),
    'TruncateMod'                       : _elemwise('mod'),
    'Unpack'                            : _unpack(),
    'Where'                             : _where(),
    'ZerosLike'                         : AttrCvt('zeros_like'),

}

def _LSTMBlockCell():
    def _impl(inputs, in_state_c, in_state_h, attr, params):
        """LSTM Block cell.
        Calculations are described in: https://github.com/tensorflow/tensorflow/blob/
        r1.8/tensorflow/contrib/rnn/python/ops/lstm_ops.py#L41-L114

        Parameters
        ----------
        inputs : relay.Expr
            Input data
        in_state_c: list of relay.Expr
            Cell state input values for all the layers
        in_state_h: list of relay.Expr
            Hidden state input values for all the layers
        attrs : dict
            Dict of operator attributes
        params : dict
            List of pretrained weights and bias

        Returns
        -------
        sym : relay.Expr
            Converted relay.Expr
        output: relay.Expr
            Output state value.
        """
        in_data = inputs[0]
        in_weight = inputs[3]
        in_bias = inputs[7]
        forget_bias = attr.pop('forget_bias')
        input_shape = attr['_input_shapes'][inputs[0]]
        weight_shape = attr['_input_shapes'][inputs[3]]
        batch_size, input_size = input_shape[0], input_shape[1]
        num_hidden_layers = weight_shape[1]
        num_hidden = num_hidden_layers // 4

        in_data = _op.reshape(in_data,
                              newshape=(batch_size, input_size))
        ixh = _op.concatenate([in_data, in_state_h], axis=1)
        in_weight = _op.transpose(in_weight, axes=None)
        gates = _op.nn.dense(ixh, in_weight,
                             units=num_hidden_layers)
        gates_bias = _op.add(gates, in_bias)
        gate_list = _op.split(gates_bias, indices_or_sections=4, axis=1)
        in_gate = _op.sigmoid(gate_list[0])
        in_transform = _op.tanh(gate_list[1])
        forget_gate = _op.add(gate_list[2], tvm.relay.const(forget_bias, attr['T'].name))
        forget_gate = _op.sigmoid(forget_gate)
        out_gate = _op.sigmoid(gate_list[3])
        next_c = _op.add(_op.multiply(forget_gate, in_state_c),
                         _op.multiply(in_gate, in_transform))
        next_h = out_gate * _op.tanh(next_c)
        out_state = _op.concatenate([next_c, next_h], axis=1)
        out_state = _op.reshape(out_state,
                                newshape=(2, batch_size, num_hidden))
        return next_h, out_state
    return _impl

# _convert_map_rnn defines maps of rnn operator name to
# converter functor(callable) for 1 to 1 mapping.
_convert_map_rnn = {
    'LSTMBlockCell'                     : _LSTMBlockCell(),
}

class RecurrentNetworks(object):
    """Recurrent network layer handlers.

    Handle Layer operations.
    ToDo: Operators like RNN/GRU layer concepts also can be handled here

    Parameters
    ----------
    nodes : list
        list of graph nodes used for tensorflow parsing.

    out_rnn : list
        List of RecurrentNetwork outputs. This output will be appended to the
        'head' nodes of the graph.

    graph : tensorflow graph definition object
        The loaded tensorflow GraphDef

    convert_map : dict
        Dict of name : callable, where name is the op's name that
        require conversion to relay, callable are functions which
        take attrs and return (new_op_name, new_attrs)
    """
    def __init__(self, nodes, out_rnn, graph, convert_map):
        self._graph = graph
        self._convert_map = convert_map
        self._nodes = nodes
        self._out_rnn = out_rnn
        self._cur_lstm_layer = 0
        self._layer_name_list = []
        self._recurrent_ops_layer_map = {
            'LSTMBlockCell'               : self._LSTMBlockCellLayer(),
        }

    def _LSTMBlockCellLayer(self):
        """LSTMBlockCell layer handler.

        Parameters
        ----------
        op_name : str
            Operator name, eg:LSTMBlockCell

        layer_name : str list
            Layer name is used for creating the state input placeholder.

        inputs : relay.Expr
            Input data

        attrs : dict
            Dict of operator attributes

        params : dict
            List of pretrained weights and bias

        num_layers : int
            Total number of LSTM layer presented in the graph

        Returns
        -------
        sym : relay.Expr
            The returned relay Expr
        """
        def _impl(op_name, layer_name, inputs, attrs, params, num_layers):
            in_state_c_name = layer_name+'_c'
            in_state_h_name = layer_name+'_h'

            def _init_state(num_layers, batch_size, num_hidden):
                """Create the initial states for the first layer in the graph."""
                in_state_c = [_expr.var(in_state_c_name,
                                        shape=(num_layers, batch_size, num_hidden),
                                        dtype='float32')]

                in_state_h = [_expr.var(in_state_h_name,
                                        shape=(num_layers, batch_size, num_hidden),
                                        dtype='float32')]
                return in_state_c, in_state_h

            def _get_cur_input_state(in_state_c, in_state_h, num_layers,
                                     layer, batch_size, num_hidden):
                """Select the appropriate states for the current layer"""
                in_state_c_tup = _op.split(in_state_c[0],
                                           indices_or_sections=num_layers, axis=0)
                in_state_h_tup = _op.split(in_state_h[0],
                                           indices_or_sections=num_layers, axis=0)
                cur_in_state_c = _op.reshape(in_state_c_tup[layer],
                                             newshape=(batch_size, num_hidden))
                cur_in_state_h = _op.reshape(in_state_h_tup[layer],
                                             newshape=(batch_size, num_hidden))
                return cur_in_state_c, cur_in_state_h

            def _LSTMBlockCellWrapper(inputs, attr, params,
                                      num_layers, layer):
                """LSTM cell warapper to prepare the inputs"""
                input_shape = attr['_input_shapes'][inputs[0]]
                weight_shape = attr['_input_shapes'][inputs[3]]

                batch_size = input_shape[0]
                num_hidden = weight_shape[1] // 4

                if layer == 0:
                    #Create initial states placeholder in case of first layer
                    in_state_c, in_state_h = _init_state(num_layers,
                                                         batch_size, num_hidden)
                else:
                    in_state_c = self._nodes[in_state_c_name]
                    in_state_h = self._nodes[in_state_h_name]

                cur_in_state_c, cur_in_state_h = _get_cur_input_state( \
                                                    in_state_c, in_state_h,
                                                    num_layers, layer,
                                                    batch_size, num_hidden)
                output, out_state = self._convert_map[op_name](inputs, cur_in_state_c,
                                                               cur_in_state_h,
                                                               attr, params)
                return output, out_state, in_state_c, in_state_h

            sym, cur_out_state, in_state_c, in_state_h = \
                    _LSTMBlockCellWrapper(inputs, attrs, params,
                                          num_layers, self._cur_lstm_layer)
            self._nodes[in_state_c_name] = in_state_c
            self._nodes[in_state_h_name] = in_state_h
            cur_out_state = _op.expand_dims(cur_out_state, axis=0, num_newaxis=1)
            self._out_rnn.append(cur_out_state)
            self._cur_lstm_layer += 1
            return sym
        return _impl

    def process_op(self, op_name, inputs, attrs, params):
        """Process recurrent layer operators.

        List '_recurrent_ops_layer_map' map each Layer based operators with its
        layer handlers. Total number of layers are calculated to form the input
        data shapes.

        Parameters
        ----------
        op_name : str
            Operator name, such as LSTMBlockCell

        inputs : relay.Expr
            Input data

        attrs : dict
            Dict of operator attributes

        params : dict
            List of pretrained weights and bias

        Returns
        -------
        sym : relay.Expr
            Returns relay.Expr
        """
        def _get_abs_layer_name(node):
            """Identify the layer name is already handled. Return the absolute name
            """
            if not self._layer_name_list:
                self._layer_name_list.append(node.name)
                return node.name

            for _name in self._layer_name_list:
                if _name in node.name:
                    abs_name = _name
                else:
                    self._layer_name_list.append(node.name)
                    abs_name = node.name
            return abs_name

        #Find number of layers of this same operator node in the graph
        #and also read the inputs name for the current op.
        num_layers = 0
        for _, node in enumerate(self._graph.node):
            if node.op == op_name:
                layer_name = _get_abs_layer_name(node)
                num_layers += 1

        sym = self._recurrent_ops_layer_map[op_name](op_name, layer_name, inputs, attrs,
                                                     params, num_layers)
        return sym

# An internal list to contain all the control flow primitives used in Tensorflow
# 1.x.
_control_flow_nodes = ['Merge', 'Switch', 'NextIteration', 'Exit', 'Enter', 'LoopCond']

class RewriteSubgraph(ExprMutator):
    """
    A helper class to rewrite expr in while loop function to variable

    Parameters
    ----------
    rewrite_map : Dict[expr, expr]
        A dictionay contains a set of expr to var mapping.
    """
    def __init__(self, rewrite_map):
        ExprMutator.__init__(self)
        self.rewrite_map = rewrite_map

    def visit(self, expr):
        if expr in self.rewrite_map:
            return self.rewrite_map[expr]
        return super().visit(expr)

def rewrite_subgraph(expr, rewrites):
    return RewriteSubgraph(rewrites).visit(expr)

def _in_while_loop(control_flow_node_map, op_name):
    """
    Check if a given control flow operator is part of a while loop execution
    frame. This is based on the fact that there is only one occurrence of
    `LoopCond` for a loop execution frame and it is only presented in the loop
    construct.

    Parameters
    ----------
    control_flow_node_map : Dict[str, Set[str]]
        A dictionay contains the unique control flow execution frame name to
        a set of primitive operators mapping.

    op_name : str
        The name of a control flow primitive.

    Returns
    -------
    ret : bool
        Return true if the operator is in a while loop execution frame,
    otherwise, return false.
    """
    return op_name in control_flow_node_map and \
            "LoopCond" in control_flow_node_map[op_name]


class Branch:
    """A class contains the components that are used to build up a Relay if
    node.

    Parameters
    ----------
    cond : tvm.relay.Expr
        The condition of a if node.

    true_branch : tvm.relay.Expr
        The body of the true branch of a if expression.

    false_branch: tvm.relay.Expr
        The body of the false branch of a if expression.

    _if : tvm.relay.Expr
        An internal variable indicates where an if expression is already created
        for a matched TF condition construct.

    Examples
    --------
    The following is a cond statement written in TensorFlow:

    .. code-block:: python

        def vanilla_cond():
            i = tf.constant(1)
            j = tf.constant(4)

             def f1():
                return tf.multiply(1, 17)

             def f2():
                return tf.add(4, 23)
            r = tf.cond(tf.less(i, j), f1, f2)

    This condition statement should be converted into Relay in the following
    form:

    .. code-block:: python

        fn (%Const: Tensor[(1,), int32],
            %Const_1: Tensor[(1,), int32],
            %cond/Mul/x: Tensor[(1,), int32],
            %cond/Mul/y: Tensor[(1,), int32],
            %cond/Add/x: Tensor[(1,), int32],
            %cond/Add/y: Tensor[(1,), int32]) {
          %0 = less(%Const, %Const_1) # ty=Tensor[(1,), bool]
          %1 = min(%0)
          if (%1) {
            %2 = multiply(%cond/Mul/x, %cond/Mul/y)
            %2
          }  else {
            %3 = add(%cond/Add/x, %cond/Add/y)
            %3
          }
        }
    """
    def __init__(self):
        self._if = None
        self.cond = None
        self.true_branch = None
        self.false_branch = None

    def _if_node(self):
        """An internal API to create a relay if node from the matched TF
        condition construct.
        """
        # `cond`  returns a tensor that contains boolean values. We add a `min`
        # operator to checks if there is any false value. If so, this condition
        # doesn't not hold.
        cond = tvm.relay.op.min(self.cond)
        return tvm.relay.If(cond, self.true_branch, self.false_branch)

    def if_node(self):
        """Create an tvm.relay.If node if it hasn't been created yet."""
        if self._if is None:
            self._if = self._if_node()
        return self._if


class Loop:
    """
    A class contains the components that are used to build up a Relay
    recursive call.

    Parameters
    ----------
    loop_vars : List[tvm.relay.Expr]
        The loop variables that used in a while loop.

    cond : tvm.relay.Expr
        The condition of a while loop.

    body : tvm.relay.Expr
        The body of a matched while loop.

    _loop : tvm.relay.Expr
        An internal variable indicates where a recursive call is already created
        for a matched TF while loop construct.

    Examples
    --------
    The following is a vanilla loop from TensorFlow:

    .. code-block:: python

        i = tf.constant(0)
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])

    It will be converted to the following recursive call in Relay:

    .. code-block:: python

        fn (%while/Less/y: Tensor[(1,), int32],
            %while/Add/y: Tensor[(1,), int32],
            %Const: Tensor[(1,), int32]) {
          %0 = fn(%loop_var0: Tensor[(1,), int32]) {
            %1 = less(%loop_var0, %while/Less/y)
            %2 = min(%1)
            if (%2) {
              %3 = add(%loop_var0, %while/Add/y)
              free_var %while_loop
              %4 = %while_loop(%3)
              %4
            }    else {
              %5 = (%loop_var0,)
              %5
            }
          }
          let %while_loop1 = %0
          %6 = %while_loop1(%Const)
          %6
        }
    """
    def __init__(self):
        self.loop_vars = []
        self.cond = None
        self.body = []
        self._loop = None

    def _while_loop(self):
        """An internal API to create a Relay recursive call for a matched TF
        `while_loop` construct.
        """
        wl = tvm.relay.var('while_loop')

        sb = tvm.relay.scope_builder.ScopeBuilder()

        loop_vars = []
        bind_map = {}
        for i, var in enumerate(self.loop_vars):
            if not isinstance(var, _expr.Var):
                var_chk = _infer_type(var)
                var_type = var_chk.checked_type
            else:
                var_type = var.type_annotation

            v = tvm.relay.var("loop_var" + str(i), type_annotation=var_type)
            loop_vars.append(v)
            bind_map[var] = v

        self.cond = rewrite_subgraph(self.cond, bind_map)
        self.body = [rewrite_subgraph(b, bind_map) for b in self.body]

        cond = tvm.relay.op.min(self.cond)

        with sb.if_scope(cond):
            sb.ret(wl(*self.body))
        with sb.else_scope():
            sb.ret(tvm.relay.Tuple(loop_vars))

        loop_fn = tvm.relay.Function(loop_vars, sb.get())
        sb = tvm.relay.scope_builder.ScopeBuilder()
        sb.let(wl, loop_fn)
        sb.ret(wl(*self.loop_vars))
        return sb.get()

    def while_loop(self):
        """Instantiate a while loop if it has not been created yet."""
        if self._loop is None:
            self._loop = self._while_loop()
            return self._loop
        return self._loop


class GraphProto(object):
    """ A helper class for handling relay graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._input_shapes = {}
        self._output_shapes = {}
        self._num_param = 0
        self._num_rnn_layer = False
        self._input_shapes = {}
        self._loops = {}
        self._branches = {}
        self._mod = IRModule({})
        self._prelude = Prelude(self._mod)

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Construct relay nodes from tensorflow graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to Relay.
        Some of the assumptions listed below.

            -> All Placeholders are considered as graph input.
            -> All Const nodes are params.
            -> Last node is assumed as graph output.
            -> _output_shapes : Graph should be frozen with add_shapes=True.
                                Or user can pass input shape dictionary optionally.
            -> DecodeJpeg, ResizeBilinear: These are dummy operators.
                                           Hence user should handle preprocessing outside.
            -> CheckNumerics: No implementation as of now for this.
                              Just copies input to output.

        Parameters
        ----------
        graph : tensorflow graph definition object
            The loaded tensorflow GraphDef

        layout : target layout to be used (Optional)
            NCHW only supported now to enable NHWC models on GPU.

        shape : Dictionary of input dimensions (Optional)
            Graph level input shape dictionary.

        outputs : List of output tensor names (Optional)
            if not specified then the last node is assumed as graph output.

        Returns
        -------
        mod : tvm.IRModule
            The module that optimizations will be performed on.

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """

        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        missing_operators = self._parse_import_prerequisites(graph)

        if missing_operators:
            raise NotImplementedError( \
                "The following operators are not implemented: {}".format(missing_operators))

        control_flow_node_map = defaultdict(set)
        for node in graph.node:
            node_name_prefix = node.name.rsplit('/', 1)[0]
            control_flow_node_map[node_name_prefix].add(node.op)
            if node.op == 'Placeholder' or node.op == 'PlaceholderWithDefault':
                # Give priority to user argument.
                if shape and node.name in shape:
                    self._input_shapes[node.name] = list(shape[node.name])
                else:
                    self._input_shapes[node.name] = \
                        tensor_util.TensorShapeProtoToList(node.attr['shape'].shape)
                    for idx, dim in enumerate(self._input_shapes[node.name]):
                        if dim < 0:
                            self._input_shapes[node.name][idx] = 1
                            warnings.warn("Use 1 instead of -1 in shape of operator %s."
                                          % node.name)

                self._output_shapes[node.name] = [self._input_shapes[node.name]]
                attr = self._parse_attr(node.attr)
                self._nodes[node.name] = [_expr.var(node.name,
                                                    shape=self._input_shapes[node.name],
                                                    dtype=attr['dtype'].name)]

                # Ignore user's input shape for Non placeholder
            elif node.op == 'Const':
                tensor_value = node.attr['value'].tensor
                self._input_shapes[node.name] = \
                    tensor_util.TensorShapeProtoToList(tensor_value.tensor_shape)
                if shape and node.name in shape:
                    warnings.warn("Ignore the passed shape. Shape in graphdef "
                                  "will be used for operator %s." % node.name)

        # Parse the nodes to re-create TF graph using Relay operators.
        for node in graph.node:
            # Tensorflow doesn't have separate list for params extraction.
            # Operator name 'Const' is treated as a parameter to build params dict.

            input_shapes = {}
            attr = self._parse_attr(node.attr)

            # Variable converted to Const will not have only value attr
            if 'value' in attr and node.op == 'Const':
                self._output_shapes[node.name] = [self._input_shapes[node.name]]
            elif '_output_shapes' in attr:
                self._output_shapes[node.name] = \
                    [tensor_util.TensorShapeProtoToList(tshape) \
                    for tshape in attr['_output_shapes']]
            else:
                # Keep the list indexable to avoid key error.
                # Actual value will be filled after node creation.
                # Will infer shapes if the graph is not frozen with add_shapes=True
                self._output_shapes[node.name] = [None]

            if node.op == "Const":
                # All Const nodes are Param nodes, lets parse
                self._num_param += 1
                for key, value in node.attr.items():
                    self._parse_param(key, value, node.name, shape)
                if node.name not in self._nodes:
                    raise NotImplementedError( \
                        "Const {} couldn't be converted to Param.".format(node.name))

                attr = self._parse_attr(node.attr)

            elif node.op != "Placeholder" and node.op != 'PlaceholderWithDefault':
                # Pass the parsed shapes instead
                attr["_output_shapes"] = output_shapes = self._output_shapes[node.name]

                # Pass the node name too in attr
                attr["_node_name"] = node.name

                # Pass the target layout
                attr["_target_layout"] = layout

                # Fill shapes for all inputs in a list
                inputs = []
                for i in node.input:
                    # Some TensorFlow operators internally maintain execution layers
                    # and their output name includes the layer number along with
                    # graph node name. E.g. the node name is 'Model/RNN/cell_0/RnnCell', but the
                    # output tensor name is 'Model/RNN/cell_0/RnnCell:0'. In this case,
                    # the number has to be ignored for single-output nodes.
                    # On the other hand, for multi-output nodes the number is the output index,
                    # and the lack of the number implies 0.
                    tensor_name = i.split(':')
                    node_name = tensor_name[0]
                    if node_name in self._nodes:
                        in_sym = self._nodes[node_name]
                        if isinstance(in_sym, _expr.TupleWrapper):
                            tensor_slot = int(tensor_name[1]) if len(tensor_name) > 1 else 0
                            in_sym = [in_sym[tensor_slot]]
                            input_shape = self._output_shapes[node_name][tensor_slot]
                        else:
                            tensor_slot = 0
                            input_shape = self._output_shapes[node_name][0]
                        inputs.append(in_sym[0])
                        input_shapes[in_sym[0]] = input_shape

                attr['_input_shapes'] = input_shapes

                if node.op in _control_flow_nodes:
                    op = self._convert_control_flow_operator(node, inputs,
                                                             attr,
                                                             control_flow_node_map)
                else:
                    op = self._convert_operator(node.op, inputs, attr, graph)

                # Check if op is converted to param
                if isinstance(op, np.ndarray):
                    self._params[node.name] = tvm.nd.array(op)
                    op = [_expr.var(node.name,
                                    shape=self._params[node.name].shape,
                                    dtype=self._params[node.name].dtype)]

                elif isinstance(op, (_expr.TupleWrapper, tuple, list)):
                    pass
                elif isinstance(op, _expr.Expr):
                    op = [op]
                else:
                    raise RuntimeError("unexpected type %s" % type(op))

                self._nodes[node.name] = op

                # Infer shapes even without specifying "add_shapes=True"
                if output_shapes == [None]:
                    out_shapes = [_infer_shape(node_item, self._mod)
                                  for node_item in self._nodes[node.name]]
                    self._output_shapes[node.name] = out_shapes

                if self._output_shapes[node.name] and shape and node.name in shape:
                    assert self._output_shapes[node.name] == list(shape[node.name])

            # Infer shapes if passed explicitly
            node_output = self._nodes[node.name]
            if shape and (not self._output_shapes[node.name][0]
                          or -1 in self._output_shapes[node.name][0]):
                out_shapes = [_infer_shape(node_item, self._mod) for node_item in node_output]
                self._output_shapes[node.name] = out_shapes

        out = []
        if outputs is None:
            if node.op == "Exit":
                out = [op[0].tuple_value]
            else:
                out = op
        else:
            for out_name in outputs:
                if ":" in out_name:
                    out_name, out_num = out_name.split(":")
                    out_num = int(out_num)
                    out.append(self._nodes[out_name][out_num])
                else:
                    out.append(self._nodes[out_name][0])

        #Add the RNN outputs also with 'head' nodes of the relay graph
        if self._num_rnn_layer:
            if len(self._out_rnn) == 1:
                out.append(self._out_rnn[0])
            else:
                out_rnn = _op.concatenate(self._out_rnn, axis=0)
                out.append(out_rnn)

        out = out[0] if len(out) == 1 else _expr.Tuple(out)
        func = _expr.Function(analysis.free_vars(out), out)
        self._mod["main"] = func
        return self._mod, self._params

    def _parse_import_prerequisites(self, graph):
        """ Calculate the named preconditions from TensorFlow `graph`.
            Return prerequisites for parsing:
            a. Set of operator names which don't have their mapping in TVM, i.e.
                which are not supported
        """
        missing_operators = set()
        for node in graph.node:
            if node.op == "Placeholder" or node.op == 'PlaceholderWithDefault':
                pass
            elif node.op == "Const":
                pass
            else:
                if any([node.op in t for t in [_identity_list, _convert_map,
                                               _convert_map_rnn,
                                               _control_flow_nodes]]):
                    pass
                else:
                    missing_operators.add(node.op)

        return missing_operators

    def _parse_param(self, key, value, name, shape):
        try:
            from tensorflow.python.framework import tensor_util
        except ImportError as e:
            raise ImportError(
                "Unable to import tensorflow which is required {}".format(e))

        if key == 'value':
            np_array = tensor_util.MakeNdarray(value.tensor)

            if np_array.dtype == np.dtype(object):
                # Object types are generally tensorflow DT_STRING (DecodeJpeg op).
                # Just leave it as placeholder.
                if shape and name in shape:
                    var_shape = shape[name]
                else:
                    var_shape = tensor_util.TensorShapeProtoToList(value.tensor.tensor_shape)
                self._nodes[name] = [_expr.var(name, shape=var_shape, dtype='uint8')]
                return

            array_ndim = len(np_array.shape)
            if array_ndim == 0:
                new_array = np.empty([1], dtype=np_array.dtype)
                new_array[0] = np_array
                self._nodes[name] = [tvm.relay.const(new_array)]
            else:
                self._params[name] = tvm.nd.array(np_array)
                self._nodes[name] = [_expr.var(name,
                                               shape=self._params[name].shape,
                                               dtype=self._params[name].dtype)]
        else:
            if key not in ('dtype', '_output_shapes', '_class'):
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
                        ret += [dtypes.as_dtype(x) for x in list(getattr(x.list, f))]
                    else:
                        ret += list(getattr(x.list, f))
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

    def _convert_rnn_operator(self, op_name, inputs,
                              attrs, params, graph, convert_map):
        """Convert RNN and its variant operators to Relay operators.
        This converter read the input states of each layers and
        also maintain the output states of each layer in a list.

        Parameters
        ----------
        op_name : str
            Operator name, such as LSTMBlockCell
        inputs : list of relay.Expr
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        params : dict
            List of pretrained weights and bias
        graph : Tensorflow graph object
            Graph is to find the number of upcoming same operator to
            calculate the number of layers.
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to relay, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : relay.Expr
            Converted relay.Expr
        """
        if not self._num_rnn_layer:
            self._out_rnn = []
            self.rnn = RecurrentNetworks(self._nodes, self._out_rnn, graph, convert_map)
            self._num_rnn_layer = True
        sym = self.rnn.process_op(op_name, inputs, attrs, params)
        return sym

    def _convert_control_flow_operator(self, node, inputs, attrs, control_flow_node_map):
        """
        Convert the Relay control flow primitive into corresponding component
        of a Relay control flow construct, i.e. `tf.cond` and `tf.while_loop`
        are converted in Relay `If` and recusrive call, respectively.

        Parameters
        ----------
        node: TensorFlow graph node object.
            A TensorFlow graph node object.

        inputs : List[tvm.relay.Expr]
            List of input symbols.

        attrs : Dict[tvm.Attrs]
            Dict of operator attributes.

        control_flow_node_map : Dict[str, Set[str]]
            A dictionary contains the execution frame name to primitives
            mapping.

        Returns
        -------
        op : tvm.relay.Expr
            Converted relay expression.
        """
        node_name_prefix = node.name.rsplit('/', 1)[0]
        if node.op == "Merge":
            if _in_while_loop(control_flow_node_map, node_name_prefix):
                op = self._nodes[node.input[0]]
                self._loops[node_name_prefix] = Loop()
            else:
                if len(self._branches) == 0:
                    raise RuntimeError("Cannot find a created "
                                       "conditional for merge node")
                branch = self._branches[node_name_prefix]
                false_br = self._nodes[node.input[0]]
                true_br = self._nodes[node.input[1]]
                assert len(true_br) == 1
                assert len(false_br) == 1
                branch.true_branch = true_br[0]
                branch.false_branch = false_br[0]
                op = [branch.if_node()]
        elif node.op == "Exit":
            loop = self._loops[node_name_prefix]
            exit_name = node.name.split('/')[-1]
            assert str.startswith(exit_name, 'Exit')

            # TensorFlow has differen naming convention on different
            # versions.
            if '_' in exit_name:
                exit_number = int("0" + exit_name[5:])
            else:
                exit_number = int("0" + exit_name[4:])

            expr = loop.while_loop()
            op = _expr.TupleGetItem(expr, exit_number)
        elif node.op == "Enter":
            op = self._nodes[node.input[0]]
        elif node.op == "LoopCond":
            op = self._nodes[node.input[0]]
            assert len(op) == 1
            self._loops[node_name_prefix].cond = op[0]
        elif node.op == "Switch":
            op = self._nodes[node.input[0]]
            assert len(op) == 1
            if _in_while_loop(control_flow_node_map, node_name_prefix):
                self._loops[node_name_prefix].loop_vars.append(op[0])
            else:
                if node_name_prefix not in self._branches:
                    self._branches[node_name_prefix] = Branch()
                chk_op = _infer_type(op[0])
                self._branches[node_name_prefix].cond = chk_op
        elif node.op == "NextIteration":
            op = self._nodes[node.input[0]]
            assert len(op) == 1
            self._loops[node_name_prefix].body.append(op[0])
        else:
            raise Exception("Cannot identify control flow operator: " +
                            "{}".format(node.op))

        return op


    def _convert_operator(self, op_name, inputs, attrs,
                          graph, identity_list=None, convert_map=None):
        """Convert from Tensorflow operator to relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Conv2D, AvgPool
        inputs : list of relay.op
            List of input symbols.
        attrs : dict
            Dict of operator attributes
        identity_list : list
            List of operators that don't require conversion
        convert_map : dict
            Dict of name : callable, where name is the op's name that
            require conversion to relay, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : relay.op
            Converted relay operator
        """
        identity_list = identity_list if identity_list else _identity_list
        convert_map = convert_map if convert_map else _convert_map
        convert_map_rnn = _convert_map_rnn
        if op_name in identity_list:
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            if _need_prelude_for_shape_inference(op_name):
                sym = convert_map[op_name](inputs, attrs, self._params, self._prelude)
            elif _need_module_for_shape_inference(op_name):
                sym = convert_map[op_name](inputs, attrs, self._params, self._mod)
            else:
                sym = convert_map[op_name](inputs, attrs, self._params)

        elif op_name in convert_map_rnn:
            sym = self._convert_rnn_operator(op_name, inputs, attrs,
                                             self._params, graph,
                                             convert_map_rnn)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_tensorflow(graph, layout="NHWC", shape=None, outputs=None):
    """Load tensorflow graph which is a python tensorflow graph object into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    layout : target layout to be used (Optional)
        NCHW only supported now to enable NHWC models on GPU.

    shape : Dictionary of input dimensions (Optional)
        Graph level input shape dictionary.

    outputs : List of output tensor names (Optional)
        if not specified then the last node is assumed as graph output.

    Returns
    -------
    mod : tvm.IRModule
        The module that optimizations will be performed on.

    params : dict of str to tvm.nd.NDArray
        Dict of converted parameters stored in tvm.nd.NDArray format
    """
    g = GraphProto()
    mod, params = g.from_tensorflow(graph, layout, shape, outputs)
    return mod, params
