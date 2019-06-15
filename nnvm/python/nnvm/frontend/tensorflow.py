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
"""TF: Tensorflow frontend."""
from __future__ import absolute_import as _abs
from __future__ import print_function

import warnings
# Numpy support
import numpy as np

import tvm
from .. import symbol as _sym
from .. import graph as _graph
from .. compiler import graph_util, build_module
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
        self._ignores.append('_node_name')
        self._ignores.append('is_training')
        self._ignores.append('_target_layout')
        self._ignores.append('_input_0d_mismatch')
        # Retain the names
        try:
            attrs['name'] = attrs['_node_name']
        except KeyError:
            pass
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
        raise tvm.error.OpAttributeUnimplemented(
            'Non-2D kernels are not supported for operator {}.'.format(prefix))
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

def _argx(func, func_name):
    """ A common wrapper for argmin and argmax operations """
    def _impl(inputs, attr, params):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_name = inputs[1].list_output_names()[0]
            axis_input_vlaue = params[axis_input_name].asnumpy()[0]
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for `{}` : `axis` should be a constant".format(func_name))
        return func(inputs[0], axis=axis_input_vlaue, keepdims=False)
    return _impl

def _elemwise(name):
    def _impl(inputs, attr, *args):
        assert len(inputs) == 2, "{} take 2 inputs, {} given".format(name, len(inputs))
        op_name = _math_name_picker(name)(attr)
        return get_nnvm_op(op_name)(*inputs)
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
            msg = 'Value {} in attribute "data_format" of operator Pooling is not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['data_format']))

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            tmp_shape = attr['_input_shapes'][inputs[0]]
            input_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _sym.transpose(inputs[0], axes=(0, 3, 1, 2))
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
            msg = 'Value {} in attribute "padding" of operator Pooling is not valid.'
            raise tvm.error.OpAttributeUnimplemented(msg.format(attr['padding']))

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
            out = _sym.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _conv(opname):
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        # NCHW Layout require weights transpose
        if attr['data_format'] == 'NCHW':
            tmp_shape = attr['_input_shapes'][inputs[1]]
            tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
            inputs[1] = _sym.transpose(inputs[1], axes=(3, 2, 0, 1))
            attr['_input_shapes'][inputs[1]] = tmp_shape

        input_shape = attr['_input_shapes'][inputs[0]]
        weights_shape = attr['_input_shapes'][inputs[1]]

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _sym.transpose(inputs[0], axes=(0, 3, 1, 2))
            if opname == 'conv':
                weights_shape = [weights_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _sym.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                weights_shape = [weights_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _sym.transpose(inputs[1], axes=(2, 3, 0, 1))

            attr['data_format'] = "NCHW"
            attr['strides'] = [attr['strides'][ii] for ii in (0, 3, 1, 2)]
            flip_layout = True

        if attr['data_format'] == 'NHWC':
            kernel_h, kernel_w, _, depth_mult = weights_shape
            attr['kernel_shape'] = (weights_shape[0], weights_shape[1])
            if opname == 'conv':
                attr['channels'] = weights_shape[3]
            else:
                attr['channels'] = input_shape[3] * depth_mult

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][1], attr['dilations'][2])
            attr['strides'] = (attr['strides'][1], attr['strides'][2])
        elif attr['data_format'] == 'NCHW':
            depth_mult, _, kernel_h, kernel_w = weights_shape
            attr['kernel_shape'] = (weights_shape[2], weights_shape[3])
            if opname == 'conv':
                attr['channels'] = weights_shape[0]
            else:
                attr['channels'] = input_shape[0] * depth_mult
                if attr['channels'] < 0:
                    attr['channels'] *= -1

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][2], attr['dilations'][3])
            attr['strides'] = (attr['strides'][2], attr['strides'][3])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is not valid.'
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
            if attr['data_format'] == 'NHWC':
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            dilation_h = attr['dilations'][0]
            dilation_w = attr['dilations'][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)

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
            msg = 'Value {} in attribute "padding" of operator Conv is not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if 'kernel_layout' not in attr:
            if opname == 'conv':
                attr['kernel_layout'] = 'HWIO' if attr['data_format'] == 'NHWC' else 'OIHW'
            else:
                attr['kernel_layout'] = 'HWOI' if attr['data_format'] == 'NHWC' else 'OIHW'

        out = AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            extras={'use_bias': len(inputs) == 3},
            custom_check=_dimension_constraint())(inputs, attr)

        if flip_layout:
            out = _sym.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _decode_image():
    def _impl(inputs, attr, params):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        warnings.warn("DecodeJpeg: It's a pass through, "
                      "please handle preprocessing before input")
        return inputs[0]
    return _impl

def _cast():
    def _impl(inputs, attr, params):
        # Convert from tensorflow Dtype to str
        attr['DstT'] = attr['DstT'].name
        return AttrCvt(op_name='cast', transforms={'DstT': 'dtype'},
                       ignores=['SrcT', 'Truncate'])(inputs, attr)
    return _impl

def _expand_dims():
    def _impl(inputs, attr, params):
        dim_input = inputs.pop(1)
        axis = params[dim_input.list_output_names()[0]]
        params.pop(dim_input.list_output_names()[0])
        return _expand_dims_0d_aware(inputs[0], attr, axis=axis.asnumpy()[0])
    return _impl

def _resize_bilinear():
    def _impl(inputs, attr, params):
        attr['size'] = attr['_output_shapes'][0][1:3]
        inputs.pop(1)
        # NHWC
        attr['layout'] = 'NHWC'

        return AttrCvt(op_name="resize",
                       ignores=['Tdim'],
                       extras={'method': "BILINEAR"})(inputs, attr)
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
            inputs[0] = _sym.transpose(inputs[0], axes=(1, 0))
        if not attr['transpose_b']:
            inputs[1] = _sym.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(op_name="dense",
                       extras={'use_bias': False, 'units': channels},
                       ignores=['transpose_a', 'transpose_b', 'T'])(inputs, attr)

    return _impl

def _undef():
    def _impl(inputs, attr, params):
        return _sym.__undef__()
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

def _pack():
    def _impl(inputs, attr, params):
        axis = int(attr["axis"])
        inputs_reshaped = [_expand_dims_0d_aware(i, attr, axis=axis, num_newaxis=1) for i in inputs]
        return _sym.concatenate(*inputs_reshaped, axis=axis, name=attr["_node_name"])

    return _impl

def _slice():
    def _impl(inputs, attr, params):
        begin = params.pop(inputs[1].list_output_names()[0]).asnumpy().tolist()
        size = params.pop(inputs[2].list_output_names()[0]).asnumpy().tolist()
        data_shape = attr['_input_shapes'][inputs[0]]
        data_dim = len(data_shape)
        end = size
        for i in range(data_dim):
            if size[i] == -1:
                end[i] = data_shape[i] - begin[i]
            else:
                end[i] += begin[i]
        return _sym.strided_slice(inputs[0], begin=begin, end=size)
    return _impl

def _reshape():
    def _impl(inputs, attr, params):
        try:
            pop_node = inputs[1]
            shape_arg = params.pop(pop_node.list_output_names()[0])
            inputs.pop(1)

            return AttrCvt(
                op_name="reshape",
                extras={'shape':tuple(shape_arg.asnumpy())},
                ignores=['Tshape'])(inputs, attr)
        except KeyError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            if all(in_node in params for in_node in inputs[1].list_input_names()):
                graph = _graph.create(_sym.Group(inputs[1]))
                params_pre = {k: params[k] for k in inputs[1].list_input_names()}
                params_new = build_module._run_graph(graph, params_pre)
                inputs.pop(1)
                return AttrCvt(
                    op_name="reshape",
                    extras={'shape':tuple(params_new[0].asnumpy().flatten())},
                    ignores=['Tshape'])(inputs, attr)
            raise tvm.error.OpAttributeUnimplemented(
                'Attribute "dynamic shape" of operator Reshape is not supported.')
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

def _fused_batch_norm():
    def _impl(inputs, attr, params):
        # Tensorflow: (data, gamma, beta, moving_mean, moving_variance)
        # NNVM:       (data, gamma, beta, moving_mean, moving_varience)
        axis = 3
        need_cast = False

        if 'data_format' in attr:
            attr['data_format'] = attr['data_format'].decode("utf-8")
            if attr['data_format'] == 'NCHW':
                axis = 1
        if 'U' in attr:
            need_cast = True
            inputs[0] = _sym.cast(inputs[0], dtype=attr['U'].name)

        out = AttrCvt(op_name='batch_norm',
                      transforms={'scale_after_normalization':'scale',
                                  'variance_epsilon':'epsilon'},
                      extras={'axis': axis},
                      ignores=['data_format', 'U'],
                      disables=['momentum'])(inputs, attr)

        if need_cast:
            out = _sym.cast(out, dtype=attr['T'].name)
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
        return _sym.clip(inputs[0], a_min=0, a_max=6, name=attr['_node_name'])
    return _impl

def _shape():
    def _impl(inputs, attr, params):
        return np.array(attr['_input_shapes'][inputs[0]], dtype='int32')
    return _impl

def _fill():
    def _impl(inputs, attr, params):
        fill_arg = params.pop(inputs.pop(1).list_output_names()[0])
        new_inputs = []
        return AttrCvt(
            op_name='full',
            extras={'shape':inputs[0],
                    'fill_value':fill_arg.asnumpy()[0], 'dtype':attr['T'].name},
            ignores=['index_type', 'T'])(new_inputs, attr)
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
        axis = params.pop(inputs[1].list_output_names()[0]).asnumpy()
        # convert to tuple for preventing invalid parameter format error
        axis = tuple(axis)
        return AttrCvt(
            op_name='sum',
            extras={'axis': axis},
            transforms={'keep_dims':'keepdims'},
            ignores=['name', 'Tidx'])(inputs[0], attr)
    return _impl

def _square():
    def _impl(inputs, attr, params):
        return _sym.elemwise_mul(inputs[0], inputs[0])
    return _impl

def _gather_v2():
    "Tensorflow now support only gatherv2"
    def _impl(inputs, attr, params):
        axis = params[inputs.pop(2).list_output_names()[0]].asnumpy()[0]
        new_input = []
        new_input.append(inputs.pop(0))
        new_input.append(inputs.pop(0))
        return AttrCvt(
            op_name="take",
            extras={'axis':axis},
            ignores=['Tindices', 'Tparams', 'validate_indices', \
                     'Taxis', '_class'])(new_input, attr)
    return _impl

def _infer_out_shapes(inputs, params):
    """A method to get the output shape of an intermediate node in the NNVM graph."""
    g = _graph.create(inputs)
    shape_dict = {k: v.shape for k, v in params.items()}
    _, out_shapes = graph_util.infer_shape(g, **shape_dict)
    return out_shapes

def _stridedSlice():
    def _impl(inputs, attr, params):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = params.pop(inputs[1].list_output_names()[0]).asnumpy().tolist()
        end = params.pop(inputs[2].list_output_names()[0]).asnumpy().tolist()
        stride = params.pop(inputs[3].list_output_names()[0]).asnumpy().tolist()
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
        out = _sym.strided_slice(inputs[0], begin=begin, end=end, stride=stride)
        out_shape = _infer_out_shapes(out, params)[0]
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
        # Prevent 0-dim tensors which are not accepted by nnvm
        if not final_output:
            final_output.append(1)
        return _sym.reshape(out, shape=tuple(final_output))
    return _impl

def _LSTMBlockCell():
    def _impl(inputs, in_state_c, in_state_h, attr, params):
        """LSTM Block cell.
        Calculations are described in: https://github.com/tensorflow/tensorflow/blob/
        r1.8/tensorflow/contrib/rnn/python/ops/lstm_ops.py#L41-L114

        Parameters
        ----------
        inputs : nnvm.Symbol
            Input data
        in_state_c: list of nnvm.Symbol
            Cell state input values for all the layers
        in_state_h: list of nnvm.Symbol
            Hidden state input values for all the layers
        attrs : dict
            Dict of operator attributes
        params : dict
            List of pretrained weights and bias

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        output: nnvm.Symbol
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

        in_data = _sym.reshape(in_data,
                               shape=(batch_size, input_size))
        ixh = _sym.concatenate(*[in_data, in_state_h], axis=1)
        in_weight = _sym.transpose(in_weight)
        gates = _sym.dense(ixh, in_weight, in_bias, use_bias=True,
                           units=num_hidden_layers)
        gate_list = _sym.split(gates, indices_or_sections=4, axis=1)
        in_gate = _sym.sigmoid(gate_list[0])
        in_transform = _sym.tanh(gate_list[1])
        forget_gate = _sym.sigmoid(gate_list[2])
        forget_gate = forget_gate + forget_bias
        out_gate = _sym.sigmoid(gate_list[3])
        next_c = _sym.broadcast_add(_sym.broadcast_mul(forget_gate, in_state_c),
                                    _sym.broadcast_mul(in_gate, in_transform))
        next_h = out_gate * _sym.tanh(next_c)
        out_state = _sym.concatenate(*[next_c, next_h])
        out_state = _sym.reshape(out_state,
                                 shape=(2, batch_size, num_hidden))
        return next_h, out_state
    return _impl


def _pad(name):
    def _impl(inputs, attr, params):
        padlist_key = inputs[1].list_output_names()[0]
        if padlist_key in params:
            padlist = params.pop(padlist_key).asnumpy()
        else:
            raise tvm.error.OpAttributeRequired(
                'Required attribute "{}" not found in operator Pad.'.format(padlist_key))
        paddings = tuple([tuple(l) for l in padlist])
        attr['pad_width'] = paddings
        attr['pad_value'] = 0
        new_inputs = [inputs[0]]
        if name == 'PadV2':
            constant_values = params.pop(inputs[2].list_output_names()[0]).asnumpy()
            attr['pad_value'] = constant_values[0]
        return AttrCvt(
            op_name='pad',
            ignores=['Tpaddings'],)(new_inputs, attr)
    return _impl


def _transpose():
    def _impl(inputs, attr, params):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        param_name = inputs[1].list_output_names()[0]
        axes = params.get(param_name, tvm.nd.array([])).asnumpy()
        return _sym.transpose(inputs[0], axes=tuple(axes))
    return _impl

def _rank():
    def _impl(inputs, attr, params):
        input_shape = attr['_input_shapes'][inputs[0]]

        name = attr["_node_name"]
        params[name] = tvm.nd.array([len(input_shape)])
        return _sym.Variable(name=name, shape=params[name].shape)
    return _impl

def _range():
    def _impl(inputs, attr, params):
        start = params.pop(inputs[0].list_output_names()[0]).asnumpy()[0]
        limit = params.pop(inputs[1].list_output_names()[0]).asnumpy()[0]
        delta = params.pop(inputs[2].list_output_names()[0]).asnumpy()[0]

        name = attr["_node_name"]
        params[name] = tvm.nd.array([start, limit, delta])
        return _sym.Variable(name=name, shape=params[name].shape)
    return _impl

def _elu():
    def _impl(inputs, attr, params):
        alpha = 1.0
        return -alpha * _sym.relu(1 - _sym.exp(inputs[0])) + _sym.relu(inputs[0])
    return _impl

def _selu():
    def _impl(inputs, attr, params):
        alpha = 1.6732632423543772848170429916717
        gamma = 1.0507009873554804934193349852946
        return gamma * (-alpha * _sym.relu(1 - _sym.exp(inputs[0])) + _sym.relu(inputs[0]))
    return _impl

def _mean():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].list_output_names()[0])
        return AttrCvt(op_name="mean", ignores=['Tdim', 'Tidx'],
                       transforms={'keep_dims': 'keepdims'},
                       extras={'axis': tuple(axis.asnumpy())})(inputs[0], attr)
    return _impl

def _broadcast(name):
    def _impl(inputs, attr, params):
        op_name = _math_name_picker(name)(attr)
        return AttrCvt(
            op_name=op_name,
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
                size_splits_input_name = inputs[1].list_output_names()[0]
                size_splits = params[size_splits_input_name].asnumpy()
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr['num_split']
            input_node = inputs[input_node_index]
            axis_input_name = inputs[input_axis_index].list_output_names()[0]
            axis_input_value = params[axis_input_name].asnumpy()[0]
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for split: `axis` and `num_or_size_splits` " \
                "should be constants")
        return _sym.split(input_node,
                          indices_or_sections=indices_or_sections,
                          axis=axis_input_value)
    return _impl

def _unpack():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        axis = attr['axis']
        input_shape = attr['_input_shapes'][input_node]
        axis_length = input_shape[axis]
        if axis_length < 0:
            raise TypeError("Unstack with unknown axis length")
        splitted = _sym.split(input_node,
                              indices_or_sections=axis_length,
                              axis=axis,
                              name=attr.get('_node_name', 'unstack'))

        return _sym.Group([_sym.squeeze(split_item, axis=axis) for split_item in splitted])
    return _impl

def _expand_dims_0d_aware(data, attr, axis, num_newaxis=1):
    if data in attr['_input_0d_mismatch']:
        return data if num_newaxis == 1 else \
            _sym.expand_dims(data, axis=axis, num_newaxis=num_newaxis-1)

    return _sym.expand_dims(data, axis=axis, num_newaxis=num_newaxis)

def _logical(name):
    def _impl(inputs, attr, params):
        return AttrCvt(op_name=name)(inputs, attr)
    return _impl

# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    'ArgMax'                            : _argx(_sym.argmax, 'argmax'),
    'ArgMin'                            : _argx(_sym.argmin, 'argmin'),
    'AvgPool'                           : _pooling('avg_pool'),
    'BatchNormWithGlobalNormalization'  : _batch_norm(),
    'BiasAdd'                           : _bias_add(),
    'Cast'                              : _cast(),
    'Ceil'                              : AttrCvt('ceil'),
    'CheckNumerics'                     : _check_numerics(),
    'Concat'                            : _concat(),
    'ConcatV2'                          : _concatV2(),
    'Conv2D'                            : _conv('conv'),
    'DecodeJpeg'                        : _decode_image(),
    'Elu'                               : _elu(),
    'ExpandDims'                        : _expand_dims(),
    'Floor'                             : AttrCvt('floor'),
    'Identity'                          : _identity(),
    'MatMul'                            : _matmul(),
    'MaxPool'                           : _pooling('max_pool'),
    'Add'                               : _elemwise('add'),
    'Sub'                               : _elemwise('sub'),
    'Mul'                               : _elemwise('mul'),
    'RealDiv'                           : _elemwise('div'),
    'Maximum'                           : _elemwise('max'),
    'Minimum'                           : _elemwise('min'),
    'Sum'                               : _sum(),
    'Square'                            : _square(),
    'Pack'                              : _pack(),
    'Slice'                             : _slice(),
    'LeakyRelu'                         : AttrCvt('leaky_relu'),
    'Relu'                              : AttrCvt('relu'),
    'Reshape'                           : _reshape(),
    'ResizeBilinear'                    : _resize_bilinear(),
    'Selu'                              : _selu(),
    'Softmax'                           : AttrCvt('softmax', {'axis': ('axis', 1)}),
    'Rsqrt'                             : _rsqrt(),
    'Squeeze'                           : _squeeze(),
    'FusedBatchNorm'                    : _fused_batch_norm(),
    'FusedBatchNormV2'                  : _fused_batch_norm(),
    'Relu6'                             : _relu6(),
    'DepthwiseConv2dNative'             : _conv('depthwise'),
    'Shape'                             : _shape(),
    'Sigmoid'                           : AttrCvt('sigmoid'),
    'Fill'                              : _fill(),
    'GatherV2'                          : _gather_v2(),
    'StridedSlice'                      : _stridedSlice(),
    'LRN'                               : _lrn(),
    'Pad'                               : _pad('Pad'),
    'PadV2'                             : _pad('PadV2'),
    'Range'                             : _range(),
    'Rank'                              : _rank(),
    'Transpose'                         : _transpose(),
    'Tanh'                              : AttrCvt('tanh'),
    'Mean'                              : _mean(),
    'LogicalAnd'                        : _logical('logical_and'),
    'LogicalOr'                         : _logical('logical_or'),
    'LogicalNot'                        : _logical('logical_not'),
    'Less'                              : _broadcast('less'),
    'Greater'                           : _broadcast('greater'),
    'LessEqual'                         : _broadcast('less_equal'),
    'GreaterEqual'                      : _broadcast('greater_equal'),
    'Equal'                             : _broadcast('equal'),
    'NotEqual'                          : _broadcast('not_equal'),
    'Split'                             : _split(False),
    'SplitV'                            : _split(True),
    'Unpack'                            : _unpack(),
}

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
        require conversion to nnvm, callable are functions which
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

        inputs : nnvm.Symbol
            Input data

        attrs : dict
            Dict of operator attributes

        params : dict
            List of pretrained weights and bias

        num_layers : int
            Total number of LSTM layer presented in the graph

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
        """
        def _impl(op_name, layer_name, inputs, attrs, params, num_layers):
            in_state_c_name = layer_name+'_c'
            in_state_h_name = layer_name+'_h'

            def _init_state(num_layers, batch_size, num_hidden):
                """Create the initial states for the first layer in the graph."""
                in_state_c = _sym.Variable(in_state_c_name,
                                           shape=(num_layers, batch_size, num_hidden))
                in_state_h = _sym.Variable(in_state_h_name,
                                           shape=(num_layers, batch_size, num_hidden))
                return in_state_c, in_state_h

            def _get_cur_input_state(in_state_c, in_state_h, num_layers,
                                     layer, batch_size, num_hidden):
                """Select the appropriate states for the current layer"""
                in_state_c_tup = _sym.split(in_state_c,
                                            indices_or_sections=num_layers, axis=0)
                in_state_h_tup = _sym.split(in_state_h,
                                            indices_or_sections=num_layers, axis=0)
                cur_in_state_c = _sym.reshape(in_state_c_tup[layer],
                                              shape=(batch_size, num_hidden))
                cur_in_state_h = _sym.reshape(in_state_h_tup[layer],
                                              shape=(batch_size, num_hidden))
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
            cur_out_state = _sym.expand_dims(cur_out_state, axis=0, num_newaxis=1)
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

        inputs : nnvm.Symbol
            Input data

        attrs : dict
            Dict of operator attributes

        params : dict
            List of pretrained weights and bias

        Returns
        -------
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
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

class GraphProto(object):
    """ A helper class for handling nnvm graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._output_shapes = {}
        self._num_param = 0
        self._num_rnn_layer = False
        self._outputs_are_0d = {}
        self._input_shapes = {}

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Construct nnvm nodes from tensorflow graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to NNVM.
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
        sym : nnvm.sym.Symbol
            The returned nnvm symbol
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
            msg = 'The following operators are not supported in frontend TensorFlow: {}'
            ops = str(list(missing_operators)).strip('[,]')
            raise tvm.error.OpNotImplemented(msg.format(ops))

        for node in graph.node:
            if node.op == 'Placeholder':
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

                self._nodes[node.name] = _sym.Variable(name=node.name,
                                                       shape=self._input_shapes[node.name])
                self._output_shapes[node.name] = [self._input_shapes[node.name]]
                self._outputs_are_0d[node.name] = [ \
                    not tshape if isinstance(tshape, list) else False \
                    for tshape in self._output_shapes[node.name]]

            # Ignore user's input shape for Non placeholder
            elif node.op == 'Const':
                tensor_value = node.attr['value'].tensor
                self._input_shapes[node.name] = \
                    tensor_util.TensorShapeProtoToList(tensor_value.tensor_shape)
                if shape and node.name in shape:
                    warnings.warn("Ignore the passed shape. "
                                  "Shape in graphdef will be used for operator %s." % node.name)

        final_op = None
        # Parse the nodes to re-create TF graph using Symbol API of NNVM
        for node in graph.node:
            # Tensorflow doesn't have separate list for params extraction.
            # Operator name 'Const' is treated as a parameter to build NNVM params dict.

            input_shapes = {}
            input_0d_mismatch = set()
            attr = self._parse_attr(node.attr)

            #  Variable converted to Const will not have only value attr
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

            self._outputs_are_0d[node.name] = [ \
                not tshape if isinstance(tshape, list) else False \
                for tshape in self._output_shapes[node.name]]

            if node.op == "Const":
                # All Const nodes are Param nodes, lets parse
                self._num_param += 1
                for key, value in node.attr.items():
                    self._parse_param(key, value, node.name)
                if node.name not in self._nodes:
                    raise NotImplementedError( \
                        "Const {} couldn't be converted to Param.".format(node.name))

                attr = self._parse_attr(node.attr)

            elif node.op != "Placeholder":
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
                        if len(in_sym.list_output_names()) > 1:
                            tensor_slot = int(tensor_name[1]) if len(tensor_name) > 1 else 0
                            in_sym = in_sym[tensor_slot]
                            input_shape = self._output_shapes[node_name][tensor_slot]
                        else:
                            tensor_slot = 0
                            input_shape = self._output_shapes[node_name][0]
                        inputs.append(in_sym)
                        input_shapes[in_sym] = input_shape
                        # This means the node is 1d in NNVM and 0d in TF.
                        # See `_expand_dims_0d_aware`.
                        if self._outputs_are_0d[node_name][tensor_slot] and input_shape:
                            input_0d_mismatch.add(in_sym)
                attr['_input_shapes'] = input_shapes
                attr['_input_0d_mismatch'] = input_0d_mismatch

                inputs = self._fix_extranodes(node.op, attr, inputs)
                op = self._convert_operator(node.op, inputs, attr, graph)

                # Check if op is converted to param
                if isinstance(op, np.ndarray):
                    self._params[node.name] = tvm.nd.array(op)
                    op = _sym.Variable(name=node.name,
                                       shape=self._params[node.name].shape)

                # Assuming only one output.
                self._nodes[node.name] = op
                final_op = op

                # Infer shapes even without specifying "add_shapes=True"
                if output_shapes == [None]:
                    g = _graph.create(final_op)
                    self._output_shapes[node.name] = \
                        list(graph_util.infer_shape(g, **self._input_shapes))[-1]

                if self._output_shapes[node.name] and shape and node.name in shape:
                    assert self._output_shapes[node.name] == list(shape[node.name])

            # Infer shapes if passed explicitely
            node_output = self._nodes[node.name]
            if shape and (not self._output_shapes[node.name][0]
                          or -1 in self._output_shapes[node.name][0]):
                g = _graph.create(node_output)
                shape_dict = {k: v.shape for k, v in self._params.items()}
                shape_dict.update(shape)
                _, out_shapes = graph_util.infer_shape(g, **shape_dict)
                self._output_shapes[node.name] = out_shapes

        out = []
        if outputs is None:
            out.append(final_op)
        else:
            for out_name in outputs:
                if ":" in out_name:
                    out_name, out_num = out_name.split(":")
                    out_num = int(out_num)
                    out.append(self._nodes[out_name][out_num])
                else:
                    out.append(self._nodes[out_name])

        #Add the RNN outputs also with 'head' nodes of the nnvm graph
        if self._num_rnn_layer:
            out_rnn = _sym.concatenate(*self._out_rnn, axis=0)
            out.append(out_rnn)

        if isinstance(out, list):
            out = _sym.Group(out) if len(out) > 1 else out[0]

        return out, self._params

    def _parse_import_prerequisites(self, graph):
        """ Calculate the named preconditions from TensorFlow `graph`.
            Return prerequisites for parsing:
            a. Set of operator names which don't have their mapping in TVM, i.e.
                which are not supported
        """
        missing_operators = set()
        for node in graph.node:
            if node.op == "Placeholder":
                pass
            elif node.op == "Const":
                pass
            else:
                if any([node.op in t for t in [_identity_list, _convert_map, _convert_map_rnn]]):
                    pass
                else:
                    missing_operators.add(node.op)

        return missing_operators

    def _parse_param(self, key, value, name):
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
                self._nodes[name] = _sym.Variable(name=name)
                return

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
        """Convert RNN and its variant operators to NNVM operators.
        This converter read the input states of each layers and
        also maintain the output states of each layer in a list.

        Parameters
        ----------
        op_name : str
            Operator name, such as LSTMBlockCell
        inputs : list of nnvm.Symbol
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
            require conversion to nnvm, callable are functions which
            take attrs and return (new_op_name, new_attrs)

        Returns
        -------
        sym : nnvm.Symbol
            Converted nnvm Symbol
        """
        if not self._num_rnn_layer:
            self._out_rnn = []
            self.rnn = RecurrentNetworks(self._nodes, self._out_rnn, graph, convert_map)
            self._num_rnn_layer = True
        sym = self.rnn.process_op(op_name, inputs, attrs, params)
        return sym

    def _convert_operator(self, op_name, inputs, attrs,
                          graph, identity_list=None, convert_map=None):
        """Convert from Tensorflow operator to nnvm operator.
        The converter must specify conversions explicitly for incompatible name, and
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
        convert_map_rnn = _convert_map_rnn
        if op_name in identity_list:
            sym = get_nnvm_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        elif op_name in convert_map_rnn:
            sym = self._convert_rnn_operator(op_name, inputs, attrs,
                                             self._params, graph,
                                             convert_map_rnn)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported in frontend TensorFlow.'.format(op_name))
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

def from_tensorflow(graph, layout="NHWC", shape=None, outputs=None):
    """Load tensorflow graph which is a python tensorflow graph object into nnvm graph.
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
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    sym, params = g.from_tensorflow(graph, layout, shape, outputs)
    return sym, params
