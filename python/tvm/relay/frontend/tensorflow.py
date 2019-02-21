# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition
"""TF: Tensorflow frontend."""
from __future__ import absolute_import as _abs
from __future__ import print_function

import logging
# Numpy support
import numpy as np

import tvm
from topi.util import get_const_tuple
from .. import ir_pass
from .. import expr as _expr
from .. import op as _op

__all__ = ['from_tensorflow']

def _get_relay_op(op_name):
    try:
        op = getattr(_op, op_name)
    except AttributeError:
        try:
            op = getattr(_op.nn, op_name)
        except AttributeError:
            op = getattr(_op.image, op_name)

    if not op:
        raise RuntimeError("Unable to map op_name {} to relay".format(op_name))
    return op

class AttrCvt(object):
    """Common attribute conveter. An AttrConverter instance is a callable:
    ```
    attr_converter = AttrConverter(op_name, transforms={'a':'b', 'c':('d', 1)})
    new_op_name, new_attr = attr_converter(attrs)
    ```

    Parameters
    ----------
    op_name : str or callable
        If set as str, returned operator name is the str.
        If set as callable, returned operator is the str returned by calling:
        `op_name = func(attr)`
    transforms : dict of `new_name, or (new_name, default_value, transform function)`
        If only a new_name is provided, it's like renaming the attribute name.
        If default_value if provded, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.
    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occured.
    disables : list
        A list of attributes that is disabled in relay. Log warnings.
    ignores : list
        A list of attributes that is ignored in relay. Debug level logging.
    extras : dict
        A series of additional attributes should be added anyway to the returned
        attribute dict.
    custom_check : callable
        A custom function takes attribute, and return True/False.
        Raise RuntimeError if not bool(True) returned.
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

        # apply custom check
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError("Check failed: {}".format(msg))
        # get new op_name
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)
        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise NotImplementedError("Attribute {} not supported yet.".format(k))
            elif k in self._disables:
                logging.warning("Attribute %s is disabled in relay.%s", k, op_name)
            elif k in self._ignores:
                logging.debug("Attribute %s is ignored in relay.%s", k, op_name)
            elif k in self._transforms:
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                if defaults is None:
                    new_attr = self._required_attr(attrs, k)
                else:
                    new_attr = attrs.get(k, None)
                if new_attr is None:
                    new_attrs[new_name] = defaults
                else:
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                new_attrs[k] = attrs[k]
        # add extras
        new_attrs.update(self._extras)
        return _get_relay_op(op_name)(*inputs, **new_attrs)

    def _parse_default(self, target):
        """Helper function to parse default values."""
        if not isinstance(target, (list, tuple)):
            k, v, t = target, None, lambda x: x
        elif len(target) == 1:
            k, v, t = target[0], None, lambda x: x
        elif len(target) == 2:
            k, v, t = target[0], target[1], lambda x: x
        elif len(target) > 2:
            k, v, t = target[0], target[1], target[2]
        else:
            k = None  # should raise
        if not isinstance(k, str):
            msg = "{} is not a valid target, (name, default) expected.".format(target)
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, str):
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise AttributeError("Required attribute {} not found.".format(key))
        return attr[key]

def _get_pad_pair(input1d, kernel1d, stride1d):
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]

def _get_name_hint(node):
    name = ''
    if hasattr(node, "name_hint"):
        name = node.name_hint
    return name

def _math_name_picker(surfix):
    def _impl(attr):
        return 'broadcast_' + surfix
    return _impl

def _dimension_picker(prefix, surfix=''):
    def _impl(attr):
        kernel = attr['kernel_shape']
        if len(kernel) == 2:
            return prefix + '2d' + surfix
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
    out_type = ir_pass.infer_type(inputs)
    out_shapes = [get_const_tuple(out_type.checked_type.shape)]
    channels = out_shapes[0][0] if not transpose else out_shapes[0][1]
    return channels

def _rsqrt():
    def _impl(inputs, attr, *args):
        inputs.append(tvm.relay.const(-0.5, attr['T'].name))
        return AttrCvt(op_name="power")(inputs, attr)
    return _impl

def _argx(func, func_name):
    """ A common wrapper for argmin and argmax operations """
    def _impl(inputs, attr, params):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_name = inputs[1].name_hint
            axis_input_vlaue = [params[axis_input_name].asnumpy()[0]]
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for `{}` : `axis` should be a constant".format(func_name))
        return func(inputs[0], axis=axis_input_vlaue, keepdims=False)
    return _impl

def _elemwise(name):
    def _impl(inputs, attr, *args):
        assert len(inputs) == 2, "Math op take 2 inputs, {} given".format(len(inputs))
        return _get_relay_op(name)(*inputs)
    return _impl

def _pooling(name):
    def _impl(inputs, attr, params):

        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        input_shape = attr['_input_shapes'][inputs[0]][0]

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (attr['ksize'][1], attr['ksize'][2])
            attr['strides'] = (attr['strides'][1], attr['strides'][2])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (attr['ksize'][2], attr['ksize'][3])
            attr['strides'] = (attr['strides'][2], attr['strides'][3])
        else:
            raise TypeError("Unsupported data_format type : {}".format(attr['data_format']))

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            tmp_shape = attr['_input_shapes'][inputs[0]][0]
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
            raise TypeError("Unsupported padding type : {}".format(attr['padding']))

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

        # NCHW Layout require weights transpose
        if attr['data_format'] == 'NCHW':
            tmp_shape = attr['_input_shapes'][inputs[1]][0]
            tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
            inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            attr['_input_shapes'][inputs[1]] = [tmp_shape]

        input_shape = attr['_input_shapes'][inputs[0]][0]
        weights_shape = attr['_input_shapes'][inputs[1]][0]

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            if opname == 'conv':
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
            else:
                attr['channels'] = input_shape[3] * depth_mult

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][0], attr['dilations'][1])
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
            raise TypeError("Unsupported data format type : {}".format(attr['data_format']))


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

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            if attr['data_format'] == 'NHWC':
                inputs[0] = _op.nn.pad(data=inputs[0],
                                       pad_width=((0, 0),
                                                  (pad_v[0], pad_v[1]),
                                                  (pad_h[0], pad_h[1]),
                                                  (0, 0)))
            else:
                inputs[0] = _op.nn.pad(data=inputs[0],
                                       pad_width=((0, 0),
                                                  (0, 0),
                                                  (pad_v[0], pad_v[1]),
                                                  (pad_h[0], pad_h[1])))

            attr['padding'] = [0, 0]

        else:
            raise TypeError("Unsupported padding type : {}".format(attr['padding']))

        if 'kernel_layout' not in attr:
            if opname == 'conv':
                attr['kernel_layout'] = 'HWIO' if attr['data_format'] == 'NHWC' else 'OIHW'
            else:
                attr['kernel_layout'] = 'HWOI' if attr['data_format'] == 'NHWC' else 'OIHW'

        use_bias = len(inputs) == 3
        channel_axis = 1 if attr['data_format'] == "NCHW" else 3

        out = AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'data_layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            custom_check=_dimension_constraint())([inputs[0], inputs[1]], attr)

        if use_bias:
            out = _op.nn.bias_add(out, inputs[2], axis=channel_axis)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _decode_image():
    def _impl(inputs, attr, params):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        print("DecodeJpeg: It's a pass through, please handle preprocessing before input")
        return inputs[0]
    return _impl

def _cast():
    def _impl(inputs, attr, params):
        return inputs[0].astype(attr['DstT'].name)
    return _impl

def _expand_dims():
    def _impl(inputs, attr, params):
        dim_input = inputs.pop(1)
        axis = params[dim_input.name_hint]
        params.pop(dim_input.name_hint)
        return AttrCvt(op_name="expand_dims", ignores=['Tdim'],
                       extras={'axis': int(axis.asnumpy()[0])})(inputs, attr)
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
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not attr['transpose_b']:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(op_name="dense",
                       extras={'units': channels},
                       ignores=['transpose_a', 'transpose_b', 'T'])(inputs, attr)

    return _impl

def _identity():
    def _impl(inputs, attr, params):
        return inputs[0]
    return _impl

def _concatV2():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(len(inputs)-1)
        axis = params[pop_node.name_hint]
        params.pop(pop_node.name_hint)
        return AttrCvt(
            op_name="concatenate", ignores=['T', 'N', 'Tidx'],
            extras={'axis': int(axis.asnumpy()[0])})([inputs], attr)
    return _impl

def _concat():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(0)
        axis = params[pop_node.name_hint]
        params.pop(pop_node.name_hint)
        return AttrCvt(
            op_name="concatenate", ignores=['N'],
            extras={'axis': int(axis.asnumpy()[0])})([inputs], attr)
    return _impl

def _pack():
    def _impl(inputs, attr, params):
        axis = int(attr["axis"])
        inputs_reshaped = [_op.expand_dims(i, axis=axis, num_newaxis=1) for i in inputs]
        return _op.concatenate(inputs_reshaped, axis)
    return _impl

def _reshape():
    def _impl(inputs, attr, params):
        try:
            pop_node = inputs[1]
            shape_arg = params.pop(pop_node.name_hint)
            inputs.pop(1)

            return AttrCvt(
                op_name="reshape",
                extras={'newshape':tuple(shape_arg.asnumpy())},
                ignores=['Tshape'])(inputs, attr)
        except KeyError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            if all(in_node in params for in_node in inputs[1].list_input_names()):
                func = _expr.Function(ir_pass.free_vars(inputs[1]), inputs[1])
                with tvm.relay.build_config(opt_level=0):
                    graph, lib, params = tvm.relay.build(func, target="llvm", params=params)
                ctx = tvm.context("llvm", 0)
                from tvm.contrib import graph_runtime
                m = graph_runtime.create(graph, lib, ctx)
                m.set_input(**params)
                m.run()
                params_new = m.get_output(0)
                inputs.pop(1)
                return AttrCvt(
                    op_name="reshape",
                    extras={'newshape':tuple(params_new.asnumpy().flatten())},
                    ignores=['Tshape'])(inputs, attr)
            raise RuntimeError("Reshape with dynamic shape input not supported yet.")
    return _impl

def _bias_add():
    def _impl(inputs, attr, params):
        return _op.add(inputs[0], inputs[1])
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
        return np.array(attr['_input_shapes'][inputs[0]][0], dtype='int32')
    return _impl

def _fill():
    def _impl(inputs, attr, params):
        fill_arg = params.pop(inputs.pop(1).name_hint)
        return _op.full(tvm.relay.const(fill_arg.asnumpy()[0], attr['T'].name),
                        attr['_output_shapes'][0], attr['T'].name)
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
        axis = params.pop(inputs[1].name_hint).asnumpy()
        # convert to tuple for preventing invalid parameter format error
        axis = tuple(axis)
        return AttrCvt(
            op_name='sum',
            extras={'axis': axis},
            transforms={'keep_dims':'keepdims'},
            ignores=['name', 'Tidx'])([inputs[0]], attr)
    return _impl

def _square():
    def _impl(inputs, attr, params):
        return _op.multiply(inputs[0], inputs[0])
    return _impl

def _gather_v2():
    "Tensorflow now support only gatherv2"
    def _impl(inputs, attr, params):
        axis = params[inputs.pop(2).name_hint].asnumpy()[0]
        new_input = []
        new_input.append(inputs.pop(0))
        new_input.append(inputs.pop(0))
        return  AttrCvt(op_name="take",
                        extras={'axis': tvm.const(axis, 'int32')},
                        ignores=['Tindices', 'Tparams', 'validate_indices', \
                                 'Taxis', '_class'])(new_input, attr)
    return _impl

def _infer_out_shapes(inputs, params):
    """A method to get the output shape of an intermediate node in the relay graph."""
    out_type = ir_pass.infer_type(inputs)
    out_shapes = [get_const_tuple(out_type.checked_type.shape)]
    return out_shapes

def _stridedSlice():
    def _impl(inputs, attr, params):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = params.pop(inputs[1].name_hint).asnumpy().tolist()
        end = params.pop(inputs[2].name_hint).asnumpy().tolist()
        stride = params.pop(inputs[3].name_hint).asnumpy().tolist()
        begin_mask = int(attr.get('begin_mask', 0))
        end_mask = int(attr.get('end_mask', 0))
        ellipsis_mask = int(attr.get('ellipsis_mask', 0))
        new_axis_mask = int(attr.get('new_axis_mask', 0))
        shrink_axis_mask = int(attr.get('shrink_axis_mask', 0))
        data_shape = attr['_input_shapes'][inputs[0]]
        data_dim = len(data_shape[0])
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
                        m_end[final_index] = data_shape[0][final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask &new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[0][final_index] \
                                                     if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = 0 if stride[index] < 0 \
                                                 else data_shape[0][final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        #Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = data_shape[0][final_index] + begin[index] \
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
        return _op.reshape(out, newshape=tuple(final_output))
    return _impl

def _pad(name):
    def _impl(inputs, attr, params):
        padlist_key = inputs[1].name_hint
        if padlist_key in params:
            padlist = params.pop(padlist_key).asnumpy()
        else:
            raise RuntimeError("Required parameter {} not fount.".format(padlist_key))
        paddings = tuple([tuple(l) for l in padlist])
        attr['pad_width'] = paddings
        attr['pad_value'] = 0
        new_inputs = [inputs[0]]
        if name == 'PadV2':
            constant_values = params.pop(inputs[2].name_hint).asnumpy()
            attr['pad_value'] = constant_values[0]
        return AttrCvt(
            op_name='pad',
            ignores=['Tpaddings'],)(new_inputs, attr)
    return _impl


def _transpose():
    def _impl(inputs, attr, params):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        param_name = _get_name_hint(inputs[1])
        if param_name in params:
            axes = tuple(params.get(param_name).asnumpy())
        else:
            axes = None
        return _op.transpose(inputs[0], axes=axes)
    return _impl

def _rank():
    def _impl(inputs, attr, params):
        input_shapes = attr['_input_shapes'][inputs[0]]
        assert len(inputs) == 1

        name = attr["_node_name"]
        params[name] = tvm.nd.array([len(input_shapes[0])])
        return [_expr.var(name,
                          shape=params[name].shape,
                          dtype='int32')]

    return _impl

def _range():
    def _impl(inputs, attr, params):
        start = params.pop(inputs[0].name_hint).asnumpy()[0]
        limit = params.pop(inputs[1].name_hint).asnumpy()[0]
        delta = params.pop(inputs[2].name_hint).asnumpy()[0]

        name = attr["_node_name"]
        params[name] = tvm.nd.array([start, limit, delta])
        return [_expr.var(name,
                          shape=params[name].shape,
                          dtype='int32')]
    return _impl

def _elu():
    def _impl(inputs, attr, params):
        alpha = tvm.relay.const(-1.0, attr['T'].name)
        return alpha * _op.nn.relu(tvm.relay.const(1, attr['T'].name) \
                                   - _op.exp(inputs[0])) + _op.nn.relu(inputs[0])
    return _impl

def _selu():
    def _impl(inputs, attr, params):
        alpha = tvm.relay.const(-1.6732632423543772848170429916717, attr['T'].name)
        gamma = tvm.relay.const(1.0507009873554804934193349852946, attr['T'].name)
        return gamma * (alpha * _op.nn.relu(tvm.relay.const(1, attr['T'].name) \
                                            - _op.exp(inputs[0])) + _op.nn.relu(inputs[0]))
    return _impl

def _mean():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].name_hint)
        return AttrCvt(op_name="mean", ignores=['Tdim', 'Tidx'],
                       transforms={'keep_dims': 'keepdims'},
                       extras={'axis': tuple(axis.asnumpy())})([inputs[0]], attr)
    return _impl

def _broadcast(name):
    def _impl(inputs, attr, params):
        return AttrCvt(
            op_name=name,
            ignores=['name', 'Tidx']
        )(inputs, attr)
    return _impl

def _softmax():
    def _impl(inputs, attr, params):
        return AttrCvt(op_name='softmax',
                       transforms={'axis': ('axis', 1)})([inputs[0]], attr)
    return _impl

# compatible operators that do NOT require any conversion.
_identity_list = []

# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
_convert_map = {
    'ArgMax'                            : _argx(_op.argmax, 'argmax'),
    'ArgMin'                            : _argx(_op.argmin, 'argmin'),
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
    'Sub'                               : _elemwise('subtract'),
    'Mul'                               : _elemwise('multiply'),
    'Maximum'                           : _elemwise('maximum'),
    'Minimum'                           : _elemwise('minimum'),
    'Sum'                               : _sum(),
    'Square'                            : _square(),
    'Pack'                              : _pack(),
    'LeakyRelu'                         : AttrCvt('leaky_relu'),
    'Relu'                              : AttrCvt('relu'),
    'Reshape'                           : _reshape(),
    'ResizeBilinear'                    : _resize_bilinear(),
    'Selu'                              : _selu(),
    'Softmax'                           : _softmax(),
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
    'Less'                              : _broadcast('less'),
    'Greater'                           : _broadcast('greater'),
    'LessEqual'                         : _broadcast('less_equal'),
    'GreaterEqual'                      : _broadcast('greater_equal'),
    'Equal'                             : _broadcast('equal'),
    'NotEqual'                          : _broadcast('not_equal'),
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
        batch_size, input_size = input_shape[0][0], input_shape[0][1]
        num_hidden_layers = weight_shape[0][1]
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
        forget_gate = _op.sigmoid(gate_list[2])
        forget_gate = _op.add(forget_gate,
                              tvm.relay.const(forget_bias, attr['T'].name))
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

                batch_size = input_shape[0][0]
                num_hidden = weight_shape[0][1] // 4

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

class GraphProto(object):
    """ A helper class for handling relay graph copying from Tensorflow GraphDef.
    Definition:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
    """
    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._output_shapes = {}
        self._num_param = 0
        self._num_rnn_layer = False

    def from_tensorflow(self, graph, layout="NHWC", shape=None, outputs=None):
        """Construct relay nodes from tensorflow  graph definition - GraphDef.

        Follow the tensorflow graph definition to parse and convert it to Relay.
        Some of the assumptions listed below.

            -> All Placeholders are considered as graph input.
            -> All Const nodes are params.
            -> Last node is assumed as graph output.
            -> _output_shapes : Graph should be frozen with add_shapes=True.
                                Or user can pass input shape dictionaly optionally.
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

        Returns
        -------
        sym : relay.op
            The returned relay operator
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

        # Parse the nodes to re-create TF graph using Relay operators.
        for node in graph.node:
            # Tensorflow doesn't have seperate list for params extraction.
            # Operator name 'Const' is treated as a parameter to build params dict.

            input_shapes = {}
            attr = self._parse_attr(node.attr)

            #Variable converted to Const will not have only value attr
            if 'value' in attr and node.op == 'Const':
                tensor_value = attr['value']
                self._output_shapes[node.name] = \
                    [tensor_util.TensorShapeProtoToList( \
                        tensor_value.tensor_shape)]
            elif '_output_shapes' in attr:
                self._output_shapes[node.name] = \
                    [tensor_util.TensorShapeProtoToList(tshape) \
                    for tshape in attr['_output_shapes']]
            elif shape:
                # Keep the list indexable to avoid key error.
                # Actual value will be filled after node creation.
                self._output_shapes[node.name] = [None]
            else:
                raise NotImplementedError( \
                    "Please freeze the graph with add_shapes=True")

            if node.op == "Placeholder":
                self._output_shapes[node.name] = [shape[node.name]]
                self._nodes[node.name] = [_expr.var(node.name,
                                                    shape=self._output_shapes[node.name][0],
                                                    dtype=attr['dtype'].name)]

            elif node.op == "Const":
                # All Const nodes are Param nodes, lets parse
                self._num_param += 1
                for key, value in node.attr.items():
                    self._parse_param(key, value, node.name, shape)
                if node.name not in self._nodes:
                    raise NotImplementedError( \
                        "Const {} couldn't be converted to Param.".format(node.name))

                attr = self._parse_attr(node.attr)

            else:
                # Pass the parsed shapes instead
                attr["_output_shapes"] = self._output_shapes[node.name]

                # Pass the node name too in attr
                attr["_node_name"] = node.name

                # Pass the target layout
                attr["_target_layout"] = layout

                #ToDo: Some of the tensorflow operators internaly maintain
                #execution layers and its output name will the layer number along with
                #graph node name.eg: Node name:- 'Model/RNN/cell_0/RnnCell', but the
                #output name will be 'Model/RNN/cell_0/RnnCell:0'. In this case,
                #the digit has to be ignored.
                if ":" in node.input[0]:
                    in_name, _ = node.input[0].split(':')
                    node.input[0] = in_name

                # Fill shapes for all inputs in a list
                inputs = []
                for i in node.input:
                    if i in self._nodes:
                        inputs.append(self._nodes[i][0])
                        input_shapes[self._nodes[i][0]] = self._output_shapes[i]
                attr['_input_shapes'] = input_shapes

                op = self._convert_operator(node.op, inputs, attr, graph)

                # Check is op is converted to param
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

            # Infer shapes if passed explicitely
            node_output = self._nodes[node.name]
            out_type = ir_pass.infer_type(node_output[0])
            self._output_shapes[node.name] = [get_const_tuple(out_type.checked_type.shape)]


        out = []
        if outputs is None:
            out = op
        else:
            out = [self._nodes[out_name][0] for out_name in outputs]

        #Add the RNN outputs also with 'head' nodes of the relay graph
        if self._num_rnn_layer:
            if len(self._out_rnn) == 1:
                out.append(self._out_rnn[0])
            else:
                out_rnn = _op.concatenate(self._out_rnn, axis=0)
                out.append(out_rnn)

        out = out[0] if len(out) == 1 else _expr.Tuple(out)
        func = _expr.Function(ir_pass.free_vars(out), out)

        return func, self._params

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
                self._nodes[name] = [_expr.var(name, shape=shape[name], dtype='uint8')]

                return

            array_ndim = len(np_array.shape)
            if array_ndim == 0:
                new_array = np.empty([1], dtype=np_array.dtype)
                new_array[0] = np_array
                self._params[name] = tvm.nd.array(new_array)
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

    def _convert_operator(self, op_name, inputs, attrs,
                          graph, identity_list=None, convert_map=None):
        """Convert from Tensorflow operator to relay operator.
        The converter must specify conversions explicity for incompatible name, and
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
            sym = _get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        elif op_name in convert_map_rnn:
            sym = self._convert_rnn_operator(op_name, inputs, attrs,
                                             self._params, graph,
                                             convert_map_rnn)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_tensorflow(graph, layout="NHWC", shape=None, outputs=None):
    """  Load tensorflow graph which is a python tensorflow graph object into relay.
    The companion parameters will be handled automatically.

    Parameters
    ----------
    graph : GraphDef object
        Tensorflow GraphDef

    Returns
    -------
    sym : relay.op
        Compatible relay operator

    params : dict of str to tvm.ndarray
        Dict of converted parameters stored in tvm.ndarray format
    """
    g = GraphProto()
    sym, params = g.from_tensorflow(graph, layout, shape, outputs)
    return sym, params
