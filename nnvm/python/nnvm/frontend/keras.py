# pylint: disable=invalid-name, import-self
"""Keras frontend."""
from __future__ import absolute_import as _abs
import sys
import numpy as np
import tvm
from .. import symbol as _sym
from .common import SymbolTable

__all__ = ['from_keras']


def _check_data_format(keras_layer):
    if hasattr(keras_layer, ('data_format')):
        if keras_layer.data_format != 'channels_last':
            raise ValueError("Keras frontend currently supports data_format = channels_last only.")


def _get_pad_pair(input1d, kernel1d, stride1d):
    out1d = (input1d + stride1d - 1) // stride1d
    pad = np.maximum((out1d - 1) * stride1d + kernel1d - input1d, 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return [pad_before, pad_after]

def _get_elu(insym, alpha):
    """ A helper method for elu.
    """
    return -alpha * _sym.relu(1 - _sym.exp(insym)) + _sym.relu(insym)

def _convert_recurrent_activation(insym, keras_layer):
    act_type = keras_layer.recurrent_activation.__name__
    return _convert_activation(insym, act_type, None)

def _convert_activation(insym, keras_layer, _):
    if isinstance(keras_layer, str):
        act_type = keras_layer
    else:
        if sys.version_info.major < 3:
            act_type = keras_layer.activation.func_name
        else:
            act_type = keras_layer.activation.__name__
    if act_type == 'linear':
        if isinstance(keras_layer, str):
            return insym
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") else 1
        beta = keras_layer.beta if hasattr(keras_layer, "beta") else 0
        return _sym.__add_scalar__(_sym.__mul_scalar__(insym, \
            scalar=alpha), scalar=beta)
    elif act_type == 'softmax':
        return _sym.softmax(insym, axis=1)
    elif act_type == 'sigmoid':
        return _sym.sigmoid(insym)
    elif act_type == 'tanh':
        return _sym.tanh(insym)
    elif act_type == 'relu':
        return _sym.relu(insym)
    elif act_type == 'softplus':
        return _sym.log(_sym.__add_scalar__(_sym.exp(insym), scalar=1))
    elif act_type == 'elu':
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") else 1
        return _get_elu(insym, alpha)
    elif act_type == 'selu':
        # Alpha, Gamma values, obtained from  https://arxiv.org/abs/1706.02515
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") \
            else 1.6732632423543772848170429916717
        gamma = keras_layer.gamma if hasattr(keras_layer, "gamma") \
            else 1.0507009873554804934193349852946
        return gamma * _get_elu(insym, alpha)
    elif act_type == 'relu6':
        return _sym.clip(insym, a_min=0, a_max=6)
    elif act_type == 'softsign':
        return insym / (1 + (_sym.relu(insym) + _sym.relu(_sym.negative(insym))))
    elif act_type == 'hard_sigmoid':
        transformX = (0.2 * insym) + 0.5
        return _sym.clip(transformX, a_min=0, a_max=1)
    else:
        raise TypeError("Unsupported activation type : {}".format(act_type))


def _convert_advanced_activation(insym, keras_layer, symtab):
    act_type = type(keras_layer).__name__
    if act_type == 'ReLU':
        if keras_layer.max_value:
            return _sym.clip(insym, a_min=0, a_max=keras_layer.max_value)
        return _sym.relu(insym)
    elif act_type == 'LeakyReLU':
        return _sym.leaky_relu(insym, alpha=keras_layer.alpha)
    elif act_type == 'ELU':
        alpha = keras_layer.alpha if hasattr(keras_layer, "alpha") else 1
        return _get_elu(insym, alpha)
    elif act_type == 'PReLU':
        assert hasattr(keras_layer, "alpha"), \
            "alpha required for PReLU."
        _check_data_format(keras_layer)
        size = len(keras_layer.alpha.shape)
        return -symtab.new_const(keras_layer.get_weights()[0] \
                                 .transpose(np.roll(range(size), 1))) \
                                 * _sym.relu(-insym) + _sym.relu(insym)
    elif act_type == 'ThresholdedReLU':
        theta = keras_layer.theta if hasattr(keras_layer, "theta") else 1.0
        theta_tensor = _sym.full_like(insym[0], fill_value=float(theta))
        return _sym.elemwise_mul(insym[0], _sym.greater(insym[0], theta_tensor, out_type="float32"))
    else:
        raise TypeError("Unsupported advanced activation type : {}".format(act_type))


def _convert_merge(insym, keras_layer, _):
    merge_type = type(keras_layer).__name__
    ret = insym[0]
    for i in range(1, len(insym)):
        if merge_type == 'Add':
            ret = _sym.elemwise_add(ret, insym[i])
        elif merge_type == 'Subtract':
            ret = _sym.elemwise_sub(ret, insym[i])
        elif merge_type == 'Multiply':
            ret = _sym.elemwise_mul(ret, insym[i])
        elif merge_type == 'Average':
            raise NotImplementedError('Average merge not implemented')
        elif merge_type == 'Maximum':
            raise NotImplementedError('Maximum merge not implemented')
        else:
            raise TypeError("Unsupported merge type : {}".format(merge_type))
    return ret


def _convert_dense(insym, keras_layer, symtab):
    weightList = keras_layer.get_weights()
    weight = symtab.new_const(weightList[0].transpose([1, 0]))
    params = {'weight':weight, 'use_bias':False, 'units':weightList[0].shape[1]}
    if keras_layer.use_bias:
        params['use_bias'] = True
        params['bias'] = symtab.new_const(weightList[1])
    out = _sym.dense(data=insym, **params)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, symtab)
    return out


def _convert_convolution(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    is_deconv = type(keras_layer).__name__ == 'Conv2DTranspose'
    is_depthconv = type(keras_layer).__name__ == 'DepthwiseConv2D'
    weightList = keras_layer.get_weights()
    if is_deconv:
        kernel_h, kernel_w, n_filters, in_channels = weightList[0].shape
        weight = weightList[0].transpose([3, 2, 0, 1])
    elif is_depthconv:
        kernel_h, kernel_w, in_channels, depth_mult = weightList[0].shape
        weight = weightList[0].transpose([2, 3, 0, 1])
    else:
        kernel_h, kernel_w, in_channels, n_filters = weightList[0].shape
        weight = weightList[0].transpose([3, 2, 0, 1])
    dilation = [1, 1]
    if isinstance(keras_layer.dilation_rate, (list, tuple)):
        dilation = [keras_layer.dilation_rate[0], keras_layer.dilation_rate[1]]
    else:
        dilation = [keras_layer.dilation_rate, keras_layer.dilation_rate]
    dilated_kernel_h = (kernel_h - 1) * dilation[0] + 1
    dilated_kernel_w = (kernel_w - 1) * dilation[1] + 1
    stride_h, stride_w = keras_layer.strides
    params = {'weight': symtab.new_const(weight),
              'kernel_size': [kernel_h, kernel_w],
              'strides': [stride_h, stride_w],
              'dilation': dilation,
              'padding': [0, 0],
              'use_bias': False}
    if is_depthconv:
        params['channels'] = in_channels * depth_mult
        params['groups'] = in_channels
    else:
        params['channels'] = n_filters
    if keras_layer.use_bias:
        params['use_bias'] = True
        params['bias'] = symtab.new_const(weightList[1])
    if keras_layer.padding == 'valid':
        pass
    # we insert a separate pad operator
    elif keras_layer.padding == 'same':
        in_h = keras_layer.input_shape[1]
        in_w = keras_layer.input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        if pad_t == pad_b and pad_l == pad_r:
            params['padding'] = (pad_t, pad_l)
        else:
            insym = _sym.pad(data=insym, pad_width=((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))
    else:
        raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
    if is_deconv:
        out = _sym.conv2d_transpose(data=insym, **params)
    else:
        out = _sym.conv2d(data=insym, **params)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, symtab)
    return out


def _convert_separable_convolution(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    weightList = keras_layer.get_weights()
    # depthwise conv
    kernel_h, kernel_w, in_channels, depth_mult = weightList[0].shape
    stride_h, stride_w = keras_layer.strides
    weight0 = weightList[0].transpose([2, 3, 0, 1])
    params0 = {'weight': symtab.new_const(weight0),
               'channels': in_channels * depth_mult,
               'groups': in_channels,
               'kernel_size': [kernel_h, kernel_w],
               'strides': [stride_h, stride_w],
               'dilation': [1, 1],
               'padding': [0, 0],
               'use_bias': False}
    if keras_layer.padding == 'valid':
        pass
    # we insert a separate pad operator
    elif keras_layer.padding == 'same':
        in_h = keras_layer.input_shape[1]
        in_w = keras_layer.input_shape[2]
        pad_t, pad_b = _get_pad_pair(in_h, kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, kernel_w, stride_w)
        insym = _sym.pad(data=insym, pad_width=(
            (0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))
    else:
        raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
    depthconv = _sym.conv2d(data=insym, **params0)
    # pointwise conv
    weight1 = weightList[1].transpose([3, 2, 0, 1])
    params1 = {'weight': symtab.new_const(weight1),
               'channels': weight1.shape[0],
               'groups': 1,
               'kernel_size': [1, 1],
               'strides': [1, 1],
               'dilation': [1, 1],
               'use_bias': False}
    if keras_layer.use_bias:
        params1['use_bias'] = True
        params1['bias'] = symtab.new_const(weightList[2])
    out = _sym.conv2d(data=depthconv, **params1)
    # defuse activation
    if sys.version_info.major < 3:
        act_type = keras_layer.activation.func_name
    else:
        act_type = keras_layer.activation.__name__
    if act_type != 'linear':
        out = _convert_activation(out, act_type, symtab)
    return out


def _convert_flatten(insym, keras_layer, _):
    _check_data_format(keras_layer)
    # NCHW -> NHWC so that dense can be correctly converted
    insym = _sym.transpose(insym, axes=[0, 2, 3, 1])
    return _sym.flatten(insym)


def _convert_pooling(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    pool_type = type(keras_layer).__name__
    # global pool in keras = global pool + flatten in nnvm
    if pool_type == 'GlobalMaxPooling2D':
        return _convert_flatten(_sym.global_max_pool2d(insym), keras_layer, symtab)
    elif pool_type == 'GlobalAveragePooling2D':
        return _convert_flatten(_sym.global_avg_pool2d(insym), keras_layer, symtab)
    else:
        pool_h, pool_w = keras_layer.pool_size
        stride_h, stride_w = keras_layer.strides
        params = {'pool_size': [pool_h, pool_w],
                  'strides': [stride_h, stride_w],
                  'padding': [0, 0]}
        if keras_layer.padding == 'valid':
            pass
        elif keras_layer.padding == 'same':
            in_h = keras_layer.input_shape[1]
            in_w = keras_layer.input_shape[2]
            pad_t, pad_b = _get_pad_pair(in_h, pool_h, stride_h)
            pad_l, pad_r = _get_pad_pair(in_w, pool_w, stride_w)
            params['padding'] = [pad_t, pad_l, pad_b, pad_r]
        else:
            raise TypeError("Unsupported padding type : {}".format(keras_layer.padding))
        if pool_type == 'MaxPooling2D':
            return _sym.max_pool2d(insym, **params)
        elif pool_type == 'AveragePooling2D':
            # TODO: in keras, padded zeros are not calculated
            return _sym.avg_pool2d(insym, **params)
        else:
            raise TypeError("Unsupported pooling type : {}".format(keras_layer))


def _convert_upsample(insym, keras_layer, _):
    _check_data_format(keras_layer)
    upsample_type = type(keras_layer).__name__
    if upsample_type == "UpSampling1D":
        h = keras_layer.size
        params = {'scale': h}
    elif upsample_type == "UpSampling2D":
        h, w = keras_layer.size
        if h != w:
            raise TypeError("Unsupported upsampling type with different axes size : {}"
                            .format(keras_layer.size))
        params = {'scale': h}
    elif upsample_type == "UpSampling3D":
        h, w, d = keras_layer.size
        if h != w or w != d:
            raise TypeError("Unsupported upsampling type with different axes size : {}"
                            .format(keras_layer.size))
        params = {'scale': h}
    else:
        raise TypeError("Unsupported upsampling type : {}".format(upsample_type))
    return _sym.upsampling(insym, **params)


def _convert_cropping(insym, keras_layer, _):
    _check_data_format(keras_layer)
    crop_type = type(keras_layer).__name__
    if crop_type == "Cropping1D":
        raise NotImplementedError("Cropping1D not implemented")
    elif crop_type == "Cropping2D":
        (_, in_h, in_w, _) = keras_layer.input_shape
        ((crop_t, crop_b), (crop_l, crop_r)) = keras_layer.cropping
    else:
        raise TypeError("Unrecognized cropping type : {}".format(crop_type))
    int32_max = np.iinfo(np.int32).max
    return _sym.strided_slice(insym, begin=[0, 0, crop_t, crop_l],
                              end=[int32_max, int32_max, in_h-crop_b, in_w-crop_r])


def _convert_batchnorm(insym, keras_layer, symtab):
    params = {'scale': False,
              'center': False,
              'epsilon': keras_layer.epsilon}
    idx = 0
    if keras_layer.scale:
        params['scale'] = True
        gamma = keras_layer.get_weights()[idx]
        params['gamma'] = symtab.new_const(gamma)
        idx += 1
    if keras_layer.center:
        params['center'] = True
        beta = keras_layer.get_weights()[idx]
        params['beta'] = symtab.new_const(beta)
        idx += 1
    moving_mean = keras_layer.get_weights()[idx]
    moving_var = keras_layer.get_weights()[idx + 1]
    params['moving_mean'] = symtab.new_const(moving_mean)
    params['moving_var'] = symtab.new_const(moving_var)
    return _sym.batch_norm(data=insym, **params)


def _convert_padding(insym, keras_layer, _):
    _check_data_format(keras_layer)
    padding_type = type(keras_layer).__name__
    padding = keras_layer.padding
    top = left = bottom = right = 0
    if padding_type == 'ZeroPadding2D':
        if isinstance(padding, int):
            top = left = bottom = right = padding
        elif isinstance(padding, tuple):
            if isinstance(padding[0], int):
                top, left = padding
                bottom, right = padding
            elif isinstance(padding[0], tuple):
                top, bottom = padding[0]
                left, right = padding[1]
            else:
                raise ValueError("Unrecognized padding option: {}".format(str(padding)))
        else:
            raise ValueError("Unrecognized padding option: {}".format(str(padding)))
    elif padding_type == 'ZeroPadding1D':
        raise NotImplementedError("ZeroPadding1D not implemented")
    else:
        raise ValueError("Unrecognized padding type: {}".format(padding_type))
    return _sym.pad(data=insym, pad_width=((0, 0), (0, 0), (top, bottom), (left, right)))


def _convert_concat(insym, keras_layer, _):
    _check_data_format(keras_layer)
    if not isinstance(insym, list):
        insym = [insym]
    return _sym.concatenate(*insym, axis=1)


def _convert_reshape(insym, keras_layer, _):
    _check_data_format(keras_layer)
    ch = keras_layer.input_shape[-1]
    assert ch == keras_layer.target_shape[-1], \
        "Only supports last dimension in target shape being equal to " \
        "the channel number of input tensor."
    shape = (-1, ch) + keras_layer.target_shape[:-1]
    return _sym.reshape(insym, shape=shape)

def _convert_lstm(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    if not isinstance(insym, list):
        buffer = np.zeros((1, keras_layer.units), 'float32')
        c_sym = symtab.new_const(buffer)
        h_sym = symtab.new_const(buffer)
        insym = [insym, h_sym, c_sym]

    in_data = insym[0]
    in_state_h = insym[1]
    in_state_c = insym[2]

    weightList = keras_layer.get_weights()

    kernel_wt = symtab.new_const(weightList[0].transpose([1, 0]))
    recurrent_wt = symtab.new_const(weightList[1].transpose([1, 0]))
    in_bias = symtab.new_const(weightList[2])

    units = list(weightList[0].shape)[1]

    in_data = _sym.flatten(in_data)
    ixh1 = _sym.dense(in_data, kernel_wt, use_bias=False, units=units)
    ixh2 = _sym.dense(in_state_h, recurrent_wt, in_bias, use_bias=True, units=units)
    gate = ixh1 + ixh2
    gates = _sym.split(gate, indices_or_sections=4, axis=1)
    in_gate = _convert_recurrent_activation(gates[0], keras_layer)
    in_transform = _convert_recurrent_activation(gates[1], keras_layer)
    next_c = in_transform * in_state_c + in_gate * _convert_activation(gates[2], keras_layer, None)
    out_gate = _convert_recurrent_activation(gates[3], keras_layer)
    next_h = out_gate * _convert_activation(next_c, keras_layer, None)

    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    out = _sym.reshape(next_h, shape=out_shape)
    return [out, next_h, next_c]

def _convert_simple_rnn(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    if not isinstance(insym, list):
        buffer = np.zeros((1, keras_layer.units), 'float32')
        prev_sym = symtab.new_const(buffer)
        insym = [insym, prev_sym]
    in_data = insym[0]
    prev_sym = insym[1]

    weightList = keras_layer.get_weights()
    kernel_wt = symtab.new_const(weightList[0].transpose([1, 0]))
    recurrent_wt = symtab.new_const(weightList[1].transpose([1, 0]))
    in_bias = symtab.new_const(weightList[2])
    units = list(weightList[0].shape)[1]

    in_data = _sym.flatten(in_data)
    ixh = _sym.dense(in_data, kernel_wt, in_bias, use_bias=True, units=units)
    prev_sym = _sym.flatten(prev_sym)
    ixh2 = _sym.dense(prev_sym, recurrent_wt, use_bias=False, units=units)
    output = ixh + ixh2
    output = _convert_activation(output, keras_layer, None)

    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    output = _sym.reshape(output, shape=out_shape)

    return [output, output]

def _convert_gru(insym, keras_layer, symtab):
    _check_data_format(keras_layer)
    if not isinstance(insym, list):
        buffer = np.zeros((1, keras_layer.units), 'float32')
        h_tm1 = symtab.new_const(buffer)
        insym = [insym, h_tm1]
    in_data = insym[0]
    h_tm1_sym = insym[1]

    weightList = keras_layer.get_weights()
    kernel_wt = symtab.new_const(weightList[0].transpose([1, 0]))
    recurrent_wt = symtab.new_const(weightList[1].transpose([1, 0]))
    in_bias = symtab.new_const(weightList[2])

    units = list(weightList[0].shape)[1]

    in_data = _sym.flatten(in_data)
    matrix_x = _sym.dense(in_data, kernel_wt, in_bias, use_bias=True, units=units)

    # inputs projected by all gate matrices at once
    split_indices = [keras_layer.units, 2 * keras_layer.units]
    gates = _sym.split(matrix_x, indices_or_sections=split_indices, axis=1)
    x_z = gates[0]
    x_r = gates[1]
    x_h = gates[2]

    # hidden state projected separately for update/reset and new
    units = 2 * keras_layer.units
    split_indices = [units]
    rec_wts = _sym.split(recurrent_wt, indices_or_sections=split_indices, axis=0)

    h_tm1_sym = _sym.flatten(h_tm1_sym)
    matrix_inner = _sym.dense(h_tm1_sym, rec_wts[0], use_bias=False, units=units)

    split_indices = [keras_layer.units]
    recurrent = _sym.split(matrix_inner, indices_or_sections=split_indices, axis=1)
    recurrent_z = recurrent[0]
    recurrent_r = recurrent[1]

    rec_act_z = _convert_recurrent_activation(x_z + recurrent_z, keras_layer)
    rec_act_r = _convert_recurrent_activation(x_r + recurrent_r, keras_layer)

    units = keras_layer.units
    recurrent_h = _sym.dense(rec_act_r * h_tm1_sym, rec_wts[1], use_bias=False, units=units)
    act_hh = _convert_activation(x_h + recurrent_h, keras_layer, None)

    # previous and candidate state mixed by update gate
    output = rec_act_z * h_tm1_sym + (1 - rec_act_z) * act_hh

    out_shape = tuple(dim if dim else 1 for dim in _as_list(keras_layer.output_shape)[0])
    output = _sym.reshape(output, shape=out_shape)
    return [output, output]

def _default_skip(insym, keras_layer, _): # pylint: disable=unused-argument
    """Layers that can be skipped because they are train time only."""
    return insym


_convert_map = {
    'Dense'                    : _convert_dense,
    'Activation'               : _convert_activation,
    'ReLU'                     : _convert_advanced_activation,
    'LeakyReLU'                : _convert_advanced_activation,
    'PReLU'                    : _convert_advanced_activation,
    'ELU'                      : _convert_advanced_activation,
    'ThresholdedReLU'          : _convert_advanced_activation,

    'AveragePooling2D'         : _convert_pooling,
    'MaxPooling2D'             : _convert_pooling,
    'GlobalAveragePooling2D'   : _convert_pooling,
    'GlobalMaxPooling2D'       : _convert_pooling,
    'Conv2D'                   : _convert_convolution,
    'Conv2DTranspose'          : _convert_convolution,
    'DepthwiseConv2D'          : _convert_convolution,
    'SeparableConv2D'          : _convert_separable_convolution,

    'Flatten'                  : _convert_flatten,
    'Reshape'                  : _convert_reshape,
    'Concatenate'              : _convert_concat,
    'BatchNormalization'       : _convert_batchnorm,

    'Add'                      : _convert_merge,
    'Subtract'                 : _convert_merge,
    'Multiply'                 : _convert_merge,
    'ZeroPadding2D'            : _convert_padding,
    'UpSampling2D'             : _convert_upsample,
    'Cropping2D'               : _convert_cropping,

    # 'ZeroPadding1D'          : _convert_padding,
    # 'AveragePooling1D'       : _convert_pooling,
    # 'MaxPooling1D'           : _convert_pooling,
    # 'GlobalAveragePooling1D' : _convert_pooling,
    # 'GlobalMaxPooling1D'     : _convert_pooling,
    # 'Cropping1D'             : _convert_cropping,
    # 'UpSampling1D'           : _convert_upsample,
    # 'UpSampling3D'           : _convert_upsample,
    # 'Conv1D'                 : _convert_convolution1d,

    'SimpleRNN'                : _convert_simple_rnn,
    'LSTM'                     : _convert_lstm,
    'GRU'                      : _convert_gru,
    # 'Bidirectional'          : _convert_bidirectional,
    # 'TimeDistributed'        : _default_skip,

    # 'Average'                : _convert_merge,
    # 'Maximum'                : _convert_merge,
    # 'Dot'                    : _convert_merge,
    # 'Permute'                : _convert_permute,
    # 'Embedding'              : _convert_embedding,
    # 'RepeatVector'           : _convert_repeat_vector,

    'InputLayer'               : _default_skip,
    'Dropout'                  : _default_skip,
    'SpatialDropout2D'         : _default_skip,
    'SpatialDropout1D'         : _default_skip,
}


def _check_unsupported_layers(model):
    for layer in model.layers:
        if type(layer).__name__ not in _convert_map:
            raise ValueError("Keras layer {} not supported.".format(type(layer).__name__))

def _as_list(arr):
    """Force being a list, ignore if already is."""
    if isinstance(arr, list):
        return arr
    return [arr]

def keras_op_to_nnvm(insym, keras_layer, outname, symtab):
    """Convert keras layer to nnvm symbol, and update symtab.

    Parameters
    ----------
    insym : nnvm.symbol.Symbol or a list of it
        The input nnvm symbol(s)

    keras_layer : keras.layers
        The keras layer to be converted

    outname : str
        Name of the output nnvm symbol

    symtab : nnvm.frontend.common.SymbolTable
        The global symbol table to be updated
    """
    if type(keras_layer).__name__ not in _convert_map:
        raise NotImplementedError("{} is not supported".format((type(keras_layer).__name__)))
    outs = _convert_map[type(keras_layer).__name__](insym, keras_layer, symtab)
    outs = _as_list(outs)

    for t_idx, out in enumerate(outs):
        name = outname + ":" + str(t_idx)
        symtab.set_var(name, out)

def from_keras(model):
    """Convert keras model to NNVM format.

    Parameters
    ----------
    model : keras.engine.training.Model
        The keras model to be converted

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import keras
    except ImportError:
        raise ImportError('Keras must be installed')

    assert isinstance(model, keras.engine.training.Model)
    if keras.backend.backend() != 'tensorflow':
        raise ValueError("Keras frontend currently supports tensorflow backend only.")
    if keras.backend.image_data_format() != 'channels_last':
        raise ValueError("Keras frontend currently supports data_format = channels_last only.")
    _check_unsupported_layers(model)

    symtab = SymbolTable()
    for keras_layer in model.layers:
        if isinstance(keras_layer, keras.engine.InputLayer):
            symtab.get_var(keras_layer.name, must_contain=False)
        else:
            inbound_nodes = keras_layer.inbound_nodes if hasattr(keras_layer, 'inbound_nodes') \
                       else keras_layer._inbound_nodes if hasattr(keras_layer, '_inbound_nodes') \
                       else None
            if inbound_nodes is None:
                raise TypeError("Unknown layer type or unsupported Keras version : {}"
                                .format(keras_layer))
            for node_idx, node in enumerate(inbound_nodes):
                insym = []

                # Since Keras allows creating multiple layers from the same name instance,
                # we append node index to the symbol name to make it unique.
                # The one exception is InputLayer.  Changing input variable names after conversion
                # would confuse users, so we should keep them as far as possible.  Fortunately,
                # they are named uniquely to input_1, input_2, input_3 ... by default.
                zip_node = zip(node.node_indices, node.tensor_indices, node.inbound_layers)
                for n_idx, t_idx, layer in zip_node:
                    if isinstance(layer, keras.engine.InputLayer):
                        sym = symtab.get_var(layer.name, must_contain=True)
                    else:
                        sym_name = layer.name + ':' + str(n_idx) + ':' + str(t_idx)
                        sym = symtab.get_var(sym_name, must_contain=True)
                    insym.append(sym)

                if len(insym) == 1:
                    insym = insym[0]
                keras_op_to_nnvm(insym, keras_layer, keras_layer.name + ':' + str(node_idx), symtab)

    #model._output_coordinates contains out_node(oc[0]), node_index(oc[1]) and tensor index(oc[2])
    #Get all output nodes in symtab using the name made from above values. The out symbols
    #were added to symtab in keras_op_to_nnvm using this name. For multiple outputs, make a list
    #with these output symbols and Group them.
    outsym = [symtab.get_var(oc[0].name + ":" + str(oc[1]) + ":" + str(oc[2]))
              for oc in model._output_coordinates]

    tvmparams = {k:tvm.nd.array(np.array(v, dtype=np.float32)) for k, v in symtab.params.items()}
    return _sym.Group(outsym), tvmparams
