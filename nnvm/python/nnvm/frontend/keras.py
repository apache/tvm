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
        return _sym.softmax(insym)
    elif act_type == 'sigmoid':
        return _sym.sigmoid(insym)
    elif act_type == 'tanh':
        return _sym.tanh(insym)
    elif act_type == 'relu':
        return _sym.relu(insym)
    elif act_type == 'softplus':
        return _sym.log(_sym.__add_scalar__(_sym.exp(insym), scalar=1))
    elif act_type == 'elu':
        raise NotImplementedError('elu not implemented')
    elif act_type == 'relu6':
        raise NotImplementedError('relu6 not implemented')
    elif act_type == 'softsign':
        raise NotImplementedError('softsign not implemented')
    elif act_type == 'hard_sigmoid':
        raise NotImplementedError('hard_sigmoid not implemented')
    else:
        raise TypeError("Unsupported activation type : {}".format(act_type))


def _convert_advanced_activation(insym, keras_layer, _):
    act_type = type(keras_layer).__name__
    if act_type == 'LeakyReLU':
        return _sym.leaky_relu(insym, alpha=keras_layer.alpha)
    elif act_type == 'ELU':
        raise NotImplementedError('ELU not implemented')
    elif act_type == 'PReLU':
        raise NotImplementedError('PReLU not implemented')
    elif act_type == 'ThresholdedReLU':
        raise NotImplementedError('ThresholdedReLU not implemented')
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
    kernel_h = (kernel_h - 1) * dilation[0] + 1
    kernel_w = (kernel_w - 1) * dilation[1] + 1
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
        in_h = keras_layer.input.shape[1].value
        in_w = keras_layer.input.shape[2].value
        pad_t, pad_b = _get_pad_pair(in_h, kernel_h, stride_h)
        pad_l, pad_r = _get_pad_pair(in_w, kernel_w, stride_w)
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
        in_h = keras_layer.input.shape[1].value
        in_w = keras_layer.input.shape[2].value
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
        # we insert a separate pad operator
        elif keras_layer.padding == 'same':
            in_h = keras_layer.input.shape[1].value
            in_w = keras_layer.input.shape[2].value
            pad_t, pad_b = _get_pad_pair(in_h, pool_h, stride_h)
            pad_l, pad_r = _get_pad_pair(in_w, pool_w, stride_w)
            insym = _sym.pad(data=insym, pad_width=(
                (0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))
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
    shape = keras_layer.shape if hasattr(keras_layer, 'shape') \
       else keras_layer.target_shape if hasattr(keras_layer, 'target_shape') \
       else None
    if shape is None:
        raise TypeError("No shape attribute in reshape layer: {}".format(keras_layer))
    return _sym.reshape(insym, shape=shape)


def _default_skip(insym, keras_layer, _): # pylint: disable=unused-argument
    """Layers that can be skipped because they are train time only."""
    return insym


_convert_map = {
    'Dense'                    : _convert_dense,
    'Activation'               : _convert_activation,
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

    # 'ZeroPadding1D'          : _convert_padding,
    # 'AveragePooling1D'       : _convert_pooling,
    # 'MaxPooling1D'           : _convert_pooling,
    # 'GlobalAveragePooling1D' : _convert_pooling,
    # 'GlobalMaxPooling1D'     : _convert_pooling,
    # 'Cropping1D'             : _convert_cropping,
    # 'Cropping2D'             : _convert_cropping,
    # 'UpSampling1D'           : _convert_upsample,
    # 'UpSampling3D'           : _convert_upsample,
    # 'Conv1D'                 : _convert_convolution1d,

    # 'GRU'                    : _convert_gru,
    # 'LSTM'                   : _convert_lstm,
    # 'SimpleRNN'              : _convert_simple_rnn,
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
    ret = _convert_map[type(keras_layer).__name__](insym, keras_layer, symtab)
    symtab.set_var(outname, ret)


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
    if keras.backend.image_data_format() != 'channels_last':
        raise ValueError("Keras frontend currently supports data_format = channels_last only.")
    _check_unsupported_layers(model)

    symtab = SymbolTable()
    for keras_layer in model.layers:
        if isinstance(keras_layer, keras.engine.topology.InputLayer):
            keras_layer.name = 'data'
            symtab.get_var(keras_layer.name, must_contain=False)
        else:
            predecessors = []
            inbound_nodes = keras_layer.inbound_nodes if hasattr(keras_layer, 'inbound_nodes') \
                       else keras_layer._inbound_nodes if hasattr(keras_layer, '_inbound_nodes') \
                       else None
            if inbound_nodes is None:
                raise TypeError("Unknown layer type or unsupported Keras version : {}"
                                .format(keras_layer))
            for node in inbound_nodes:
                for pred in node.inbound_layers:
                    predecessors.append(pred.name)
            if len(predecessors) == 1:
                insym = symtab.get_var(predecessors[0], must_contain=True)
            else:
                insym = [symtab.get_var(pred, must_contain=True) for pred in predecessors]
            keras_op_to_nnvm(insym, keras_layer, keras_layer.name, symtab)

    returns = [symtab.get_var(i.name, must_contain=False) for i in model.output_layers]
    tvmparams = {k:tvm.nd.array(np.array(v, dtype=np.float32)) for k, v in symtab.params.items()}
    return returns[0], tvmparams
