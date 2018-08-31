# pylint: disable=invalid-name, unused-argument
"""CoreML frontend."""
from __future__ import absolute_import as _abs
import numpy as np

import tvm
from .. import symbol as _sym
from .._base import string_types
from .common import SymbolTable

__all__ = ['from_coreml']


def NeuralNetworkImageScaler(op, insym, symtab):
    # this changes the symbol
    biases = np.array([op.blueBias, op.greenBias, op.redBias]).reshape([3, 1, 1])
    bias = symtab.new_const(biases)
    ret = _sym.__mul_scalar__(insym, scalar=op.channelScale)
    ret = _sym.broadcast_add(ret, bias)
    return ret


def NeuralNetworkMeanImage(op, insym, symtab):
    # this changes the symbol
    ret = _sym.elemwise_sub(insym, scalar=op.meanImage)
    return ret


def ConvolutionLayerParams(op, insym, symtab):
    """Convolution layer params."""
    weights = symtab.new_const(np.array(list(op.weights.floatValue)).reshape(
        tuple([op.outputChannels, op.kernelChannels] + list(op.kernelSize))))
    if op.hasBias:
        biases = symtab.new_const(list(op.bias.floatValue))
    dilation = list(op.dilationFactor)
    if not dilation:
        dilation = [1, 1]
    params = {'channels':op.outputChannels,
              'kernel_size':list(op.kernelSize),
              'strides':list(op.stride),
              'dilation': dilation,
              'use_bias': op.hasBias,
              'groups':op.nGroups}

    if op.WhichOneof('ConvolutionPaddingType') == 'valid':
        valid = op.valid
        padding = [b.startEdgeSize for b in valid.paddingAmounts.borderAmounts]
        padding2 = [b.endEdgeSize for b in valid.paddingAmounts.borderAmounts]
        for i, j in zip(padding, padding2):
            assert i == j, "Asymmetry padding not supported"
        if padding:
            params['padding'] = padding
    elif op.WhichOneof('ConvolutionPaddingType') == 'same':
        kernel = params['kernel_size']
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
        pad_t = pad_h // 2
        pad_l = pad_w // 2
        pad_b = pad_h - pad_t
        pad_r = pad_w - pad_l
        assert pad_t == pad_r and pad_l == pad_b, "Asymmetry padding not supported"
        params['padding'] = [pad_t, pad_l]
    else:
        raise NotImplementedError("Valid/Same convolution padding implemented")

    if op.hasBias:
        pos = [insym, weights, biases]
    else:
        pos = [insym, weights]

    if op.isDeconvolution:
        ret = _sym.conv2d_transpose(*pos, **params)
    else:
        ret = _sym.conv2d(*pos, **params)
    # consume padding layer
    if symtab.in_padding:
        params['padding'] = [sum(x) for x in zip(params.get('padding', [0, 0]), symtab.paddings)]
        symtab.clear_padding()
    return ret

def BatchnormLayerParams(op, insym, symtab):
    """Get layer of batchnorm parameter"""
    # this changes the symbol
    if op.instanceNormalization:
        raise NotImplementedError("instance normalization not implemented")
    else:
        params = {'gamma':symtab.new_const(list(op.gamma.floatValue)),
                  'beta':symtab.new_const(list(op.beta.floatValue)),
                  'moving_mean':symtab.new_const(list(op.mean.floatValue)),
                  'moving_var': symtab.new_const(list(op.variance.floatValue)),
                  'epsilon': op.epsilon}
        return _sym.batch_norm(data=insym, **params)

def ActivationParams(op, insym, symtab):
    """Get activation parameters"""
    whichActivation = op.WhichOneof('NonlinearityType')
    par = getattr(op, whichActivation)
    if whichActivation == 'linear':
        return _sym.__add_scalar__(_sym.__mul_scalar__(insym, scalar=par.alpha), scalar=par.beta)
    elif whichActivation == 'ReLU':
        return _sym.relu(insym)
    elif whichActivation == 'leakyReLU':
        return _sym.leaky_relu(insym, alpha=par.alpha)
    elif whichActivation == 'thresholdedReLU':
        alpha_tensor = _sym.full_like(insym, fill_value=float(par.alpha))
        return _sym.elemwise_mul(insym, _sym.greater(insym, alpha_tensor))
    elif whichActivation == 'PReLU':
        return _sym.prelu(insym, alpha=par.alpha)
    elif whichActivation == 'tanh':
        return _sym.tanh(insym)
    elif whichActivation == 'scaledTanh':
        return _sym.__mul_scalar__(_sym.tanh(_sym.__mul_scalar__(
            insym, scalar=par.beta)), scalar=par.alpha)
    elif whichActivation == 'sigmoid':
        return _sym.sigmoid(insym)
    elif whichActivation == 'sigmoidHard':
        transformX = (par.alpha * insym) + par.beta
        return _sym.clip(transformX, a_min=0, a_max=1)
    elif whichActivation == 'ELU':
        return _sym.__mul_scalar__(_sym.__add_scalar__(
            _sym.exp(insym), scalar=-1), scalar=par.alpha)
    elif whichActivation == 'softsign':
        return insym / (1 + (_sym.relu(insym) + _sym.relu(_sym.negative(insym))))
    elif whichActivation == 'softplus':
        return _sym.log(_sym.__add_scalar__(_sym.exp(insym), scalar=1))
    elif whichActivation == 'parametricSoftplus':
        alpha = list(par.alpha.floatValue)
        beta = list(par.alpha.floatValue)
        if len(alpha) == 1:
            return _sym.__mul_scalar__(_sym.log(_sym.__add_scalar__(
                _sym.exp(insym), scalar=beta[0])), scalar=alpha[0])
        alpha = np.array(alpha).reshape((len(alpha), 1, 1))
        beta = np.array(beta).reshape((len(beta), 1, 1))
        alphasym = symtab.new_const(alpha)
        betasym = symtab.new_const(beta)
        return _sym.broadcast_mul(_sym.log(_sym.broadcast_add(
            _sym.exp(insym), betasym)), alphasym)
    else:
        raise NotImplementedError('%s not implemented' % whichActivation)

def ScaleLayerParams(op, insym, symtab):
    """Scale layer params."""
    scale = symtab.new_const(np.array(list(op.scale.floatValue)).reshape(
        tuple(list(op.shapeScale) + [1, 1])))
    # scale = _sym.reshape(scale, shape=tuple(list(op.shapeScale) + [1,1]))
    ret = _sym.broadcast_mul(insym, scale)
    if op.hasBias:
        bias = symtab.new_const(np.array(list(op.bias.floatValue)).reshape(
            tuple(list(op.shapeBias) + [1, 1])))
        # bias = _sym.reshape(bias, shape=tuple(list(op.shapeBias) + [1,1]))
        ret = _sym.broadcast_add(ret, bias)
    return ret

def PoolingLayerParams(op, insym, symtab):
    """get pooling parameters"""
    if op.globalPooling:
        if op.type == 0:
            return _sym.global_max_pool2d(insym)
        elif op.type == 1:
            return _sym.global_avg_pool2d(insym)
        else:
            raise NotImplementedError("Only max and average pooling implemented")

    else:
        params = {'pool_size':list(op.kernelSize),
                  'strides':list(op.stride)}

        if op.WhichOneof('PoolingPaddingType') == 'valid':
            valid = op.valid
            padding = [b.startEdgeSize for b in valid.paddingAmounts.borderAmounts]
            padding2 = [b.endEdgeSize for b in valid.paddingAmounts.borderAmounts]
            for i, j in zip(padding, padding2):
                assert i == j
            params['padding'] = padding
        elif op.WhichOneof('PoolingPaddingType') == 'includeLastPixel':
            # I don't know if this is correct
            valid = op.includeLastPixel
            padding = list(valid.paddingAmounts)
            params['padding'] = padding
            params['ceil_mode'] = True
        else:
            raise NotImplementedError("Other convolution padding not implemented")

        # consume padding layer
        if symtab.in_padding:
            params['padding'] = [sum(x) for x in zip(
                params.get('padding', [0, 0]), symtab.paddings)]
            symtab.clear_padding()

        if op.type == 0:
            return _sym.max_pool2d(insym, **params)
        elif op.type == 1:
            return _sym.avg_pool2d(insym, **params)
        else:
            raise NotImplementedError("Only max and average pooling implemented")

def SoftmaxLayerParams(op, insym, symtab):
    return _sym.softmax(_sym.flatten(insym))

def InnerProductLayerParams(op, insym, symtab):
    weights = symtab.new_const(np.array(op.weights.floatValue).reshape(
        (op.outputChannels, op.inputChannels)))
    par = {'weight':weights, 'use_bias':False, 'units':op.outputChannels}
    if op.hasBias:
        bias = symtab.new_const(np.array(op.bias.floatValue))
        par['bias'] = bias
        par['use_bias'] = True
    return _sym.dense(data=insym, **par)

def AddLayerParams(op, insyms, symtab):
    if not isinstance(insyms, list):
        insyms = [insyms]
    ret = insyms[0]
    for i in range(1, len(insyms)):
        ret = _sym.elemwise_add(ret, insyms[i])
    if op.alpha > 0:
        ret = _sym.__add_scalar__(ret, scalar=op.alpha)
    return ret

def MultiplyLayerParams(op, insyms, symtab):
    if not isinstance(insyms, list):
        insyms = [insyms]
    ret = insyms[0]
    for i in range(1, len(insyms)):
        ret = _sym.elemwise_mul(ret, insyms[i])
    if op.alpha != 1:
        ret = _sym.__mul_scalar__(ret, scalar=op.alpha)
    return ret

def ConcatLayerParams(op, insyms, symtab):
    if not isinstance(insyms, list):
        insyms = [insyms]
    if op.sequenceConcat:
        raise NotImplementedError("Sequence Concat not supported")
    ret = _sym.concatenate(*insyms, axis=1)
    return ret

def FlattenLayerParams(op, insym, symtab):
    if op.mode == 1:
        insym = _sym.transpose(_sym.reshape(insym, shape=(0, 0, -1)), axes=(0, 2, 1))
    return _sym.flatten(insym)

def PaddingLayerParams(op, insym, symtab):
    """Hacking for padding layer params."""
    if op.WhichOneof('PaddingType') == 'constant':
        constant = op.constant
        if constant.value != 0:
            raise NotImplementedError("Padding value {} not supported.".format(constant.value))
        padding = [b.startEdgeSize for b in op.paddingAmounts.borderAmounts]
        padding2 = [b.endEdgeSize for b in op.paddingAmounts.borderAmounts]
        for i, j in zip(padding, padding2):
            assert i == j
        symtab.set_padding(padding)
    else:
        raise NotImplementedError("Only constant padding is supported now.")
    return insym

def PermuteLayerParams(op, insym, symtab):
    axes = tuple(op.axis)
    return _sym.transpose(insym, axes=axes)

def UpsampleLayerParams(op, insym, symtab):
    if op.scalingFactor[0] != op.scalingFactor[1]:
        raise NotImplementedError("Upsampling only supported with same \
            height and width scaling factor.")
    interpolationMode = 'NEAREST_NEIGHBOR' if op.mode == 0 else 'BILINEAR'
    return _sym.upsampling(insym, scale=op.scalingFactor[0], method=interpolationMode)

def L2NormalizeLayerParams(op, insym, symtab):
    return _sym.l2_normalize(insym, eps=op.epsilon, axis=1)

def LRNLayerParams(op, insym, symtab):
    par = {}
    par['size'] = op.localSize
    par['bias'] = op.k
    par['alpha'] = op.alpha
    par['beta'] = op.beta
    par['axis'] = 1 #default layout is nchw
    return _sym.lrn(data=insym, **par)

def AverageLayerParams(op, insyms, symtab):
    if not isinstance(insyms, list) or len(insyms) < 2:
        raise ValueError("Expect minimum 2 inputs")
    count = len(insyms)
    _sum = insyms[0]
    for i in range(1, count):
        _sum = _sym.broadcast_add(_sum, insyms[i])
    return _sum / count

def MaxLayerParams(op, insyms, symtab):
    if not isinstance(insyms, list) or len(insyms) < 2:
        raise ValueError("Expect minimum 2 inputs")
    _max = insyms[0]
    for i in range(1, len(insyms)):
        _max = _sym.broadcast_max(_max, insyms[i])
    return _max

def MinLayerParams(op, insyms, symtab):
    if not isinstance(insyms, list) or len(insyms) < 2:
        raise ValueError("Expect minimum 2 inputs")
    _min = insyms[0]
    for i in range(1, len(insyms)):
        _min = _sym.broadcast_min(_min, insyms[i])
    return _min

_convert_map = {
    'NeuralNetworkMeanImage': NeuralNetworkMeanImage,
    'NeuralNetworkImageScaler': NeuralNetworkImageScaler,
    'ConvolutionLayerParams':ConvolutionLayerParams,
    'BatchnormLayerParams':BatchnormLayerParams,
    'ActivationParams':ActivationParams,
    'ScaleLayerParams':ScaleLayerParams,
    'PoolingLayerParams':PoolingLayerParams,
    'SoftmaxLayerParams':SoftmaxLayerParams,
    'InnerProductLayerParams':InnerProductLayerParams,
    'AddLayerParams':AddLayerParams,
    'MultiplyLayerParams':MultiplyLayerParams,
    'FlattenLayerParams':FlattenLayerParams,
    'ConcatLayerParams':ConcatLayerParams,
    'PaddingLayerParams':PaddingLayerParams,
    'PermuteLayerParams':PermuteLayerParams,
    'UpsampleLayerParams':UpsampleLayerParams,
    'L2NormalizeLayerParams':L2NormalizeLayerParams,
    'LRNLayerParams':LRNLayerParams,
    'AverageLayerParams':AverageLayerParams,
    'MaxLayerParams':MaxLayerParams,
    'MinLayerParams':MinLayerParams,
}

def coreml_op_to_nnvm(op, inname, outname, symtab):
    """Convert coreml layer to nnvm layer.

    Parameters
    ----------
    coremlop: a coreml protobuf bit

    prevsym: previous nnvm symbol

    Returns:
    -------
    nnvm.sym.Symbol
        Converted symbol
    """
    classname = type(op).__name__
    if classname not in _convert_map:
        raise NotImplementedError("%s is not supported" % (classname))
    if isinstance(inname, string_types):
        insym = symtab.get_var(inname)
    else:
        insym = [symtab.get_var(i) for i in inname]
    ret = _convert_map[classname](op, insym, symtab)
    if outname:
        symtab.set_var(outname, ret)
    if classname != 'PaddingLayerParams':
        assert not symtab.in_padding, "Previous padding not consumed by conv/pool"

def from_coreml(model):
    """Convert from coreml model into NNVM format.

    Parameters
    ----------
    model:
        coremltools.models.MLModel of a NeuralNetworkClassifier

    Returns
    -------
    sym : nnvm.Symbol
        Compatible nnvm symbol

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import coremltools as cm
    except ImportError:
        raise ImportError('The coremltools package must be installed')

    assert isinstance(model, cm.models.MLModel)
    spec = model.get_spec()
    modeltype = spec.WhichOneof('Type')
    assert modeltype in ['neuralNetworkClassifier', 'neuralNetwork', 'neuralNetworkRegressor']
    cc = getattr(spec, modeltype)

    symtab = SymbolTable()
    for i in spec.description.input:
        symtab.get_var(i.name, must_contain=False)

    for pp in cc.preprocessing:
        whichpp = pp.WhichOneof('preprocessor')
        ppmethod = getattr(pp, whichpp)
        # the NeuralNetworkImageScalar doesn't seem to have a featureName?
        if whichpp == 'scaler':
            for i in spec.description.input:
                coreml_op_to_nnvm(ppmethod, i.name, i.name, symtab)
        else:
            coreml_op_to_nnvm(ppmethod, pp.featureName, pp.featureName, symtab)

    for l in cc.layers:
        layertype = l.WhichOneof('layer')
        layerop = getattr(l, layertype)
        assert len(l.output) == 1
        if len(l.input) == 1:
            coreml_op_to_nnvm(layerop, l.input[0], l.output[0], symtab)
        else:
            coreml_op_to_nnvm(layerop, list(l.input), l.output[0], symtab)
    returns = [symtab.get_var(i.name, must_contain=False) for i in spec.description.output]
    tvmparams = {k:tvm.nd.array(np.array(v, dtype=np.float32)) for k, v in symtab.params.items()}
    # for now return first output
    return returns[0], tvmparams
