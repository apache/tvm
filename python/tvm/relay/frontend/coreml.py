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
# pylint: disable=invalid-name, import-self, unused-argument, unused-variable, no-else-return
# pylint: disable=inconsistent-return-statements, import-outside-toplevel
"""CoreML frontend."""
import math
import numpy as np
import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from ... import nd as _nd
from ..._ffi import base as _base
from .common import ExprTable
from .common import infer_shape as _infer_shape

__all__ = ["from_coreml"]


def _NeuralNetworkImageScaler(op, inexpr, etab):
    # TODO: we need to support more colorspace, such as rgb.
    # this changes the symbol
    biases = np.array([op.blueBias, op.greenBias, op.redBias]).reshape([3, 1, 1])
    bias = etab.new_const(biases)
    ret = _op.multiply(inexpr, _expr.const(op.channelScale, dtype="float32"))
    ret = _op.add(ret, bias)
    return ret


def _NeuralNetworkMeanImage(op, inexpr, etab):
    # this changes the symbol
    ret = _op.subtract(inexpr, _expr.const(op.meanImage, dtype="float32"))
    return ret


def _ConvolutionLayerParams(op, inexpr, etab):
    """Convolution layer params."""
    if op.isDeconvolution:
        weights = etab.new_const(
            np.array(list(op.weights.floatValue)).reshape(
                tuple([op.kernelChannels, op.outputChannels] + list(op.kernelSize))
            )
        )
    else:
        weights = etab.new_const(
            np.array(list(op.weights.floatValue)).reshape(
                tuple([op.outputChannels, op.kernelChannels] + list(op.kernelSize))
            )
        )
    dilation = list(op.dilationFactor)
    if not dilation:
        dilation = [1, 1]
    N, C, H, W = _infer_shape(inexpr)
    params = {
        "channels": op.outputChannels,
        "kernel_size": list(op.kernelSize),
        "strides": list(op.stride),
        "dilation": dilation,
        "groups": op.nGroups,
    }

    if op.WhichOneof("ConvolutionPaddingType") == "valid":
        valid = op.valid
        if valid.paddingAmounts.borderAmounts:
            assert len(valid.paddingAmounts.borderAmounts) == 2
            pad_t = valid.paddingAmounts.borderAmounts[0].startEdgeSize
            pad_l = valid.paddingAmounts.borderAmounts[1].startEdgeSize
            pad_b = valid.paddingAmounts.borderAmounts[0].endEdgeSize
            pad_r = valid.paddingAmounts.borderAmounts[1].endEdgeSize
            if not all(v == 0 for v in (pad_t, pad_l, pad_b, pad_r)):
                params["padding"] = (pad_t, pad_l, pad_b, pad_r)
    elif op.WhichOneof("ConvolutionPaddingType") == "same":
        assert op.same.asymmetryMode == 0, (
            "Only support BOTTOM_RIGHT_HEAVY mode, " "which is used by tf/caffe and so on"
        )
        kernel = params["kernel_size"]
        strides = params["strides"]
        pad_t, pad_b = get_pad_value(H, kernel[0], strides[0])
        pad_l, pad_r = get_pad_value(W, kernel[1], strides[1])
        params["padding"] = (pad_t, pad_l, pad_b, pad_r)
    else:
        raise NotImplementedError("Valid/Same convolution padding implemented")

    if op.isDeconvolution:
        ret = _op.nn.conv2d_transpose(data=inexpr, weight=weights, **params)
    else:
        ret = _op.nn.conv2d(data=inexpr, weight=weights, **params)
    if op.hasBias:
        biases = etab.new_const(list(op.bias.floatValue))
        ret = _op.nn.bias_add(ret, biases)

    return ret


def _BatchnormLayerParams(op, inexpr, etab):
    """Get layer of batchnorm parameter"""
    # this changes the symbol
    if op.instanceNormalization:
        raise tvm.error.OpNotImplemented(
            'Operator "instance normalization" is not supported in frontend CoreML.'
        )
    params = {
        "gamma": etab.new_const(list(op.gamma.floatValue)),
        "beta": etab.new_const(list(op.beta.floatValue)),
        "moving_mean": etab.new_const(list(op.mean.floatValue)),
        "moving_var": etab.new_const(list(op.variance.floatValue)),
        "epsilon": op.epsilon,
    }
    result, moving_mean, moving_var = _op.nn.batch_norm(data=inexpr, **params)
    return result


def _ActivationParams(op, inexpr, etab):
    """Get activation parameters"""
    whichActivation = op.WhichOneof("NonlinearityType")
    par = getattr(op, whichActivation)
    if whichActivation == "linear":
        alpha = _expr.const(par.alpha, dtype="float32")
        beta = _expr.const(par.beta, dtype="float32")
        return _op.add(_op.multiply(inexpr, alpha), beta)
    if whichActivation == "ReLU":
        return _op.nn.relu(inexpr)
    if whichActivation == "leakyReLU":
        return _op.nn.leaky_relu(inexpr, alpha=par.alpha)
    elif whichActivation == "thresholdedReLU":
        alpha_tensor = _op.full_like(inexpr, fill_value=_expr.const(par.alpha, dtype="float32"))
        return _op.multiply(inexpr, _op.greater(inexpr, alpha_tensor).as_type("float32"))
    if whichActivation == "PReLU":
        return _op.nn.prelu(inexpr, alpha=_expr.const(par.alpha, dtype="float32"))
    if whichActivation == "tanh":
        return _op.tanh(inexpr)
    if whichActivation == "scaledTanh":
        alpha = _expr.const(par.alpha, dtype="float32")
        beta = _expr.const(par.beta, dtype="float32")
        return _op.multiply(_op.tanh(_op.multiply(inexpr, beta)), alpha)
    if whichActivation == "sigmoid":
        return _op.sigmoid(inexpr)
    if whichActivation == "sigmoidHard":
        alpha = _expr.const(par.alpha, dtype="float32")
        beta = _expr.const(par.beta, dtype="float32")
        transformX = (alpha * inexpr) + beta
        return _op.clip(transformX, a_min=0.0, a_max=1.0)
    if whichActivation == "ELU":
        return _op.multiply(
            _op.add(_op.exp(inexpr), _expr.const(-1, dtype="float32")),
            _expr.const(par.alpha, dtype="float32"),
        )
    if whichActivation == "softsign":
        return inexpr / (
            _expr.const(1, dtype="float32")
            + (op.nn.relu(inexpr) + _op.nn.relu(_op.negative(inexpr)))
        )
    if whichActivation == "softplus":
        return _op.log(_op.add(_op.exp(inexpr), _expr.const(1, dtype="float32")))
    if whichActivation == "parametricSoftplus":
        alpha = list(par.alpha.floatValue)
        beta = list(par.alpha.floatValue)
        if len(alpha) == 1:
            return _op.multiply(
                _op.log(_op.add(_op.exp(inexpr), _expr.const(beta[0], dtype="float32"))),
                _expr.const(alpha[0], dtype="float32"),
            )
        alpha = np.array(alpha).reshape((len(alpha), 1, 1))
        beta = np.array(beta).reshape((len(beta), 1, 1))
        alpha_expr = etab.new_const(alpha)
        beta_expr = etab.new_const(beta)
        return _op.multiply(_op.log(_op.add(_op.exp(inexpr), beta_expr)), alpha_expr)
    raise tvm.error.OpNotImplemented(
        "Operator {} is not supported in frontend CoreML.".format(whichActivation)
    )


def _ScaleLayerParams(op, inexpr, etab):
    """Scale layer params."""
    scale = etab.new_const(
        np.array(list(op.scale.floatValue)).reshape(tuple(list(op.shapeScale) + [1, 1]))
    )
    ret = _op.multiply(inexpr, scale)
    if op.hasBias:
        bias = etab.new_const(
            np.array(list(op.bias.floatValue)).reshape(tuple(list(op.shapeBias) + [1, 1]))
        )
        ret = _op.add(ret, bias)
    return ret


def _PoolingLayerParams(op, inexpr, etab):
    """get pooling parameters"""
    if op.globalPooling:
        if op.type == 0:
            return _op.nn.global_max_pool2d(inexpr)
        if op.type == 1:
            return _op.nn.global_avg_pool2d(inexpr)
        raise tvm.error.OpNotImplemented(
            "Only Max and Average Pooling are supported in frontend CoreML."
        )

    params = {"pool_size": list(op.kernelSize), "strides": list(op.stride)}

    if op.WhichOneof("PoolingPaddingType") == "valid":
        valid = op.valid
        if valid.paddingAmounts.borderAmounts:
            assert len(valid.paddingAmounts.borderAmounts) == 2
            pad_t = valid.paddingAmounts.borderAmounts[0].startEdgeSize
            pad_l = valid.paddingAmounts.borderAmounts[1].startEdgeSize
            pad_b = valid.paddingAmounts.borderAmounts[0].endEdgeSize
            pad_r = valid.paddingAmounts.borderAmounts[1].endEdgeSize
            if not all(v == 0 for v in (pad_t, pad_l, pad_b, pad_r)):
                params["padding"] = [pad_t, pad_l, pad_b, pad_r]
    elif op.WhichOneof("PoolingPaddingType") == "includeLastPixel":
        # I don't know if this is correct
        valid = op.includeLastPixel
        padding = list(valid.paddingAmounts)
        params["padding"] = padding
        params["ceil_mode"] = True
    else:
        msg = "PoolingPaddingType {} is not supported in operator Pooling."
        op_name = op.WhichOneof("PoolingPaddingType")
        raise tvm.error.OpAttributeUnImplemented(msg.format(op_name))

    if op.type == 0:
        return _op.nn.max_pool2d(inexpr, **params)
    if op.type == 1:
        return _op.nn.avg_pool2d(inexpr, **params)
    raise tvm.error.OpNotImplemented("Only Max and Average Pooling are supported in CoreML.")


def _SoftmaxLayerParams(op, inexpr, etab):
    return _op.nn.softmax(_op.nn.batch_flatten(inexpr))


def _InnerProductLayerParams(op, inexpr, etab):
    weights = etab.new_const(
        np.array(op.weights.floatValue).reshape((op.outputChannels, op.inputChannels))
    )
    out = _op.nn.dense(data=inexpr, weight=weights, units=op.outputChannels)
    if op.hasBias:
        bias = etab.new_const(np.array(op.bias.floatValue))
        out = _op.nn.bias_add(out, bias)
    return out


def _AddLayerParams(op, inexpr, etab):
    if not isinstance(inexpr, list):
        inexpr = [inexpr]
    ret = inexpr[0]
    for i in range(1, len(inexpr)):
        ret = _op.add(ret, inexpr[i])
    if op.alpha > 0:
        ret = _op.add(ret, _expr.const(op.alpha, dtype="float32"))
    return ret


def _MultiplyLayerParams(op, inexpr, etab):
    if not isinstance(inexpr, list):
        inexpr = [inexpr]
    ret = inexpr[0]
    for i in range(1, len(inexpr)):
        ret = _op.multiply(ret, inexpr[i])
    if op.alpha != 1:
        ret = _op.multiply(ret, _expr.const(op.alpha, dtype="float32"))
    return ret


def _ConcatLayerParams(op, inexpr, etab):
    if not isinstance(inexpr, list):
        inexpr = [inexpr]
    if op.sequenceConcat:
        raise tvm.error.OpNotImplemented(
            "Operator Sequence Concat is not supported in frontend CoreML."
        )
    ret = _op.concatenate(inexpr, axis=1)
    return ret


def _FlattenLayerParams(op, inexpr, etab):
    if op.mode == 1:
        inexpr = _op.transpose(_op.reshape(inexpr, newshape=(0, 0, -1)), axes=(0, 2, 1))
    return _op.nn.batch_flatten(inexpr)


def _PaddingLayerParams(op, inexpr, etab):
    """Padding layer params."""
    if op.WhichOneof("PaddingType") == "constant":
        constant = op.constant
        if constant.value != 0:
            raise tvm.error.OpAttributeUnImplemented(
                "{} is not supported in operator Padding.".format(constant.value)
            )
        pad_t = op.paddingAmounts.borderAmounts[0].startEdgeSize
        pad_l = op.paddingAmounts.borderAmounts[1].startEdgeSize
        pad_b = op.paddingAmounts.borderAmounts[0].endEdgeSize
        pad_r = op.paddingAmounts.borderAmounts[1].endEdgeSize
        return _op.nn.pad(data=inexpr, pad_width=((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))
    raise tvm.error.OpNotImplemented("Non-constant padding is not supported in frontend CoreML.")


def _PermuteLayerParams(op, inexpr, etab):
    axes = tuple(op.axis)
    return _op.transpose(inexpr, axes=axes)


def _UpsampleLayerParams(op, inexpr, etab):
    if op.scalingFactor[0] != op.scalingFactor[1]:
        raise tvm.error.OpAttributeUnimplemented("Upsample height and width must be equal.")
    interpolationMode = "nearest_neighbor" if op.mode == 0 else "bilinear"
    return _op.nn.upsampling(
        inexpr, scale_h=op.scalingFactor[0], scale_w=op.scalingFactor[1], method=interpolationMode
    )


def _L2NormalizeLayerParams(op, inexpr, etab):
    return _op.nn.l2_normalize(inexpr, eps=op.epsilon, axis=[1])


def _LRNLayerParams(op, inexpr, etab):
    par = {}
    par["size"] = op.localSize
    par["bias"] = op.k
    par["alpha"] = op.alpha
    par["beta"] = op.beta
    par["axis"] = 1  # default layout is nchw
    return _op.nn.lrn(data=inexpr, **par)


def _AverageLayerParams(op, inexpr, etab):
    if not isinstance(inexpr, list) or len(inexpr) < 2:
        raise ValueError("Expect minimum 2 inputs")
    count = len(inexpr)
    _sum = inexpr[0]
    for i in range(1, count):
        _sum = _op.add(_sum, inexpr[i])
    return _sum / _expr.const(count, dtype="float32")


def _MaxLayerParams(op, inexpr, etab):
    if not isinstance(inexpr, list) or len(inexpr) < 2:
        raise ValueError("Expect minimum 2 inputs")
    _max = inexpr[0]
    for i in range(1, len(inexpr)):
        _max = _op.maximum(_max, inexpr[i])
    return _max


def _MinLayerParams(op, inexpr, etab):
    if not isinstance(inexpr, list) or len(inexpr) < 2:
        raise ValueError("Expect minimum 2 inputs")
    _min = inexpr[0]
    for i in range(1, len(inexpr)):
        _min = _op.minimum(_min, inexpr[i])
    return _min


def _UnaryFunctionLayerParams(op, inexpr, etab):
    op_type = op.type
    if op_type == op.SQRT:
        return _op.sqrt(inexpr)
    elif op_type == op.RSQRT:
        epsilon = _expr.const(op.epsilon)
        return _op.rsqrt(inexpr + epsilon)
    elif op_type == op.INVERSE:
        epsilon = _expr.const(op.epsilon)
        return _expr.const(1.0) / (inexpr + epsilon)
    elif op_type == op.POWER:
        alpha = _expr.const(op.alpha)
        return _op.power(inexpr, alpha)
    elif op_type == op.EXP:
        return _op.exp(inexpr)
    elif op_type == op.LOG:
        return _op.log(inexpr)
    elif op_type == op.ABS:
        return _op.abs(inexpr)
    elif op_type == op.THRESHOLD:
        alpha = _expr.const(op.alpha)
        return _op.maximum(inexpr, alpha)
    else:
        msg = "Unary Op type value {} is not supported in frontend CoreML."
        raise tvm.error.OpAttributeUnImplemented(msg.format(op_type))


def _ReduceLayerParams(op, inexpr, etab):
    axis = op.axis
    if axis == op.CHW:
        axis = [-3, -2, -1]
    elif axis == op.HW:
        axis = [-2, -1]
    elif axis == op.C:
        axis = -3
    elif axis == op.H:
        axis = -2
    elif axis == op.W:
        axis = -1
    else:
        msg = "Reduce axis value {} is not supported in frontend CoreML."
        raise tvm.error.OpAttributeUnImplemented(msg.format(axis))

    mode = op.mode
    if mode == op.SUM:
        return _op.sum(inexpr, axis=axis, keepdims=True)
    elif mode == op.AVG:
        return _op.mean(inexpr, axis=axis, keepdims=True)
    elif mode == op.PROD:
        return _op.prod(inexpr, axis=axis, keepdims=True)
    elif mode == op.MIN:
        return _op.min(inexpr, axis=axis, keepdims=True)
    elif mode == op.MAX:
        return _op.max(inexpr, axis=axis, keepdims=True)
    elif mode == op.ARGMAX:
        return _op.argmax(inexpr, axis=axis, keepdims=True)
    else:
        msg = "Reduce mode value {} is not supported in frontend CoreML."
        raise tvm.error.OpAttributeUnImplemented(msg.format(mode))


def _ReshapeLayerParams(op, inexpr, etab):
    return _op.reshape(inexpr, op.targetShape)


def _SplitLayerParams(op, inexpr, etab):
    return _op.split(inexpr, op.nOutputs, axis=-3)


_convert_map = {
    "NeuralNetworkMeanImage": _NeuralNetworkMeanImage,
    "NeuralNetworkImageScaler": _NeuralNetworkImageScaler,
    "ConvolutionLayerParams": _ConvolutionLayerParams,
    "BatchnormLayerParams": _BatchnormLayerParams,
    "ActivationParams": _ActivationParams,
    "ScaleLayerParams": _ScaleLayerParams,
    "PoolingLayerParams": _PoolingLayerParams,
    "SoftmaxLayerParams": _SoftmaxLayerParams,
    "InnerProductLayerParams": _InnerProductLayerParams,
    "AddLayerParams": _AddLayerParams,
    "MultiplyLayerParams": _MultiplyLayerParams,
    "FlattenLayerParams": _FlattenLayerParams,
    "ConcatLayerParams": _ConcatLayerParams,
    "PaddingLayerParams": _PaddingLayerParams,
    "PermuteLayerParams": _PermuteLayerParams,
    "UpsampleLayerParams": _UpsampleLayerParams,
    "L2NormalizeLayerParams": _L2NormalizeLayerParams,
    "LRNLayerParams": _LRNLayerParams,
    "AverageLayerParams": _AverageLayerParams,
    "MaxLayerParams": _MaxLayerParams,
    "MinLayerParams": _MinLayerParams,
    "UnaryFunctionLayerParams": _UnaryFunctionLayerParams,
    "ReduceLayerParams": _ReduceLayerParams,
    "ReshapeLayerParams": _ReshapeLayerParams,
    "SplitLayerParams": _SplitLayerParams,
}

# SAME padding: https://www.tensorflow.org/api_guides/python/nn
def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding

    Parameters
    ----------
    data:
        1D input data

    kernel:
        1D input kernel

    stride:
        1D input stride

    Returns
    -------
        pad tuple of value
    """

    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def coreml_op_to_relay(op, inname, outnames, etab):
    """Convert coreml layer to a Relay expression and update the expression table.

    Parameters
    ----------
    op: a coreml protobuf bit

    inname : str or list of str
        Name of the input Relay expression.

    outnames : str or list of str
        Name of the output Relay expression.

    etab : relay.frontend.common.ExprTable
        The global expression table to be updated.
    """
    classname = type(op).__name__
    if classname not in _convert_map:
        raise tvm.error.OpNotImplemented(
            "Operator {} is not supported in frontend CoreML.".format(classname)
        )
    if isinstance(inname, _base.string_types):
        insym = etab.get_expr(inname)
    else:
        insym = [etab.get_expr(i) for i in inname]
    outs = _convert_map[classname](op, insym, etab)

    if outnames:
        if isinstance(outnames, _base.string_types) or len(outnames) == 1:
            outname = outnames if isinstance(outnames, _base.string_types) else outnames[0]
            etab.set_expr(outname, outs, force_override=True)
        else:
            # the number of outputs from model op and tvm relay must be same
            assert len(outnames) == len(outs)
            for outname, out in zip(outnames, outs):
                etab.set_expr(outname, out, force_override=True)


def from_coreml(model, shape=None):
    """Convert from coreml model into Relay Function.

    Parameters
    ----------
    model:
        coremltools.models.MLModel of a NeuralNetworkClassifier

    shape : dict of str to int list/tuple, optional
        The input shapes

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by Relay.
    """
    try:
        import coremltools as cm
    except ImportError:
        raise ImportError("The coremltools package must be installed")

    assert isinstance(model, cm.models.MLModel)
    spec = model.get_spec()
    modeltype = spec.WhichOneof("Type")
    assert modeltype in ["neuralNetworkClassifier", "neuralNetwork", "neuralNetworkRegressor"]
    cc = getattr(spec, modeltype)

    etab = ExprTable()
    for i in spec.description.input:
        input_shape = list(shape[i.name]) if shape is not None and i.name in shape else None
        etab.set_expr(i.name, _expr.var(i.name, shape=input_shape))

    for pp in cc.preprocessing:
        whichpp = pp.WhichOneof("preprocessor")
        ppmethod = getattr(pp, whichpp)
        if whichpp == "scaler":
            # Be careful we maybe only preprocess one input when we have multi inputs
            # which is stored in pp.featureName. See unit testing verify_image_scaler
            # in test_forward.py for CoreML.
            for i in spec.description.input:
                # we have multi inputs
                if len(spec.description.input) > 1:
                    assert pp.featureName != ""
                    if i.name == pp.featureName:
                        coreml_op_to_relay(ppmethod, i.name, i.name, etab)
                else:
                    assert pp.featureName == ""
                    coreml_op_to_relay(ppmethod, i.name, i.name, etab)
        else:
            coreml_op_to_relay(ppmethod, pp.featureName, pp.featureName, etab)

    for l in cc.layers:
        layertype = l.WhichOneof("layer")
        layerop = getattr(l, layertype)
        if len(l.input) == 1:
            coreml_op_to_relay(layerop, l.input[0], l.output, etab)
        else:
            coreml_op_to_relay(layerop, list(l.input), l.output, etab)

    outexpr = [
        etab.get_expr(o.name) if o.name in etab.exprs else _expr.var(o.name)
        for o in spec.description.output
    ]

    # check there are multiple outputs in the model and all are there in etab
    multi_out = all([bool(o.name in etab.exprs) for o in spec.description.output])
    outexpr = _expr.Tuple(outexpr) if multi_out else outexpr[0]

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    params = {k: _nd.array(np.array(v, dtype=np.float32)) for k, v in etab.params.items()}
    return IRModule.from_expr(func), params
