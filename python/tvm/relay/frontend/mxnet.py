# pylint: disable=invalid-name, import-self, len-as-condition
"""MXNet symbol frontend."""
from __future__ import absolute_import as _abs

import json
from .. import ir_pass
from .. import expr as _expr
from .. import op as _op
from ... import nd as _nd
from .common import StrAttrsDict
from .nnvm_common import _rename, _binop_scalar, _rbinop_scalar, _reduce
from .nnvm_common import _arg_reduce, _init_op, _softmax_op, _cast
from .nnvm_common import _clip, _transpose, _upsampling
from .nnvm_common import _elemwise_sum, _reshape
from .nnvm_common import _warn_not_used

__all__ = ['from_mxnet']

def _mx_fully_connected(inputs, attrs):
    import mxnet as mx
    units = attrs.get_int("num_hidden")
    use_bias = not attrs.get_bool("no_bias", False)
    try:
        _ = mx.sym.FullyConnected(mx.sym.var("x"), num_hidden=1, flatten=True)
        has_flatten = True
    except mx.base.MXNetError:
        # no flatten attribute in old mxnet
        has_flatten = False
    use_flatten = attrs.get_bool("flatten", True)
    if has_flatten and use_flatten:
        inputs[0] = _op.nn.batch_flatten(inputs[0])
    res = _op.nn.dense(inputs[0], inputs[1], units=units)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2])
    return res


def _get_channel_axis(layout, op_name):
    if layout == "NCHW":
        return 1
    if layout == "NHWC":
        return 3
    raise RuntimeError("layout: {} is not supported in {}".format(layout, op_name))


def _mx_activations(inputs, attrs):
    act_type = attrs.get_str("act_type")
    assert len(inputs) == 1
    if act_type == "sigmoid":
        return _op.sigmoid(inputs[0])
    if act_type == "tanh":
        return _op.tanh(inputs[0])
    if act_type == "relu":
        return _op.nn.relu(inputs[0])
    if act_type == "softrelu":
        def _stable_softrelu(x):
            # log(1 + exp(-abs(x))) + relu(x)
            one = _expr.const(1, dtype="float32")
            exp_neg_abs_x = _op.exp(_op.negative(_op.abs(x)))
            return _op.add(_op.log(_op.add(one, exp_neg_abs_x)),
                           _op.nn.relu(x))
        return _stable_softrelu(inputs[0])
    raise RuntimeError("Do not support act_type: {}".format(act_type))


def _mx_conv2d(inputs, attrs):
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise RuntimeError("non-2d kernel is not supported in conv2d")
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv2d(inputs[0], inputs[1], **new_attrs)
    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_conv2d_transpose(inputs, attrs):
    if "target_shape" in attrs.attrs:
        raise RuntimeError("target_shape is not supported in conv2d_transpose")
    kernel_size = attrs.get_int_tuple("kernel")
    if len(kernel_size) != 2:
        raise RuntimeError("non-2d kernel is not supported in conv2d")
    data_layout = attrs.get_str("layout", "NCHW")
    channel_axis = _get_channel_axis(data_layout, "conv2d_transpose")

    if "kernel_layout" in attrs.attrs:
        kernel_layout = attrs.get_str("kernel_layout")
    else:
        kernel_layout = "HWIO" if data_layout == "NHWC" else "OIHW"

    new_attrs = {}
    new_attrs["channels"] = attrs.get_int("num_filter")
    new_attrs["kernel_size"] = kernel_size
    new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
    new_attrs["output_padding"] = attrs.get_int_tuple("adj", (0, 0))
    new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
    new_attrs["dilation"] = attrs.get_int_tuple("dilate", (1, 1))
    new_attrs["groups"] = attrs.get_int("num_group", 1)
    new_attrs["data_layout"] = data_layout
    new_attrs["kernel_layout"] = kernel_layout
    use_bias = not attrs.get_bool("no_bias", False)
    res = _op.nn.conv2d_transpose(inputs[0], inputs[1], **new_attrs)

    if use_bias:
        assert len(inputs) == 3
        res = _op.nn.bias_add(res, inputs[2], axis=channel_axis)
    return res


def _mx_pooling(inputs, attrs):
    global_pool = attrs.get_bool("global_pool", False)
    pool_type = attrs.get_str("pool_type")

    def _pool2d(new_op, is_avg):
        kernel_size = attrs.get_int_tuple("kernel")
        if len(kernel_size) != 2:
            raise RuntimeError("non-2d kernel is not supported in pool2d")
        new_attrs = {}
        new_attrs["pool_size"] = kernel_size
        new_attrs["strides"] = attrs.get_int_tuple("stride", (1, 1))
        new_attrs["padding"] = attrs.get_int_tuple("pad", (0, 0))
        new_attrs["ceil_mode"] = (attrs.get_str("pooling_convention", "valid") == "full")
        if is_avg:
            new_attrs["count_include_pad"] = attrs.get_bool("count_include_pad", True)
        return new_op(inputs[0], **new_attrs)

    if pool_type == "max":
        if global_pool:
            return _op.nn.global_max_pool2d(inputs[0])
        return _pool2d(_op.nn.max_pool2d, False)
    if pool_type == "avg":
        if global_pool:
            return _op.nn.global_avg_pool2d(inputs[0])
        return _pool2d(_op.nn.avg_pool2d, True)
    raise RuntimeError("Do not support pool_type:{}".format(pool_type))


def _mx_dropout(inputs, attrs):
    rate = attrs.get_float("p", 0.5)
    return _op.nn.dropout(inputs[0], rate=rate)


def _mx_batch_norm(inputs, attrs):
    if attrs.get_bool("output_mean_var", False):
        raise RuntimeError("batch_norm do not support output_mean_var")
    if attrs.get_bool("use_global_stats", False):
        _warn_not_used("use_global_stats", "batch_norm")
    new_attrs = {}
    new_attrs["axis"] = attrs.get_int("axis", 1)
    new_attrs["epsilon"] = attrs.get_float("eps", 0.001)
    new_attrs["center"] = True
    new_attrs["scale"] = not attrs.get_bool("fix_gamma", False)
    return _op.nn.batch_norm(*inputs, **new_attrs)


def _mx_slice(inputs, attrs):
    new_attrs = {}
    begin = attrs.get_int_tuple('begin', None)
    end = attrs.get_int_tuple('end', None)
    stride = attrs.get_int_tuple('step', None)
    if begin is None or end is None:
        raise RuntimeError("begin and end are required parameters.")
    if None in begin or None in end:
        raise RuntimeError("None in begin or end is not supported yet.")
    new_attrs = {'begin': begin, 'end': end}
    if stride is not None:
        new_attrs['strides'] = stride
    return _op.strided_slice(inputs[0], **new_attrs)


def _mx_split(inputs, attrs):
    axis = attrs.get_int("axis", 1)
    new_attrs = {}
    new_attrs["indices_or_sections"] = attrs.get_int("num_outputs")
    new_attrs["axis"] = axis
    res = _op.split(inputs[0], **new_attrs)
    if attrs.get_bool("squeeze_axis", False):
        return tuple([_op.squeeze(x, axis=[axis]) for x in res])
    return res


def _mx_softmax_activation(inputs, attrs):
    mode = attrs.get_str("mode", "instance")
    axis = 0 if mode == "instance" else 1
    return _op.nn.softmax(inputs[0], axis=axis)


def _mx_softmax_output(inputs, attrs):
    if attrs.get_bool("multi_output", False):
        return _op.nn.softmax(inputs[0], axis=1)
    return _op.nn.softmax(inputs[0])


def _mx_concat(inputs, attrs):
    axis = attrs.get_int("dim", 1)
    return _op.concatenate(tuple(inputs), axis=axis)


def _mx_expand_dims(inputs, attrs):
    axis = attrs.get_int("axis")
    return _op.expand_dims(inputs[0], axis=axis)


def _mx_leaky_relu(inputs, attrs):
    act_type = attrs.get_str("act_type")
    if act_type == "leaky":
        return _op.nn.leaky_relu(inputs[0], alpha=attrs.get_float("slope", 0.25))
    if act_type == "prelu":
        assert len(inputs) == 2
        return _op.nn.prelu(*inputs)
    if act_type == "elu":
        # -slope * relu(1-exp(x)) + relu(x)
        slope = attrs.get_float("slope", 0.25)
        one = _expr.const(1, dtype="float32")
        x = inputs[0]
        mslope = _op.nn.relu(_op.subtract(one, _op.exp(x)))
        mslope = _op.multiply(mslope, _expr.const(-slope, dtype="float32"))
        return _op.add(mslope, _op.nn.relu(x))
    if act_type == "rrelu":
        # NOTE this is only converted for inference.
        lower_bound = attrs.get_float("lower_bound")
        upper_bound = attrs.get_float("upper_bound")
        alpha = (lower_bound + upper_bound) / 2.0
        return _op.nn.leaky_relu(inputs[0], alpha=alpha)
    raise RuntimeError("act_type: {} is not supported".format(act_type))


def _mx_lrn(inputs, attrs):
    new_attrs = {}
    new_attrs["alpha"] = attrs.get_float("alpha", 0.0001)
    new_attrs["beta"] = attrs.get_float("beta", 0.75)
    new_attrs["bias"] = attrs.get_float("knorm", 2)
    # NCHW format and normalization along channel axis
    new_attrs["axis"] = 1
    new_attrs["size"] = attrs.get_int("nsize")
    assert len(inputs) == 1
    return _op.nn.lrn(inputs[0], **new_attrs)


def _mx_multibox_prior(inputs, attrs):
    new_attrs = {}
    new_attrs["sizes"] = attrs.get_float_tuple("sizes", (1.0, ))
    new_attrs["steps"] = attrs.get_float_tuple("steps", (-1.0, -1.0))
    new_attrs["offsets"] = attrs.get_float_tuple("offsets", (0.5, 0.5))
    new_attrs["ratios"] = attrs.get_float_tuple("ratios", (1.0, ))
    new_attrs["clip"] = attrs.get_bool("clip", False)
    return _op.vision.multibox_prior(inputs[0], **new_attrs)


def _mx_multibox_detection(inputs, attrs):
    new_attrs0 = {}
    new_attrs0["clip"] = attrs.get_bool("clip", True)
    new_attrs0["threshold"] = attrs.get_float("threshold", 0.01)
    new_attrs0["variances"] = attrs.get_float_tuple("variances", (0.1, 0.1,
                                                                  0.2, 0.2))

    new_attrs1 = {}
    new_attrs1["overlap_threshold"] = attrs.get_float("nms_threshold", 0.5)
    new_attrs1["force_suppress"] = attrs.get_bool("force_suppress", False)
    new_attrs1["topk"] = attrs.get_int("nms_topk", -1)

    ret = _op.vision.multibox_transform_loc(inputs[0], inputs[1],
                                            inputs[2], **new_attrs0)
    return _op.vision.nms(ret[0], ret[1], **new_attrs1)


def _mx_arange(inputs, attrs):
    assert len(inputs) == 0
    if attrs.get_int("repeat", 1) != 1:
        raise RuntimeError("arange doesn't support repeat")
    new_attrs = {}
    new_attrs["start"] = attrs.get_float("start", 0)
    new_attrs["stop"] = attrs.get_float("stop")
    new_attrs["step"] = attrs.get_float("step", 1)
    new_attrs["dtype"] = attrs.get_str("dtype", "float32")
    return _op.arange(**new_attrs)


def _mx_roi_align(inputs, attrs):
    new_attrs = {}
    new_attrs["pooled_size"] = attrs.get_int_tuple("pooled_size")
    new_attrs["spatial_scale"] = attrs.get_float("spatial_scale")
    new_attrs["sample_ratio"] = attrs.get_int("sample_ratio", -1)
    new_attrs["layout"] = "NCHW"
    return _op.vision.roi_align(inputs[0], inputs[1], **new_attrs)


# Note: due to attribute conversion constraint
# ops in the identity set must be attribute free
_identity_list = [
    "log",
    "exp",
    "sigmoid",
    "tanh",
    "exp",
    "negative",
    "reshape_like",
    "slice_like",
    "zeros_like",
    "ones_like",
    "where",
]

_convert_map = {
    "_copy"         : _rename(_op.copy),
    "relu"          : _rename(_op.nn.relu),
    "broadcast_add" : _rename(_op.add),
    "broadcast_sub" : _rename(_op.subtract),
    "broadcast_mul" : _rename(_op.multiply),
    "broadcast_div" : _rename(_op.divide),
    "elemwise_add"  : _rename(_op.add),
    "elemwise_sub"  : _rename(_op.subtract),
    "elemwise_mul"  : _rename(_op.multiply),
    "elemwise_div"  : _rename(_op.divide),
    "flatten"       : _rename(_op.nn.batch_flatten),
    "Flatten"       : _rename(_op.nn.batch_flatten),
    "_plus_scalar"  : _binop_scalar(_op.add),
    "__add_scalar__": _binop_scalar(_op.add),
    "__sub_scalar__": _binop_scalar(_op.subtract),
    "_minus_scalar" : _binop_scalar(_op.subtract),
    "__mul_scalar__": _binop_scalar(_op.multiply),
    "_mul_scalar"   : _binop_scalar(_op.multiply),
    "__div_scalar__": _binop_scalar(_op.divide),
    "_div_scalar"   : _binop_scalar(_op.divide),
    "__pow_scalar__": _binop_scalar(_op.power),
    "_rminus_scalar": _rbinop_scalar(_op.subtract),
    "__rsub_scalar__": _rbinop_scalar(_op.subtract),
    "_rdiv_scalar"  : _rbinop_scalar(_op.divide),
    "__rdiv_scalar__"  : _rbinop_scalar(_op.divide),
    "__rpow_scalar__": _rbinop_scalar(_op.power),
    # reduction ops
    "max"           : _reduce(_op.max),
    "min"           : _reduce(_op.min),
    "sum"           : _reduce(_op.sum),
    "max_axis"      : _reduce(_op.max),
    "min_axis"      : _reduce(_op.min),
    "sum_axis"      : _reduce(_op.sum),
    "argmax"        : _arg_reduce(_op.argmax),
    "argmin"        : _arg_reduce(_op.argmin),
    # init ops
    "_ones"         : _init_op(_op.ones),
    "_zeros"        : _init_op(_op.zeros),
    # softmax
    "softmax"       : _softmax_op(_op.nn.softmax),
    "log_softmax"   : _softmax_op(_op.nn.log_softmax),
    "Softmax"       : _softmax_op(_op.nn.softmax),
    # per op specialization
    "Reshape"       : _reshape,
    "reshape"       : _reshape,
    "Cast"          : _cast,
    "clip"          : _clip,
    "transpose"     : _transpose,
    "UpSampling"    : _upsampling,
    "add_n"         : _elemwise_sum,
    # MXNet specific implementations
    "FullyConnected": _mx_fully_connected,
    "Activation"    : _mx_activations,
    "Convolution"   : _mx_conv2d,
    "Convolution_v1": _mx_conv2d,
    "Deconvolution" : _mx_conv2d_transpose,
    "Pooling"       : _mx_pooling,
    "Pooling_v1"    : _mx_pooling,
    "Dropout"       : _mx_dropout,
    "BatchNorm"     : _mx_batch_norm,
    "BatchNorm_v1"  : _mx_batch_norm,
    "LRN"           : _mx_lrn,
    "slice"         : _mx_slice,
    "SliceChannel"  : _mx_split,
    "split"         : _mx_split,
    "expand_dims"   : _mx_expand_dims,
    "Concat"        : _mx_concat,
    "concat"        : _mx_concat,
    "LeakyReLU"     : _mx_leaky_relu,
    "_arange"       : _mx_arange,
    "SoftmaxOutput" : _mx_softmax_output,
    "SoftmaxActivation" : _mx_softmax_activation,
    # vision
    "_contrib_MultiBoxPrior" : _mx_multibox_prior,
    "_contrib_MultiBoxDetection" : _mx_multibox_detection,
    "_contrib_ROIAlign" : _mx_roi_align,
    # List of missing operators that are present in NNVMv1
    # TODO(tvm-tvm): support all operators.
    #
    # "broadcast_to",
    # "gather_nd",
    # "Crop"          : _crop_like,

}

# set identity list
_convert_map.update({k : _rename(k) for k in _identity_list})


def _from_mxnet_impl(symbol, shape_dict, dtype_info):
    """Convert mxnet symbol to compatible relay Function.

    Reconstruct a relay Function by traversing the mxnet symbol.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol
        Incompatible symbol from mxnet.
        The op_name and attrs inside are not always compatible.

    shape_dict : dict
        Known parameter shapes

    dtype_info : dict or str.
        Known parameter dtypes

    Returns:
    -------
    func : tvm.relay.Function
        Converted relay Function
    """
    assert symbol is not None
    jgraph = json.loads(symbol.tojson())
    jnodes = jgraph["nodes"]
    node_map = {}

    for nid, node in enumerate(jnodes):
        children = [node_map[e[0]][e[1]] for e in node["inputs"]]
        attrs = StrAttrsDict(node.get("attrs", {}))
        node_name = node["name"]
        op_name = node["op"]
        if op_name == "null":
            shape = shape_dict[node_name] if node_name in shape_dict else None
            if isinstance(dtype_info, dict):
                dtype = dtype_info[node_name] if node_name in dtype_info else "float32"
            else:
                dtype = dtype_info
            node_map[nid] = [_expr.var(node_name, shape=shape, dtype=dtype)]
        elif op_name in _convert_map:
            res = _convert_map[op_name](children, attrs)
            if isinstance(res, (_expr.TupleWrapper, tuple, list)):
                pass
            elif isinstance(res, _expr.Expr):
                res = [res]
            else:
                raise RuntimeError("unexpected type %s" % type(res))
            node_map[nid] = res
        else:
            raise RuntimeError("{} is not supported in relay frontend".format(op_name))

    outputs = [node_map[e[0]][e[1]] for e in jgraph["heads"]]
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    func = _expr.Function(ir_pass.free_vars(outputs), outputs)
    return func


def _update_shape_dtype(shape, dtype, params):
    """Update shape dtype given params information"""
    shape = {} if shape is None else shape
    if not params:
        return shape, dtype
    shape = shape.copy()
    shape.update({k : v.shape for k, v in params.items()})
    if isinstance(dtype, str):
        for k, v in params.items():
            if v.dtype != dtype:
                raise ValueError(
                    "%s: dtype not expected %s vs %s" % (k, dtype, v.dtype))
    else:
        dtype = dtype.copy()
        dtype.update({k : str(v.dtype) for k, v in params.items()})
    return shape, dtype


def from_mxnet(symbol,
               shape=None,
               dtype="float32",
               arg_params=None,
               aux_params=None):
    """Convert from MXNet"s model into compatible relay Function.

    Parameters
    ----------
    symbol : mxnet.Symbol or mxnet.gluon.HybridBlock
        MXNet symbol.

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    arg_params : dict of str to mx.NDArray
        The argument parameters in mxnet

    aux_params : dict of str to mx.NDArray
        The auxiliary parameters in mxnet

    Returns
    -------
    sym : tvm.relay.Function
        Compatible relay Function

    params : dict of str to tvm.NDArray
        The parameter dict to be used by nnvm
    """
    try:
        import mxnet as mx
    except ImportError as e:
        raise ImportError("{}. MXNet is required to parse symbols.".format(e))

    if isinstance(symbol, mx.sym.Symbol):
        params = {}
        arg_params = arg_params if arg_params else {}
        aux_params = aux_params if aux_params else {}
        for k, v in arg_params.items():
            params[k] = _nd.array(v.asnumpy())
        for k, v in aux_params.items():
            params[k] = _nd.array(v.asnumpy())
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        sym = _from_mxnet_impl(symbol, shape, dtype)
    elif isinstance(symbol, mx.gluon.HybridBlock):
        if arg_params is not None or aux_params is not None:
            raise ValueError("arg_params and aux_params ae not used when importing HybridBlock")
        params = {}
        for k, v in symbol.collect_params().items():
            params[k] = _nd.array(v.data().asnumpy())
        data = mx.sym.Variable("data")
        sym = symbol(data)
        shape, dtype = _update_shape_dtype(shape, dtype, params)
        sym = _from_mxnet_impl(sym, shape, dtype)
    elif isinstance(symbol, mx.gluon.Block):
        raise NotImplementedError("Only Hybrid Blocks are supported now.")
    else:
        msg = "mxnet.Symbol or gluon.HybridBlock expected, got {}".format(type(symbol))
        raise ValueError(msg)
    return sym, params
