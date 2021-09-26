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
"""Paddle: PArallel Distributed Deep LEarning."""

import numpy as np

import tvm
from tvm.ir import IRModule

from .. import analysis
from .. import ty as _ty
from .. import expr as _expr
from .. import function as _function
from .. import ty as _ty
from .. import op as _op
from .common import (
    fold_constant,
    get_relay_op,
    infer_shape,
    infer_type,
    infer_value,
    new_var,
)

__all__ = ["from_paddle"]


def _get_pad_size(in_size, dilated_kernel_size, stride_size):
    """calculate the paddings size"""

    if stride_size == 1 or in_size % stride_size == 0:
        pad = max(dilated_kernel_size - stride_size, 0)
    else:
        pad = max(dilated_kernel_size - (in_size % stride_size), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]


def _dtype_shape_promotion(inputs):
    """promote data type and shape for list of tensors."""

    dtype_order = ["bool", "int8", "int16", "int32", "int64", "float32", "float64"]

    ranks = [len(infer_shape(x)) for x in inputs]
    if set(ranks) == set([1, 0]):
        for i, r in enumerate(ranks):
            if r == 0:
                inputs[i] = _op.expand_dims(inputs[i], axis=0)

    dtypes = set(dtype_order.index(infer_type(x).checked_type.dtype) for x in inputs)
    if len(dtypes) == 1:
        return inputs
    max_dtype = dtype_order[max(dtypes)]
    for i, input_op in enumerate(inputs):
        if infer_type(input_op).checked_type.dtype != max_dtype:
            inputs[i] = input_op.astype(max_dtype)
    return inputs


def shape_of(x, dtype="int32"):
    """Get shape of a tensor"""

    ttype = infer_type(x).checked_type
    if not _ty.is_dynamic(ttype):
        shape = list(ttype.shape)
        return _expr.const(np.array(shape), dtype)
    return _op.shape_of(x, dtype)


def _infer_value(x, params):
    """Try running infer_value, and if successful, return the inferred value.
    Otherwise, return input"""

    try:
        value = infer_value(x, params)
        return value.numpy().tolist()
    except Exception:  # pylint: disable=broad-except
        return x


def _convert_dtype_value(val):
    """converts a Paddle type id to a string."""

    convert_dtype_map = {
        21: "int8",
        20: "uint8",
        6: "float64",
        5: "float32",
        4: "float16",
        3: "int64",
        2: "int32",
        1: "int16",
        0: "bool",
    }
    if val not in convert_dtype_map:
        msg = "Paddle data type value %d is not handled yet." % (val)
        raise NotImplementedError(msg)
    return convert_dtype_map[val]


def convert_unary_op(g, op, block):
    """Operator converter for all the activation."""

    op_map = {
        "isinf_v2": _op.isinf,
        "isfinite_v2": _op.isfinite,
        "isnan_v2": _op.isnan,
    }
    if op.type in op_map:
        unary_func = op_map[op.type]
    else:
        unary_func = get_relay_op(op.type)
    out = unary_func(g.get_node(op.input("X")[0]))
    g.add_node(op.output("Out")[0], out)


def convert_addmm(g, op, block):
    """Operator converter for addmm."""

    input_x = g.get_node(op.input("Input")[0])
    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Y")[0])

    alpha = op.attr("Alpha")
    beta = op.attr("Beta")
    dtype = block.var(op.output("Out")[0]).dtype
    dtype = str(dtype).strip().split(".")[1]

    if not isinstance(alpha, _expr.Expr) and alpha != 1:
        alpha = _expr.const(alpha, dtype)
        x *= alpha

    if not isinstance(beta, _expr.Expr) and beta != 1:
        beta = _expr.const(beta, dtype)
        input_x *= beta

    transposed_y = _op.transpose(y, axes=[1, 0])
    dense_out = _op.nn.dense(x, transposed_y)
    out = dense_out + input_x
    g.add_node(op.output("Out")[0], out)


def convert_arg_max(g, op, block):
    """Operator converter for arg_max."""

    axis = op.attr("axis")
    keepdims = op.attr("keepdims")
    flatten = op.attr("flatten")
    dtype = op.attr("dtype")
    dtype = _convert_dtype_value(dtype)

    x = g.get_node(op.input("X")[0])
    if axis is None or flatten:
        x = _op.reshape(x, [-1])
        out = _op.argmax(x, axis=None, keepdims=True)
    else:
        out = _op.argmax(x, axis=axis, keepdims=keepdims)
    if dtype != infer_type(out).checked_type.dtype:
        out = _op.cast(out, dtype)
    g.add_node(op.output("Out")[0], out)


def convert_arg_min(g, op, block):
    """Operator converter for arg_min."""

    axis = op.attr("axis")
    keepdims = op.attr("keepdims")
    flatten = op.attr("flatten")
    dtype = op.attr("dtype")
    dtype = _convert_dtype_value(dtype)

    x = g.get_node(op.input("X")[0])
    if axis is None or flatten:
        x = _op.reshape(x, [-1])
        out = _op.argmin(x, axis=None, keepdims=True)
    else:
        out = _op.argmin(x, axis=axis, keepdims=keepdims)
    if dtype != infer_type(out).checked_type.dtype:
        out = _op.cast(out, dtype)
    g.add_node(op.output("Out")[0], out)


def convert_argsort(g, op, block):
    """Operator converter for argsort."""

    x = g.get_node(op.input("X")[0])
    axis = op.attr("axis")
    descending = op.attr("descending")

    out = _op.sort(x, axis, not descending)
    out_indice = _op.argsort(x, axis, not descending, dtype="int64")
    g.add_node(op.output("Out")[0], out)
    g.add_node(op.output("Indices")[0], out_indice)


def convert_assign(g, op, block):
    """Operator converter for assign."""

    out = g.get_node(op.input("X")[0])
    g.add_node(op.output("Out")[0], out)


def convert_assign_value(g, op, block):
    """Operator converter for assign_value."""

    keys = ["bool_values", "fp32_values", "int32_values", "int64_values"]
    dtypes = ["bool", "float32", "int32", "int64"]
    for i, key in enumerate(keys):
        dtype = dtypes[i]
        value = np.array(op.attr(key)).astype(dtype)
        if value is not None and value.size >= 1:
            break
    shape = op.attr("shape")
    value = value.reshape(shape)
    out = _op.const(value, dtype=dtype)
    g.add_node(op.output("Out")[0], out)


def convert_batch_norm(g, op, block):
    """Operator converter for batch_norm."""

    ipt_name = op.input("X")[0]
    scale_name = op.input("Scale")[0]
    bias_name = op.input("Bias")[0]
    mean_name = op.input("Mean")[0]
    variance_name = op.input("Variance")[0]
    epsilon = op.attr("epsilon")
    out = _op.nn.batch_norm(
        g.get_node(ipt_name),
        g.get_node(scale_name),
        g.get_node(bias_name),
        g.get_node(mean_name),
        g.get_node(variance_name),
        epsilon=epsilon,
    )
    g.add_node(op.output("Y")[0], out[0])


def convert_bmm(g, op, block):
    """Operator converter for bmm."""

    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Y")[0])
    y = _op.transpose(y, [0, 2, 1])
    out = _op.nn.batch_matmul(x, y)
    g.add_node(op.output("Out")[0], out)


def convert_cast(g, op, block):
    """Operator converter for cast."""

    dtype = op.attr("out_dtype")
    dtype = _convert_dtype_value(dtype)
    x = g.get_node(op.input("X")[0])
    out = _op.cast(x, dtype=dtype)
    g.add_node(op.output("Out")[0], out)


def convert_clip(g, op, block):
    """Operator converter for clip."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    is_dynamic = False
    if op.input("Min"):
        min_value = g.get_node(op.input("Min")[0])
        min_value = _infer_value(min_value, g.get_params())
        if isinstance(min_value, _expr.Expr):
            is_dynamic = True
        else:
            min_value = min_value[0]
    else:
        min_value = op.attr("min")
    if op.input("Max"):
        max_value = g.get_node(op.input("Max")[0])
        max_value = _infer_value(max_value, g.get_params())
        if isinstance(max_value, _expr.Expr):
            if not is_dynamic:
                is_dynamic = True
                min_value = _op.const(min_value, dtype)
        else:
            max_value = max_value[0]
            if is_dynamic:
                max_value = _op.const(max_value, dtype)
    else:
        max_value = op.attr("max")
        if is_dynamic:
            max_value = _op.const(max_value, dtype)

    if not is_dynamic:
        out = _op.clip(x, min_value, max_value)
    else:
        out = _op.maximum(x, min_value)
        out = _op.minimum(out, max_value)
    g.add_node(op.output("Out")[0], out)


def convert_concat(g, op, block):
    """Operator converter for concat."""

    inputs = [g.get_node(op.input("X")[i]) for i in range(len(op.input("X")))]
    axis = op.attr("axis")
    inputs = _dtype_shape_promotion(inputs)
    out = _op.concatenate(inputs, axis=axis)
    g.add_node(op.output("Out")[0], out)


def convert_conv2d(g, op, block):
    """Operator converter for conv2d."""

    dilations = op.attr("dilations")
    groups = op.attr("groups")
    paddings = op.attr("paddings")
    padding_algorithm = op.attr("padding_algorithm")
    strides = op.attr("strides")

    kernel = g.get_node(op.input("Filter")[0])
    input_x = g.get_node(op.input("Input")[0])
    out_channels, _, k_h, k_w = infer_shape(kernel)
    if padding_algorithm == "VALID":
        paddings = [0, 0]
    elif padding_algorithm == "SAME":
        if strides[0] == 1 and strides[1] == 1:
            pad_h = _get_pad_size(0, (k_h - 1) * dilations[0] + 1, strides[0])
            pad_w = _get_pad_size(0, (k_w - 1) * dilations[1] + 1, strides[1])
        else:
            input_shape = shape_of(input_x)
            h_w = _op.strided_slice(input_shape, [2], [4])
            try:
                in_h, in_w = infer_value(h_w, g.get_params()).numpy().tolist()
            except Exception as e:
                msg = "Dynamic shape is not supported in SAME padding algorithm while stride!=1"
                raise tvm.error.OpAttributeInvalid(msg) from e
            pad_h = _get_pad_size(in_h, (k_h - 1) * dilations[0] + 1, strides[0])
            pad_w = _get_pad_size(in_w, (k_w - 1) * dilations[1] + 1, strides[1])
        paddings = [pad_h[0], pad_w[0], pad_h[1], pad_w[1]]
    elif padding_algorithm == "EXPLICIT":
        if len(paddings) == 2:
            paddings = [paddings[0], paddings[1], paddings[0], paddings[1]]
        if len(paddings) == 4:
            paddings = [paddings[0], paddings[2], paddings[1], paddings[3]]
    else:
        msg = 'Value {} in attribute "padding" of operator Conv is not "valid."'
        raise tvm.error.OpAttributeInvalid(msg.format(padding_algorithm))

    out = _op.nn.conv2d(
        input_x,
        kernel,
        strides=strides,
        padding=paddings,
        dilation=dilations,
        groups=groups,
        channels=out_channels,
        kernel_size=[k_h, k_w],
    )
    g.add_node(op.output("Output")[0], out)


def convert_crop(g, op, block):
    """Operator converter for crop."""

    x = g.get_node(op.input("X")[0])
    dims = len(infer_shape(x))
    input_shape = op.input("Shape")
    input_offsets = op.input("Offsets")
    if input_shape:
        shape = g.get_node(input_shape[0])
        shape = _infer_value(shape, g.get_params())
    else:
        shape = op.attr("shape")

    if input_offsets:
        offsets = g.get_node(input_offsets[0])
        offsets = _infer_value(offsets, g.get_params())
    else:
        offsets = op.attr("offsets")

    if not isinstance(shape, _expr.Expr):
        shape = _op.const(shape, "int32")
    if not isinstance(offsets, _expr.Expr):
        offsets = _op.const(offsets, "int32")
    slice_start = offsets
    slice_end = _op.add(shape, offsets)
    strides = _op.const([1] * dims, dtype="int32")

    out = _op.strided_slice(x, slice_start, slice_end, strides)
    g.add_node(op.output("Out")[0], out)


def convert_cumsum(g, op, block):
    """Operator converter for cumsum."""

    axis = op.attr("axis")
    exclusive = op.attr("exclusive")
    flatten = op.attr("flatten")
    reverse = op.attr("reverse")

    x = g.get_node(op.input("X")[0])
    if axis is None or flatten:
        x = _op.reshape(x, [-1])
    if reverse:
        x = _op.reverse(x, axis=axis)
        out = _op.cumsum(x, axis=axis, exclusive=exclusive)
        out = _op.reverse(out, axis=axis)
    else:
        out = _op.cumsum(x, axis=axis, exclusive=exclusive)
    g.add_node(op.output("Out")[0], out)


def convert_dropout(g, op, block):
    """Operator converter for dropout."""

    x = g.get_node(op.input("X")[0])
    g.add_node(op.output("Out")[0], x)


def convert_elu(g, op, block):
    """Operator converter for elu."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    alpha = op.attr("alpha")
    alpha = _expr.const(-1.0 * alpha, dtype=dtype)
    out = alpha * _op.nn.relu(_expr.const(1, dtype=dtype) - _op.exp(x)) + _op.nn.relu(x)
    g.add_node(op.output("Out")[0], out)


def convert_dist(g, op, block):
    """Operator converter for dist."""

    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Y")[0])
    dtype = infer_type(x).checked_type.dtype
    p = op.attr("p")

    x -= y
    if p == np.inf:
        out = _op.reduce.max(_op.abs(x))
    elif p == np.NINF:
        out = _op.reduce.min(_op.abs(x))
    else:
        reci_order = _expr.const(1.0 / p, dtype=dtype)
        p = _expr.const(p)
        out = _op.power(
            _op.reduce.sum(_op.power(_op.abs(x), p)),
            reci_order,
        )
    out = _op.expand_dims(out, axis=0)
    g.add_node(op.output("Out")[0], out)


def convert_dot(g, op, block):
    """Operator converter for dot."""

    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Y")[0])

    out = _op.sum(_op.multiply(x, y), axis=[-1], keepdims=True)
    g.add_node(op.output("Out")[0], out)


def convert_elementwise_op(g, op, block):
    """Operator converter for all the elementwise operators."""

    op_map = {
        "elementwise_div": "divide",
        "elementwise_add": "add",
        "elementwise_mul": "multiply",
        "elementwise_sub": "subtract",
        "elementwise_mod": "mod",
        "elementwise_max": "maximum",
        "elementwise_min": "minimum",
        "elementwise_pow": "power",
        "elementwise_floordiv": "floor_divide",
        "floor_mod": "floor_mod",
        "equal": "equal",
        "greater_equal": "greater_equal",
        "greater_than": "greater",
        "less_equal": "less_equal",
        "less_than": "less",
        "not_equal": "not_equal",
    }
    op_func = op_map[op.type]
    ipt0 = g.get_node(op.input("X")[0])
    ipt1 = g.get_node(op.input("Y")[0])
    ipt0_shape = infer_shape(ipt0)
    ipt1_shape = infer_shape(ipt1)
    axis = op.attr("axis")
    if len(ipt0_shape) != len(ipt1_shape):
        if axis < 0:
            axis = axis + len(ipt0_shape)
        if axis != len(ipt0_shape) - 1:
            ipt1 = _op.expand_dims(ipt1, axis=axis, num_newaxis=(len(ipt0_shape) - axis - 1))
    op_func = get_relay_op(op_func)
    out = op_func(ipt0, ipt1)
    g.add_node(op.output("Out")[0], out)


def convert_expand(g, op, block):
    """Operator converter for expand."""

    x = g.get_node(op.input("X")[0])
    if op.input("Shape"):
        sizes = g.get_node(op.input("Shape")[0])
        sizes = _infer_value(sizes, g.get_params())
    else:
        sizes = op.attr("shape")

    out = _op.broadcast_to(x, sizes)
    g.add_node(op.output("Out")[0], out)


def convert_expand_as(g, op, block):
    """Operator converter for expand_as."""

    x = g.get_node(op.input("X")[0])
    target_shape = op.attr("target_shape")
    out = _op.broadcast_to(x, target_shape)
    g.add_node(op.output("Out")[0], out)


def convert_feed(g, op, block):
    """Converter for model input node."""

    if block is not None:
        ipt_name = op.output("Out")[0]
        dtype = op.attr("dtype")
        dtype = _convert_dtype_value(dtype)
    else:
        ipt_shape = op.shape
        ipt_dtype = str(op.dtype).strip().split(".")[1]
        ipt_name = op.name
    if g.shape_dict is not None:
        ipt_shape = g.shape_dict[ipt_name]

    if isinstance(ipt_shape, tuple):
        ipt_shape = list(ipt_shape)
    for i, s in enumerate(ipt_shape):
        if s < 0:
            ipt_shape[i] = _ty.Any()
    out = new_var(ipt_name, shape=ipt_shape, dtype=ipt_dtype)
    g.add_node(ipt_name, out)


def convert_fill_any_like(g, op, block):
    """Operator converter for fill_any_like."""

    dtype = op.attr("dtype")
    dtype = _convert_dtype_value(dtype)
    x = g.get_node(op.input("X")[0])
    value = _expr.const(op.attr("value"), dtype=dtype)
    out = _op.transform.full_like(x, value)
    g.add_node(op.output("Out")[0], out)


def convert_fill_constant(g, op, block):
    """Operator converter for fill_constant."""

    value = op.attr("value")
    shape = block.var(op.output("Out")[0]).shape
    dtype = op.attr("dtype")
    dtype = _convert_dtype_value(dtype)
    value = _expr.const(value).astype(dtype)
    if "ValueTensor" in op.input_names and op.input("ValueTensor"):
        shape = g.get_node(op.input("ValueTensor")[0])
        shape = _infer_value(shape, g.get_params())
    if "ShapeTensor" in op.input_names and op.input("ShapeTensor"):
        shape = g.get_node(op.input("ShapeTensor")[0])
        shape = _infer_value(shape, g.get_params())

    out = _op.full(value, shape=shape, dtype=dtype)
    g.add_node(op.output("Out")[0], out)


def convert_gelu(g, op, block):
    """Operator converter for gelu."""

    x = g.get_node(op.input("X")[0])
    out = x * (
        _expr.const(0.5, dtype="float32")
        + _op.erf(x * _expr.const(0.5 ** 0.5, dtype="float32")) * _expr.const(0.5, dtype="float32")
    )
    g.add_node(op.output("Out")[0], out)


def convert_hard_shrink(g, op, block):
    """Operator converter for hard_shrink."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    threshold = op.attr("threshold")
    threshold = _op.const(threshold, dtype)
    out = _op.logical_or(x < _op.const(-1.0, dtype) * threshold, x > threshold)
    out = _op.cast(out, dtype) * x
    g.add_node(op.output("Out")[0], out)


def convert_hard_sigmoid(g, op, block):
    """Operator converter for hard_sigmoid."""

    slope = op.attr("slope")
    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    out = x * _expr.const(slope, dtype) + _expr.const(0.5, dtype)
    out = _op.clip(out, 0, 1)
    g.add_node(op.output("Out")[0], out)


def convert_hard_swish(g, op, block):
    """Operator converter for hard_swish."""

    offset = op.attr("offset")
    scale = op.attr("scale")
    threshold = op.attr("threshold")
    assert np.isclose(offset, 3.0), "Only support offset==3.0 for PaddlePaddle's hard_swish"
    assert np.isclose(scale, 6.0), "Only support scale==6.0 for PaddlePaddle's hard_swish"
    assert np.isclose(threshold, 6.0), "Only support threshold==6.0 for PaddlePaddle's hard_swish"
    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    out = _op.clip(x, -1 * offset, offset)
    out = out / _expr.const(threshold, dtype) + _expr.const(0.5, dtype)
    out = x * out
    g.add_node(op.output("Out")[0], out)


def convert_hard_tanh(g, op, block):
    """Operator converter for hard_tanh."""

    x = g.get_node(op.input("X")[0])
    t_max = op.attr("t_max")
    t_min = op.attr("t_min")
    out = _op.tensor.clip(x, t_min, t_max)
    g.add_node(op.output("Out")[0], out)


def convert_layer_norm(g, op, block):
    """Operator converter for layer_norm."""

    begin_norm_axis = op.attr("begin_norm_axis")
    epsilon = op.attr("epsilon")
    x = g.get_node(op.input("X")[0])
    bias_input = op.input("Bias")
    scale_input = op.input("Scale")

    x_shape = infer_shape(x)
    assert begin_norm_axis in (
        len(x_shape) - 1,
        -1,
    ), "Support only normalization over last one dimension."

    if bias_input:
        bias = g.get_node(bias_input[0])
    else:
        bias = _expr.const(np.zeros(x_shape[begin_norm_axis]))

    if scale_input:
        scale = g.get_node(scale_input[0])
    else:
        scale = _expr.const(np.ones(x_shape[begin_norm_axis]))

    out = _op.nn.layer_norm(
        x, gamma=scale, beta=bias, axis=begin_norm_axis, epsilon=epsilon, center=True, scale=True
    )
    g.add_node(op.output("Y")[0], out)


def convert_leaky_relu(g, op, block):
    """Operator converter for leaky_relu."""

    alpha = op.attr("alpha")
    x = g.get_node(op.input("X")[0])
    out = _op.nn.leaky_relu(x, alpha=alpha)
    g.add_node(op.output("Out")[0], out)


def convert_log1p(g, op, block):
    """Operator converter for log1p."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    one = _expr.const(1, dtype=dtype)
    out = _op.log(x + one)
    g.add_node(op.output("Out")[0], out)


def convert_binary_logical_op(g, op, block):
    """Operator converter for logical op."""

    ipt0 = g.get_node(op.input("X")[0])
    ipt1 = g.get_node(op.input("Y")[0])
    op_func = get_relay_op(op.type)
    out = op_func(ipt0, ipt1)
    g.add_node(op.output("Out")[0], out)


def convert_logical_not(g, op, block):
    """Operator converter for logical_not op."""

    ipt0 = g.get_node(op.input("X")[0])
    op_func = get_relay_op(op.type)
    out = op_func(ipt0)
    g.add_node(op.output("Out")[0], out)


def convert_logsigmoid(g, op, block):
    """Operator converter for logsigmoid."""

    x = g.get_node(op.input("X")[0])
    out = _op.log(_op.tensor.sigmoid(x))
    g.add_node(op.output("Out")[0], out)


def convert_logsoftmax(g, op, block):
    """Operator converter for logsoftmax."""

    x = g.get_node(op.input("X")[0])
    axis = op.attr("axis")
    ndim = len(infer_shape(x))
    if axis < 0:
        axis += ndim
    m = _op.max(x, [axis], keepdims=True)
    e = _op.exp(x - m)
    s = _op.sum(e, [axis], keepdims=True)
    out = x - m - _op.log(s)
    g.add_node(op.output("Out")[0], out)


def convert_matmul(g, op, block):
    """Operator converter for matmul."""

    inputs = [g.get_node(op.input("X")[0]), g.get_node(op.input("Y")[0])]
    a_shape = infer_shape(inputs[0])
    b_shape = infer_shape(inputs[1])
    if op.has_attr("trans_x"):
        # for matmul_v2
        trans_x = op.attr("trans_x")
        trans_y = op.attr("trans_y")
    else:
        # for matmul
        trans_x = op.attr("transpose_X")
        trans_y = op.attr("transpose_Y")
    if trans_x:
        perm = list(range(len(a_shape)))
        perm[-2] = len(a_shape) - 1
        perm[-1] = len(a_shape) - 2
        inputs[0] = _op.transpose(inputs[0], axes=perm)
    if trans_y:
        perm = list(range(len(b_shape)))
        perm[-2] = len(b_shape) - 1
        perm[-1] = len(b_shape) - 2
        inputs[1] = _op.transpose(inputs[1], axes=perm)

    # This implemention almost keeps same with ONNX
    # Need to check input shape as batch matmul must be supported.
    a_shape = shape_of(inputs[0])
    a_rank = infer_shape(a_shape)[0]
    b_shape = shape_of(inputs[1])
    b_rank = infer_shape(b_shape)[0]
    # When performing a batch matmul, we need to properly handle N-dim shapes.
    if a_rank > 2 or b_rank > 2:

        def flatten_to_nd(x, x_shape, nd=3):
            ndims = infer_shape(x_shape)[0]
            if ndims == nd:
                return x
            newshape = _op.concatenate(
                [
                    _expr.const([-1], dtype=infer_type(x_shape).checked_type.dtype),
                    _op.strided_slice(x_shape, [ndims - nd + 1], [ndims]),
                ],
                0,
            )
            out = _op.reshape(x, fold_constant(newshape))
            return out

        b_type = infer_type(inputs[1])
        # Convert to dense if the second matrix is 2d and non-dynamic
        if b_rank == 2 and not _ty.is_dynamic(b_type.checked_type):
            a = flatten_to_nd(inputs[0], a_shape, 2)
            b = _op.transpose(inputs[1])
            output = _op.nn.dense(a, b)
        else:
            # Convert a and b into 3 dimensional tensors.
            a = flatten_to_nd(inputs[0], a_shape, 3)
            b = flatten_to_nd(inputs[1], b_shape, 3)
            # Transpose matrix dimensions of b.
            b = _op.transpose(b, [0, 2, 1])
            # Perform a batch matmul.
            output = _op.nn.batch_matmul(a, b)
        # Determine the output batch dimension.
        if a_rank > b_rank:
            out_batch = _op.strided_slice(a_shape, [0], [a_rank - 2])
        elif a_rank < b_rank:
            out_batch = _op.strided_slice(b_shape, [0], [b_rank - 2])
        # If its unclear how broadcasting should be applied, the output
        # shape is determined by choosing the maximum value from each input.
        else:
            out_batch = _op.concatenate(
                [
                    _op.maximum(
                        _op.strided_slice(a_shape, [i], [i + 1]),
                        _op.strided_slice(b_shape, [i], [i + 1]),
                    )
                    for i in range(a_rank - 2)
                ],
                0,
            )
        # Reshape output to original dimensions.
        final_shape = _op.concatenate(
            [
                out_batch,
                _op.strided_slice(
                    a_shape, [infer_shape(a_shape)[0] - 2], [infer_shape(a_shape)[0] - 1]
                ),
                _op.strided_slice(
                    b_shape, [infer_shape(b_shape)[0] - 1], [infer_shape(b_shape)[0]]
                ),
            ],
            0,
        )
        out = _op.reshape(output, fold_constant(final_shape))
    else:
        if b_rank == 1:
            inputs[1] = _op.expand_dims(inputs[1], 1, 1)
        # Otherwise a simple dense op will get the job done.
        input_1_t = _op.transpose(inputs[1], axes=(1, 0))
        out = _op.nn.dense(inputs[0], input_1_t)
        if b_rank == 1:
            out = _op.squeeze(out, axis=[-1])
    if op.has_attr("alpha"):
        alpha = op.attr("alpha")
        if not np.isclose(alpha, 1.0):
            out = out * _expr.const(alpha).astype("float32")
    g.add_node(op.output("Out")[0], out)


def convert_meshgrid(g, op, block):
    """Operator converter for meshgrid."""

    inputs = op.input("X")
    x = [g.get_node(i) for i in inputs]
    outs = _op.meshgrid(x, indexing="ij")
    for i, out in enumerate(outs):
        g.add_node(op.output("Out")[i], out)


def convert_mul(g, op, block):
    """Operator converter for mul."""

    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Y")[0])
    x_num_col_dims = op.attr("x_num_col_dims")
    y_num_col_dims = op.attr("y_num_col_dims")
    x_shape = _op.shape_of(x)
    y_shape = _op.shape_of(y)
    x_dim = infer_shape(x_shape)[0]
    y_dim = infer_shape(y_shape)[0]
    if x_num_col_dims < 0:
        x_num_col_dims += x_dim
    if y_num_col_dims < 0:
        y_num_col_dims += y_dim
    if x_num_col_dims == 1:
        x = _op.nn.batch_flatten(x)
    else:
        pre_shape = _op.prod(_op.strided_slice(x_shape, [0], [x_num_col_dims], [1]), keepdims=True)
        post_shape = _op.prod(
            _op.strided_slice(x_shape, [x_num_col_dims], [x_dim], [1]), keepdims=True
        )
        new_shape = _op.concatenate([pre_shape, post_shape], axis=0)
        new_shape = fold_constant(new_shape)
        x = _op.reshape(x, new_shape)
    if y_num_col_dims == 1:
        y = _op.nn.batch_flatten(y)
    else:
        pre_shape = _op.prod(_op.strided_slice(y_shape, [0], [y_num_col_dims], [1]), keepdims=True)
        post_shape = _op.prod(
            _op.strided_slice(y_shape, [y_num_col_dims], [y_dim], [1]), keepdims=True
        )
        new_shape = _op.concatenate([pre_shape, post_shape], axis=0)
        new_shape = fold_constant(new_shape)
        y = _op.reshape(y, new_shape)
    y = _op.transpose(y)
    out = _op.nn.dense(x, y)
    out_pre_shape = _op.strided_slice(x_shape, [0], [x_num_col_dims], [1])
    out_post_shape = _op.strided_slice(y_shape, [y_num_col_dims], [y_dim], [1])
    out_shape = _op.concatenate([out_pre_shape, out_post_shape], axis=0)
    out_shape = fold_constant(out_shape)
    out = _op.reshape(out, out_shape)
    g.add_node(op.output("Out")[0], out)


def convert_mv(g, op, block):
    """Operator converter for mv."""

    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Vec")[0])
    y = _op.expand_dims(y, axis=-1)
    y = _op.transpose(y)
    out = _op.nn.dense(x, y)
    out = _op.squeeze(out, axis=[-1])
    g.add_node(op.output("Out")[0], out)


def convert_numel(g, op, block):
    """Operator converter for numel."""

    input_x = g.get_node(op.input("Input")[0])
    out = _op.ndarray_size(input_x, dtype="int64")
    out = _op.expand_dims(out, axis=0)
    g.add_node(op.output("Out")[0], out)


def convert_nonzero(g, op, block):
    """Operator converter for nonzero."""

    input_x = g.get_node(op.input("Condition")[0])
    out = _op.transform.argwhere(input_x)
    # Paddle NonZero always outputs int64
    out = _op.cast(out, "int64")
    g.add_node(op.output("Out")[0], out)


def convert_pool2d(g, op, block):
    """Operator converter for pool2d."""

    adaptive = op.attr("adaptive")
    ceil_mode = op.attr("ceil_mode")
    global_pooling = op.attr("global_pooling")
    ksize = op.attr("ksize")
    paddings = op.attr("paddings")
    padding_algorithm = op.attr("padding_algorithm")
    pooling_type = op.attr("pooling_type")
    if global_pooling:
        adaptive = True
        ksize = [1, 1]

    input_x = g.get_node(op.input("X")[0])
    _, _, in_h, in_w = infer_shape(input_x)

    op_map = {
        "avg": "avg_pool2d",
        "max": "max_pool2d",
    }
    strides = op.attr("strides")
    if isinstance(strides, int):
        strides = [strides, strides]
    if isinstance(ksize, int):
        ksize = [ksize, ksize]
    if isinstance(paddings, int):
        paddings = [paddings] * 2

    if padding_algorithm == "VALID":
        paddings = [0, 0]
    elif padding_algorithm == "SAME":
        if strides[0] == 1 and strides[1] == 1:
            pad_h = _get_pad_size(0, ksize[0], strides[0])
            pad_w = _get_pad_size(0, ksize[1], strides[1])
        else:
            input_shape = shape_of(input_x)
            h_w = _op.strided_slice(input_shape, [2], [4])
            try:
                in_h, in_w = infer_value(h_w, g.get_params()).numpy().tolist()
            except Exception as e:
                msg = "The SAME padding algorithm of Conv not support dynamic shape"
                raise tvm.error.OpAttributeInvalid(msg) from e
            pad_h = _get_pad_size(in_h, ksize[0], strides[0])
            pad_w = _get_pad_size(in_w, ksize[1], strides[1])
        paddings = [pad_h[0], pad_w[0], pad_h[1], pad_w[1]]
    elif padding_algorithm == "EXPLICIT":
        if len(paddings) == 2:
            paddings = [paddings[0], paddings[1], paddings[0], paddings[1]]
        if len(paddings) == 4:
            paddings = [paddings[0], paddings[2], paddings[1], paddings[3]]
    else:
        msg = 'Value {} in attribute "padding" of operator Pool2d is not "valid."'
        raise tvm.error.OpAttributeInvalid(msg.format(padding_algorithm))

    if not isinstance(in_h, _op.Expr) and in_h < ksize[0]:
        ksize[0] = in_h
    if not isinstance(in_w, _op.Expr) and in_w < ksize[1]:
        ksize[1] = in_w

    if not adaptive:
        out = getattr(_op.nn, op_map[pooling_type])(
            input_x, pool_size=ksize, strides=strides, padding=paddings, ceil_mode=ceil_mode
        )
    else:
        out = getattr(_op.nn, "adaptive_" + op_map[pooling_type])(input_x, output_size=ksize)
    g.add_node(op.output("Out")[0], out)


def convert_reshape(g, op, block):
    """Operator converter for reshape."""

    input_shape = op.input("Shape")
    input_shape_tensor = op.input("ShapeTensor")
    data = g.get_node(op.input("X")[0])
    if input_shape:
        new_shape = g.get_node(input_shape[0])
    elif input_shape_tensor:
        new_shape = []
        for shape_name in input_shape_tensor:
            shape = g.get_node(shape_name)
            if len(infer_shape(shape)) == 0:
                shape = _op.reshape(shape, [-1])
            new_shape.append(shape.astype("int64"))
        new_shape = _op.concatenate(new_shape, axis=0)
        new_shape = _infer_value(new_shape, g.get_params())
    else:
        new_shape = op.attr("shape")
    out = _op.reshape(data, new_shape)
    g.add_node(op.output("Out")[0], out)


def convert_scale(g, op, block):
    """Operator converter for scale."""

    scale = op.attr("scale")
    bias = op.attr("bias")
    bias_after_scale = op.attr("bias_after_scale")
    x = g.get_node(op.input("X")[0])
    if np.isclose(scale, 1.0) and np.isclose(bias, 0.0):
        out = x
    else:
        x_dtype = infer_type(x).checked_type.dtype
        if x_dtype != "float32":
            x = x.astype("float32")
        if np.isclose(bias, 0.0):
            out = x * _expr.const(np.array(scale).astype("float32"))
        elif np.isclose(scale, 1.0):
            out = x + _expr.const(np.array(bias).astype("float32"))
        else:
            if bias_after_scale:
                out = x * _expr.const(np.array(scale).astype("float32")) + _expr.const(
                    np.array(bias).astype("float32")
                )
            else:
                out = (x + _expr.const(np.array(bias).astype("float32"))) * _expr.const(
                    np.array(scale).astype("float32")
                )
        if x_dtype != "float32":
            out = out.astype(x_dtype)
    g.add_node(op.output("Out")[0], out)


def convert_scatter(g, op, block):
    """Operator converter for scatter."""

    x = g.get_node(op.input("X")[0])
    index = g.get_node(op.input("Ids")[0])
    updates = g.get_node(op.input("Updates")[0])
    overwrite = op.attr("overwrite")

    shape = infer_shape(updates)
    ndims = len(shape)
    index = _op.expand_dims(index, axis=-1, num_newaxis=ndims - 1)
    index = _op.transform.broadcast_to(index, shape)

    if overwrite:
        out = _op.scatter(x, index, updates, axis=0)
    else:
        out = _op.scatter_add(_op.zeros_like(x), index, updates, axis=0)
        out += _op.scatter(x, index, _op.zeros_like(updates), axis=0)
    g.add_node(op.output("Out")[0], out)


def convert_scatter_nd_add(g, op, block):
    """Operator converter for scatter_nd_add."""

    x = g.get_node(op.input("X")[0])
    index = g.get_node(op.input("Index")[0])
    updates = g.get_node(op.input("Updates")[0])
    indices_dim = len(infer_shape(index))
    axes = list(range(indices_dim))
    index = _op.transpose(index, axes[-1:] + axes[:-1])
    out = _op.scatter_nd(x, index, updates, mode="add")
    g.add_node(op.output("Out")[0], out)


def convert_selu(g, op, block):
    """Operator converter for selu."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    alpha = _op.const(op.attr("alpha"), dtype)
    scale = _op.const(op.attr("scale"), dtype)
    out = (
        _expr.const(-1.0, dtype=dtype)
        * alpha
        * _op.nn.relu(_expr.const(1.0, dtype=dtype) - _op.exp(x))
    )
    out = scale * (out + _op.nn.relu(x))
    g.add_node(op.output("Out")[0], out)


def convert_shape(g, op, block):
    """Operator converter for shape."""

    x = g.get_node(op.input("Input")[0])
    out = shape_of(x)
    g.add_node(op.output("Out")[0], out)


def convert_slice(g, op, block):
    """Operator converter for slice."""

    data = g.get_node(op.input("Input")[0])
    dims = len(infer_shape(data))

    axes = op.attr("axes")
    indices = _expr.const(axes, dtype="int64")

    decrease_axis = op.attr("decrease_axis")
    if isinstance(decrease_axis, int):
        decrease_axis = [decrease_axis]

    if op.input("StartsTensor"):
        starts = g.get_node(op.input("StartsTensor")[0])
        starts = _infer_value(starts, g.get_params())
    elif op.input("StartsTensorList"):
        starts = []
        for start_index in op.input("StartsTensorList"):
            start_index = g.get_node(start_index).astype("int64")
            starts.append(start_index)
        starts = _op.concatenate(starts, axis=0)
        starts = _infer_value(starts, g.get_params())
    else:
        starts = op.attr("starts")

    if len(axes) < dims:
        if isinstance(starts, _expr.Expr):
            starts = _op.scatter(
                _op.const([0] * dims, dtype=infer_type(starts).checked_type.dtype),
                indices,
                starts,
                axis=0,
            )
        else:
            base = [0] * dims
            for i, axis in enumerate(axes):
                base[axis] = starts[i]
            starts = base

    if op.input("EndsTensor"):
        ends = g.get_node(op.input("EndsTensor")[0])
        ends = _infer_value(ends, g.get_params())
    elif op.input("EndsTensorList"):
        ends = []
        for end_index in op.input("EndsTensorList"):
            end_index = g.get_node(end_index).astype("int64")
            ends.append(end_index)
        ends = _op.concatenate(ends, axis=0)
        ends = _infer_value(ends, g.get_params())
    else:
        ends = op.attr("ends")

    if len(axes) < dims:
        if isinstance(ends, _expr.Expr):
            ends = _op.scatter(
                _expr.const(
                    np.array([np.iinfo(np.int32).max] * dims),
                    dtype=infer_type(ends).checked_type.dtype,
                ),
                indices,
                ends,
                axis=0,
            )
        else:
            base = [np.iinfo(np.int32).max] * dims
            for i, axis in enumerate(axes):
                base[axis] = ends[i]
            ends = base

    strides = None
    if "StridesTensor" in op.input_names and op.input("StridesTensor"):
        strides = g.get_node(op.input("StridesTensor")[0])
        strides = _infer_value(strides, g.get_params())
    elif "StridesTensorList" in op.input_names and op.input("StridesTensorList"):
        strides = []
        for strides_index in op.input("StridesTensorList"):
            strides_index = g.get_node(strides_index).astype("int64")
            strides.append(strides_index)
        strides = _op.concatenate(strides, axis=0)
        strides = _infer_value(strides, g.get_params())
    elif op.has_attr("strides"):
        strides = op.attr("strides")

    if len(axes) < dims:
        if isinstance(strides, _expr.Expr):
            strides = _op.scatter(
                _expr.const(
                    np.array([1] * dims),
                    dtype=infer_type(strides).checked_type.dtype,
                ),
                indices,
                strides,
                axis=0,
            )
        elif strides:
            base = [1] * dims
            for i, axis in enumerate(axes):
                base[axis] = strides[i]
            strides = base
    if not strides:
        strides = _op.const([1] * dims, dtype="int64")

    out = _op.strided_slice(data, begin=starts, end=ends, strides=strides)
    if decrease_axis:
        out = _op.squeeze(out, axis=decrease_axis)
    g.add_node(op.output("Out")[0], out)


def convert_softmax(g, op, block):
    """Operator converter for softmax."""

    axis = op.attr("axis")
    input_shape = block.var(op.input("X")[0]).shape
    if axis < 0:
        axis = len(input_shape) + axis
    x = g.get_node(op.input("X")[0])
    m = _op.max(x, axis, keepdims=True)
    e = _op.exp(x - m)
    out = e / _op.sum(e, axis, keepdims=True)
    g.add_node(op.output("Out")[0], out)


def convert_unsqueeze(g, op, block):
    """Operator converter for unsqueeze."""

    x = g.get_node(op.input("X")[0])
    axes = sorted(op.attr("axes"))
    for axis in axes:
        x = _op.expand_dims(x, axis=axis, num_newaxis=1)
    g.add_node(op.output("Out")[0], x)


_convert_map = {
    "abs": convert_unary_op,
    "acos": convert_unary_op,
    "addmm": convert_addmm,
    "arg_max": convert_arg_max,
    "arg_min": convert_arg_min,
    "argsort": convert_argsort,
    "asin": convert_unary_op,
    "assign": convert_assign,
    "assign_value": convert_assign_value,
    "atan": convert_unary_op,
    "batch_norm": convert_batch_norm,
    "bmm": convert_bmm,
    "brelu": convert_hard_tanh,
    "cast": convert_cast,
    "ceil": convert_unary_op,
    "clip": convert_clip,
    "concat": convert_concat,
    "conv2d": convert_conv2d,
    "cos": convert_unary_op,
    "cosh": convert_unary_op,
    "crop_tensor": convert_crop,
    "cumsum": convert_cumsum,
    "depthwise_conv2d": convert_conv2d,
    "dist": convert_dist,
    "dot": convert_dot,
    "dropout": convert_dropout,
    "elementwise_add": convert_elementwise_op,
    "elementwise_div": convert_elementwise_op,
    "elementwise_mul": convert_elementwise_op,
    "elementwise_sub": convert_elementwise_op,
    "elementwise_mod": convert_elementwise_op,
    "elementwise_max": convert_elementwise_op,
    "elementwise_min": convert_elementwise_op,
    "elementwise_pow": convert_elementwise_op,
    "elementwise_floordiv": convert_elementwise_op,
    "elu": convert_elu,
    "equal": convert_elementwise_op,
    "erf": convert_unary_op,
    "exp": convert_unary_op,
    "expand_v2": convert_expand,
    "expand_as_v2": convert_expand_as,
    "feed": convert_feed,
    "fill_any_like": convert_fill_any_like,
    "fill_constant": convert_fill_constant,
    "floor": convert_unary_op,
    "floor_mod": convert_elementwise_op,
    "gelu": convert_gelu,
    "greater_equal": convert_elementwise_op,
    "greater_than": convert_elementwise_op,
    "hard_shrink": convert_hard_shrink,
    "hard_sigmoid": convert_hard_sigmoid,
    "hard_swish": convert_hard_swish,
    "isfinite": convert_unary_op,
    "isfinite_v2": convert_unary_op,
    "isinf": convert_unary_op,
    "isinf_v2": convert_unary_op,
    "isnan": convert_unary_op,
    "isnan_v2": convert_unary_op,
    "layer_norm": convert_layer_norm,
    "leaky_relu": convert_leaky_relu,
    "less_equal": convert_elementwise_op,
    "less_than": convert_elementwise_op,
    "log": convert_unary_op,
    "log2": convert_unary_op,
    "log10": convert_unary_op,
    "log1p": convert_log1p,
    "logical_and": convert_binary_logical_op,
    "logical_not": convert_logical_not,
    "logical_or": convert_binary_logical_op,
    "logical_xor": convert_binary_logical_op,
    "logsigmoid": convert_logsigmoid,
    "log_softmax": convert_logsoftmax,
    "matmul": convert_matmul,
    "matmul_v2": convert_matmul,
    "meshgrid": convert_meshgrid,
    "mv": convert_mv,
    "mul": convert_mul,
    "not_equal": convert_elementwise_op,
    "pool2d": convert_pool2d,
    "relu": convert_unary_op,
    "reshape2": convert_reshape,
    "round": convert_unary_op,
    "rsqrt": convert_unary_op,
    "scale": convert_scale,
    "scatter": convert_scatter,
    "scatter_nd_add": convert_scatter_nd_add,
    "selu": convert_selu,
    "shape": convert_shape,
    "sigmoid": convert_unary_op,
    "sign": convert_unary_op,
    "sin": convert_unary_op,
    "sinh": convert_unary_op,
    "size": convert_numel,
    "slice": convert_slice,
    "softmax": convert_softmax,
    "sqrt": convert_unary_op,
    "strided_slice": convert_slice,
    "tan": convert_unary_op,
    "tanh": convert_unary_op,
    "unsqueeze2": convert_unsqueeze,
}


class GraphProto:
    """A helper class for handling relay functions from PaddlePaddle model."""

    def __init__(self, freeze_params=False):
        self.nodes = {}
        self.params = {}
        self.shape_dict = None
        self.freeze_params = freeze_params

    def get_node(self, name):
        """get node from graph"""

        assert name in self.nodes, "Node: {} not found".format(name)
        return self.nodes[name]

    def add_node(self, name, node):
        """add a node to graph"""
        if self.shape_dict:
            self.nodes[name] = fold_constant(node)
        else:
            self.nodes[name] = node

    def get_params(self, name=None):
        """get params from graph"""

        if name is None:
            return self.params
        assert name in self.params
        return self.params[name]

    def set_params(self, params):
        """set params for graph"""

        self.params = params

    def extract_parameters(self, program, scope=None):
        """Extract all the weights from PaddlePaddle program."""

        self.params = {}
        variables = program.global_block().vars
        for name in variables:
            var = program.global_block().var(name)
            if name.endswith("feed") or name.endswith("fetch"):
                continue
            if not var.persistable:
                continue
            if isinstance(scope, dict):
                self.params[name] = scope[name]
            else:
                self.params[name] = np.array(scope.var(name).get_tensor())
            if self.freeze_params:
                self.nodes[name] = _expr.const(self.params[name])
            else:
                self.nodes[name] = _expr.var(
                    name, shape=self.params[name].shape, dtype=str(self.params[name].dtype)
                )

    def check_unsupported_ops(self, program):
        """Check whether all the operators are supported."""

        unsupported_ops = set()
        for block in program.blocks:
            for op in block.ops:
                if op.type == "fetch":
                    continue
                if op.type not in _convert_map:
                    unsupported_ops.add(op.type)
        if len(unsupported_ops) > 0:
            msg = "The following operators are not supported for frontend Paddle: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def ops_to_relay(self, program, input_specs=None):
        """Convert PaddlePaddle operators to TVM relay functions."""

        if input_specs is not None:
            for input_spec in input_specs:
                convert_feed(self, input_spec, None)
        global_block = program.blocks[0]
        for op in global_block.ops:
            if op.type == "fetch":
                continue
            convert_func = _convert_map[op.type]
            convert_func(self, op, global_block)

    def from_program(self, program, shape_dict, scope):
        """Construct the TVM relay expression from PaddlePaddle program."""

        self.shape_dict = shape_dict
        if scope is None:
            import paddle

            scope = paddle.fluid.global_scope()
        self.check_unsupported_ops(program)
        self.extract_parameters(program, scope)
        self.ops_to_relay(program)

        output_names = list()
        for block in program.blocks:
            for op in block.ops:
                if op.type == "fetch":
                    output_names.append(op.input("X")[0])

        outputs = [self.get_node(name) for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        if self.freeze_params:
            self.params = {}
        return mod, self.params

    def from_translated_layer(self, layer, shape_dict):
        """Construct the TVM relay expression from PaddlePaddle TranslatedLayer."""

        self.shape_dict = shape_dict
        program = layer.program()
        parameters = dict()
        for param in layer.parameters():
            parameters[param.name] = np.array(param.value().get_tensor())
        self.check_unsupported_ops(program)
        self.extract_parameters(program, parameters)

        input_specs = layer._input_spec()
        self.ops_to_relay(program, input_specs)

        output_names = [x.name for x in layer._output_spec()]

        outputs = [self.get_node(name) for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        if self.freeze_params:
            self.params = {}
        return mod, self.params


def from_paddle(program_or_layer, shape_dict=None, scope=None, freeze_params=False):
    """Convert a PaddlePaddle model into an equivalent Relay Function.
    PaddlePaddle Program/TranslatedLayer represent the computation
    graph of PaddlePaddle model, and PaddlePaddle scope stores all the
    weights of PaddlePaddle model.
    """

    import paddle

    g = GraphProto(freeze_params)
    if isinstance(program_or_layer, paddle.jit.TranslatedLayer):
        # model is loaded by `paddle.jit.load`
        mod, params = g.from_translated_layer(program_or_layer, shape_dict)
    elif isinstance(program_or_layer, paddle.static.Program):
        # model is loaded by `paddle.static.load_inference_model`
        mod, params = g.from_program(program_or_layer, shape_dict, scope)
    else:
        raise Exception("Only PaddlePaddle's Program and TranslatedLayer are supported.")
    return mod, params
