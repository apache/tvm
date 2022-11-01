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

import warnings
import numpy as np

import tvm
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import ty as _ty
from .. import expr as _expr
from .. import function as _function
from .. import ty as _ty
from .. import op as _op
from .common import (
    autopad,
    fold_constant,
    get_relay_op,
    infer_shape,
    infer_type,
    infer_value,
    shape_of,
    try_infer_value,
    new_var,
)

__all__ = ["from_paddle"]


def _dtype_shape_promotion(inputs):
    """Promote data type and shape for list of tensors."""

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


def _convert_dtype_value(val):
    """Converts a Paddle type id to a string."""

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
    """Operator converter for all the unary operators."""

    # op_map stores mapping relationship between paddlepaddle and relay
    op_map = {
        "isinf_v2": _op.isinf,
        "isfinite_v2": _op.isfinite,
        "isnan_v2": _op.isnan,
    }
    if op.type in op_map:
        unary_func = op_map[op.type]
    else:
        # while paddle operator's name is same with relay
        unary_func = get_relay_op(op.type)
    out = unary_func(g.get_node(op.input("X")[0]))
    g.add_node(op.output("Out")[0], out)


def convert_binary_logical_op(g, op, block):
    """Operator converter for logical op."""

    ipt0 = g.get_node(op.input("X")[0])
    ipt1 = g.get_node(op.input("Y")[0])
    op_func = get_relay_op(op.type)
    out = op_func(ipt0, ipt1)
    g.add_node(op.output("Out")[0], out)


def convert_addmm(g, op, block):
    """Operator converter for addmm."""

    input_x = g.get_node(op.input("Input")[0])
    x = g.get_node(op.input("X")[0])
    y = g.get_node(op.input("Y")[0])

    alpha = op.attr("Alpha")
    beta = op.attr("Beta")
    dtype = block.var(op.output("Out")[0]).dtype
    dtype = _convert_dtype_value(dtype)

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


def convert_arg_max_min(g, op, block):
    """Operator converter for arg_max and arg_min."""

    axis = op.attr("axis")
    keepdims = op.attr("keepdims")
    flatten = op.attr("flatten")
    dtype = op.attr("dtype")
    dtype = _convert_dtype_value(dtype)

    func = _op.argmax if op.type == "arg_max" else _op.argmin
    x = g.get_node(op.input("X")[0])
    if axis is None or flatten:
        x = _op.reshape(x, [-1])
        out = func(x, axis=None, keepdims=True)
    else:
        out = func(x, axis=axis, keepdims=keepdims)
    if dtype != infer_type(out).checked_type.dtype:
        out = _op.cast(out, dtype)
    g.add_node(op.output("Out")[0], out)


def convert_argsort(g, op, block):
    """Operator converter for argsort."""

    x = g.get_node(op.input("X")[0])
    axis = op.attr("axis")
    descending = op.attr("descending")

    out_indices = _op.argsort(x, axis, not descending, dtype="int64")
    out = _op.gather(x, axis, out_indices)
    g.add_node(op.output("Out")[0], out)
    g.add_node(op.output("Indices")[0], out_indices)


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


def convert_brelu(g, op, block):
    """Operator converter for brelu."""

    x = g.get_node(op.input("X")[0])
    t_max = op.attr("t_max")
    t_min = op.attr("t_min")
    out = _op.tensor.clip(x, t_min, t_max)
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
    # if the min/max value is a tensor
    min_max_is_tensor = False
    if op.input("Min"):
        min_value = g.get_node(op.input("Min")[0])
        min_value, infered = try_infer_value(min_value, g.get_params())
        if infered:
            min_value = min_value.tolist()[0]
        if isinstance(min_value, _expr.Expr):
            min_max_is_tensor = True
    else:
        min_value = op.attr("min")

    if op.input("Max"):
        max_value = g.get_node(op.input("Max")[0])
        max_value, infered = try_infer_value(max_value, g.get_params())
        if infered:
            max_value = max_value.tolist()[0]
        if isinstance(max_value, _expr.Expr):
            min_max_is_tensor = True
    else:
        max_value = op.attr("max")

    if min_max_is_tensor:
        if not isinstance(min_value, _expr.Expr):
            min_value = _op.const(min_value, dtype)
        if not isinstance(max_value, _expr.Expr):
            max_value = _op.const(max_value, dtype)
        out = _op.maximum(x, min_value)
        out = _op.minimum(out, max_value)
    else:
        out = _op.clip(x, min_value, max_value)
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
        # Handle history issue of PaddlePaddle
        # while padding_algorithm == "SAME"
        # dilations will be set to [1, 1]
        dilations = [1, 1]
        input_x = autopad(input_x, strides, [k_h, k_w], dilations)
        paddings = [0, 0]
    elif padding_algorithm == "EXPLICIT":
        if len(paddings) == 2:
            paddings = [paddings[0], paddings[1], paddings[0], paddings[1]]
        elif len(paddings) == 4:
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


def convert_conv2d_transpose(g, op, block):
    """Operator converter for conv2d_transpose."""

    dilations = op.attr("dilations")
    groups = op.attr("groups")
    paddings = op.attr("paddings")
    padding_algorithm = op.attr("padding_algorithm")
    strides = op.attr("strides")
    output_padding = op.attr("output_padding") if op.attr("output_padding") else [0, 0]

    kernel = g.get_node(op.input("Filter")[0])
    input_x = g.get_node(op.input("Input")[0])
    _, out_channels, k_h, k_w = infer_shape(kernel)
    k_size = [k_h, k_w]
    if padding_algorithm == "VALID":
        paddings = [0, 0]
    elif padding_algorithm == "SAME":
        # SAME padding of conv2d_transpose is not same with conv2d
        # We cannot use auto_pad here, only static shape is supported now
        dilations = [1, 1]
        input_shape = shape_of(input_x)
        h_w = _op.strided_slice(input_shape, [2], [4])
        try:
            h_w = infer_value(h_w, g.get_params()).numpy().tolist()
        except Exception as e:
            msg = "The SAME padding algorithm of conv2d_transpose not support dynamic shape"
            raise tvm.error.OpAttributeInvalid(msg) from e
        paddings = []
        for i in range(2):
            if strides[i] == 1 or h_w[i] % strides[i] == 0:
                pad = max(k_size[i] - strides[i], 0)
            else:
                pad = max(k_size[i] - (h_w[i] % strides[i]), 0)
            pad_before = pad // 2
            pad_after = pad - pad_before
            paddings.insert(-1, pad_before)
            paddings.append(pad_after)
    elif padding_algorithm == "EXPLICIT":
        if len(paddings) == 2:
            paddings = [paddings[0], paddings[1], paddings[0], paddings[1]]
        elif len(paddings) == 4:
            paddings = [paddings[0], paddings[2], paddings[1], paddings[3]]
    else:
        msg = 'Value {} in attribute "padding" of operator Conv is not "valid."'
        raise tvm.error.OpAttributeInvalid(msg.format(padding_algorithm))

    out = _op.nn.conv2d_transpose(
        input_x,
        kernel,
        strides=strides,
        padding=paddings,
        dilation=dilations,
        groups=groups,
        channels=out_channels * groups,
        kernel_size=k_size,
        output_padding=output_padding,
    )
    g.add_node(op.output("Output")[0], out)


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


def convert_dot(g, op, block):
    """Operator converter for dot."""

    # x, y should be 1D or 2D tensor
    # when it's 2D tensor, the first dimension means batch dimension
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


def convert_elu(g, op, block):
    """Operator converter for elu."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    alpha = op.attr("alpha")
    alpha = _expr.const(-1.0 * alpha, dtype=dtype)
    out = alpha * _op.nn.relu(_expr.const(1, dtype=dtype) - _op.exp(x)) + _op.nn.relu(x)
    g.add_node(op.output("Out")[0], out)


def convert_expand(g, op, block):
    """Operator converter for expand."""

    x = g.get_node(op.input("X")[0])
    if op.input("Shape"):
        sizes = g.get_node(op.input("Shape")[0])
    else:
        sizes = op.attr("shape")

    if isinstance(sizes, _expr.Expr):
        sizes = try_infer_value(sizes, parameters=g.get_params())[0]

    if isinstance(sizes, np.ndarray):
        sizes = sizes.tolist()

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
        ipt_shape = block.var(ipt_name).shape
        ipt_dtype = block.var(ipt_name).dtype
        ipt_dtype = str(ipt_dtype).strip().split(".")[1]
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
    out = _op.transform.full_like(x, value).astype(dtype)
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
    if "ShapeTensor" in op.input_names and op.input("ShapeTensor"):
        shape = g.get_node(op.input("ShapeTensor")[0])

    if isinstance(shape, _expr.Expr):
        shape = try_infer_value(shape, parameters=g.get_params())[0]

    if isinstance(shape, np.ndarray):
        shape = shape.tolist()

    out = _op.full(value, shape=shape, dtype=dtype)
    g.add_node(op.output("Out")[0], out)


def convert_fill_constant_batch_size_like(g, op, block):
    """Operator converter for fill_constant_batch_size_like."""

    x = g.get_node(op.input("Input")[0])
    value = op.attr("value")
    shape = op.attr("shape")
    input_dim_idx = op.attr("input_dim_idx")
    output_dim_idx = op.attr("output_dim_idx")
    dtype = op.attr("dtype")

    dtype = _convert_dtype_value(dtype)
    input_shape = shape_of(x)
    batch = _op.strided_slice(input_shape, begin=[input_dim_idx], end=[input_dim_idx + 1]).astype(
        "int32"
    )
    shape_before = shape[:output_dim_idx]
    shape_before = _expr.const(shape_before, dtype="int32")
    shape_after = shape[output_dim_idx + 1 :]
    shape_after = _expr.const(shape_after, dtype="int32")

    out_shape = _op.concatenate([shape_before, batch, shape_after], axis=0)
    out_shape, infered = try_infer_value(out_shape, g.get_params())
    if infered:
        out_shape = out_shape.tolist()
    constant = _expr.const(value, dtype=dtype).astype(dtype)
    out = _op.full(constant, out_shape, dtype=dtype)

    g.add_node(op.output("Out")[0], out)


def convert_flatten(g, op, block):
    """Operator converter for flatten."""

    x = g.get_node(op.input("X")[0])
    input_shape = list(infer_shape(x))

    start = op.attr("start_axis")
    end = op.attr("stop_axis")
    ndim = len(input_shape)
    if end < 0:
        end += ndim
    new_shape = [0] * start

    new_shape.append(-1)
    squeeze_axes = []
    for i in range(start + 1, end + 1):
        new_shape.append(1)
        squeeze_axes.append(i)
    for _ in range(end + 1, ndim):
        new_shape.append(0)
    out = _op.reshape(x, new_shape)
    if squeeze_axes:
        out = _op.squeeze(out, axis=squeeze_axes)

    g.add_node(op.output("Out")[0], out)


def convert_gather(g, op, block):
    """Operator converter for gather."""

    x = g.get_node(op.input("X")[0])
    index = g.get_node(op.input("Index")[0])
    axis = op.attr("axis")
    out = _op.take(x, index, axis)
    g.add_node(op.output("Out")[0], out)


def convert_gather_nd(g, op, block):
    """Operator converter for gather_nd."""

    x = g.get_node(op.input("X")[0])
    index = g.get_node(op.input("Index")[0])
    shape = infer_shape(index)
    perm = list(range(0, len(shape) - 1))
    perm.insert(0, len(shape) - 1)
    index = _op.transpose(index, axes=perm)
    out = _op.gather_nd(x, index, 0, shape[-1])
    g.add_node(op.output("Out")[0], out)


def convert_gelu(g, op, block):
    """Operator converter for gelu."""

    x = g.get_node(op.input("X")[0])
    out = x * (
        _expr.const(0.5, dtype="float32")
        + _op.erf(x * _expr.const(0.5**0.5, dtype="float32")) * _expr.const(0.5, dtype="float32")
    )
    g.add_node(op.output("Out")[0], out)


def convert_group_norm(g, op, block):
    """Operator converter for group_norm."""

    x = g.get_node(op.input("X")[0])
    num_groups = op.attr("groups")
    epsilon = op.attr("epsilon")
    gamma = g.get_node(op.input("Scale")[0])
    beta = g.get_node(op.input("Bias")[0])
    out = _op.nn.group_norm(
        x,
        gamma=gamma,
        beta=beta,
        num_groups=num_groups,
        axis=1,
        epsilon=epsilon,
        center=True,
        scale=True,
    )
    g.add_node(op.output("Y")[0], out)


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
    out = x * _expr.const(slope) + _expr.const(0.5)
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
    out = _op.clip(x, -1 * offset, offset)
    out = out / _expr.const(threshold) + _expr.const(0.5)
    out = x * out
    g.add_node(op.output("Out")[0], out)


def convert_interpolate(g, op, block):
    """Operator converter for interpolate."""

    def get_interpolate_mode(op):
        """Get parameters for interpolation methods."""

        interp_method = op.attr("interp_method")
        align_corners = op.attr("align_corners")
        align_mode = op.attr("align_mode")

        rounding_method = ""
        if interp_method == "nearest":
            interp_method = "nearest_neighbor"
            coordinate_transformation_mode = "asymmetric"
            rounding_method = "floor"
        elif interp_method == "bilinear":
            interp_method = "linear"
            if not align_corners and align_mode == 0:
                coordinate_transformation_mode = "half_pixel"
            else:
                if align_corners:
                    coordinate_transformation_mode = "align_corners"
                else:
                    coordinate_transformation_mode = "asymmetric"
        elif interp_method == "bicubic":
            interp_method = "cubic"
            if align_corners:
                coordinate_transformation_mode = "align_corners"
            else:
                coordinate_transformation_mode = "half_pixel"
        else:
            msg = "interp_method {} is not supported for PaddlePaddle's interpolate"
            raise tvm.error.OpAttributeInvalid(msg.format(interp_method))
        return rounding_method, interp_method, coordinate_transformation_mode

    layout = op.attr("data_layout")
    out_h = op.attr("out_h")
    out_w = op.attr("out_w")
    scale = op.attr("scale")
    if not isinstance(scale, (list, tuple)):
        scale = [scale, scale]

    x = g.get_node(op.input("X")[0])
    x_shape = infer_shape(x)
    assert len(x_shape) == 4, "Only 4D input tensor is supported for PaddlePaddle's interpolate"
    input_out_size = op.input("OutSize")
    input_size_tensor = op.input("SizeTensor")
    input_scale = op.input("Scale")
    rounding_method, interp_method, coordinate_transformation_mode = get_interpolate_mode(op)

    if input_size_tensor:
        # if out_size is a list of tensor
        out_size = list()
        for name in input_size_tensor:
            size = g.get_node(name)
            if len(infer_shape(size)) == 0:
                size = _op.reshape(size, [-1])
            out_size.append(size)
        out_size = _op.concatenate(out_size, axis=0)
        out_size, infered = try_infer_value(out_size, parameters=g.get_params())
        if infered:
            out_size = out_size.tolist()
    elif input_scale:
        # if out_size is not defined, but scale is defined
        input_scale = g.get_node(input_scale[0])
        input_shape = shape_of(x).astype("float32")
        if layout.startswith("NC"):
            out_size = _op.strided_slice(input_shape, begin=[2], end=[4]) * input_scale
        else:
            out_size = _op.strided_slice(input_shape, begin=[1], end=[3]) * input_scale
        out_size = out_size.astype("int32")
        out_size, infered = try_infer_value(out_size, parameters=g.get_params())
        if infered:
            out_size = out_size.tolist()
    elif scale and scale[0] > 0 and scale[1] > 0:
        # use attribute scale
        input_shape = shape_of(x).astype("float32")
        input_scale = _expr.const(np.array([scale[0], scale[1]]).astype("float32"))
        if layout.startswith("NC"):
            out_size = _op.strided_slice(input_shape, begin=[2], end=[4]) * input_scale
        else:
            out_size = _op.strided_slice(input_shape, begin=[1], end=[3]) * input_scale
        out_size = out_size.astype("int32")
        out_size, infered = try_infer_value(out_size, parameters=g.get_params())
        if infered:
            out_size = out_size.tolist()
    elif input_out_size:
        # if out_size is a tensor
        out_size = g.get_node(input_out_size[0])
        out_size, infered = try_infer_value(out_size, parameters=g.get_params())
        if infered:
            out_size = out_size.tolist()
    else:
        # if out_size is a constant value
        out_size = [out_h, out_w]

    out = _op.image.resize2d(
        x,
        size=out_size,
        layout=layout,
        method=interp_method,
        coordinate_transformation_mode=coordinate_transformation_mode,
        rounding_method=rounding_method,
        cubic_alpha=-0.75,
    )
    g.add_node(op.output("Out")[0], out)


def convert_instance_norm(g, op, block):
    """Operator converter for instance_norm."""

    x = g.get_node(op.input("X")[0])
    gamma = g.get_node(op.input("Scale")[0])
    beta = g.get_node(op.input("Bias")[0])
    epsilon = op.attr("epsilon")

    scale = center = True
    out = _op.nn.instance_norm(x, gamma, beta, axis=1, epsilon=epsilon, center=center, scale=scale)
    g.add_node(op.output("Y")[0], out)


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


def convert_logsumexp(g, op, block):
    """Operator converter for logsumexp."""

    input_x = g.get_node(op.input("X")[0])
    axis = op.attr("axis")
    if op.attr("reduce_all"):
        axis = None
    keepdims = op.attr("keepdim")
    out = get_relay_op("logsumexp")(input_x, axis=axis, keepdims=keepdims)
    if not axis and not keepdims:
        out = _op.expand_dims(out, axis=0)
    g.add_node(op.output("Out")[0], out)


def convert_lookup_table(g, op, block):
    """Operator converter for lookup_table_v2."""

    indices = g.get_node(op.input("Ids")[0])
    padding_idx = op.attr("padding_idx")
    weights = g.get_node(op.input("W")[0])
    if padding_idx != -1:
        if op.input("W")[0] in g.get_params():
            weights = g.get_params(op.input("W")[0])
            weights[padding_idx] = 0.0
            weights = _expr.const(weights)
        else:
            shape, infered = try_infer_value(shape_of(weights), g.get_params())
            if infered:
                shape = shape.tolist()
            assert not isinstance(
                shape, _expr.Expr
            ), "Shape of weight has to be fixed for PaddlePaddle's lookup_table"
            filters = np.ones(shape).astype(infer_type(weights).checked_type.dtype)
            filters[padding_idx] = 0.0
            filters = _expr.const(filters)
            weights = weights * filters
    out = _op.take(weights, indices.astype("int32"), axis=0)
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
    a_shape = shape_of(inputs[0], dtype="int32")
    a_rank = infer_shape(a_shape)[0]
    b_shape = shape_of(inputs[1], dtype="int32")
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
    x_shape = shape_of(x, dtype="int32")
    y_shape = shape_of(y, dtype="int32")
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


def convert_padding(g, op, block):
    """Operator converter for padding."""

    input_x = g.get_node(op.input("X")[0])
    input_padding = op.input("Paddings")
    if input_padding:
        padding = g.get_node(input_padding[0])
        padding = infer_value(padding, g.get_params()).numpy().tolist()
    else:
        padding = op.attr("paddings")
    padding = op.attr("paddings")
    value = op.attr("value")
    data_format = op.attr("data_format")
    mode = op.attr("mode")
    assert mode != "circular", "Don't support mod='circular' for PaddlePaddle's padding"
    if mode == "replicate":
        mode = "edge"

    pad_len = len(padding)
    new_paddings = [0] * (pad_len + 4)
    for i in range(0, pad_len, 2):
        index = -1 - i
        if data_format[:2] != "NC":
            index = -3 - i
        new_paddings[index] = padding[i + 1]
        new_paddings[index - 1] = padding[i]

    new_paddings = [new_paddings[i : i + 2] for i in range(0, len(new_paddings), 2)]

    out = _op.nn.pad(input_x, new_paddings, pad_value=value, pad_mode=mode)
    g.add_node(op.output("Out")[0], out)


def convert_pixel_shuffle(g, op, block):
    """Operator converter for pixel_shuffle."""

    x = g.get_node(op.input("X")[0])
    upscale_factor = op.attr("upscale_factor")
    out = _op.nn.depth_to_space(x, upscale_factor, mode="CRD")
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
    data_format = op.attr("data_format")

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
        input_x = autopad(input_x, strides, ksize)
        paddings = [0, 0]
    elif padding_algorithm == "EXPLICIT":
        if len(paddings) == 2:
            paddings = [paddings[0], paddings[1], paddings[0], paddings[1]]
        elif len(paddings) == 4:
            paddings = [paddings[0], paddings[2], paddings[1], paddings[3]]
    else:
        msg = 'Value {} in attribute "padding" of operator Pool2d is not "valid."'
        raise tvm.error.OpAttributeInvalid(msg.format(padding_algorithm))

    # handle with special case
    # while kernel size less than input size
    # shrink kernel size to input size
    if (
        not isinstance(in_h, _op.Expr)
        and padding_algorithm == "EXPLICIT"
        and in_h + paddings[0] + paddings[2] < ksize[0]
    ):
        ksize[0] = in_h
    if (
        not isinstance(in_w, _op.Expr)
        and padding_algorithm == "EXPLICIT"
        and in_w + paddings[1] + paddings[3] < ksize[1]
    ):
        ksize[1] = in_w

    if not adaptive:
        if pooling_type == "avg":
            exclusive = op.attr("exclusive")
            out = _op.nn.avg_pool2d(
                input_x,
                pool_size=ksize,
                strides=strides,
                padding=paddings,
                ceil_mode=ceil_mode,
                count_include_pad=not exclusive,
            )
        else:
            out = getattr(_op.nn, op_map[pooling_type])(
                input_x, pool_size=ksize, strides=strides, padding=paddings, ceil_mode=ceil_mode
            )
    else:
        out = getattr(_op.nn, "adaptive_" + op_map[pooling_type])(
            input_x, output_size=ksize, layout=data_format
        )
    g.add_node(op.output("Out")[0], out)


def convert_pow(g, op, block):
    """Operator converter for pow."""

    x = g.get_node(op.input("X")[0])
    dtype = block.var(op.output("Out")[0]).dtype
    dtype = _convert_dtype_value(dtype)
    factor = op.attr("factor")
    factor = _expr.const(factor, dtype=dtype)
    out = _op.power(x, factor)
    g.add_node(op.output("Out")[0], out)


def convert_prelu(g, op, block):
    """Operator converter for prelu."""

    x = g.get_node(op.input("X")[0])
    alpha = g.get_node(op.input("Alpha")[0])
    ndims = len(infer_shape(x))
    axis = 0 if ndims <= 1 else 1
    mode = op.attr("mode")
    if mode == "all":
        if ndims == 1:
            shape = _op.strided_slice(shape_of(x), [0], [1])
        else:
            shape = _op.strided_slice(shape_of(x), [1], [2])
        alpha = _op.broadcast_to(alpha, fold_constant(shape))
    out = _op.nn.prelu(x, alpha, axis)
    g.add_node(op.output("Out")[0], out)


def convert_range(g, op, block):
    """Operator converter for range."""

    start = g.get_node(op.input("Start")[0])
    stop = g.get_node(op.input("End")[0])
    step = g.get_node(op.input("Step")[0])
    dtype = infer_type(start).checked_type.dtype

    params = []
    for param in (start, stop, step):
        param, infered = try_infer_value(param, g.get_params())
        if infered:
            param = param.tolist()
        if isinstance(param, list):
            param = param[0]
        if isinstance(param, _expr.Expr):
            param = _op.squeeze(param)
        else:
            param = _op.const(param, dtype=dtype)
        params.append(param)

    out = _op.transform.arange(params[0], params[1], params[2], dtype=dtype)
    g.add_node(op.output("Out")[0], out)


def convert_reciprocal(g, op, block):
    """Operator converter for reciprocal."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    out = _expr.const(1.0, dtype) / x
    g.add_node(op.output("Out")[0], out)


def convert_reduce(g, op, block):
    """Operator converter for series of reduce operators."""

    op_map = {
        "reduce_all": "all",
        "reduce_any": "any",
        "reduce_max": "max",
        "reduce_min": "min",
        "reduce_prod": "prod",
        "reduce_sum": "sum",
        "reduce_mean": "mean",
    }
    op_name = op_map[op.type]
    input_x = g.get_node(op.input("X")[0])
    axis = op.attr("dim")
    if op.attr("reduce_all"):
        axis = None
    keepdims = op.attr("keep_dim")
    out = get_relay_op(op_name)(input_x, axis=axis, keepdims=keepdims)
    if not axis and not keepdims:
        # use `expand_dims` to solve the following situation
        # for TVM, the shape of `out` will be (, )
        # for Paddle, the shape of `out` will be [1]
        out = _op.expand_dims(out, axis=0)
    g.add_node(op.output("Out")[0], out)


def convert_relu6(g, op, block):
    """Operator converter for relu6."""

    x = g.get_node(op.input("X")[0])
    out = _op.clip(x, 0.0, 6.0)
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
            new_shape.append(shape)
        new_shape = _op.concatenate(new_shape, axis=0)
        new_shape, infered = try_infer_value(new_shape, parameters=g.get_params())
        if infered:
            new_shape = new_shape.tolist()
    else:
        new_shape = op.attr("shape")
    out = _op.reshape(data, new_shape)
    g.add_node(op.output("Out")[0], out)


def convert_rnn(g, op, block):
    """Operator converter for rnn."""

    def generate_lstm(
        input_seqs,
        hidden_state,
        cell_state,
        w_inp,
        w_hid,
        b_inp,
        b_hid,
        f_act,
        g_act,
        h_act,
        backwards=False,
    ):
        """Implementation of LSTM cell for paddlepaddle of TVM"""

        h_list = []
        seq_length = len(input_seqs)
        for i in range(seq_length):
            step = input_seqs[i] if not backwards else input_seqs[seq_length - (i + 1)]
            step = _op.squeeze(step, axis=[0])
            gates = _op.nn.dense(step, w_inp) + _op.nn.dense(hidden_state, w_hid)
            if b_inp is not None:
                gates += b_inp
            if b_hid is not None:
                gates += b_hid
            i, f, c, o = _op.split(gates, 4, axis=-1)

            i = f_act(i)
            f = f_act(f)

            c = g_act(c)
            C = f * cell_state + i * c

            o = f_act(o)

            H = o * h_act(C)

            hidden_state = H
            cell_state = C
            h_list.append(_op.expand_dims(H, axis=0))

        if backwards:
            h_list = h_list[::-1]

        # Concatenate outputs and add back in direction axis.
        concatenated = _op.concatenate(h_list, 0)
        output = _op.expand_dims(concatenated, axis=1)
        hidden_state = _op.expand_dims(hidden_state, axis=0)
        cell_state = _op.expand_dims(cell_state, axis=0)

        return output, hidden_state, cell_state

    def generate_gru(
        input_seqs, hidden_state, w_inp, w_hid, b_inp, b_hid, rz_act, n_act, backwards=False
    ):
        """Implementation of GRU cell for paddlepaddle of TVM"""

        h_list = []
        seq_length = len(input_seqs)
        for i in range(seq_length):
            step = input_seqs[i] if not backwards else input_seqs[seq_length - (i + 1)]
            step = _op.squeeze(step, axis=[0])
            xwt = _op.nn.dense(step, w_inp)
            hwt = _op.nn.dense(hidden_state, w_hid)
            if b_inp is not None:
                xwt += b_inp
            if b_hid is not None:
                hwt += b_hid
            i_r, i_z, i_n = _op.split(xwt, 3, axis=-1)
            h_r, h_z, h_n = _op.split(hwt, 3, axis=-1)

            r_gate = rz_act(i_r + h_r)
            z_gate = rz_act(i_z + h_z)
            n_gate = n_act(i_n + r_gate * h_n)

            hidden_state = (hidden_state - n_gate) * z_gate + n_gate
            h_list.append(_op.expand_dims(hidden_state, axis=0))

        if backwards:
            h_list = h_list[::-1]

        # Concatenate outputs and add back in direction axis.
        concatenated = _op.concatenate(h_list, 0)
        output = _op.expand_dims(concatenated, axis=1)
        hidden_state = _op.expand_dims(hidden_state, axis=0)

        return output, hidden_state

    def generate_simplernn(
        input_seqs, hidden_state, w_inp, w_hid, b_inp, b_hid, n_act, backwards=False
    ):
        """Implementation of SimpleRNN cell for paddlepaddle of TVM"""

        h_list = []
        seq_length = len(input_seqs)
        for i in range(seq_length):
            step = input_seqs[i] if not backwards else input_seqs[seq_length - (i + 1)]
            step = _op.squeeze(step, axis=[0])
            xwt = _op.nn.dense(step, w_inp)
            hwt = _op.nn.dense(hidden_state, w_hid)
            if b_inp is not None:
                xwt += b_inp
            if b_hid is not None:
                hwt += b_hid

            n_gate = n_act(xwt + hwt)

            hidden_state = n_gate
            h_list.append(_op.expand_dims(hidden_state, axis=0))

        if backwards:
            h_list = h_list[::-1]

        # Concatenate outputs and add back in direction axis.
        concatenated = _op.concatenate(h_list, 0)
        output = _op.expand_dims(concatenated, axis=1)
        hidden_state = _op.expand_dims(hidden_state, axis=0)

        return output, hidden_state

    def make_param_inputs(g, node, layer, hidden_size, num_layers):
        """Param for weight and bias."""

        bidirect_len = 4 if node.attr("is_bidirec") else 2
        all_layer_param_len = len(node.input("WeightList"))
        weight_list = node.input("WeightList")[: all_layer_param_len // 2]
        bias_list = node.input("WeightList")[all_layer_param_len // 2 :]

        layer_weight_list = weight_list[layer * bidirect_len : layer * bidirect_len + bidirect_len]
        layer_bias_list = bias_list[layer * bidirect_len : layer * bidirect_len + bidirect_len]
        param_list = layer_weight_list + layer_bias_list
        param_list_len = len(param_list)

        input_weights = param_list[0 : param_list_len // 2 : 2]
        hidden_weights = param_list[1 : param_list_len // 2 : 2]

        input_bias = param_list[param_list_len // 2 : param_list_len : 2]
        hidden_bias = param_list[param_list_len // 2 + 1 : param_list_len : 2]

        return input_weights, hidden_weights, input_bias, hidden_bias

    def make_init_param_inputs(g, node, layer):
        """Init param for inputs."""

        mode = node.attr("mode")
        if mode == "LSTM":
            all_init_h, all_init_c = node.input("PreState")
            bidirect_len = 2 if node.attr("is_bidirec") else 1
            init_h = _op.strided_slice(
                g.get_node(all_init_h),
                [layer * bidirect_len],
                [layer * bidirect_len + bidirect_len],
                axes=[0],
            )
            init_c = _op.strided_slice(
                g.get_node(all_init_c),
                [layer * bidirect_len],
                [layer * bidirect_len + bidirect_len],
                axes=[0],
            )
            return init_h, init_c
        all_init_h = node.input("PreState")[0]
        bidirect_len = 2 if node.attr("is_bidirec") else 1
        init_h = _op.strided_slice(
            g.get_node(all_init_h),
            [layer * bidirect_len],
            [layer * bidirect_len + bidirect_len],
            axes=[0],
        )
        return init_h

    hidden_size = op.attr("hidden_size")
    num_layers = op.attr("num_layers")
    is_bidirec = op.attr("is_bidirec")
    mode = op.attr("mode")

    input_x = g.get_node(op.input("Input")[0])

    num_directions = 1
    if is_bidirec:
        num_directions = 2

    x_shape = infer_shape(input_x)
    time_steps = x_shape[0]
    x_steps = _op.split(input_x, indices_or_sections=time_steps, axis=0)
    for layer in range(num_layers):
        input_weights, hidden_weights, input_bias, hidden_bias = make_param_inputs(
            g, op, layer, hidden_size, num_layers
        )
        if mode == "LSTM":
            init_h, init_c = make_init_param_inputs(g, op, layer)
            init_hs = _op.split(init_h, num_directions)
            init_cs = _op.split(init_c, num_directions)
            result_output = []
            result_H = []
            result_C = []
            for i in range(num_directions):
                H_t = _op.squeeze(init_hs[i], axis=[0])
                C_t = _op.squeeze(init_cs[i], axis=[0])
                W = g.get_node(input_weights[i])
                R = g.get_node(hidden_weights[i])
                WB = g.get_node(input_bias[i])
                RB = g.get_node(hidden_bias[i])
                output, H, C = generate_lstm(
                    input_seqs=x_steps,
                    hidden_state=H_t,
                    cell_state=C_t,
                    w_inp=W,
                    w_hid=R,
                    b_inp=WB,
                    b_hid=RB,
                    f_act=_op.sigmoid,
                    g_act=_op.tanh,
                    h_act=_op.tanh,
                    backwards=i == 1,
                )
                result_output.append(output)
                result_H.append(H)
                result_C.append(C)
            output = _op.concatenate(result_output, axis=1)
            H = _op.concatenate(result_H, axis=0)
            C = _op.concatenate(result_C, axis=0)
        elif mode == "GRU":
            init_h = make_init_param_inputs(g, op, layer)
            init_hs = _op.split(init_h, num_directions)
            result_output = []
            result_H = []
            for i in range(num_directions):
                H_t = _op.squeeze(init_hs[i], axis=[0])
                W = g.get_node(input_weights[i])
                R = g.get_node(hidden_weights[i])
                WB = g.get_node(input_bias[i])
                RB = g.get_node(hidden_bias[i])
                output, H = generate_gru(
                    input_seqs=x_steps,
                    hidden_state=H_t,
                    w_inp=W,
                    w_hid=R,
                    b_inp=WB,
                    b_hid=RB,
                    rz_act=_op.sigmoid,
                    n_act=_op.tanh,
                    backwards=i == 1,
                )
                result_output.append(output)
                result_H.append(H)
            output = _op.concatenate(result_output, axis=1)
            H = _op.concatenate(result_H, axis=0)
        elif mode == "RNN_TANH":
            init_h = make_init_param_inputs(g, op, layer)
            init_hs = _op.split(init_h, num_directions)
            result_output = []
            result_H = []
            for i in range(num_directions):
                H_t = _op.squeeze(init_hs[i], axis=[0])
                W = g.get_node(input_weights[i])
                R = g.get_node(hidden_weights[i])
                WB = g.get_node(input_bias[i])
                RB = g.get_node(hidden_bias[i])
                output, H = generate_simplernn(
                    input_seqs=x_steps,
                    hidden_state=H_t,
                    w_inp=W,
                    w_hid=R,
                    b_inp=WB,
                    b_hid=RB,
                    n_act=_op.tanh,
                    backwards=i == 1,
                )
                result_output.append(output)
                result_H.append(H)
            output = _op.concatenate(result_output, axis=1)
            H = _op.concatenate(result_H, axis=0)

        output = _op.transpose(output, axes=[0, 2, 1, 3])
        output = _op.reshape(output, newshape=(0, 0, -1))
        x_steps = _op.split(output, indices_or_sections=time_steps, axis=0)

    g.add_node(op.output("Out")[0], output)


def convert_scale(g, op, block):
    """Operator converter for scale."""

    scale = op.attr("scale")
    bias = op.attr("bias")
    bias_after_scale = op.attr("bias_after_scale")
    x = g.get_node(op.input("X")[0])
    if np.isclose(scale, 1.0) and np.isclose(bias, 0.0):
        out = x
    else:
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
    out = shape_of(x, dtype="int32")
    g.add_node(op.output("Out")[0], out)


def convert_size(g, op, block):
    """Operator converter for size."""

    input_x = g.get_node(op.input("Input")[0])
    out = _op.ndarray_size(input_x, dtype="int64")
    out = _op.expand_dims(out, axis=0)
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
        starts, infered = try_infer_value(starts, g.get_params())
        if infered:
            starts = starts.tolist()
    elif op.input("StartsTensorList"):
        starts = []
        for start_index in op.input("StartsTensorList"):
            start_index = g.get_node(start_index).astype("int64")
            starts.append(start_index)
        starts = _op.concatenate(starts, axis=0)
        starts, infered = try_infer_value(starts, g.get_params())
        if infered:
            starts = starts.tolist()
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
        ends, infered = try_infer_value(ends, g.get_params())
        if infered:
            ends = ends.tolist()
    elif op.input("EndsTensorList"):
        ends = []
        for end_index in op.input("EndsTensorList"):
            end_index = g.get_node(end_index).astype("int64")
            ends.append(end_index)
        ends = _op.concatenate(ends, axis=0)
        ends, infered = try_infer_value(ends, g.get_params())
        if infered:
            ends = ends.tolist()
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
        strides, infered = try_infer_value(strides, g.get_params())
        if infered:
            strides = strides.tolist()
    elif "StridesTensorList" in op.input_names and op.input("StridesTensorList"):
        strides = []
        for strides_index in op.input("StridesTensorList"):
            strides_index = g.get_node(strides_index).astype("int64")
            strides.append(strides_index)
        strides = _op.concatenate(strides, axis=0)
        strides, infered = try_infer_value(strides, g.get_params())
        if infered:
            strides = strides.tolist()
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


def convert_softplus(g, op, block):
    """Operator converter for softplus."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    beta = op.attr("beta")
    beta = _expr.const(beta, dtype=dtype)
    out = _op.log(_op.exp(x * beta) + _expr.const(1.0, dtype=dtype)) / beta
    g.add_node(op.output("Out")[0], out)


def convert_softsign(g, op, block):
    """Operator converter for softsign."""

    x = g.get_node(op.input("X")[0])
    dtype = infer_type(x).checked_type.dtype
    out = x / (_op.const(1.0, dtype) + _op.abs(x))
    g.add_node(op.output("Out")[0], out)


def convert_split(g, op, block):
    """Operator converter for split."""

    x = g.get_node(op.input("X")[0])
    axis = op.input("AxisTensor")
    if axis:
        axis = g.get_node(axis[0])
        axis, infered = try_infer_value(axis, g.get_params())
        if infered:
            axis = axis.tolist()[0]
    else:
        axis = op.attr("axis")

    sections = op.input("SectionsTensorList")
    if sections:
        tmp_section = []
        for i in sections:
            i = g.get_node(i)
            i, infered = try_infer_value(i, g.get_params())
            if infered:
                i = i.tolist()
            else:
                raise ValueError("Dynamic Split not yet supported.")
            tmp_section.extend(i)
        sections = tmp_section
    else:
        sections = op.attr("sections")
    if sections:
        indices = []
        split_index = 0
        for i in sections[:-1]:
            if i == -1:
                input_shape = infer_shape(x)[axis]
                i = input_shape - np.sum(sections) - 1
            split_index += i
            indices.append(split_index)
    else:
        indices = op.attr("num")

    out = _op.split(x, indices, axis)
    for i, out_i in enumerate(out):
        g.add_node(op.output("Out")[i], out_i)


def convert_square(g, op, block):
    """Operator converter for square."""

    x = g.get_node(op.input("X")[0])
    dtype = block.var(op.output("Out")[0]).dtype
    dtype = _convert_dtype_value(dtype)
    out = _op.power(x, _expr.const(2, dtype))
    g.add_node(op.output("Out")[0], out)


def convert_squeeze(g, op, block):
    """Operator converter for squeeze2."""

    x = g.get_node(op.input("X")[0])
    axes = op.attr("axes")
    if not axes:
        axes = None
    x = _op.squeeze(x, axis=axes)
    g.add_node(op.output("Out")[0], x)


def convert_swish(g, op, block):
    """Operator converter for swish."""

    x = g.get_node(op.input("X")[0])
    beta = op.attr("beta")
    assert beta == 1.0, "Only support beta==1.0 for PaddlePaddle's swish"
    out = x * _op.tensor.sigmoid(x)
    g.add_node(op.output("Out")[0], out)


def convert_transpose(g, op, block):
    """Operator converter for transpose."""

    perm = op.attr("axis")
    out = _op.transpose(g.get_node(op.input("X")[0]), axes=perm)
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
    "arg_max": convert_arg_max_min,
    "arg_min": convert_arg_max_min,
    "argsort": convert_argsort,
    "asin": convert_unary_op,
    "assign": convert_assign,
    "assign_value": convert_assign_value,
    "atan": convert_unary_op,
    "batch_norm": convert_batch_norm,
    "bicubic_interp_v2": convert_interpolate,
    "bilinear_interp_v2": convert_interpolate,
    "bmm": convert_bmm,
    "brelu": convert_brelu,
    "cast": convert_cast,
    "ceil": convert_unary_op,
    "clip": convert_clip,
    "concat": convert_concat,
    "conv2d": convert_conv2d,
    "conv2d_transpose": convert_conv2d_transpose,
    "cos": convert_unary_op,
    "cosh": convert_unary_op,
    "cumsum": convert_cumsum,
    "depthwise_conv2d": convert_conv2d,
    "depthwise_conv2d_transpose": convert_conv2d_transpose,
    "dot": convert_dot,
    "dropout": convert_dropout,
    "elementwise_add": convert_elementwise_op,
    "elementwise_div": convert_elementwise_op,
    "elementwise_floordiv": convert_elementwise_op,
    "elementwise_max": convert_elementwise_op,
    "elementwise_min": convert_elementwise_op,
    "elementwise_mod": convert_elementwise_op,
    "elementwise_mul": convert_elementwise_op,
    "elementwise_pow": convert_elementwise_op,
    "elementwise_prod": convert_elementwise_op,
    "elementwise_sub": convert_elementwise_op,
    "elu": convert_elu,
    "equal": convert_elementwise_op,
    "erf": convert_unary_op,
    "exp": convert_unary_op,
    "expand_v2": convert_expand,
    "expand_as_v2": convert_expand_as,
    "feed": convert_feed,
    "fill_any_like": convert_fill_any_like,
    "fill_constant": convert_fill_constant,
    "fill_constant_batch_size_like": convert_fill_constant_batch_size_like,
    "flatten_contiguous_range": convert_flatten,
    "floor": convert_unary_op,
    "floor_mod": convert_elementwise_op,
    "gather": convert_gather,
    "gather_nd": convert_gather_nd,
    "gelu": convert_gelu,
    "greater_equal": convert_elementwise_op,
    "greater_than": convert_elementwise_op,
    "group_norm": convert_group_norm,
    "hard_shrink": convert_hard_shrink,
    "hard_sigmoid": convert_hard_sigmoid,
    "hard_swish": convert_hard_swish,
    "instance_norm": convert_instance_norm,
    "isfinite_v2": convert_unary_op,
    "isinf_v2": convert_unary_op,
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
    "logsumexp": convert_logsumexp,
    "lookup_table_v2": convert_lookup_table,
    "matmul": convert_matmul,
    "matmul_v2": convert_matmul,
    "meshgrid": convert_meshgrid,
    "mul": convert_mul,
    "mv": convert_mv,
    "nearest_interp_v2": convert_interpolate,
    "not_equal": convert_elementwise_op,
    "pad1d": convert_padding,
    "pad2d": convert_padding,
    "pad3d": convert_padding,
    "pixel_shuffle": convert_pixel_shuffle,
    "pool2d": convert_pool2d,
    "pow": convert_pow,
    "prelu": convert_prelu,
    "range": convert_range,
    "relu": convert_unary_op,
    "relu6": convert_relu6,
    "reshape2": convert_reshape,
    "round": convert_unary_op,
    "reciprocal": convert_reciprocal,
    "reduce_all": convert_reduce,
    "reduce_any": convert_reduce,
    "reduce_max": convert_reduce,
    "reduce_min": convert_reduce,
    "reduce_prod": convert_reduce,
    "reduce_sum": convert_reduce,
    "reduce_mean": convert_reduce,
    "rnn": convert_rnn,
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
    "size": convert_size,
    "slice": convert_slice,
    "softmax": convert_softmax,
    "softplus": convert_softplus,
    "softsign": convert_softsign,
    "split": convert_split,
    "strided_slice": convert_slice,
    "sqrt": convert_unary_op,
    "square": convert_square,
    "squeeze2": convert_squeeze,
    "swish": convert_swish,
    "tan": convert_unary_op,
    "tanh": convert_unary_op,
    "transpose2": convert_transpose,
    "unsqueeze2": convert_unsqueeze,
}


class GraphProto:
    """A helper class for handling relay functions from PaddlePaddle model."""

    def __init__(self):
        self.nodes = {}
        self.params = {}
        self.shape_dict = None

    def get_node(self, name):
        """get node from graph"""

        assert name in self.nodes
        return self.nodes[name]

    def add_node(self, name, node):
        """add a node to graph"""

        self.nodes[name] = fold_constant(node)

    def get_params(self, name=None):
        """Get params from graph."""

        if name is None:
            return self.params
        assert name in self.params
        return self.params[name]

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
                self.params[name] = _nd.array(scope[name])
            else:
                self.params[name] = _nd.array(np.array(scope.var(name).get_tensor()))
            shape = self.params[name].shape
            dtype = self.params[name].dtype
            self.nodes[name] = new_var(name, shape=shape, dtype=dtype)

    def check_input_shape(self, op, block):
        """Check the shape information of model's inputs, fixed shape is recommended."""

        ipt_name = op.input(op.input_names[0])
        ipt_shape = block.var(ipt_name).shape
        for i in ipt_shape:
            if i < 0:
                warning_msg = "Input {}(shape={}) has unkown dimension shapes. \
                               Specifying static values may improve performance".format(
                    ipt_name, ipt_shape
                )
                warnings.warn(warning_msg)

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
        for block in program.blocks:
            for op in block.ops:
                if op.type == "fetch":
                    continue
                convert_func = _convert_map[op.type]
                convert_func(self, op, block)

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

        outputs = [self.nodes[name] for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        return mod, self.params

    def from_translated_layer(self, layer, shape_dict):
        """Construct the TVM relay expression from PaddlePaddle TranslatedLayer."""

        self.shape_dict = shape_dict
        program = layer.program()
        parameters = dict()
        for param in layer.parameters() + layer.buffers():
            parameters[param.name] = np.array(param.value().get_tensor())
        self.check_unsupported_ops(program)
        self.extract_parameters(program, parameters)

        input_specs = layer._input_spec()
        self.ops_to_relay(program, input_specs)

        output_names = [x.name for x in layer._output_spec()]

        outputs = [self.nodes[name] for name in output_names]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        free_vars = analysis.free_vars(outputs)
        func = _function.Function(free_vars, outputs)
        mod = IRModule.from_expr(func)
        # remove unused parameters
        final_params = dict()
        for var in free_vars:
            if var.name_hint in self.params:
                final_params[var.name_hint] = self.params[var.name_hint]
        self.params = final_params
        return mod, self.params


def from_paddle(program_or_layer, shape_dict=None, scope=None):
    """Convert a PaddlePaddle model into an equivalent Relay Function.
    PaddlePaddle Program/TranslatedLayer represent the computation graph of PaddlePaddle model,
    and PaddlePaddle scope stores all the weights of PaddlePaddle model.

    Parameters
    ----------
    program_or_layer : object of `paddle.static.Program` or `paddle.jit.TranslatedLayer`
        Loaded model by `paddle.static.load_inference_model` or `paddle.jit.load`

    shape_dict : dict of str to tuple/list, optional
        The input shape of model

    scope : object of `paddle.static.Scope`, optional
        The scope that saves all the weights of model, use `paddle.static.global_scope` by default

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
    """

    import paddle

    # disable system signal capturing in paddle framework
    # the signal capturing may cause conflict while running autotvm with paddle frontend
    paddle.disable_signal_handler()

    g = GraphProto()
    if isinstance(program_or_layer, paddle.jit.TranslatedLayer):
        # model is loaded by `paddle.jit.load`
        mod, params = g.from_translated_layer(program_or_layer, shape_dict)
    elif isinstance(program_or_layer, paddle.static.Program):
        # model is loaded by `paddle.static.load_inference_model`
        mod, params = g.from_program(program_or_layer, shape_dict, scope)
    else:
        raise Exception("Only PaddlePaddle's Program and TranslatedLayer are supported.")
    return mod, params
