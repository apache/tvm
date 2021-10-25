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


def convert_cast(g, op, block):
    """Operator converter for cast."""

    dtype = op.attr("out_dtype")
    dtype = _convert_dtype_value(dtype)
    x = g.get_node(op.input("X")[0])
    out = _op.cast(x, dtype=dtype)
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


def convert_expand(g, op, block):
    """Operator converter for expand."""

    x = g.get_node(op.input("X")[0])
    if op.input("Shape"):
        sizes = g.get_node(op.input("Shape")[0])
        sizes = try_infer_value(sizes, g.get_params())[0]
    else:
        sizes = op.attr("shape")

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
        shape = try_infer_value(shape, g.get_params())[0]
    if "ShapeTensor" in op.input_names and op.input("ShapeTensor"):
        shape = g.get_node(op.input("ShapeTensor")[0])
        shape = try_infer_value(shape, g.get_params())[0]

    if isinstance(shape, np.ndarray):
        shape = shape.tolist()

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


def convert_lookup_table(g, op, block):
    """Operator converter for lookup_table_v2."""

    indices = g.get_node(op.input("Ids")[0])
    padding_idx = op.attr("padding_idx")
    if padding_idx != -1:
        g.get_params[op.input("W")[0]][padding_idx] = 0.0
        g.add_node(op.input("W")[0], _expr.const(g.params[op.input("W")[0]]))
    weights = g.get_node(op.input("W")[0])
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
    if not isinstance(in_h, _op.Expr) and in_h < ksize[0]:
        ksize[0] = in_h
    if not isinstance(in_w, _op.Expr) and in_w < ksize[1]:
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
        tmp_shape = []
        for shape_name in input_shape_tensor:
            shape = g.get_node(shape_name)
            if len(infer_shape(shape)) == 0:
                shape = _op.reshape(shape, [-1])
            if isinstance(shape, _expr.Constant):
                tmp_shape.append(shape)
            elif isinstance(shape, _expr.Expr):
                tmp_shape.append(shape)
            else:
                tmp_shape.append(_expr.const(np.array(shape).astype("int64")))
        new_shape = _op.concatenate(tmp_shape, axis=0)
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
        out = _op.copy(x)
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


def convert_shape(g, op, block):
    """Operator converter for shape."""

    x = g.get_node(op.input("Input")[0])
    out = shape_of(x, dtype="int32")
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
        # if `starts` is a tensor
        starts = g.get_node(op.input("StartsTensor")[0])
        starts = _infer_value(starts, g.get_params())
    elif op.input("StartsTensorList"):
        # if `starts` is a list of tensor
        starts = []
        for start_index in op.input("StartsTensorList"):
            start_index = g.get_node(start_index).astype("int64")
            starts.append(start_index)
        starts = _op.concatenate(starts, axis=0)
        starts = _infer_value(starts, g.get_params())
    else:
        # if `starts` is constant value
        starts = op.attr("starts")

    if len(axes) < dims:
        # make the numel of `starts` be same with the rank of input tensor
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
        # if `ends` is a tensor
        ends = g.get_node(op.input("EndsTensor")[0])
        ends = _infer_value(ends, g.get_params())
    elif op.input("EndsTensorList"):
        # if `ends` is a list of tensor
        ends = []
        for end_index in op.input("EndsTensorList"):
            end_index = g.get_node(end_index).astype("int64")
            ends.append(end_index)
        ends = _op.concatenate(ends, axis=0)
        ends = _infer_value(ends, g.get_params())
    else:
        # if `ends` is constant value
        ends = op.attr("ends")

    if len(axes) < dims:
        # make the numel of `ends` be same with the rank of input tensor
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
        # if `strides` is a input tensor
        strides = g.get_node(op.input("StridesTensor")[0])
        strides = _infer_value(strides, g.get_params())
    elif "StridesTensorList" in op.input_names and op.input("StridesTensorList"):
        # if `strides` is a list of tensor
        strides = []
        for strides_index in op.input("StridesTensorList"):
            strides_index = g.get_node(strides_index).astype("int64")
            strides.append(strides_index)
        strides = _op.concatenate(strides, axis=0)
        strides = _infer_value(strides, g.get_params())
    elif op.has_attr("strides"):
        # if `strides` is constant value
        strides = op.attr("strides")
    else:
        # default value for `strides`
        strides = [1] * dims

    if len(axes) < dims:
        # make the numel of `strides` be same with the rank of input tensor
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
        else:
            base = [1] * dims
            for i, axis in enumerate(axes):
                base[axis] = strides[i]
            strides = base

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


def convert_squeeze(g, op, block):
    """Operator converter for squeeze2."""

    x = g.get_node(op.input("X")[0])
    axes = op.attr("axes")
    if not axes:
        axes = None
    x = _op.squeeze(x, axis=axes)
    g.add_node(op.output("Out")[0], x)


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
    "arg_max": convert_arg_max_min,
    "arg_min": convert_arg_max_min,
    "argsort": convert_argsort,
    "asin": convert_unary_op,
    "assign": convert_assign,
    "assign_value": convert_assign_value,
    "atan": convert_unary_op,
    "batch_norm": convert_batch_norm,
    "cast": convert_cast,
    "ceil": convert_unary_op,
    "concat": convert_concat,
    "conv2d": convert_conv2d,
    "cos": convert_unary_op,
    "cosh": convert_unary_op,
    "cumsum": convert_cumsum,
    "depthwise_conv2d": convert_conv2d,
    "dot": convert_dot,
    "dropout": convert_dropout,
    "elementwise_add": convert_elementwise_op,
    "elementwise_div": convert_elementwise_op,
    "elementwise_mul": convert_elementwise_op,
    "elementwise_sub": convert_elementwise_op,
    "equal": convert_elementwise_op,
    "erf": convert_unary_op,
    "exp": convert_unary_op,
    "expand_v2": convert_expand,
    "expand_as_v2": convert_expand_as,
    "feed": convert_feed,
    "fill_any_like": convert_fill_any_like,
    "fill_constant": convert_fill_constant,
    "floor": convert_unary_op,
    "gelu": convert_gelu,
    "hard_sigmoid": convert_hard_sigmoid,
    "hard_swish": convert_hard_swish,
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
    "logical_and": convert_binary_logical_op,
    "logical_or": convert_binary_logical_op,
    "logical_xor": convert_binary_logical_op,
    "lookup_table_v2": convert_lookup_table,
    "matmul": convert_matmul,
    "matmul_v2": convert_matmul,
    "mul": convert_mul,
    "pad3d": convert_padding,
    "pool2d": convert_pool2d,
    "relu": convert_unary_op,
    "reshape2": convert_reshape,
    "round": convert_unary_op,
    "rsqrt": convert_unary_op,
    "scale": convert_scale,
    "shape": convert_shape,
    "sigmoid": convert_unary_op,
    "sign": convert_unary_op,
    "sin": convert_unary_op,
    "sinh": convert_unary_op,
    "slice": convert_slice,
    "softmax": convert_softmax,
    "sqrt": convert_unary_op,
    "squeeze2": convert_squeeze,
    "tan": convert_unary_op,
    "tanh": convert_unary_op,
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
        for param in layer.parameters():
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
