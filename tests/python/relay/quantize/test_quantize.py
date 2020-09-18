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
import tvm
from tvm import relay
from tvm.relay.transform.quantize import (
    Quantizer,
    Conv2DPattern,
    Conv2DBiasAddPattern,
    DensePattern,
    DenseBiasAddPattern,
    AddPattern,
    MultiplyPattern,
)
from tvm.relay.op.nn.utils import get_pad_tuple2d
from tvm.relay.frontend.common import infer_type
import numpy as np


def quantize_and_check(
    pre_func, expected_func, quantizer_pattern_list, skip_first=False, skip_last=False
):
    quantizer = Quantizer(
        pre_func, None, quantizer_pattern_list, skip_first=skip_first, skip_last=skip_last
    )
    q_func = infer_type(quantizer.quantized_func)
    expected_func = infer_type(expected_func)
    assert tvm.ir.structural_equal(q_func, expected_func)


def create_scale_zps(lhs_name, rhs_name, channels=None):
    data_scale_var = relay.var(lhs_name + "_scale_0", shape=(), dtype="float32")
    data_zp_var = relay.var(lhs_name + "_zero_pt_0", shape=(), dtype="int32")

    if not channels:
        weight_scale_var = relay.var(rhs_name + "_scale_1", shape=(), dtype="float32")
        weight_zp_var = relay.var(rhs_name + "_zero_pt_1", shape=(), dtype="int32")
    else:
        weight_scale_var = relay.var(rhs_name + "_scale_1", shape=(channels,), dtype="float32")
        weight_zp_var = relay.var(rhs_name + "_zero_pt_1", shape=(channels,), dtype="int32")

    return data_scale_var, data_zp_var, weight_scale_var, weight_zp_var


def get_conv2d_axes(attrs):
    kernel_layout = attrs["kernel_layout"]
    data_layout = attrs["data_layout"]

    if kernel_layout == "OIHW":
        weight_channel_axis = 0
    elif kernel_layout == "HWIO":
        weight_channel_axis = 3
    else:
        raise ValueError(
            "We don't support layouts other than OIHW or HWIO, but got %s. Please provide a compatible layout to the test. ",
            kernel_layout,
        )

    if data_layout == "NCHW":
        data_channel_axis = 1
    elif data_layout == "NHWC":
        data_channel_axis = 3
    else:
        raise ValueError(
            "We don't support layouts other than NCHW or NHWC, but got %s. Please provide a compatible layout to the test. ",
            data_layout,
        )

    return data_channel_axis, weight_channel_axis


def create_conv2d_func(data_shape, weight_shape, attrs):
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))

    # Pre quantize input
    conv2d = relay.op.nn.conv2d(data, weight, **attrs)
    pre_func = relay.Function([data, weight], conv2d)
    return pre_func, data, weight


def create_q_conv2d_func(data, weight, weight_shape, attrs):
    data_channel_axis, weight_channel_axis = get_conv2d_axes(attrs)
    # Post quantize output
    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps(
        "conv2d_data", "conv2d_weight"
    )

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var, axis=data_channel_axis)
    q_weight = relay.qnn.op.quantize(
        weight, weight_scale_var, weight_zp_var, axis=weight_channel_axis
    )

    if "padding" in attrs.keys():
        padding = attrs["padding"]
    else:
        padding = None

    kernel_layout = attrs["kernel_layout"]
    data_layout = attrs["data_layout"]

    if padding is not None:
        top, left, bottom, right = get_pad_tuple2d(padding)
        if kernel_layout == "OIHW":
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
        elif kernel_layout == "HWIO":
            pad_width = (
                (top, bottom),
                (left, right),
                (0, 0),
                (0, 0),
            )
        pad_val = 0
        q_data = relay.op.nn.pad(q_data, pad_width, pad_val)

    if kernel_layout == "OIHW":
        kernel_size = tuple(weight_shape[2:4])
    elif kernel_layout == "HWIO":
        kernel_size = tuple(weight_shape[0:2])
    else:
        raise ValueError(
            "We don't support layouts other than OIHW or HWIO, but got %s. Please provide a compatible layout to the test. ",
            kernel_layout,
        )

    q_conv2d = relay.qnn.op.conv2d(
        q_data,
        q_weight,
        data_zp_var,
        weight_zp_var,
        data_scale_var,
        weight_scale_var,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=kernel_size,
        channels=weight_shape[weight_channel_axis],
    )

    deq_conv2d = relay.qnn.op.dequantize(
        q_conv2d,
        data_scale_var * weight_scale_var,
        relay.const(0, dtype="int32"),
        out_dtype="float32",
        axis=data_channel_axis,
    )
    quantized_func = relay.Function(
        [data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_conv2d
    )
    return quantized_func


def verify_conv2d(data_shape, weight_shape, attrs):
    pre_func, data, weight = create_conv2d_func(data_shape, weight_shape, attrs)

    quantized_func = create_q_conv2d_func(data, weight, weight_shape, attrs)
    quantize_and_check(pre_func, quantized_func, [Conv2DPattern(None)])


def create_conv2d_bias_func(data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add"):
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    bias = relay.const(np.random.rand(*bias_shape).astype("float32"), "float32")

    conv2d = relay.op.nn.conv2d(data, weight, **attrs)
    data_channel_axis, _ = get_conv2d_axes(attrs)
    if bias_type == "normal_add":
        bias_add = relay.op.add(conv2d, bias)
    elif bias_type == "bias_add":
        bias_add = relay.op.nn.bias_add(conv2d, bias, axis=data_channel_axis)
    else:
        raise ValueError(
            "Please pass in a valid bias type to the test function, got %s" % bias_type
        )
    pre_func = relay.Function([data, weight], bias_add)
    return pre_func, data, weight, bias


def create_q_conv2d_bias_func(data, weight, bias, weight_shape, attrs, bias_type="bias_add"):

    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps(
        "conv2d_data", "conv2d_weight"
    )
    data_channel_axis, weight_channel_axis = get_conv2d_axes(attrs)

    q_data = relay.qnn.op.quantize(
        data, data_scale_var, data_zp_var, axis=data_channel_axis
    )  # Put axis in
    q_weight = relay.qnn.op.quantize(
        weight, weight_scale_var, weight_zp_var, axis=weight_channel_axis
    )

    if "padding" in attrs.keys():
        padding = attrs["padding"]
    else:
        padding = None

    kernel_layout = attrs["kernel_layout"]
    data_layout = attrs["data_layout"]

    if padding is not None:
        top, left, bottom, right = get_pad_tuple2d(padding)
        kernel_layout = attrs["kernel_layout"]
        if kernel_layout == "OIHW":
            pad_width = ((0, 0), (0, 0), (top, bottom), (left, right))
        elif kernel_layout == "HWIO":
            pad_width = (
                (top, bottom),
                (left, right),
                (0, 0),
                (0, 0),
            )
        pad_val = 0
        q_data = relay.op.nn.pad(q_data, pad_width, pad_val)

    if kernel_layout == "OIHW":
        kernel_size = tuple(weight_shape[2:4])
    elif kernel_layout == "HWIO":
        kernel_size = tuple(weight_shape[0:2])
    else:
        raise ValueError(
            "We don't support layouts other than OIHW or HWIO, but got %s. Please provide a compatible layout to the test. ",
            kernel_layout,
        )

    q_conv2d = relay.qnn.op.conv2d(
        q_data,
        q_weight,
        data_zp_var,
        weight_zp_var,
        data_scale_var,
        weight_scale_var,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        kernel_size=kernel_size,
        channels=weight_shape[weight_channel_axis],
    )

    bias_add = relay.op.nn.bias_add(
        q_conv2d,
        relay.qnn.op.quantize(bias, data_scale_var, data_zp_var, axis=0, out_dtype="int32"),
        axis=data_channel_axis,
    )

    if bias_type == "normal_add":
        bias_add = relay.op.add(
            q_conv2d,
            relay.qnn.op.quantize(bias, data_scale_var, data_zp_var, axis=0, out_dtype="int32"),
        )
    elif bias_type == "bias_add":
        bias_add = relay.op.nn.bias_add(
            q_conv2d,
            relay.qnn.op.quantize(bias, data_scale_var, data_zp_var, axis=0, out_dtype="int32"),
            axis=data_channel_axis,
        )
    else:
        raise ValueError(
            "Please pass in a valid bias type to the test function, got %s" % bias_type
        )

    deq_conv2d = relay.qnn.op.dequantize(
        bias_add,
        data_scale_var * weight_scale_var,
        relay.const(0, dtype="int32"),
        out_dtype="float32",
        axis=data_channel_axis,
    )
    quantized_func = relay.Function(
        [data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_conv2d
    )
    return quantized_func


def verify_conv2d_bias(data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add"):
    pre_func, data, weight, bias = create_conv2d_bias_func(
        data_shape, weight_shape, bias_shape, attrs, bias_type
    )
    quantized_func = create_q_conv2d_bias_func(data, weight, bias, weight_shape, attrs, bias_type)
    quantize_and_check(pre_func, quantized_func, [Conv2DBiasAddPattern(None)])


def create_dense_func(data_shape, weight_shape, attrs):
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))

    pre_func = relay.Function([data, weight], relay.nn.dense(data, weight, **attrs))

    return pre_func, data, weight


def create_q_dense_func(data, weight, attrs):
    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps(
        "dense_data", "dense_weight"
    )

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var)
    q_weight = relay.qnn.op.quantize(weight, weight_scale_var, weight_zp_var, axis=0)

    q_dense = relay.qnn.op.dense(
        q_data, q_weight, data_zp_var, weight_zp_var, data_scale_var, weight_scale_var, **attrs
    )
    deq_dense = relay.qnn.op.dequantize(
        q_dense, data_scale_var * weight_scale_var, relay.const(0, dtype="int32"), axis=1
    )
    quantized_func = relay.Function(
        [data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_dense
    )

    return quantized_func


def verify_dense(data_shape, weight_shape, attrs):
    pre_func, data, weight = create_dense_func(data_shape, weight_shape, attrs)
    quantized_func = create_q_dense_func(data, weight, attrs)
    quantize_and_check(pre_func, quantized_func, [DensePattern(None)])


def create_dense_bias_func(data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add"):
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    bias = relay.const(np.random.rand(*bias_shape).astype("float32"), "float32")
    dense = relay.nn.dense(data, weight, **attrs)

    if bias_type == "normal_add":
        bias_add = relay.op.add(dense, bias)
    elif bias_type == "bias_add":
        bias_add = relay.op.nn.bias_add(dense, bias, axis=1)
    else:
        raise ValueError(
            "Please pass in a valid bias type to the test function, got %s" % bias_type
        )

    pre_func = relay.Function([data, weight], bias_add)

    return pre_func, data, weight, bias


def create_q_dense_bias_func(data, weight, bias, attrs, bias_type="bias_add"):
    data_scale_var, data_zp_var, weight_scale_var, weight_zp_var = create_scale_zps(
        "dense_data", "dense_weight"
    )

    q_data = relay.qnn.op.quantize(data, data_scale_var, data_zp_var)
    q_weight = relay.qnn.op.quantize(weight, weight_scale_var, weight_zp_var, axis=0)
    q_bias = relay.qnn.op.quantize(bias, data_scale_var, data_zp_var, axis=0, out_dtype="int32")

    q_dense = relay.qnn.op.dense(
        q_data, q_weight, data_zp_var, weight_zp_var, data_scale_var, weight_scale_var, **attrs
    )

    if bias_type == "normal_add":
        bias_add = relay.op.add(q_dense, q_bias)
    elif bias_type == "bias_add":
        bias_add = relay.op.nn.bias_add(q_dense, q_bias, axis=1)
    else:
        raise ValueError(
            "Please pass in a valid bias type to the test function, got %s" % bias_type
        )

    deq_dense = relay.qnn.op.dequantize(
        bias_add, data_scale_var * weight_scale_var, relay.const(0, dtype="int32"), axis=1
    )
    quantized_func = relay.Function(
        [data, weight, data_scale_var, data_zp_var, weight_scale_var, weight_zp_var], deq_dense
    )
    return quantized_func


def verify_dense_bias(data_shape, weight_shape, bias_shape, attrs, bias_type="bias_add"):
    pre_func, data, weight, bias = create_dense_bias_func(
        data_shape, weight_shape, bias_shape, attrs, bias_type
    )
    quantized_func = create_q_dense_bias_func(data, weight, bias, attrs, bias_type)
    quantize_and_check(pre_func, quantized_func, [DenseBiasAddPattern(None)])


def create_add_func(lhs_shape, rhs_shape):
    lhs = relay.var("lhs", relay.TensorType(lhs_shape, dtype="float32"))
    rhs = relay.var("rhs", relay.TensorType(rhs_shape, dtype="float32"))
    pre_func = relay.Function([lhs, rhs], relay.add(lhs, rhs))

    return pre_func, lhs, rhs


def create_q_add_func(lhs, rhs):
    lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var = create_scale_zps("add_lhs", "add_rhs")
    q_lhs = relay.qnn.op.quantize(lhs, lhs_scale_var, lhs_zp_var)
    q_rhs = relay.qnn.op.quantize(rhs, rhs_scale_var, rhs_zp_var)

    deq_lhs = relay.qnn.op.dequantize(q_lhs, lhs_scale_var, relay.const(0, dtype="int32"))
    deq_rhs = relay.qnn.op.dequantize(q_rhs, rhs_scale_var, relay.const(0, dtype="int32"))

    add_scale = relay.op.add(lhs_scale_var, rhs_scale_var)

    requantized_lhs = relay.qnn.op.quantize(deq_lhs, add_scale, relay.const(0, dtype="int32"))
    requantized_rhs = relay.qnn.op.quantize(deq_rhs, add_scale, relay.const(0, dtype="int32"))

    add = relay.op.add(requantized_lhs, requantized_rhs)
    deq_add = relay.qnn.op.dequantize(add, add_scale, relay.const(0, dtype="int32"))

    quantized_func = relay.Function(
        [lhs, rhs, lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var], deq_add
    )
    return quantized_func


def verify_add(lhs_shape, rhs_shape):
    pre_func, lhs, rhs = create_add_func(lhs_shape, rhs_shape)
    quantized_func = create_q_add_func(lhs, rhs)

    quantize_and_check(pre_func, quantized_func, [AddPattern(None)])


def create_mul_func(lhs_shape, rhs_shape):
    lhs = relay.var("lhs", relay.TensorType(lhs_shape, dtype="float32"))
    rhs = relay.var("rhs", relay.TensorType(rhs_shape, dtype="float32"))
    pre_func = relay.Function([lhs, rhs], relay.multiply(lhs, rhs))

    return pre_func, lhs, rhs


def create_q_mul_func(lhs, rhs):
    lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var = create_scale_zps("mul_lhs", "mul_rhs")
    q_lhs = relay.qnn.op.quantize(lhs, lhs_scale_var, lhs_zp_var)
    q_rhs = relay.qnn.op.quantize(rhs, rhs_scale_var, rhs_zp_var)

    zeroed_q_lhs = relay.op.subtract(relay.op.cast(q_lhs, "int32"), lhs_zp_var)
    zeroed_q_rhs = relay.op.subtract(relay.op.cast(q_rhs, "int32"), rhs_zp_var)

    multiply = relay.op.multiply(zeroed_q_lhs, zeroed_q_rhs)
    deq_multiply = relay.qnn.op.dequantize(
        multiply, lhs_scale_var * rhs_scale_var, relay.const(0, dtype="int32")
    )

    quantized_func = relay.Function(
        [lhs, rhs, lhs_scale_var, lhs_zp_var, rhs_scale_var, rhs_zp_var], deq_multiply
    )
    return quantized_func


def verify_mul(lhs_shape, rhs_shape):
    pre_func, lhs, rhs = create_mul_func(lhs_shape, rhs_shape)
    quantized_func = create_q_mul_func(lhs, rhs)
    quantize_and_check(pre_func, quantized_func, [MultiplyPattern(None)])


def verify_skip_layers(data_shape, weight_shape, attrs):
    # We'll test skip_layers with the dense op
    data = relay.var("data", relay.TensorType(data_shape, "float32"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "float32"))
    pre_func = relay.Function([data, weight], relay.nn.dense(data, weight))

    quantize_and_check(pre_func, pre_func, [DensePattern(None)], skip_first=True, skip_last=False)
    quantize_and_check(pre_func, pre_func, [DensePattern(None)], skip_first=False, skip_last=True)
    quantize_and_check(pre_func, pre_func, [DensePattern(None)], skip_first=True, skip_last=True)


def test_conv2d():
    verify_conv2d(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_conv2d(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
    )


def test_conv2d_bias():
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (32,),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
    )
    verify_conv2d_bias(
        (2, 3, 32, 32),
        (32, 3, 3, 3),
        (1, 32, 1, 1),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "OIHW",
            "data_layout": "NCHW",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
    )
    verify_conv2d_bias(
        (2, 32, 32, 3),
        (3, 3, 3, 32),
        (1, 1, 1, 32),
        {
            "kernel_size": [3, 3],
            "kernel_layout": "HWIO",
            "data_layout": "NHWC",
            "padding": [0, 0, 0, 0],
        },
        bias_type="normal_add",
    )


def test_dense():
    verify_dense((1, 8), (16, 8), {"units": 16})
    verify_dense((1, 4), (3, 4), {"units": 3})


def test_dense_bias():
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16})
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3})
    verify_dense_bias((1, 8), (16, 8), (16,), {"units": 16}, bias_type="normal_add")
    verify_dense_bias((1, 4), (3, 4), (3,), {"units": 3}, bias_type="normal_add")


def test_add():
    verify_add((1, 2, 3), (1, 2, 3))


def test_mul():
    verify_mul((1, 2, 3), (1, 2, 3))


def test_skip_layers():
    verify_skip_layers((1, 8), (16, 8), {"units": 16})
    verify_skip_layers((1, 4), (3, 4), {"units": 3})


if __name__ == "__main__":
    test_conv2d()
    test_conv2d_bias()
    test_dense()
    test_dense_bias()
    test_add()
    test_mul()
    test_skip_layers()
