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
# pylint: disable=unused-wildcard-import
import numpy as np
import pytest
import tvm
from tvm import relay


def compare_fq_to_int(expr, args, allow_rounding_error=False):
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)

    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod_int)

    result = (
        relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )

    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)


def test_fake_quantize_conv():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.conv2d(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
            kernel_size=[5, 5],
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_quantize_conv_per_channel():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
        one = relay.const([1.0] * 16)
        zero = relay.const([0] * 16)

        op = relay.op.nn.conv2d(
            relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(0)),
            relay.qnn.op.dequantize(
                w, relay.const(np.random.random([16]).astype("float32")), zero, axis=0
            ),
            kernel_size=[5, 5],
            channels=16,
        )
        op = relay.qnn.op.quantize(op, relay.const(1.0), relay.const(0), out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np], allow_rounding_error=True)


def test_fake_quantize_transposeconv():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[3, 16, 5, 5], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.conv2d_transpose(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
            kernel_size=[5, 5],
            data_layout="NCHW",
            kernel_layout="IOHW",
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_quantize_dense():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[128, 64], dtype="int8")
        w = relay.var("w", shape=[256, 64], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.dense(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[128, 64], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[256, 64], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_quantize_dense_per_channel():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[128, 64], dtype="int8")
        w = relay.var("w", shape=[256, 64], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.dense(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(
                w,
                relay.const(np.random.random([256]).astype("float32")),
                relay.const([0] * 256),
                axis=0,
            ),
            units=256,
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[128, 64], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[256, 64], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np], allow_rounding_error=True)


def test_fake_quantize_batch_matmul():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 128, 64], dtype="int8")
        w = relay.var("w", shape=[1, 256, 64], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.batch_matmul(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        x_np = np.random.randint(-128, 127, size=[1, 128, 64], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[1, 256, 64], dtype="int8")

        compare_fq_to_int(op, [x_np, w_np])


def test_fake_transpose_quantize_conv():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

    compare_fq_to_int(op, [x_np, w_np])


def test_fake_transpose_quantize_conv_bias_add():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(op, relay.qnn.op.dequantize(bias, one, zero))
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np])


def test_fake_transpose_quantize_conv_bias_add_per_channel():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)
    w_scale = (np.random.random([16]).astype("float32") - 0.5) / 10 + 0.5
    w_zp = relay.const([0] * 16)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(w_scale), w_zp, axis=0), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(
        op, relay.qnn.op.dequantize(bias, relay.const(2.0 * w_scale), w_zp, axis=0)
    )
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np], allow_rounding_error=True)


def test_fake_transpose_quantize_conv_bias_add_mismatch():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    two = relay.const(2.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(op, relay.qnn.op.dequantize(bias, two, zero))
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np])


def test_fake_quantize_maxpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.max_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_avgpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.avg_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_global_avg_pool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.global_avg_pool2d(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.reshape(x, [1, 3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_expand_dims():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.expand_dims(x, axis=1)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_squeeze():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.squeeze(x, axis=[0])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_strided_slice():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.strided_slice(x, begin=[0, 0, 0, 0], end=[1, 1, 112, 112])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_split():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.split(x, axis=3, indices_or_sections=2)
    op = relay.qnn.op.quantize(op[0], relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])

    op = relay.op.split(x, axis=3, indices_or_sections=[56, 112, 168])
    op = relay.qnn.op.quantize(op[1], relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_batch_flatten():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.batch_flatten(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_transpose_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.transpose(x, [1, 0, 2, 3])
    op = relay.op.reshape(op, [3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_concat():
    zero = relay.const(0)
    inputs = []
    for i in range(4):
        inputs.append(
            relay.qnn.op.dequantize(
                relay.var("x%d" % i, shape=[1, 4], dtype="int8"), relay.const(i + 0.5), zero
            )
        )
    concat = relay.op.concatenate(inputs, axis=1)
    out = relay.qnn.op.quantize(concat, relay.const(3.5), zero)

    inputs_np = []
    for i in range(4):
        inputs_np.append(np.random.randint(-128, 127, size=[1, 4], dtype="int8"))

    compare_fq_to_int(out, inputs_np)


def test_fake_quantize_clip():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.clip(x, 0, 6)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_clip_per_channel():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(
        x, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), axis=1
    )
    op = relay.op.clip(x, 0, 6)
    op = relay.qnn.op.quantize(
        op, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), out_dtype="uint8", axis=1
    )

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_relu():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.nn.relu(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_relu_per_channel():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(
        x, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), axis=1
    )
    op = relay.op.nn.relu(x)
    op = relay.qnn.op.quantize(
        op, relay.const([1.0, 2.0, 3.0]), relay.const([96, 114, 128]), out_dtype="uint8", axis=1
    )

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np])


@pytest.mark.parametrize(
    "operator",
    [relay.op.add, relay.op.multiply, relay.op.subtract, relay.op.minimum, relay.op.maximum],
)
def test_fake_quantize_binary(operator):
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(0.1), relay.const(0))

    y = relay.var("y", shape=[1, 3, 224, 224], dtype="int8")
    y = relay.qnn.op.dequantize(y, relay.const(0.2), relay.const(0))

    op = operator(x, y)
    if operator == relay.op.multiply:
        out_scale = relay.const(20.0)
    else:
        out_scale = relay.const(0.1)

    op = relay.qnn.op.quantize(op, out_scale, relay.const(0), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")
    y_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np, y_np])


@pytest.mark.parametrize(
    "operator",
    [
        relay.op.add,
        relay.op.multiply,
        relay.op.subtract,
        relay.op.subtract,
        relay.op.minimum,
        relay.op.maximum,
    ],
)
def test_fake_quantize_binary_const(operator):
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(0.1), relay.const(10))

    y = relay.const(1.0)

    op = operator(x, y)
    op = relay.qnn.op.quantize(op, relay.const(0.1), relay.const(10), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_pad():
    x = relay.var("x", shape=[1, 383, 128], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(1.0), relay.const(10))
    op = relay.op.nn.pad(x, [[0, 0], [0, 1], [0, 0]], 0.0)
    op = relay.qnn.op.quantize(op, relay.const(1.0), relay.const(10), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 383, 128], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_depth_to_space():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.depth_to_space(x, 4)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])
