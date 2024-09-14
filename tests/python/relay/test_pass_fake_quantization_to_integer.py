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
import tvm.testing
from tvm import relay
from tvm.relay.transform import fake_quantization_to_integer


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
        zero_point = relay.const([np.random.randint(0, 255)] * 16)

        op = relay.op.nn.conv2d(
            relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(0)),
            relay.qnn.op.dequantize(
                w, relay.const(np.random.random([16]).astype("float32")), zero_point, axis=0
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
        w_np = np.random.randint(-128, 127, size=[3, 16, 5, 5], dtype="int8")

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


def test_fake_quantize_dense_bias():
    out_dtype = "int8"
    x = relay.var("x", shape=[128, 64], dtype="int8")
    w = relay.var("w", shape=[256, 64], dtype="int8")
    bias = relay.var("bias", shape=[256], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)
    w_scale = np.random.random([256]).astype("float32")

    op = relay.op.nn.dense(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(
            w,
            relay.const(w_scale),
            zero,
            axis=0,
        ),
        units=256,
    )

    op += relay.qnn.op.dequantize(
        bias,
        relay.const(2.0 * w_scale),
        zero,
    )

    op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

    x_np = np.random.randint(-128, 127, size=[128, 64], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[256, 64], dtype="int8")
    bias_np = np.random.randint(-128, 127, size=[256], dtype="int32")

    compare_fq_to_int(op, [x_np, w_np, bias_np], allow_rounding_error=True)


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


@pytest.mark.parametrize("const_bias", [False, True])
def test_fake_transpose_quantize_conv_bias_add(const_bias):
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)
    if const_bias:
        bias = relay.const(np.random.random(16).astype("float32"))
    else:
        bias = relay.qnn.op.dequantize(relay.var("bias", shape=[16], dtype="int32"), one, zero)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(0.5), zero), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(op, bias)
    op = relay.qnn.op.quantize(op, one, zero)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")
    args = [x_np, w_np]

    if not const_bias:
        args.append(bias_np)
    compare_fq_to_int(op, args)


def test_fake_transpose_quantize_conv_bias_add_per_channel():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)
    w_scale = (np.random.random([16]).astype("float32") - 0.5) / 10 + 0.5
    noise = (np.random.random([16]).astype("float32") - 0.5) * 1e-15
    w_zp = relay.const([0] * 16)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(
        x, relay.qnn.op.dequantize(w, relay.const(w_scale), w_zp, axis=0), kernel_size=[5, 5]
    )
    op = relay.op.nn.bias_add(
        op, relay.qnn.op.dequantize(bias, relay.const(2.0 * w_scale + noise), w_zp, axis=0)
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


@pytest.mark.parametrize("output_size", [None, 1])
def test_fake_quantize_adaptive_avgpool1d(output_size):
    x = relay.var("x", shape=[1, 128, 768], dtype="int8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(-12))
    op = relay.op.nn.adaptive_avg_pool1d(x, output_size)
    op = relay.qnn.op.quantize(op, relay.const(0.5), relay.const(10))

    x_np = np.random.randint(-128, 127, size=[1, 128, 768], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_avgpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(-12))
    op = relay.op.nn.avg_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(0.5), relay.const(10))

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_global_avg_pool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(-12))
    op = relay.op.nn.global_avg_pool2d(x)
    op = relay.qnn.op.quantize(op, relay.const(0.5), relay.const(10))

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], True)


class TestUnaryQNNOp:
    def helper_test_fake_quantize_unary_op(self, fp32_op, pos_values=False):
        for dtype in ["int8", "uint8"]:
            x = relay.var("x", shape=[1, 3, 3, 3], dtype=dtype)

            zero = -128 if dtype == "int8" else 0
            if pos_values:
                # Use a positive range for quanitzed ops that only work on positive values
                input_mid_point = relay.const(zero)
                output_mid_point = relay.const(zero)
            else:
                input_mid_point = relay.const(np.random.randint(0, 255) + zero)
                output_mid_point = relay.const(np.random.randint(0, 255) + zero)

            input_scale = relay.const(np.random.rand())
            output_scale = relay.const(np.random.rand())

            x = relay.qnn.op.dequantize(x, input_scale, input_mid_point)
            op = fp32_op(x)

            op = relay.qnn.op.quantize(op, output_scale, output_mid_point, out_dtype=dtype)

            x_np = np.random.randint(0 + zero, 255 + zero, size=[1, 3, 3, 3], dtype=dtype)

            compare_fq_to_int(op, [x_np], True)

    def test_sqrt(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.sqrt, pos_values=True)

    def test_rsqrt(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.rsqrt, pos_values=True)

    def test_exp(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.exp)

    def test_erf(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.erf)

    def test_sigmoid(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.sigmoid)

    def test_tanh(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.tanh)

    def test_log(self):
        self.helper_test_fake_quantize_unary_op(fp32_op=relay.log, pos_values=True)


def test_fake_quantize_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.reshape(x, [1, 3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_image_resize_bilinear():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.image.resize2d(x, size=[4, 4], method="linear")
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np], allow_rounding_error=True)


def test_fake_quantize_abs():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.abs(x)
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


@pytest.mark.parametrize("k", [0, 1, 5])
@pytest.mark.parametrize("axis", [0, -1, 1])
@pytest.mark.parametrize("is_ascend", [True, False])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_fake_quantize_topk(k, axis, is_ascend, dtype):
    x = relay.var("x", shape=[20, 100], dtype=dtype)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.topk(x, k, axis, "values", is_ascend, "float32")
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero, out_dtype=dtype)
    x_np = np.random.randint(0, 127, size=[20, 100], dtype=dtype)

    compare_fq_to_int(op, [x_np])


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


def test_fake_quantize_mean():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.mean(x)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np], allow_rounding_error=True)


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


def test_fake_quantize_leaky_relu():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.nn.leaky_relu(x, 0.1)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    compare_fq_to_int(op, [x_np], True)


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
    [relay.op.add, relay.op.multiply, relay.op.subtract, relay.op.minimum, relay.op.maximum],
)
def test_fake_quantize_binary_per_channel(operator):
    def verify_binary_per_channel(lhs_scale, rhs_scale, lhs_zp, rhs_zp, out_zp, lhs_axis, rhs_axis):
        if operator == relay.op.multiply:
            out_scale = relay.const(2.0)
            rhs_axis = lhs_axis  # TODO: Support different axes for per-channel quantized multiply
        else:
            out_scale = relay.const(0.1)

        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        x = relay.qnn.op.dequantize(x, relay.const(lhs_scale), relay.const(lhs_zp), axis=lhs_axis)

        y = relay.var("y", shape=[1, 3, 224, 224], dtype="int8")
        y = relay.qnn.op.dequantize(y, relay.const(rhs_scale), relay.const(rhs_zp), axis=rhs_axis)

        op = operator(x, y)

        op = relay.qnn.op.quantize(op, out_scale, relay.const(out_zp), out_dtype="int8")
        x_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")
        y_np = np.random.randint(-25, 25, size=[1, 3, 224, 224], dtype="int8")

        compare_fq_to_int(op, [x_np, y_np], allow_rounding_error=True)

    # Same axis
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 3),
        rhs_scale=np.random.uniform(1.0, 5.0, 3),
        lhs_zp=0,
        rhs_zp=0,
        out_zp=0,
        lhs_axis=1,
        rhs_axis=1,
    )
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 3),
        rhs_scale=np.random.uniform(1.0, 5.0, 3),
        lhs_zp=np.random.randint(1, 3),
        rhs_zp=np.random.randint(1, 3),
        out_zp=0,
        lhs_axis=1,
        rhs_axis=1,
    )
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 3),
        rhs_scale=np.random.uniform(1.0, 5.0, 3),
        lhs_zp=np.random.randint(1, 3),
        rhs_zp=np.random.randint(1, 3),
        out_zp=np.random.randint(1, 3),
        lhs_axis=1,
        rhs_axis=1,
    )
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 224),
        rhs_scale=np.random.uniform(1.0, 5.0, 224),
        lhs_zp=np.random.randint(1, 3),
        rhs_zp=np.random.randint(1, 3),
        out_zp=np.random.randint(1, 3),
        lhs_axis=-1,
        rhs_axis=-1,
    )

    # Different axes
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 224),
        rhs_scale=np.random.uniform(1.0, 5.0, 224),
        lhs_zp=0,
        rhs_zp=0,
        out_zp=0,
        lhs_axis=2,
        rhs_axis=3,
    )
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 224),
        rhs_scale=np.random.uniform(1.0, 5.0, 224),
        lhs_zp=np.random.randint(1, 3),
        rhs_zp=np.random.randint(1, 3),
        out_zp=0,
        lhs_axis=2,
        rhs_axis=3,
    )
    verify_binary_per_channel(
        lhs_scale=np.random.uniform(1.0, 5.0, 224),
        rhs_scale=np.random.uniform(1.0, 5.0, 224),
        lhs_zp=np.random.randint(1, 3),
        rhs_zp=np.random.randint(1, 3),
        out_zp=np.random.randint(1, 3),
        lhs_axis=2,
        rhs_axis=3,
    )


@pytest.mark.parametrize(
    "operator",
    [
        relay.op.add,
        relay.op.multiply,
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


def test_fake_quantize_subtract_different_output_zp():
    for dtype in ["uint8"]:
        x = relay.var("x", shape=[1, 128, 128, 3], dtype=dtype)
        x = relay.qnn.op.dequantize(x, relay.const(0.1), relay.const(0), axis=1)

        y = relay.const(0.5)

        op = relay.subtract(x, y)
        op = relay.transpose(op, axes=[0, 3, 1, 2])
        op = relay.qnn.op.quantize(op, relay.const(0.2), relay.const(128), out_dtype=dtype, axis=1)

        x_np = np.random.randint(0, 255, size=[1, 128, 128, 3], dtype=dtype)

        compare_fq_to_int(op, [x_np], True)


def test_fake_quantize_pad():
    x = relay.var("x", shape=[1, 383, 128], dtype="int8")
    x = relay.qnn.op.dequantize(x, relay.const(1.0), relay.const(10))
    op = relay.op.nn.pad(x, [[0, 0], [0, 1], [0, 0]], 0.0)
    op = relay.qnn.op.quantize(op, relay.const(1.0), relay.const(10), out_dtype="int8")

    x_np = np.random.randint(-25, 25, size=[1, 383, 128], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_pad_with_float_min():
    in_shape = [1, 383, 128]
    x = relay.var("x", shape=in_shape, dtype="float32")
    op = relay.qnn.quantize(x, relay.const(1.0), relay.const(0), out_dtype="uint8")
    op = relay.qnn.dequantize(op, relay.const(1.0), relay.const(0), out_dtype="float32")
    op = relay.op.nn.pad(
        op, pad_width=[[0, 0], [0, 1], [0, 0]], pad_value=relay.const(-3.40282e38, dtype="float32")
    )
    op = relay.qnn.op.quantize(op, relay.const(1.0), relay.const(0), out_dtype="uint8")
    x_np = np.random.randint(0, 256, size=in_shape)
    x_as_float = x_np.astype("float32")
    compare_fq_to_int(op, [x_as_float], True)


def test_fake_quantize_depth_to_space():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.depth_to_space(x, 4)
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_max_min():
    def run_test_case(partial_func):
        x = relay.var("x", shape=[1, 3, 10, 10], dtype="int8")

        zero = relay.const(0)
        x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
        # To be a little more realistic since max/min will rarely be by themselves
        x = relay.op.nn.depth_to_space(x, 4)
        op = partial_func(x)
        op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

        x_np = np.random.randint(-128, 127, size=[1, 3, 10, 10], dtype="int8")
        compare_fq_to_int(op, [x_np])

    run_test_case(relay.op.max)
    run_test_case(relay.op.min)

    # Test forwarding kwargs works
    run_test_case(lambda x: relay.op.max(x, axis=1))
    run_test_case(lambda x: relay.op.min(x, axis=1))


def test_fq_avg_pool_conv2d():
    dtype = "uint8"
    shape_x = [1, 4, 24, 24]
    shape_w = [8, 4, 1, 1]
    x = relay.var("x", shape=shape_x, dtype=dtype)
    w = relay.var("w", shape=shape_w, dtype=dtype)
    zero = relay.const(0)
    one = relay.const(1.0)

    # Tested expression.
    op0 = relay.qnn.op.dequantize(x, relay.const(0.64), relay.const(2))
    op1 = relay.op.nn.avg_pool2d(op0, [3, 3])
    op2 = relay.qnn.op.dequantize(w, relay.const(0.5), relay.const(10))
    op3 = relay.op.nn.conv2d(op1, op2, kernel_size=[1, 1])
    expr = relay.qnn.op.quantize(op3, one, zero, out_dtype="uint8")

    x_np = np.random.randint(0, 255, size=shape_x, dtype=dtype)
    w_np = np.random.randint(0, 255, size=shape_w, dtype=dtype)
    compare_fq_to_int(expr, [x_np, w_np])


def test_fq_hard_fail():
    @tvm.ir.register_op_attr("nn.conv2d", "FTVMFakeQuantizationToInteger", level=11)
    def conv2d(expr, type_map):  # pylint: disable=unused-variable
        raise NotImplementedError

    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)

    op = relay.op.nn.conv2d(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        kernel_size=[5, 5],
    )
    op = relay.qnn.op.quantize(op, one, zero, out_dtype="int8")
    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    mod_int = tvm.relay.transform.FakeQuantizationToInteger(hard_fail=False)(mod)
    tvm.ir.assert_structural_equal(mod_int, mod)
    # Catch a generic exception because the tvm FFI eats the python exception type
    with pytest.raises(Exception):
        mod_int = tvm.relay.transform.FakeQuantizationToInteger(hard_fail=True)(mod)


def compare_expected_fq_qat_to_int(expr, expected_expr, args, allow_rounding_error=False):
    mod = tvm.IRModule.from_expr(expr)
    mod_def = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger(False, True)(mod_def)
    mod_exp = tvm.relay.transform.InferType()(tvm.IRModule.from_expr(expected_expr))
    assert not tvm.ir.structural_equal(mod, mod_int)
    tvm.ir.assert_structural_equal(mod_int, mod_exp)
    result_def = (
        relay.create_executor("vm", mod=mod_def, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_exp = (
        relay.create_executor("vm", mod=mod_exp, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    if allow_rounding_error:
        assert np.all(np.abs(result_def.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result_def, result_int)

    assert np.array_equal(result_int, result_exp)


def test_fq_qat_op_positive_part():
    # Only the first operation is converted, since the next operation("add") is not enabled.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    a = relay.var("a", shape=shape_x, dtype="int8")
    b = relay.var("b", shape=shape_w, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))
    op1 = relay.qnn.op.dequantize(b, relay.const(6.0), relay.const(0))
    op2 = relay.op.nn.batch_matmul(op0, op1)
    op3 = relay.op.add(op2, relay.const(1.0))
    expr = relay.op.erf(op3)

    op0 = relay.qnn.op.qnn.batch_matmul(
        a, b, relay.const(0), relay.const(0), relay.const(2.0), relay.const(6.0)
    )
    op1 = relay.qnn.op.qnn.dequantize(op0, relay.const(12.0), relay.const(0))
    op2 = relay.op.add(op1, relay.const(1.0))
    expected_expr = relay.op.erf(op2)

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np, w_np])


def test_fq_qat_negative_all():
    # None of the operations are converted, since the first operation("add") is not enabled.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    a = relay.var("a", shape=shape_x, dtype="int8")
    b = relay.var("b", shape=shape_w, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))
    op1 = relay.qnn.op.dequantize(b, relay.const(6.0), relay.const(0))
    op2 = relay.op.add(op1, relay.const(1.0))
    op3 = relay.op.nn.batch_matmul(op0, op2)
    expr = relay.op.erf(op3)

    expected_expr = expr

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np, w_np])


def test_fq_qat_positive_single():
    # The single operation is converted.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    a = relay.var("a", shape=shape_x, dtype="int8")
    b = relay.var("b", shape=shape_w, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))
    op1 = relay.qnn.op.dequantize(b, relay.const(6.0), relay.const(0))
    expr = relay.op.nn.batch_matmul(op0, op1)

    op0 = relay.qnn.op.qnn.batch_matmul(
        a, b, relay.const(0), relay.const(0), relay.const(2.0), relay.const(6.0)
    )
    expected_expr = relay.qnn.op.qnn.dequantize(op0, relay.const(12.0), relay.const(0))

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np, w_np])


def test_fq_qat_positive_nothing_to_do():
    # All operations are converted by the non-QAT pass.
    shape_x = [1, 4, 2]
    shape_w = [1, 4, 2]
    a = relay.var("a", shape=shape_x, dtype="int8")
    b = relay.var("b", shape=shape_w, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))
    op1 = relay.qnn.op.dequantize(b, relay.const(6.0), relay.const(0))
    op2 = relay.op.nn.batch_matmul(op0, op1)
    op3 = relay.op.add(op2, relay.const(1.0))
    expr = relay.qnn.op.quantize(op3, relay.const(1.0), relay.const(0), out_dtype="int8")

    op0 = relay.qnn.op.batch_matmul(
        a, b, relay.const(0), relay.const(0), relay.const(2.0), relay.const(6.0)
    )
    op1 = relay.qnn.op.quantize(
        relay.const(1.0), relay.const(12.0), relay.const(0), out_dtype="int32"
    )
    op2 = relay.op.add(
        op0,
        op1,
    )
    expected_expr = relay.qnn.op.requantize(
        op2, relay.const(12.0), relay.const(0), relay.const(1.0), relay.const(0), out_dtype="int8"
    )

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np, w_np])


def test_fq_qat_positive_couple():
    # Several consecutive operations are converted.
    shape_x = [1, 2, 4]
    shape_w = [2]
    a = relay.var("a", shape=shape_x, dtype="int8")
    b = relay.var("b", shape=shape_w, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))
    op1 = relay.qnn.op.dequantize(b, relay.const(6.0), relay.const(0))
    op2 = relay.op.reshape(op0, (1, 4, 2))
    op3 = relay.op.broadcast_to(op1, (2, 2, 2))
    op4 = relay.op.nn.batch_matmul(op2, op3)
    expr = relay.op.erf(op4)

    op0 = relay.op.reshape(a, (1, 4, 2))
    op1 = relay.op.broadcast_to(b, (2, 2, 2))
    op3 = relay.qnn.op.qnn.batch_matmul(
        op0, op1, relay.const(0), relay.const(0), relay.const(2.0), relay.const(6.0)
    )
    op4 = relay.qnn.op.qnn.dequantize(op3, relay.const(12.0), relay.const(0))
    expected_expr = relay.op.erf(op4)

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np, w_np])


def test_fq_positive_single_arg_part():
    # The single-argument operation is converted.
    shape_x = [1, 2, 4]
    a = relay.var("a", shape=shape_x, dtype="int8")

    op0 = relay.qnn.op.dequantize(a, relay.const(2.0), relay.const(0))

    op1 = relay.op.reshape(op0, (1, 4, 2))
    expr = relay.op.erf(op1)

    op0 = relay.op.reshape(a, (1, 4, 2))
    op1 = relay.qnn.op.dequantize(op0, relay.const(2.0), relay.const(0))
    expected_expr = relay.op.erf(op1)
    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np])


def test_fq_qat_intermediate_infertype():
    # Complex conversion of non-QAT and QAT passes that form FakeQuantizationToInteger.
    shape_x = [1, 2, 4]
    x = relay.var("x", shape=shape_x, dtype="float32")
    const_0 = relay.const(np.random.uniform(size=[1, 4, 2]).astype("float32"))

    op0 = relay.qnn.op.quantize(x, relay.const(17.0), relay.const(0), out_dtype="int8")
    op1 = relay.qnn.op.dequantize(op0, relay.const(17.0), relay.const(0))
    op2 = relay.op.reshape(op1, (1, 4, 2))
    op3 = relay.qnn.op.quantize(op2, relay.const(10.0), relay.const(0), out_dtype="int8")
    op4 = relay.qnn.op.quantize(const_0, relay.const(1.0), relay.const(8), out_dtype="int8")
    op5 = relay.qnn.op.dequantize(op3, relay.const(10.0), relay.const(0))
    op6 = relay.qnn.op.dequantize(op4, relay.const(4.0), relay.const(9))
    op7 = relay.op.nn.batch_matmul(op5, op6)
    expr = relay.op.add(op7, relay.const(5.0))

    op0 = relay.qnn.op.quantize(x, relay.const(17.0), relay.const(0), out_dtype="int8")
    op1 = relay.op.reshape(op0, (1, 4, 2))
    op2 = relay.qnn.op.requantize(
        op1, relay.const(17.0), relay.const(0), relay.const(10.0), relay.const(0), out_dtype="int8"
    )
    op3 = relay.qnn.op.quantize(const_0, relay.const(1.0), relay.const(8), out_dtype="int8")
    op4 = relay.qnn.op.batch_matmul(
        op2, op3, relay.const(0), relay.const(9), relay.const(10.0), relay.const(4.0)
    )
    op5 = relay.qnn.op.dequantize(op4, relay.const(40.0), relay.const(0))
    expected_expr = relay.op.add(op5, relay.const(5.0))

    x_np = np.random.randint(-128, 127, size=shape_x, dtype="int32").astype("float32")
    compare_expected_fq_qat_to_int(expr, expected_expr, [x_np])


def test_fake_quantize_take():
    x = relay.var("x", shape=[33, 11], dtype="int8")
    indices_np = np.random.randint(0, 33, size=[37], dtype="int32")
    indices = relay.const(indices_np)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.take(x, indices, axis=0)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    x_np = np.random.randint(-25, 25, size=[33, 11], dtype="int8")

    compare_fq_to_int(op, [x_np])


def test_fake_quantize_softmax():
    shape = [5, 10]
    x_ = relay.var("x", shape=shape, dtype="int8")

    is_sorted = lambda a: np.all(a[:-1] <= a[1:])

    for scale in [1.0, 0.1, 0.01]:
        x = relay.qnn.op.dequantize(x_, relay.const(scale), relay.const(0))
        op = relay.op.nn.softmax(x, axis=1)
        op = relay.qnn.op.quantize(
            op, relay.const(1.0 / 256.0), relay.const(-128), out_dtype="int8"
        )

        x_np = np.random.randint(-128, 127, size=shape, dtype="int8")
        x_np = np.sort(x_np)
        args = [x_np]

        mod = tvm.IRModule.from_expr(op)
        mod = tvm.relay.transform.InferType()(mod)
        mod_int = tvm.relay.transform.FakeQuantizationToInteger(
            hard_fail=True, optional_qnn_ops=["nn.softmax"]
        )(mod)
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

        # Check at least the softmax output is in ascending order,
        # since it is difficult to use allclose due to not-so-good accuracy.
        for qdq, qop in zip(result, result_int):
            assert is_sorted(qdq)
            assert is_sorted(qop)

        try:
            np.testing.assert_allclose(result_int, result, atol=1)
        except AssertionError as e:
            # To see the difference
            print(e)


if __name__ == "__main__":
    tvm.testing.main()
