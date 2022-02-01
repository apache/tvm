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
"""Test function extraction"""
import tvm
from tvm import relay


def test_fake_quantize_conv():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    zero = relay.const(0)

    op = relay.op.nn.conv2d(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        kernel_size=[5, 5],
    )
    op = relay.qnn.op.quantize(op, relay.const(1.0), zero, out_dtype="int8")

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert dict(fake_quantized_op_freqs) == {"nn.conv2d": 1}


def test_fake_quantize_dense():
    x = relay.var("x", shape=[128, 64], dtype="int8")
    w = relay.var("w", shape=[256, 64], dtype="int8")
    zero = relay.const(0)

    op = relay.op.nn.dense(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
    )
    op = relay.qnn.op.quantize(op, relay.const(1.0), zero, out_dtype="int8")

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert dict(fake_quantized_op_freqs) == {"nn.dense": 1}


def test_fake_quantize_multiple_regions():
    x = relay.var("x", shape=[128, 64], dtype="int8")
    w = relay.var("w", shape=[256, 64], dtype="int8")
    zero = relay.const(0)

    op = relay.op.nn.dense(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
    )
    op = relay.qnn.op.quantize(op, relay.const(1.0), zero, out_dtype="int8")

    op = relay.qnn.op.dequantize(op, relay.const(2.0), relay.const(114))
    op = relay.op.nn.relu(op)
    op = relay.qnn.op.quantize(op, relay.const(1.0), zero, out_dtype="int8")

    w2 = relay.var("w2", shape=[64, 256], dtype="int8")
    op = relay.op.nn.dense(
        relay.qnn.op.dequantize(op, relay.const(1.0), zero),
        relay.qnn.op.dequantize(w2, relay.const(0.5), zero),
    )
    op = relay.qnn.op.quantize(op, relay.const(1.0), zero, out_dtype="int8")

    # We expect to ignore this sigmoid op since it's just outside a fake
    # quantized region
    op = relay.op.sigmoid(op)

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert dict(fake_quantized_op_freqs) == {"nn.dense": 2, "nn.relu": 1}


def test_fake_quantize_maxpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.max_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert dict(fake_quantized_op_freqs) == {"nn.max_pool2d": 1}


def test_fake_quantize_transpose_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.transpose(x, [1, 0, 2, 3])
    op = relay.op.reshape(op, [3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert dict(fake_quantized_op_freqs) == {"transpose": 1, "reshape": 1}


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
    op = relay.qnn.op.quantize(concat, relay.const(3.5), zero)

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert dict(fake_quantized_op_freqs) == {"concatenate": 1}
