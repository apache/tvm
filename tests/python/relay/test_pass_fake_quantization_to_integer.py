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


def test_fake_quantize_conv():
    for out_dtype in ["int8", "uint8"]:
        x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
        w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
        one = relay.const(1.0)
        zero = relay.const(0)

        op = relay.op.nn.conv2d(
            relay.qnn.op.dequantize(x, relay.const(2.0), zero),
            relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        )
        op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

        mod = tvm.IRModule.from_expr(op)
        mod = tvm.relay.transform.InferType()(mod)

        x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
        w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

        mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
        assert not tvm.ir.structural_equal(mod, mod2)
        mod2 = tvm.relay.transform.FoldConstant()(mod2)

        ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
        result = ex.evaluate()(x_np, w_np).asnumpy()

        ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
        result2 = ex.evaluate()(x_np, w_np).asnumpy()

        assert np.array_equal(result, result2)


def test_fake_transpose_quantize_conv():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(x, relay.qnn.op.dequantize(w, relay.const(0.5), zero))
    op = relay.qnn.op.quantize(op, one, zero)

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np, w_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np, w_np).asnumpy()

    assert np.array_equal(result, result2)


def test_fake_transpose_quantize_conv_bias_add():
    x = relay.var("x", shape=[1, 224, 224, 3], dtype="int8")
    w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
    bias = relay.var("bias", shape=[16], dtype="int32")
    one = relay.const(1.0)
    zero = relay.const(0)

    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    x = relay.transpose(x, [0, 3, 1, 2])
    op = relay.op.nn.conv2d(x, relay.qnn.op.dequantize(w, relay.const(0.5), zero))
    op = relay.op.nn.bias_add(op, relay.qnn.op.dequantize(bias, one, zero))
    op = relay.qnn.op.quantize(op, one, zero)

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(-128, 127, size=[1, 224, 224, 3], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")
    bias_np = np.random.randint(-32768, 32767, size=[16], dtype="int32")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np, w_np, bias_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np, w_np, bias_np).asnumpy()

    assert np.array_equal(result, result2)


def test_fake_quantize_maxpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.max_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np).asnumpy()

    assert np.array_equal(result, result2)


def test_fake_quantize_avgpool():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.nn.avg_pool2d(x, [3, 3])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np).asnumpy()

    assert np.all(np.abs(result - result2) <= 1)


def test_fake_quantize_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.reshape(x, [1, 3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np).asnumpy()

    assert np.array_equal(result, result2)


def test_fake_quantize_transpose_reshape():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")

    zero = relay.const(0)
    x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
    op = relay.op.transpose(x, [1, 0, 2, 3])
    op = relay.op.reshape(op, [3, -1])
    op = relay.qnn.op.quantize(op, relay.const(2.0), zero)

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np).asnumpy()

    assert np.array_equal(result, result2)


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

    mod = tvm.IRModule.from_expr(out)
    mod = tvm.relay.transform.InferType()(mod)

    inputs_np = []
    for i in range(4):
        inputs_np.append(np.random.randint(-128, 127, size=[1, 4], dtype="int8"))

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(*inputs_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(*inputs_np).asnumpy()

    assert np.array_equal(result, result2)


def test_fake_quantize_clip():
    x = relay.var("x", shape=[1, 3, 224, 224], dtype="uint8")

    x = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(114))
    op = relay.op.clip(x, 0, 6)
    op = relay.qnn.op.quantize(op, relay.const(2.0), relay.const(114), out_dtype="uint8")

    mod = tvm.IRModule.from_expr(op)
    mod = tvm.relay.transform.InferType()(mod)

    x_np = np.random.randint(0, 255, size=[1, 3, 224, 224], dtype="uint8")

    mod2 = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod2)
    mod2 = tvm.relay.transform.FoldConstant()(mod2)

    ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
    result = ex.evaluate()(x_np).asnumpy()

    ex = relay.create_executor("vm", mod=mod2, device=tvm.cpu(), target="llvm")
    result2 = ex.evaluate()(x_np).asnumpy()

    assert np.array_equal(result, result2)
