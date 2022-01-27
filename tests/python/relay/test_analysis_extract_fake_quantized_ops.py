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
import numpy as np
import pytest
import tvm
from tvm import relay


def test_fake_quantize_conv():
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
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)
        
    assert len(fake_quantized_op_freqs) == 1
    assert fake_quantized_op_freqs["nn.conv2d"] == 1


def test_fake_quantize_dense():
    x = relay.var("x", shape=[128, 64], dtype="int8")
    w = relay.var("w", shape=[256, 64], dtype="int8")
    one = relay.const(1.0)
    zero = relay.const(0)

    op = relay.op.nn.dense(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
    )
    op = relay.qnn.op.quantize(op, one, zero, out_dtype="int8")

    mod = tvm.IRModule.from_expr(op)
    fake_quantized_op_freqs = relay.analysis.list_fake_quantized_op_freqs(mod)

    assert len(fake_quantized_op_freqs) == 1
    assert fake_quantized_op_freqs["nn.dense"] == 1