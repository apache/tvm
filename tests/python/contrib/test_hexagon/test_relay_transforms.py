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
# pylint: disable=unused-wildcard-import, invalid-name

"""
Test hexagon relay transforms
"""
import tvm
from tvm import relay
from tvm.contrib.hexagon.transform import rewrite_qdistilbert, remove_empty_pad
from tvm import testing


def test_rewrite_qdistilbert():
    """Test case for rewrite_qdistilbert"""
    A = relay.var("A", shape=(12, 128, 64), dtype="int8")
    B = relay.var("B", shape=(12, 64, 128), dtype="int8")

    z = tvm.tir.IntImm("int64", 0)
    s1 = tvm.tir.IntImm("int64", 1)
    tx = tvm.tir.IntImm("int64", 128)
    ty = tvm.tir.IntImm("int64", 64)
    expand_dims = []
    for i in range(12):
        d1 = relay.const(13, dtype="int32")
        d2 = relay.const(1, dtype="int32")
        d3 = relay.const(0.0541715, dtype="float32")
        d4 = relay.const(0.0489368, dtype="float32")

        q1 = relay.const(0.00265098, dtype="float32")
        q2 = relay.const(0, dtype="int32")
        q3 = relay.const(0.728874, dtype="float32")
        q4 = relay.const(-14, dtype="int32")

        x = tvm.tir.IntImm("int64", i)
        y = tvm.tir.IntImm("int64", i + 1)

        SA = relay.op.strided_slice(
            A, begin=[x, z, z], end=[y, tx, ty], strides=[s1, s1, s1], axes=None
        )
        RA = relay.op.reshape(SA, [128, 64])
        SB = relay.op.strided_slice(
            B, begin=[x, z, z], end=[y, ty, tx], strides=[s1, s1, s1], axes=None
        )
        RB = relay.op.reshape(SB, [64, 128])
        TB = relay.op.transpose(RB, [1, 0])
        dense = relay.qnn.op.dense(RA, TB, d1, d2, d3, d4, units=None, out_dtype="int32")
        requantize = relay.qnn.op.requantize(dense, q1, q2, q3, q4)
        expand_dims.append(relay.op.expand_dims(requantize, axis=0))

    t = relay.expr.Tuple(expand_dims)
    graph = relay.op.concatenate(t, axis=0)

    func = relay.Function(relay.analysis.free_vars(graph), graph)
    mod = tvm.IRModule.from_expr(func)
    mod = rewrite_qdistilbert(mod)

    d1 = relay.const(13, dtype="int32")
    d2 = relay.const(1, dtype="int32")
    d3 = relay.const(0.0541715, dtype="float32")
    d4 = relay.const(0.0489368, dtype="float32")

    q1 = relay.const(0.00265098, dtype="float32")
    q2 = relay.const(0, dtype="int32")
    q3 = relay.const(0.728874, dtype="float32")
    q4 = relay.const(-14, dtype="int32")

    ref = relay.op.transpose(B, [0, 2, 1])
    ref = relay.qnn.op.batch_matmul(A, ref, d1, d2, d3, d4, out_dtype="int32")
    ref = relay.qnn.op.requantize(ref, q1, q2, q3, q4, out_dtype="int8")
    ref_func = relay.Function(relay.analysis.free_vars(ref), ref)
    ref_mod = tvm.IRModule.from_expr(ref_func)

    assert tvm.ir.structural_equal(mod["main"], ref_mod["main"])

    # If the pattern does not match, should return the original.
    func = relay.expr.Tuple(expand_dims)  # omitting concatenate
    mod = tvm.IRModule.from_expr(func)
    out_mod = rewrite_qdistilbert(mod)  # out does not return ref_mod but the original mod

    assert tvm.ir.structural_equal(mod["main"], out_mod["main"])


def test_remove_empty_pad():
    """Test case for remove_empty_pad"""
    A = relay.var("A", shape=(32, 32), dtype="float16")
    B = relay.var("B", shape=(32, 32), dtype="float16")

    p0 = relay.cast(relay.const(0, dtype="float32"), dtype="float16")
    p1 = relay.nn.pad(A, pad_value=p0, pad_width=((0, 0), (0, 0)))
    graph = relay.nn.matmul(p1, B)

    func = relay.Function(relay.analysis.free_vars(graph), graph)
    mod = tvm.IRModule.from_expr(func)

    mod = remove_empty_pad(mod)

    ref = relay.nn.matmul(A, B)
    ref_func = relay.Function(relay.analysis.free_vars(ref), ref)
    ref_mod = tvm.IRModule.from_expr(ref_func)

    assert tvm.ir.structural_equal(mod["main"], ref_mod["main"])


if __name__ == "__main__":
    testing.main()
