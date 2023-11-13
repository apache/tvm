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
# pylint: disable=missing-docstring
from typing import Callable, List, Literal
import numpy as np
import pytest
from tvm.ir.module import IRModule

import tvm.testing
from tvm import dlight as dl, tir
from tvm.script import tir as T
from tvm.target import Target


def do_numeric_test(
    before: tir.PrimFunc,
    input_np: List[np.ndarray],
    compute: Callable[[List[np.ndarray]], np.ndarray],
    target: Target,
    dev: tvm.runtime.Device,
    rule_name: Literal["mma", "wmma"] = "mma",
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    before_mod = IRModule.from_expr(before.without_attr("global_symbol"))
    rule = (
        dl.gpu.MatmulTensorizationMMA() if rule_name == "mma" else dl.gpu.MatmulTensorizationWMMA()
    )
    with target:
        after_mod = dl.ApplyDefaultSchedule(rule)(before_mod)
    after = after_mod["main"]
    # build
    with tvm.transform.PassContext(config={"tir.use_async_copy": 1}):
        ex = tvm.build(after, target=target)

    tvm_input = [tvm.nd.array(x, dev) for x in input_np]
    ex(*tvm_input)
    tvm_result_np = tvm_input[-1].numpy()

    np_result = compute(*input_np[:-1])
    assert np.allclose(np_result, tvm_result_np, atol=atol, rtol=rtol)


@pytest.mark.parametrize("rule_name", ["mma", "wmma"])
def test_nt_matmul_mixed_precision(rule_name):
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(256), T.int64(256)), "float16")
        B = T.match_buffer(p_B, (T.int64(256), T.int64(256)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(256), T.int64(256)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(256), T.int64(256)))
        for i0, i1, i2, k in T.grid(b, T.int64(256), T.int64(256), T.int64(256)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_i2, v_k])
        for i0, i1, i2 in T.grid(b, T.int64(256), T.int64(256)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    b = 2
    inputs = [
        np.random.normal(size=(b, 256, 256)).astype(np.float16),
        np.random.normal(size=(256, 256)).astype(np.float16),
        np.zeros((b, 256, 256), dtype=np.float16),
    ]
    np_compute = lambda x, y: np.matmul(x, y.T)
    do_numeric_test(before, inputs, np_compute, Target("cuda"), tvm.cuda(), rule_name)


@pytest.mark.parametrize("rule_name", ["mma"])
def test_nn_matmul_mixed_precision(rule_name):
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(256), T.int64(256)), "float16")
        B = T.match_buffer(p_B, (T.int64(256), T.int64(256)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(256), T.int64(256)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(256), T.int64(256)))
        for i0, i1, i2, k in T.grid(b, T.int64(256), T.int64(256), T.int64(256)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", A[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_k, v_i2])
        for i0, i1, i2 in T.grid(b, T.int64(256), T.int64(256)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    b = 2
    inputs = [
        np.random.normal(size=(b, 256, 256)).astype(np.float16),
        np.random.normal(size=(256, 256)).astype(np.float16),
        np.zeros((b, 256, 256), dtype=np.float16),
    ]
    np_compute = lambda x, y: np.matmul(x, y)
    do_numeric_test(before, inputs, np_compute, Target("cuda"), tvm.cuda(), rule_name)


@pytest.mark.parametrize("rule_name", ["mma"])
def test_tn_matmul_mixed_precision(rule_name):
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(256), T.int64(256)), "float16")
        B = T.match_buffer(p_B, (T.int64(256), T.int64(256)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(256), T.int64(256)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(256), T.int64(256)))
        for i0, i1, i2, k in T.grid(b, T.int64(256), T.int64(256), T.int64(256)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", A[v_i0, v_k, v_i1]) * T.Cast("float32", B[v_k, v_i2])
        for i0, i1, i2 in T.grid(b, T.int64(256), T.int64(256)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    b = 2
    inputs = [
        np.random.normal(size=(b, 256, 256)).astype(np.float16),
        np.random.normal(size=(256, 256)).astype(np.float16),
        np.zeros((b, 256, 256), dtype=np.float16),
    ]
    np_compute = lambda x, y: np.matmul(x.transpose(0, 2, 1), y)
    do_numeric_test(before, inputs, np_compute, Target("cuda"), tvm.cuda(), rule_name)


@pytest.mark.parametrize("rule_name", ["mma"])
def test_tt_matmul_mixed_precision(rule_name):
    @T.prim_func
    def before(p_A: T.handle, p_B: T.handle, p_O: T.handle):
        b = T.int64()
        A = T.match_buffer(p_A, (b, T.int64(256), T.int64(256)), "float16")
        B = T.match_buffer(p_B, (T.int64(256), T.int64(256)), "float16")
        O = T.match_buffer(p_O, (b, T.int64(256), T.int64(256)), "float16")
        var_matmul_intermediate = T.alloc_buffer((b, T.int64(256), T.int64(256)))
        for i0, i1, i2, k in T.grid(b, T.int64(256), T.int64(256), T.int64(256)):
            with T.block("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[
                    v_i0, v_i1, v_i2
                ] + T.Cast("float32", A[v_i0, v_k, v_i1]) * T.Cast("float32", B[v_i2, v_k])
        for i0, i1, i2 in T.grid(b, T.int64(256), T.int64(256)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                O[v_i0, v_i1, v_i2] = T.Cast("float16", var_matmul_intermediate[v_i0, v_i1, v_i2])

    b = 2
    inputs = [
        np.random.normal(size=(b, 256, 256)).astype(np.float16),
        np.random.normal(size=(256, 256)).astype(np.float16),
        np.zeros((b, 256, 256), dtype=np.float16),
    ]
    np_compute = lambda x, y: np.matmul(x.transpose(0, 2, 1), y.T)
    do_numeric_test(before, inputs, np_compute, Target("cuda"), tvm.cuda(), rule_name)


if __name__ == "__main__":
    tvm.testing.main()
