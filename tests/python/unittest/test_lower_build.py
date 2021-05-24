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

import numpy as np

import tvm
from tvm import te, tir
from tvm.ir.module import IRModule
from tvm.script import ty
import tvm.testing


def _check_module_with_numpy(mod, shape=(128, 128, 128)):
    m, n, k = shape
    a = tvm.nd.array(np.random.rand(m, k).astype("float32"))
    b = tvm.nd.array(np.random.rand(n, k).astype("float32"))
    c = tvm.nd.array(np.zeros((m, n), dtype="float32"))
    c_np = np.dot(a.numpy(), b.numpy().transpose())
    mod(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-5)


# pylint: disable=no-self-argument, missing-class-docstring, missing-function-docstring
@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
class LoweredModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [128, 128])
        B = tir.match_buffer(b, [128, 128])
        C = tir.match_buffer(c, [128, 128])
        # body
        for x, y in tir.grid(128, 128):
            C.data[x * 128 + y] = 0.0
            for k in tir.serial(0, 128):
                C.data[x * 128 + y] = tir.load("float32", C.data, x * 128 + y) + tir.load(
                    "float32", A.data, x * 128 + k
                ) * tir.load("float32", B.data, y * 128 + k)


def test_lower_build_te_schedule():
    m, n, k = 128, 128, 128
    axis_k = te.reduce_axis((0, k), "k")
    A = te.placeholder((m, k), name="A")
    B = te.placeholder((k, n), name="B")
    C = te.compute((m, n), lambda x, y: te.sum(A[x, axis_k] * B[y, axis_k], axis=axis_k), name="C")
    s = te.create_schedule(C.op)
    # check lowering
    ir_mod = tvm.lower(s, [A, B, C])
    tvm.ir.assert_structural_equal(ir_mod, LoweredModule())
    # check building
    mod = tvm.build(s, [A, B, C], target="llvm")
    _check_module_with_numpy(mod)


def test_lower_build_tir_func():
    # check lowering
    ir_mod = tvm.lower(matmul)
    tvm.ir.assert_structural_equal(ir_mod, LoweredModule())
    # check building
    mod = tvm.build(matmul, target="llvm")
    _check_module_with_numpy(mod)


def test_lower_build_tir_module():
    func = matmul.with_attr("global_symbol", "main")
    func = func.with_attr("tir.noalias", True)
    ir_mod = IRModule({"main": func})
    # check lowering
    lowered_mod = tvm.lower(ir_mod)
    tvm.ir.assert_structural_equal(lowered_mod, LoweredModule())
    # check building
    mod = tvm.build(ir_mod, target="llvm")
    _check_module_with_numpy(mod)


def test_lower_build_lowered_module():
    # check lowering
    ir_mod = tvm.lower(LoweredModule())
    tvm.ir.assert_structural_equal(ir_mod, LoweredModule())
    # check building
    mod = tvm.build(ir_mod, target="llvm")
    _check_module_with_numpy(mod)


if __name__ == "__main__":
    test_lower_build_te_schedule()
    test_lower_build_tir_func()
    test_lower_build_tir_module()
    test_lower_build_lowered_module()
