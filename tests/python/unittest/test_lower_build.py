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
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import tir as T
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
@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.float32(0)
        for k in range(128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.ir_module
class LoweredModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "from_legacy_te_schedule": True, "tir.noalias": True})
        A_flat = T.Buffer([16384], data=A.data)
        B_flat = T.Buffer([16384], data=B.data)
        C_flat = T.Buffer([16384], data=C.data)
        # body
        for x, y in T.grid(128, 128):
            C_flat[x * 128 + y] = 0.0
            for k in T.serial(0, 128):
                C_flat[x * 128 + y] = (
                    C_flat[x * 128 + y] + A_flat[x * 128 + k] * B_flat[y * 128 + k]
                )


@tvm.script.ir_module
class LoweredTIRModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A_flat = T.Buffer([16384], data=A.data)
        B_flat = T.Buffer([16384], data=B.data)
        C_flat = T.Buffer([16384], data=C.data)
        # body
        for x, y in T.grid(128, 128):
            C_flat[x * 128 + y] = 0.0
            for k in T.serial(0, 128):
                C_flat[x * 128 + y] = (
                    C_flat[x * 128 + y] + A_flat[x * 128 + k] * B_flat[y * 128 + k]
                )


def test_lower_build_te_schedule():
    m, n, k = 128, 128, 128
    axis_k = te.reduce_axis((0, k), "k")
    A = te.placeholder((m, k), name="A")
    B = te.placeholder((k, n), name="B")
    C = te.compute((m, n), lambda x, y: te.sum(A[x, axis_k] * B[y, axis_k], axis=axis_k), name="C")
    s = te.create_schedule(C.op)
    # check lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        ir_mod = tvm.lower(s, [A, B, C])
    tvm.ir.assert_structural_equal(ir_mod, LoweredModule)
    # check building
    mod = tvm.build(s, [A, B, C], target="llvm")
    _check_module_with_numpy(mod)


def test_lower_build_tir_func():
    # check lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        ir_mod = tvm.lower(matmul)
    tvm.ir.assert_structural_equal(ir_mod, LoweredTIRModule)
    # check building
    mod = tvm.build(matmul, target="llvm")
    _check_module_with_numpy(mod)


def test_lower_build_tir_module():
    func = matmul.with_attr("global_symbol", "main")
    func = func.with_attr("tir.noalias", True)
    ir_mod = IRModule({"main": func})
    # check lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        lowered_mod = tvm.lower(ir_mod)
    tvm.ir.assert_structural_equal(lowered_mod, LoweredTIRModule)
    # check building
    mod = tvm.build(ir_mod, target="llvm")
    _check_module_with_numpy(mod)


def test_lower_build_lowered_module():
    # check lowering with the CSE pass disabled as otherwise it would do some commoning
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["tir.CommonSubexprElimTIR"]):
        ir_mod = tvm.lower(LoweredTIRModule)
    tvm.ir.assert_structural_equal(ir_mod, LoweredTIRModule)
    # check building
    mod = tvm.build(ir_mod, target="llvm")
    _check_module_with_numpy(mod)


if __name__ == "__main__":
    test_lower_build_te_schedule()
    test_lower_build_tir_func()
    test_lower_build_tir_module()
    test_lower_build_lowered_module()
