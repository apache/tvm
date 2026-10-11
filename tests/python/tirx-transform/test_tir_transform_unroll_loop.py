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
from __future__ import annotations

import tvm_ffi

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


def test_unroll_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(Ab: T.Tensor((n,), "int64"), n: T.int64):  # noqa: F821
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    mod = Module
    stmt = mod["main"].body[0]

    assert isinstance(stmt, tvm.ir.For)

    with tvm.transform.PassContext(config={"tirx.UnrollLoop": {"auto_max_step": 16}}):
        ret = tvm.tirx.transform.UnrollLoop()(mod)["main"].body
        assert not any(isinstance(stmt, tvm.ir.For) for stmt in ret)

    with tvm.transform.PassContext(config={"tirx.UnrollLoop": {"auto_max_step": 15}}):
        ret = tvm.tirx.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret[0], tvm.ir.For)

    with tvm.transform.PassContext(
        config={"tirx.UnrollLoop": {"auto_max_step": 16, "explicit_unroll": False}}
    ):
        ret = tvm.tirx.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret[0], tvm.ir.For)
        assert ret[0].kind == tvm.ir.ForKind.UNROLLED

    @I.ir_module
    class ModuleWithPolicy:
        @T.function
        def main(Ab: T.Tensor((n,), "int64"), n: T.int64):  # noqa: F821
            for i in T.serial(n, n + 2, annotations={"auto_unroll_max_step": 16}):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    with tvm.transform.PassContext(
        config={"tirx.UnrollLoop": {"auto_max_depth": 8, "explicit_unroll": False}}
    ):
        ret = tvm.tirx.transform.UnrollLoop()(ModuleWithPolicy)["main"].body
        assert isinstance(ret[0], tvm.ir.For)
        assert ret[0].kind == tvm.ir.ForKind.UNROLLED
        assert isinstance(ret[1], tvm.ir.For)
        assert ret[1].kind != tvm.ir.ForKind.UNROLLED


def test_unroll_fake_loop():
    @I.ir_module
    class Module:
        @T.function
        def main(Ab: T.Tensor((n,), "int32"), n: T.int64):  # noqa: F821
            for i in T.serial(1):
                Ab[i * 2] = 3
                for j in T.serial(10):
                    Ab[j + 1] = Ab[i] + 1

    with tvm.transform.PassContext(
        config={
            "tirx.UnrollLoop": {"auto_max_depth": 8, "auto_max_extent": 1, "explicit_unroll": False}
        }
    ):
        ret = tvm.tirx.transform.UnrollLoop()(Module)["main"].body
        assert isinstance(ret[0], tvm.ir.TensorStore)


def test_unroll_allocations():
    @I.ir_module
    class Before:
        @T.function
        def main():
            for i in T.unroll(2):
                buf = T.alloc_tensor(ty_args=[T.Tensor([16], "float32")])
                buf[0] = 0.0

    @I.ir_module
    class Expected:
        @T.function
        def main():
            buf1 = T.alloc_tensor(ty_args=[T.Tensor([16], "float32")])
            buf1[0] = 0.0
            buf2 = T.alloc_tensor(ty_args=[T.Tensor([16], "float32")])
            buf2[0] = 0.0

    after = tvm.tirx.transform.UnrollLoop()(Before)

    tvm.ir.assert_structural_equal(after, Expected)


def test_unroll_local_access():
    @I.ir_module
    class Before:
        @T.function
        def main(B: T.Tensor((64,), "float32")):
            with T.launch_thread("blockIdx.x", 4) as bx:  # noqa: F841
                with T.launch_thread("threadIdx.x", 4) as tx:  # noqa: F841
                    A_local = T.alloc_tensor(ty_args=[T.Tensor((4,), scope="local")])
                    for i in T.serial(4):
                        A_local[i] = T.float32(i)

    @I.ir_module
    class Expected:
        @T.function
        def main(B: T.Tensor((64,), "float32")):
            with T.launch_thread("blockIdx.x", 4) as bx:  # noqa: F841
                with T.launch_thread("threadIdx.x", 4) as tx:  # noqa: F841
                    A_local = T.alloc_tensor(ty_args=[T.Tensor((4,), scope="local")])
                    A_local[0] = T.float32(0)
                    A_local[1] = T.float32(1)
                    A_local[2] = T.float32(2)
                    A_local[3] = T.float32(3)

    with tvm.transform.PassContext(
        config={
            "tirx.UnrollLoop": {
                "auto_max_depth": 0,
                "auto_max_extent": 1,
                "explicit_unroll": True,
                "unroll_local_access": True,
            }
        }
    ):
        after = tvm.tirx.transform.UnrollLoop()(Before)
        after = tvm.tirx.transform.StmtSimplify()(after)

    tvm.ir.assert_structural_equal(after, Expected)


def test_vectorized_loop_unroll_policy():
    @T.function
    def before(A: T.Tensor((4, 4), "int32")):
        for i in T.vectorized(
            4, annotations={"auto_unroll_max_step": 16, "unroll_explicit": False}
        ):
            for inherited in range(4):
                A[inherited, i] = inherited + i
            for disabled in T.serial(4, annotations={"auto_unroll_max_step": 0}):
                A[disabled, i] = disabled - i
            for explicit in T.serial(4, annotations={"unroll_explicit": True}):
                A[explicit, i] = explicit * i
        for sibling in range(4):
            A[sibling, 0] = sibling

    mod = tvm.IRModule({"main": before.with_attr("target", tvm.target.Target("llvm"))})
    after = tvm.tirx.transform.VectorizeLoop()(mod)
    after = tvm.tirx.transform.UnrollLoop()(after)
    loops = []
    tvm_ffi.structural_walk(
        after["main"].body,
        lambda node: loops.append(node) if isinstance(node, tvm.ir.For) else None,
    )
    assert [(loop.loop_var.name, loop.kind) for loop in loops] == [
        ("inherited", tvm.ir.ForKind.UNROLLED),
        ("disabled", tvm.ir.ForKind.DEFAULT),
        ("sibling", tvm.ir.ForKind.DEFAULT),
    ]
    assert all(not loop.annotations for loop in loops)


def test_disabled_vectorization_preserves_unroll_policy():
    @T.function
    def before(A: T.Tensor((4, 4), "int32")):
        for i in T.vectorized(
            4, annotations={"auto_unroll_max_step": 16, "unroll_explicit": False}
        ):
            for j in range(4):
                A[j, i] = i + j

    mod = tvm.IRModule({"main": before})
    after = tvm.tirx.transform.VectorizeLoop(enable_vectorize=False)(mod)
    after = tvm.tirx.transform.UnrollLoop()(after)
    outer = after["main"].body[0]
    assert outer.kind == tvm.ir.ForKind.UNROLLED
    assert outer.body[0].kind == tvm.ir.ForKind.UNROLLED


if __name__ == "__main__":
    test_unroll_local_access()
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_allocations()
