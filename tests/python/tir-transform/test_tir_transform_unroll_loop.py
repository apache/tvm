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
import tvm
from tvm.script import ir as I
from tvm.script import tir as T


def test_unroll_loop():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle, n: T.int64):
            Ab = T.match_buffer(A, (n,), "int64")
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    mod = Module
    stmt = mod["main"].body

    assert isinstance(stmt, tvm.tir.For)

    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        ret = tvm.tir.transform.UnrollLoop()(mod)["main"].body
        assert not isinstance(ret, tvm.tir.For)

    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 15}}):
        ret = tvm.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret, tvm.tir.For)

    with tvm.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_step": 16, "explicit_unroll": False}}
    ):
        ret = tvm.tir.transform.UnrollLoop()(mod)["main"].body
        assert isinstance(ret, tvm.tir.For)
        assert ret.kind == tvm.tir.ForKind.UNROLLED

    @I.ir_module
    class ModuleWithPragma:
        @T.prim_func
        def main(A: T.handle, n: T.int64):
            Ab = T.match_buffer(A, (n,), "int64")
            with T.attr(T.int32(0), "pragma_auto_unroll_max_step", 16):
                for i in T.serial(n, n + 2):
                    for j in T.unroll(8):
                        Ab[j + 1] = Ab[i] + T.int64(1)
            for i in T.serial(n, n + 2):
                for j in T.unroll(8):
                    Ab[j + 1] = Ab[i] + T.int64(1)

    with tvm.transform.PassContext(
        config={"tir.UnrollLoop": {"auto_max_depth": 8, "explicit_unroll": False}}
    ):
        ret = tvm.tir.transform.UnrollLoop()(ModuleWithPragma)["main"].body
        assert isinstance(ret[0], tvm.tir.For)
        assert ret[0].kind == tvm.tir.ForKind.UNROLLED
        assert isinstance(ret[1], tvm.tir.For)
        assert ret[1].kind != tvm.tir.ForKind.UNROLLED


def test_unroll_fake_loop():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.handle, n: T.int64):
            Ab = T.match_buffer(A, (n,), "int32")
            for i in T.serial(1):
                Ab[i * 2] = 3
                for j in T.serial(10):
                    Ab[j + 1] = Ab[i] + 1

    with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {"auto_max_depth": 8, "auto_max_extent": 1, "explicit_unroll": False}
        }
    ):
        ret = tvm.tir.transform.UnrollLoop()(Module)["main"].body
        assert isinstance(ret[0], tvm.tir.BufferStore)


def test_unroll_allocations():
    @I.ir_module
    class Before:
        @T.prim_func
        def main():
            for i in T.unroll(2):
                with T.decl_buffer([16], "float32") as buf:
                    buf[0] = 0.0

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            with T.decl_buffer([16], "float32") as buf1:
                buf1[0] = 0.0
            with T.decl_buffer([16], "float32") as buf2:
                buf2[0] = 0.0

    after = tvm.tir.transform.UnrollLoop()(Before)

    tvm.ir.assert_structural_equal(after, Expected)


def test_unroll_local_access():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(B: T.Buffer((64,), "float32")):
            for bx in T.thread_binding(4, thread="blockIdx.x"):
                for tx in T.thread_binding(4, thread="threadIdx.x"):
                    A_local_data = T.allocate([4], dtype="float32", scope="local")
                    A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                    for i in T.serial(4):
                        A_local[i] = T.float32(i)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(B: T.Buffer((64,), "float32")):
            for bx in T.thread_binding(4, thread="blockIdx.x"):
                for tx in T.thread_binding(4, thread="threadIdx.x"):
                    A_local_data = T.allocate([4], dtype="float32", scope="local")
                    A_local = T.Buffer([4], dtype="float32", data=A_local_data)
                    A_local[0] = T.float32(0)
                    A_local[1] = T.float32(1)
                    A_local[2] = T.float32(2)
                    A_local[3] = T.float32(3)

    with tvm.transform.PassContext(
        config={
            "tir.UnrollLoop": {
                "auto_max_depth": 0,
                "auto_max_extent": 1,
                "explicit_unroll": True,
                "unroll_local_access": True,
            }
        }
    ):
        after = tvm.tir.transform.UnrollLoop()(Before)
        after = tvm.tir.transform.Simplify()(after)

    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    test_unroll_local_access()
    test_unroll_loop()
    test_unroll_fake_loop()
    test_unroll_allocations()
