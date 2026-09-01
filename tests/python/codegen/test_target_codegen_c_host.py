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
import tvm.testing
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.support import utils


def test_add():
    nn = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def test_fadd(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
            C: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                with T.sblock("C"):
                    v_i0 = T.axis.spatial(1024, i0)
                    T.reads(A[v_i0], B[v_i0])
                    T.writes(C[v_i0])
                    C[v_i0] = A[v_i0] + B[v_i0]

    def check_c():
        mhost = tvm.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_fadd"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        b = tvm.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        c = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_reinterpret():
    nn = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def test_reinterpret(
            A: T.Buffer((1024,), "int32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                with T.sblock("B"):
                    v_i0 = T.axis.spatial(1024, i0)
                    T.reads(A[v_i0])
                    T.writes(B[v_i0])
                    B[v_i0] = T.reinterpret("float32", A[v_i0] + 2)

    def check_c():
        mhost = tvm.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_reinterpret"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.runtime.tensor(np.random.randint(-(2**30), 2**30, size=n).astype("int32"), dev)
        b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fadd(a, b)
        tvm.testing.assert_allclose(b.numpy(), (2 + a.numpy()).view("float32"))

    check_c()


def test_ceil():
    nn = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def test_ceil(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                with T.sblock("B"):
                    v_i0 = T.axis.spatial(1024, i0)
                    T.reads(A[v_i0])
                    T.writes(B[v_i0])
                    B[v_i0] = T.ceil(A[v_i0])

    def check_c():
        mhost = tvm.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fceil = m["test_ceil"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.runtime.tensor(np.random.rand(n).astype("float32"), dev)
        b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fceil(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.ceil(a.numpy()).view("float32")))

    check_c()


def test_floor():
    nn = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def test_floor(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                with T.sblock("B"):
                    v_i0 = T.axis.spatial(1024, i0)
                    T.reads(A[v_i0])
                    T.writes(B[v_i0])
                    B[v_i0] = T.floor(A[v_i0])

    def check_c():
        mhost = tvm.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        ffloor = m["test_floor"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.runtime.tensor(np.random.rand(n).astype("float32"), dev)
        b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        ffloor(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.floor(a.numpy()).view("float32")))

    check_c()


def test_round():
    nn = 1024

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def test_round(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                with T.sblock("B"):
                    v_i0 = T.axis.spatial(1024, i0)
                    T.reads(A[v_i0])
                    T.writes(B[v_i0])
                    B[v_i0] = T.round(A[v_i0])

    def check_c():
        mhost = tvm.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fround = m["test_round"]
        dev = tvm.cpu(0)
        n = nn
        # Exact midpoints first: this is where ties-to-even (np.round, and the
        # semantics every other backend and the constant folder use) differs
        # from ties-away-from-zero. np.random.rand never produces them, so the
        # random tail alone cannot exercise the tie rule.
        midpoints = np.array(
            [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype="float32"
        )
        a_np = np.concatenate(
            [midpoints, np.random.rand(n - len(midpoints)).astype("float32")]
        ).astype("float32")
        a = tvm.runtime.tensor(a_np, dev)
        b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fround(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.round(a.numpy()).view("float32")))

    check_c()


def test_subroutine_call():
    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer(1, dtype="float32")):
            Module.subroutine(A.data)

        @T.prim_func(private=True, s_tir=True)
        def subroutine(A_data: T.handle("float32")):
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 42.0

    built = tvm.tirx.build(Module, target="c")

    source = built.inspect_source()
    assert source.count("__tvm_ffi_main(void*") == 2, (
        "Expected two occurrences, for forward-declaration and definition"
    )
    assert source.count("subroutine(float*") == 2, (
        "Expected two occurrences, for forward-declaration and definition"
    )
    assert source.count("subroutine(") == 3, (
        "Expected three occurrences, for forward-declaration, definition, and call from main."
    )


def test_workspace_allocation_cast():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer((256,), "float32")):
            workspace = T.alloc_buffer((256,), "float32", scope="global")
            for i in range(256):
                workspace[i] = A[i]
            for i in range(256):
                A[i] = workspace[i]

    built = tvm.tirx.build(Module, target="c")
    assert "((float*)TVMBackendAllocWorkspace(" in built.inspect_source()

    temp = utils.tempdir()
    built.export_library(temp.relpath("workspace.so"))


def test_local_alloc_buffer_uses_plain_c_pointer():
    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(s_tir=True)
        def main(A: T.Buffer((1,), "float32")):
            B = T.alloc_buffer((1,), "float32", scope="local")
            for i in range(1):
                with T.sblock("copy"):
                    vi = T.axis.spatial(1, i)
                    T.reads(A[vi])
                    T.writes(B[vi], A[vi])
                    B[vi] = A[vi] + T.float32(1)
                    A[vi] = B[vi]

    built = tvm.tirx.build(Module, target="c")
    assert "local float*" not in built.inspect_source()

    temp = utils.tempdir()
    path_dso = temp.relpath("local_alloc.so")
    built.export_library(path_dso)
    loaded = tvm.runtime.load_module(path_dso)

    data = tvm.runtime.tensor(np.array([1.0], dtype="float32"))
    loaded["main"](data)
    tvm.testing.assert_allclose(data.numpy(), np.array([2.0], dtype="float32"))


def test_vector_access_ptr_address_uses_ramp_base():
    buffer = tvm.tirx.decl_buffer((8,), "float32x2", name="A")
    access_ptr = buffer.access_ptr(access_mask=3, offset=2, extent=4)
    body = tvm.tirx.Evaluate(tvm.tirx.call_extern("void", "consume", access_ptr))
    func = tvm.tirx.PrimFunc([buffer], body).with_attr("global_symbol", "main")

    source = tvm.tirx.build(tvm.IRModule.from_expr(func), target="c").inspect_source()
    call = next(line.strip() for line in source.splitlines() if line.strip().startswith("consume("))
    assert "int32_t2" not in call
    assert "float2*" in call
    assert " + 4" in call


if __name__ == "__main__":
    tvm.testing.main()
