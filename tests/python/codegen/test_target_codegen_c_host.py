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
from tvm.contrib import utils
from tvm.script import ir as I
from tvm.script import tir as T


def test_add():
    nn = 1024

    @I.ir_module
    class Module:
        @T.prim_func
        def test_fadd(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
            C: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
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

    @I.ir_module
    class Module:
        @T.prim_func
        def test_reinterpret(
            A: T.Buffer((1024,), "int32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
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

    @I.ir_module
    class Module:
        @T.prim_func
        def test_ceil(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
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

    @I.ir_module
    class Module:
        @T.prim_func
        def test_floor(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
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

    @I.ir_module
    class Module:
        @T.prim_func
        def test_round(
            A: T.Buffer((1024,), "float32"),
            B: T.Buffer((1024,), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
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
        a = tvm.runtime.tensor(np.random.rand(n).astype("float32"), dev)
        b = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fround(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.round(a.numpy()).view("float32")))

    check_c()


def test_subroutine_call():
    @I.ir_module
    class Module:
        @T.prim_func
        def main(A: T.Buffer(1, dtype="float32")):
            Module.subroutine(A.data)

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32")):
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 42.0

    built = tvm.tir.build(Module, target="c")

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


if __name__ == "__main__":
    tvm.testing.main()
