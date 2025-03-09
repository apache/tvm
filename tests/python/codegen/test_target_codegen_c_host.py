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
import tvm.testing

from tvm import te
from tvm.contrib import utils
from tvm.script import tir as T, ir as I

import numpy as np


def test_add():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="C")

    def check_c():
        mhost = tvm.compile(
            tvm.IRModule.from_expr(
                te.create_prim_func([A, B, C]).with_attr("global_symbol", "test_fadd")
            ),
            target="c",
        )
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_fadd"]
        dev = tvm.cpu(0)
        # launch the kernel.
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_reinterpret():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="int32")
    B = te.compute(
        A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.reinterpret", 2 + A(*i)), name="B"
    )

    def check_c():
        mhost = tvm.compile(
            tvm.IRModule.from_expr(
                te.create_prim_func([A, B]).with_attr("global_symbol", "test_reinterpret")
            ),
            target="c",
        )
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fadd = m["test_reinterpret"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.randint(-(2**30), 2**30, size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fadd(a, b)
        tvm.testing.assert_allclose(b.numpy(), (2 + a.numpy()).view("float32"))

    check_c()


def test_ceil():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.ceil", A(*i)), name="B")

    def check_c():
        mhost = tvm.compile(
            tvm.IRModule.from_expr(
                te.create_prim_func([A, B]).with_attr("global_symbol", "test_ceil")
            ),
            target="c",
        )
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fceil = m["test_ceil"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fceil(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.ceil(a.numpy()).view("float32")))

    check_c()


def test_floor():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.floor", A(*i)), name="B")

    def check_c():
        mhost = tvm.compile(
            tvm.IRModule.from_expr(
                te.create_prim_func([A, B]).with_attr("global_symbol", "test_floor")
            ),
            target="c",
        )
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        ffloor = m["test_floor"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        ffloor(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.floor(a.numpy()).view("float32")))

    check_c()


def test_round():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.compute(A.shape, lambda *i: tvm.tir.call_intrin("float32", "tir.round", A(*i)), name="B")

    def check_c():
        mhost = tvm.compile(
            tvm.IRModule.from_expr(
                te.create_prim_func([A, B]).with_attr("global_symbol", "test_round")
            ),
            target="c",
        )
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm.runtime.load_module(path_dso)
        fround = m["test_round"]
        dev = tvm.cpu(0)
        n = nn
        a = tvm.nd.array(np.random.rand(n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        fround(a, b)
        tvm.testing.assert_allclose(b.numpy(), (np.round(a.numpy()).view("float32")))

    check_c()


def test_subroutine_call():
    @I.ir_module
    class mod:
        @T.prim_func
        def main(A: T.Buffer(1, dtype="float32")):
            mod.subroutine(A.data)

        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32")):
            A = T.decl_buffer(1, dtype="float32", data=A_data)
            A[0] = 42.0

    built = tvm.tir.build(mod, target="c")

    func_names = list(built["get_func_names"]())
    assert (
        "main" in func_names
    ), "Externally exposed functions should be listed in available functions."
    assert (
        "subroutine" not in func_names
    ), "Internal function should not be listed in available functions."

    source = built.get_source()
    assert (
        source.count("main(void*") == 2
    ), "Expected two occurrences, for forward-declaration and definition"
    assert (
        source.count("subroutine(float*") == 2
    ), "Expected two occurrences, for forward-declaration and definition"
    assert (
        source.count("subroutine(") == 3
    ), "Expected three occurrences, for forward-declaration, definition, and call from main."


if __name__ == "__main__":
    tvm.testing.main()
