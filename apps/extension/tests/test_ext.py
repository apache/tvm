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
import tvm_ext
import tvm
import tvm._ffi.registry
import tvm.testing
from tvm import te
import numpy as np


def test_bind_add():
    def add(a, b):
        return a + b

    f = tvm_ext.bind_add(add, 1)
    assert f(2) == 3


def test_ext_dev():
    n = 10
    A = te.placeholder((n,), name="A")
    B = te.compute((n,), lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)

    def check_llvm():
        if not tvm.testing.device_enabled("llvm"):
            return
        f = tvm.build(s, [A, B], "ext_dev", "llvm")
        dev = tvm.ext_dev(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), a.numpy() + 1)

    check_llvm()


def test_sym_add():
    a = te.var("a")
    b = te.var("b")
    c = tvm_ext.sym_add(a, b)
    assert c.a == a and c.b == b


def test_ext_vec():
    ivec = tvm_ext.ivec_create(1, 2, 3)
    assert isinstance(ivec, tvm_ext.IntVec)
    assert ivec[0] == 1
    assert ivec[1] == 2

    def ivec_cb(v2):
        assert isinstance(v2, tvm_ext.IntVec)
        assert v2[2] == 3

    tvm.runtime.convert(ivec_cb)(ivec)


def test_extract_ext():
    fdict = tvm._ffi.registry.extract_ext_funcs(tvm_ext._LIB.TVMExtDeclare)
    assert fdict["mul"](3, 4) == 12


def test_extern_call():
    n = 10
    A = te.placeholder((n,), name="A")
    B = te.compute(
        (n,), lambda *i: tvm.tir.call_extern("float32", "TVMTestAddOne", A(*i)), name="B"
    )
    s = te.create_schedule(B.op)

    def check_llvm():
        if not tvm.testing.device_enabled("llvm"):
            return
        f = tvm.build(s, [A, B], "llvm")
        dev = tvm.cpu(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), dev)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), a.numpy() + 1)

    check_llvm()


def test_nd_subclass():
    a = tvm_ext.NDSubClass.create(additional_info=3)
    b = tvm_ext.NDSubClass.create(additional_info=5)
    assert isinstance(a, tvm_ext.NDSubClass)
    c = a + b
    d = a + a
    e = b + b
    assert a.additional_info == 3
    assert b.additional_info == 5
    assert c.additional_info == 8
    assert d.additional_info == 6
    assert e.additional_info == 10


if __name__ == "__main__":
    test_nd_subclass()
    test_extern_call()
    test_ext_dev()
    test_ext_vec()
    test_bind_add()
    test_sym_add()
    test_extract_ext()
