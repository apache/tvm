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
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.tir import Buffer
import numpy as np


def test_buffer():
    m = te.size_var("m")
    n = te.size_var("n")
    l = te.size_var("l")
    Ab = tvm.tir.decl_buffer((m, n), "float32")
    Bb = tvm.tir.decl_buffer((n, l), "float32")

    assert isinstance(Ab, tvm.tir.Buffer)
    assert Ab.dtype == "float32"
    assert tuple(Ab.shape) == (m, n)


def test_buffer_access_ptr():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32", strides=[n + 1, 1])
    aptr = Ab.access_ptr("rw")
    assert tvm.ir.structural_equal(aptr.args[3], Ab.strides[0] * m)
    assert aptr.args[0].dtype == Ab.dtype
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("w")
    assert aptr.args[4].value == Buffer.WRITE


def test_buffer_access_ptr_offset():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32")
    aptr = Ab.access_ptr("rw", offset=100)
    tvm.testing.assert_prim_expr_equal(aptr.args[2], 100)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    v = te.size_var("int32")
    aptr = Ab.access_ptr("rw", offset=100 + 100 + v)
    tvm.testing.assert_prim_expr_equal(aptr.args[2], 200 + v)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("rw", offset=tvm.tir.call_extern("int32", "test_call", 100 + 100 + v))
    tvm.testing.assert_prim_expr_equal(
        aptr.args[2], tvm.tir.call_extern("int32", "test_call", 200 + v)
    )
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE


def test_buffer_access_ptr_extent():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32")
    aptr = Ab.access_ptr("rw")
    assert tvm.ir.structural_equal(aptr.args[3], m * n)
    aptr = Ab.access_ptr("rw", offset=100)
    assert tvm.ir.structural_equal(aptr.args[3], m * n - 100)
    Ab = tvm.tir.decl_buffer((m, n), "float32", strides=[n + 1, 1])
    aptr = Ab.access_ptr("rw", offset=100)
    assert tvm.ir.structural_equal(aptr.args[3], Ab.strides[0] * m - 100)


def test_buffer_vload():
    m = te.size_var("m")
    n = te.size_var("n")
    Ab = tvm.tir.decl_buffer((m, n), "float32", elem_offset=100)
    load = Ab.vload([2, 3])
    tvm.testing.assert_prim_expr_equal(load.index, n * 2 + 103)


def test_buffer_vload_nullptr():
    var = tvm.tir.Var("v", dtype="int32")
    buf = tvm.tir.decl_buffer((1,), name="buf")
    buf_load = tvm.tir.expr.BufferLoad(buffer=buf, indices=tvm.runtime.convert([0]))
    buf_load_stmt = tvm.tir.stmt.Evaluate(buf_load)
    for_loop = tvm.tir.stmt.For(
        loop_var=var, kind=0, min_val=0, extent=buf_load, body=buf_load_stmt
    )
    buf_func = tvm.tir.PrimFunc(params={}, body=for_loop)
    mod = tvm.IRModule({"main": buf_func})
    # Trigger nullptr buffer bug by pass
    with pytest.raises(tvm.error.TVMError) as cm:
        mod = tvm.transform.Sequential(
            [
                tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
                tvm.tir.transform.CompactBufferAllocation(),
                tvm.tir.transform.FlattenBuffer(),
            ]
        )(mod)
        assert "(n != nullptr) is false" in str(cm.execption)


def test_buffer_index_merge_mult_mod():
    m = te.size_var("m")
    n = te.size_var("n")
    s = te.size_var("s")
    k0 = te.size_var("k0")
    k1 = te.size_var("k1")
    A = tvm.tir.decl_buffer((m, n), "float32")
    A_stride = tvm.tir.decl_buffer((m, n), "float32", strides=(s, 1))

    def assert_simplified_equal(index_simplified, index_direct):
        assert tvm.ir.structural_equal(
            index_simplified, index_direct
        ), "index_simplified=%s, index_direct=%s" % (index_simplified, index_direct)

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod
    # Test Case1
    index_simplified = A_stride.vload(
        (idxd(idxm(k0, k1), s), idxm(idxm(k0, k1), s) + idxd(k0, k1) * k1)
    )
    index_direct = A_stride.vload((0, k0))
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case2
    index_simplified = A.vload(
        (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, s)), n) + idxm(k0, k1))
    )
    index_direct = A.vload((0, idxm(k0, k1) + idxm(k0, idxd(k1, s))))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case3
    index_simplified = A.vload(
        (
            idxd((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxd(idxm(k0, idxd(k1, s)), n),
            idxm((idxd(k0, idxd(k1, s)) * idxd(k1, s)), n) + idxm(idxm(k0, idxd(k1, s)), n),
        )
    )
    index_direct = A.vload((0, k0))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case4 (not able to simplify)
    index_simplified = A.vload(
        (idxd(idxm(k0, idxd(k1, s)), n), idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1))
    )
    index_direct = A.vload(
        (0, idxd(idxm(k0, idxd(k1, s)), n) * n + (idxm(idxm(k0, idxd(k1, n)), n) + idxm(k0, k1)))
    )
    assert_simplified_equal(index_simplified, index_direct)

    # Test Case5
    B = tvm.tir.decl_buffer((1, 14, 14, 1024))
    i = te.size_var("i")
    j = te.size_var("j")
    k = te.size_var("k")

    index_simplified = B.vload(
        (
            idxd(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd(idxd((i * 50176 + j * 28672 + k), 1024), 14), 14),
            idxm(idxd((i * 50176 + j * 28672 + k), 1024), 14),
            idxm((i * 50176 + j * 28672 + k), 1024),
        )
    )
    index_direct = B.vload((0, 0, 0, (i * 50176 + j * 28672 + k)))
    assert_simplified_equal(index_simplified, index_direct)


@tvm.testing.requires_llvm
def test_buffer_broadcast():
    m0, m1, m2 = te.size_var("m0"), te.size_var("m1"), te.size_var("m2")
    n0, n1, n2 = te.size_var("n0"), te.size_var("n1"), te.size_var("n2")
    o0, o1, o2 = te.size_var("o0"), te.size_var("o1"), te.size_var("o2")

    A = te.placeholder((m0, m1, m2), name="A")
    B = te.placeholder((n0, n1, n2), name="B")

    C = te.compute((o0, o1, o2), lambda i, j, k: A[i, j, k] + B[i, j, k], name="C")

    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="auto_broadcast")
    Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="auto_broadcast")
    s = te.create_schedule(C.op)

    def check():
        fadd = tvm.build(s, [A, B, C], target="llvm", name="bcast_add", binds={A: Ab, B: Bb})
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(2, 4, 3)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(2, 1, 1)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((2, 4, 3), dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check()


@tvm.testing.requires_llvm
def test_buffer_broadcast_expr():
    n0, m0, x = te.size_var("n0"), te.size_var("m0"), te.size_var("x")
    n1, m1 = te.size_var("n1"), te.size_var("m1")
    o0, o1 = te.size_var("o0"), te.size_var("o1")

    A = te.placeholder((m0, n0), name="A")
    B = te.placeholder((m1, n1), name="B")
    C = te.compute((o0, o1 // x), lambda i, j: A[i, j] + B[i, j], name="C")

    Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="auto_broadcast")
    Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="auto_broadcast")
    Cc = tvm.tir.decl_buffer(C.shape, C.dtype, name="Cc", buffer_type="auto_broadcast")
    s = te.create_schedule(C.op)

    def check_stride():
        fadd = tvm.build(
            s, [A, B, C, o1, x], target="llvm", name="bcast_add", binds={A: Ab, B: Bb, C: Cc}
        )
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(2, 4)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(2, 4)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((2, 4), dtype=C.dtype), dev)
        fadd(a, b, c, 4, 1)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    def check_no_stride():
        fadd = tvm.build(
            s, [A, B, C, o1, x], target="llvm", name="bcast_add", binds={A: Ab, B: Bb, C: Cc}
        )
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(1, 4)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(2, 4)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((2, 4), dtype=C.dtype), dev)
        fadd(a, b, c, 4, 1)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    def check_auto_bind():
        # Let build bind buffers
        fadd = tvm.build(s, [A, B, C, o1, x], target="llvm", name="bcast_add")
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(1, 4)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(2, 4)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((2, 4), dtype=C.dtype), dev)
        fadd(a, b, c, 4, 1)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_stride()
    check_no_stride()
    check_auto_bind()


if __name__ == "__main__":
    pytest.main([__file__])
