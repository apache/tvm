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
from tvm.schedule import Buffer
import numpy as np

def test_buffer():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    Ab = tvm.decl_buffer((m, n), tvm.float32)
    Bb = tvm.decl_buffer((n, l), tvm.float32)

    assert isinstance(Ab, tvm.schedule.Buffer)
    assert Ab.dtype == tvm.float32
    assert tuple(Ab.shape) == (m, n)

def test_buffer_access_ptr():
    m = tvm.var('m')
    n = tvm.var('n')
    Ab = tvm.decl_buffer((m, n), tvm.float32, strides=[n + 1 , 1])
    aptr = Ab.access_ptr("rw")
    assert tvm.ir_pass.Equal(aptr.args[3], Ab.strides[0] * m)
    assert aptr.args[0].dtype == Ab.dtype
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("w")
    assert aptr.args[4].value == Buffer.WRITE

def test_buffer_access_ptr_offset():
    m = tvm.var('m')
    n = tvm.var('n')
    Ab = tvm.decl_buffer((m, n), tvm.float32)
    aptr = Ab.access_ptr("rw", offset=100)
    offset = tvm.ir_pass.Simplify(aptr.args[2])
    assert tvm.ir_pass.Equal(offset, 100)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    v = tvm.var('int32')
    aptr = Ab.access_ptr("rw", offset=100 + 100 + v)
    offset = tvm.ir_pass.Simplify(aptr.args[2])
    assert tvm.ir_pass.Equal(offset, 200 + v)
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE
    aptr = Ab.access_ptr("rw", offset=tvm.call_extern('int32', "test_call", 100 + 100 + v))
    offset = tvm.ir_pass.Simplify(aptr.args[2])
    assert tvm.ir_pass.Equal(offset, tvm.call_extern('int32', "test_call", 200 + v))
    assert aptr.args[4].value == Buffer.READ | Buffer.WRITE

def test_buffer_access_ptr_extent():
    m = tvm.var('m')
    n = tvm.var('n')
    Ab = tvm.decl_buffer((m, n), tvm.float32)
    aptr = Ab.access_ptr("rw")
    assert tvm.ir_pass.Equal(aptr.args[3], m * n)
    aptr = Ab.access_ptr("rw", offset=100)
    assert tvm.ir_pass.Equal(aptr.args[3], m * n - 100)
    Ab = tvm.decl_buffer((m, n), tvm.float32, strides=[n + 1 , 1])
    aptr = Ab.access_ptr("rw", offset=100)
    assert tvm.ir_pass.Equal(aptr.args[3], Ab.strides[0] * m - 100)

def test_buffer_vload():
    m = tvm.var('m')
    n = tvm.var('n')
    Ab = tvm.decl_buffer((m, n), tvm.float32, elem_offset=100)
    load = Ab.vload([2, 3])
    offset = tvm.ir_pass.Simplify(load.index)
    assert tvm.ir_pass.Equal(offset, n * 2 + 103)

def test_buffer_index_merge_mult_mod():
    m = tvm.var('m')
    n = tvm.var('n')
    s = tvm.var('s')
    k0 = tvm.var('k0')
    k1 = tvm.var('k1')
    A = tvm.decl_buffer((m, n), tvm.float32)
    A_stride = tvm.decl_buffer((m, n), tvm.float32, strides=(s, 1))
    def assert_simplified_equal(index_simplified, index_direct):
        assert tvm.ir_pass.Equal(index_simplified, index_direct),\
        "index_simplified=%s, index_direct=%s" %(index_simplified, index_direct)
    # Test Case1
    index_simplified = A_stride.vload(((k0 % k1) / s, (k0 % k1) % s + (k0 / k1) * k1))
    index_direct = A_stride.vload((0, k0))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case2
    index_simplified = A.vload(((k0 % (k1 / s)) / n,
                                (k0 % (k1 / s)) % n + (k0 % k1)))
    index_direct = A.vload((0, k0 % k1 + k0 % (k1 / s)))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case3
    index_simplified = A.vload((((k0 / (k1 / s)) * (k1 / s)) / n + (k0 % (k1 / s)) / n,
                                ((k0 / (k1 / s)) * (k1 / s)) % n + (k0 % (k1 / s)) % n))
    index_direct = A.vload((0, k0))
    assert_simplified_equal(index_simplified, index_direct)
    # Test Case4 (not able to simplify)
    index_simplified = A.vload(((k0 % (k1 / s)) / n,
                                (k0 % (k1 / n)) % n + (k0 % k1)))
    index_direct = A.vload((0, ((k0 % (k1 / s)) / n) * n + ((k0 % (k1 / n)) % n + (k0 % k1))))
    assert_simplified_equal(index_simplified, index_direct)

def test_buffer_broadcast():
    m0, m1, m2 = tvm.var("m0"), tvm.var("m1"), tvm.var("m2")
    n0, n1, n2 = tvm.var("n0"), tvm.var("n1"), tvm.var("n2")
    o0, o1, o2 = tvm.var("o0"), tvm.var("o1"), tvm.var("o2")

    A = tvm.placeholder((m0, m1, m2), name='A')
    B = tvm.placeholder((n0, n1, n2), name='B')

    C = tvm.compute((o0, o1, o2), lambda i, j, k: A[i, j, k] + B[i, j, k], name='C')

    Ab = tvm.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="auto_broadcast")
    Bb = tvm.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="auto_broadcast")
    s = tvm.create_schedule(C.op)

    def check():
        if not tvm.module.enabled("llvm"):
            return
        fadd = tvm.build(s, [A, B, C], target='llvm', name='bcast_add', binds={A:Ab, B:Bb})
        ctx = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(2, 4, 3)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(2, 1, 1)).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros((2, 4, 3), dtype=C.dtype), ctx)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

    check()


if __name__ == "__main__":
    test_buffer()
    test_buffer_access_ptr()
    test_buffer_access_ptr_offset()
    test_buffer_access_ptr_extent()
    test_buffer_vload()
    test_buffer_index_merge_mult_mod()
    test_buffer_broadcast()
