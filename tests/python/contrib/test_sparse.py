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
"""Configure pytest"""
# pylint: disable=invalid-name
from collections import namedtuple
import numpy as np
import tvm
import tvm.testing
from tvm import te
import tvm.contrib.sparse as tvmsp
import tvm.runtime.ndarray as _nd


def test_static_tensor():
    """Tests static tensor"""
    dtype = "float32"
    target = "llvm"
    dev = tvm.device(target, 0)
    m = te.size_var("m")
    n = te.size_var("n")
    A = tvmsp.placeholder(shape=(m, n), name="A", dtype=dtype)
    assert A.stype == "csr"
    n = 3
    a = np.maximum(np.random.uniform(size=(n, n)).astype(dtype) - 0.6, 0.0)
    a = tvmsp.array(a, dev)
    A.data = te.placeholder(a.data.shape, dtype, name="A_data")
    Ab = tvm.tir.decl_buffer(a.data.shape, dtype, name="A_data")
    binds = {A.data: Ab}
    C = te.compute(A.data.shape, lambda i: A.data[i] * 2.0, tag="cs_scatter")
    s = te.create_schedule(C.op)
    f = tvm.build(s, [A.data, C], target, binds=binds)
    c = tvmsp.array(np.zeros((n, n), dtype), dev)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data, c.data)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() * 2.0, rtol=1e-5)


def test_dynamic_tensor():
    """Tests dynamic tensor"""
    dtype = "float32"
    target = "llvm"
    dev = tvm.device(target, 0)
    nr, nc, n = te.size_var("nr"), te.size_var("nc"), te.size_var("n")
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, name="A", dtype=dtype)
    assert A.stype == "csr"
    C = te.compute(A.data.shape, lambda i: A.data[i] * 2.0, tag="cs_scatter")
    s = te.create_schedule(C.op)
    _nr, _nc = 3, 5
    a = np.maximum(np.random.uniform(size=(_nr, _nc)).astype(dtype) - 0.6, 0.0)
    a = tvmsp.array(a, dev)
    assert a.data.dtype == a.dtype
    Ab = namedtuple("CSRBuffer", ["data", "indices", "indptr"])
    Ab.data = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="A_data")
    Ab.indices = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="A_indices")
    binds = {A.data: Ab.data, A.indices: Ab.indices}
    f = tvm.build(s, [nr, A.data, C], target, binds=binds)
    c = tvmsp.array(np.zeros((_nr, _nc), dtype), dev)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data.shape[0], a.data, c.data)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() * 2.0, rtol=1e-5)


def test_sparse_array_tuple():
    """Tests array when it is sparse"""
    dtype, itype = "float32", "int32"
    target = "llvm"
    dev = tvm.device(target, 0)
    nr, nc, n = te.size_var("nr"), te.size_var("nc"), te.size_var("n")
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, name="A", dtype=dtype)
    assert A.stype == "csr"
    C = te.compute(A.data.shape, lambda i: A.data[i] * 2.0, tag="cs_scatter")
    s = te.create_schedule(C.op)
    _nr, _nc = 3, 5
    a = np.maximum(np.random.uniform(size=(_nr, _nc)).astype(dtype) - 0.6, 0.0)
    # convert to sparse array tuple
    source_array = a
    ridx, cidx = np.nonzero(source_array)
    data = source_array[ridx, cidx]
    a_data = _nd.array(data, dev)
    indices = np.nonzero(source_array)[1].astype(itype)
    a_indices = _nd.array(indices, dev)
    indptr = [0] + np.apply_along_axis(np.count_nonzero, axis=1, arr=source_array).tolist()
    indptr = np.cumsum(np.array(indptr, itype)).astype(itype)
    a_indptr = _nd.array(indptr, dev)
    a_init = (a_data, a_indices, a_indptr)
    # construct tvm sparse array with tuple
    a = tvmsp.array(a_init, shape=source_array.shape, device=dev)
    assert a.data.dtype == a.dtype
    Ab = namedtuple("CSRBuffer", ["data", "indices", "indptr"])
    Ab.data = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="A_data")
    Ab.indices = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="A_indices")
    binds = {A.data: Ab.data, A.indices: Ab.indices}
    f = tvm.build(s, [nr, A.data, C], target, binds=binds)
    c = tvmsp.array(np.zeros((_nr, _nc), dtype), dev)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data.shape[0], a.data, c.data)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() * 2.0, rtol=1e-5)


if __name__ == "__main__":
    test_static_tensor()
    test_dynamic_tensor()
    test_sparse_array_tuple()
