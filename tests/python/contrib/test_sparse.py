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
    input_a = tvmsp.placeholder(shape=(m, n), name="input_a", dtype=dtype)
    assert input_a.stype == "csr"
    n = 3
    a = np.maximum(np.random.uniform(size=(n, n)).astype(dtype) - 0.6, 0.0)
    a = tvmsp.array(a, dev)
    input_a.data = te.placeholder(a.data.shape, dtype, name="input_a_data")
    result_b = tvm.tir.decl_buffer(a.data.shape, dtype, name="input_a_data")
    binds = {input_a.data: result_b}
    result_c = te.compute(input_a.data.shape, lambda i: input_a.data[i] * 2.0, tag="cs_scatter")
    s = te.create_schedule(result_c.op)
    f = tvm.build(s, [input_a.data, result_c], target, binds=binds)
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
    n_row, n_col, n = te.size_var("n_row"), te.size_var("n_col"), te.size_var("n")
    input_a = tvmsp.placeholder(shape=(n_row, n_col), nonzeros=n, name="input_a", dtype=dtype)
    assert input_a.stype == "csr"
    result_c = te.compute(input_a.data.shape, lambda i: input_a.data[i] * 2.0, tag="cs_scatter")
    s = te.create_schedule(result_c.op)
    _n_row, _n_col = 3, 5
    a = np.maximum(np.random.uniform(size=(_n_row, _n_col)).astype(dtype) - 0.6, 0.0)
    a = tvmsp.array(a, dev)
    assert a.data.dtype == a.dtype
    result_b = namedtuple("CSRBuffer", ["data", "indices", "indptr"])
    result_b.data = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="input_a_data")
    result_b.indices = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="input_a_indices")
    binds = {input_a.data: result_b.data, input_a.indices: result_b.indices}
    f = tvm.build(s, [n_row, input_a.data, result_c], target, binds=binds)
    c = tvmsp.array(np.zeros((_n_row, _n_col), dtype), dev)
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
    n_row, n_col, n = te.size_var("n_row"), te.size_var("n_col"), te.size_var("n")
    input_a = tvmsp.placeholder(shape=(n_row, n_col), nonzeros=n, name="input_a", dtype=dtype)
    assert input_a.stype == "csr"
    result_c = te.compute(input_a.data.shape, lambda i: input_a.data[i] * 2.0, tag="cs_scatter")
    s = te.create_schedule(result_c.op)
    _n_row, _n_col = 3, 5
    a = np.maximum(np.random.uniform(size=(_n_row, _n_col)).astype(dtype) - 0.6, 0.0)
    # convert to sparse array tuple
    source_array = a
    row_idx, col_idx = np.nonzero(source_array)
    data = source_array[row_idx, col_idx]
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
    result_b = namedtuple("CSRBuffer", ["data", "indices", "indptr"])
    result_b.data = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="input_a_data")
    result_b.indices = tvm.tir.decl_buffer(a.data.shape, a.data.dtype, name="input_a_indices")
    binds = {input_a.data: result_b.data, input_a.indices: result_b.indices}
    f = tvm.build(s, [n_row, input_a.data, result_c], target, binds=binds)
    c = tvmsp.array(np.zeros((_n_row, _n_col), dtype), dev)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data.shape[0], a.data, c.data)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() * 2.0, rtol=1e-5)


if __name__ == "__main__":
    test_static_tensor()
    test_dynamic_tensor()
    test_sparse_array_tuple()
