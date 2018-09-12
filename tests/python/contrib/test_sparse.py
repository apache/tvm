import tvm
import tvm.contrib.sparse as tvmsp
import tvm.ndarray as _nd
import numpy as np
from collections import namedtuple

def test_static_tensor():
    dtype = 'float32'
    stype = 'csr'
    target = 'llvm'
    ctx = tvm.context(target, 0)
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvmsp.placeholder(shape=(m, n), name='A', dtype=dtype)
    assert(A.stype == 'csr')
    n = 3
    a = np.maximum(np.random.uniform(size=(n,n)).astype(dtype)-.6, 0.)
    a = tvmsp.array(a, ctx)
    A.data = tvm.placeholder(a.data.shape, dtype, name='A_data')
    Ab = tvm.decl_buffer(a.data.shape, dtype, name='A_data')
    binds = {A.data: Ab}
    C = tvm.compute(A.data.shape, lambda i: A.data[i] * 2., tag='cs_scatter')
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A.data, C], target, binds=binds)
    c = tvmsp.array(np.zeros((n,n), dtype), ctx)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data, c.data)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2., rtol=1e-5)

def test_dynamic_tensor():
    dtype = 'float32'
    stype = 'csr'
    target = 'llvm'
    ctx = tvm.context(target, 0)
    nr, nc, n = tvm.var('nr'), tvm.var('nc'), tvm.var('n')
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, name='A', dtype=dtype)
    assert(A.stype == 'csr')
    C = tvm.compute(A.data.shape, lambda i: A.data[i] * 2., tag='cs_scatter')
    s = tvm.create_schedule(C.op)
    _nr, _nc = 3, 5
    a = np.maximum(np.random.uniform(size=(_nr, _nc)).astype(dtype)-.6, 0.)
    a = tvmsp.array(a, ctx)
    assert a.data.dtype == a.dtype
    Ab = namedtuple('CSRBuffer', ['data', 'indices', 'indptr'])
    Ab.data = tvm.decl_buffer(a.data.shape, a.data.dtype, name='A_data')
    Ab.indices = tvm.decl_buffer(a.data.shape, a.data.dtype, name='A_indices')
    binds = {A.data: Ab.data, A.indices: Ab.indices}
    f = tvm.build(s, [nr, A.data, C], target, binds=binds)
    c = tvmsp.array(np.zeros((_nr, _nc), dtype), ctx)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data.shape[0], a.data, c.data)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2., rtol=1e-5)

def test_sparse_array_tuple():
    dtype, itype = 'float32', 'int32'
    stype = 'csr'
    target = 'llvm'
    ctx = tvm.context(target, 0)
    nr, nc, n = tvm.var('nr'), tvm.var('nc'), tvm.var('n')
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, name='A', dtype=dtype)
    assert(A.stype == 'csr')
    C = tvm.compute(A.data.shape, lambda i: A.data[i] * 2., tag='cs_scatter')
    s = tvm.create_schedule(C.op)
    _nr, _nc = 3, 5
    a = np.maximum(np.random.uniform(size=(_nr, _nc)).astype(dtype)-.6, 0.)
    # convert to sparse array tuple
    source_array = a
    ridx, cidx = np.nonzero(source_array)
    data = source_array[ridx, cidx]
    a_data = _nd.array(data, ctx)
    indices = np.nonzero(source_array)[1].astype(itype)
    a_indices = _nd.array(indices, ctx)
    indptr = [0]+np.apply_along_axis(np.count_nonzero, axis=1, arr=source_array).tolist()
    indptr = np.cumsum(np.array(indptr, itype)).astype(itype)
    a_indptr = _nd.array(indptr, ctx)
    a_init = (a_data, a_indices, a_indptr)
    # construct tvm sparse array with tuple
    a = tvmsp.array(a_init, shape=source_array.shape, ctx=ctx)
    assert a.data.dtype == a.dtype
    Ab = namedtuple('CSRBuffer', ['data', 'indices', 'indptr'])
    Ab.data = tvm.decl_buffer(a.data.shape, a.data.dtype, name='A_data')
    Ab.indices = tvm.decl_buffer(a.data.shape, a.data.dtype, name='A_indices')
    binds = {A.data: Ab.data, A.indices: Ab.indices}
    f = tvm.build(s, [nr, A.data, C], target, binds=binds)
    c = tvmsp.array(np.zeros((_nr, _nc), dtype), ctx)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data.shape[0], a.data, c.data)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2., rtol=1e-5)

if __name__ == "__main__":
    test_static_tensor()
    test_dynamic_tensor()
    test_sparse_array_tuple()

