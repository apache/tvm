import tvm
from tvm.schedule import Buffer

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

if __name__ == "__main__":
    test_buffer()
    test_buffer_access_ptr()
    test_buffer_access_ptr_offset()
    test_buffer_index_merge_mult_mod()
