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

if __name__ == "__main__":
    test_buffer()
    test_buffer_access_ptr()
