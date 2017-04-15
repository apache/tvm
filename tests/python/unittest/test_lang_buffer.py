import tvm

def test_buffer():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    Ab = tvm.decl_buffer((m, n), tvm.float32)
    Bb = tvm.decl_buffer((n, l), tvm.float32)

    assert isinstance(Ab, tvm.schedule.Buffer)
    assert Ab.dtype == tvm.float32
    assert tuple(Ab.shape) == (m, n)


if __name__ == "__main__":
    test_buffer()
