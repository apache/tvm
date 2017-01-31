import tvm

def test_buffer():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    Ab = tvm.Buffer((m, n), tvm.float32)
    Bb = tvm.Buffer((n, l), tvm.float32)

    assert isinstance(Ab, tvm.schedule.Buffer)
    assert Ab.dtype == tvm.float32
    assert tuple(Ab.shape) == (m, n)


if __name__ == "__main__":
    test_buffer()
