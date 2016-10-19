import tvm

def test_buffer():
    buf = tvm.Buffer(tvm.Scope.Thread)
    shape = [32, 16]
    domain = [tvm.Range(v) for v in shape]
    buf.reshape(domain)
    x = tvm.Var('x')
    y = tvm.Var('y')
    assert tvm.format_str(buf(y, x)) == '%s[(%s + (%s * %s))]' % (buf.name, x.name, y.name, shape[1])


if __name__ == '__main__':
    test_buffer()
