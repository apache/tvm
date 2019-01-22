import tvm
import numpy as np

def test_get_global():
    targs = (10, 10.0, "hello")
    # register into global function table
    @tvm.register_func
    def my_packed_func(*args):
        assert(tuple(args) == targs)
        return 10
    # get it out from global function table
    f = tvm.get_global_func("my_packed_func")
    assert isinstance(f, tvm.Function)
    y = f(*targs)
    assert y == 10

def test_get_callback_with_node():
    x = tvm.convert(10)
    def test(y):
        assert y.handle != x.handle
        return y

    f2 = tvm.convert(test)
    # register into global function table
    @tvm.register_func
    def my_callback_with_node(y, f):
        assert y == x
        return f(y)

    # get it out from global function table
    f = tvm.get_global_func("my_callback_with_node")
    assert isinstance(f, tvm.Function)
    y = f(x, f2)
    assert(y.value == 10)


def test_return_func():
    def addy(y):
        def add(x):
            return tvm.convert(x + y)
        return add
    myf = tvm.convert(addy)
    f = myf(10)
    assert f(11).value == 21


def test_convert():
    # convert a function to tvm function
    targs = (10, 10.0, "hello", 10)
    def myfunc(*args):
        assert(tuple(args) == targs)

    f = tvm.convert(myfunc)
    assert isinstance(f, tvm.Function)

def test_byte_array():
    s = "hello"
    a = bytearray(s, encoding="ascii")

    def myfunc(ss):
        assert ss == a
    f = tvm.convert(myfunc)
    f(a)


def test_empty_array():
    def myfunc(ss):
        assert tuple(ss) == ()
    x = tvm.convert(())
    tvm.convert(myfunc)(x)


def test_ctx():
    def test_ctx_func(ctx):
        assert tvm.gpu(7) == ctx
        return tvm.cpu(0)
    x = test_ctx_func(tvm.gpu(7))
    assert x == tvm.cpu(0)
    x = tvm.opencl(10)
    x = tvm._api_internal._context_test(x, x.device_type, x.device_id)
    assert x == tvm.opencl(10)

def test_trace_default_action():
    n = 2
    x = tvm.placeholder((n,n,n), name="X", dtype="float32")
    y = tvm.compute(x.shape, lambda i, j, k: tvm.trace([i, j, k, x[i][j][k]]))
    s = tvm.create_schedule(y.op)
    f = tvm.build(s, [x, y], target="llvm")
    xnd = tvm.nd.array(np.ones((n,n,n), dtype=x.dtype))
    ynd = tvm.nd.array(np.zeros((n,n,n), dtype=y.dtype))
    f(xnd, ynd)

def test_trace_expr_assign():
    @tvm.register_func("tvm.trace_callback2")
    def trace_buffer(x):
        return

    def check_assign(dtype):
        n = 4
        x = tvm.placeholder((n,n,n), name="X", dtype=dtype)
        y = tvm.compute(x.shape, lambda i, j, k: tvm.trace([x[i][j][k]], "tvm.trace_callback2"))
        z = tvm.compute(x.shape, lambda i, j, k: tvm.trace([y[i][j][k]], "tvm.trace_callback2"))
        s = tvm.create_schedule(z.op)
        f = tvm.build(s, [x, y, z], "llvm")

        xnd = tvm.nd.array(np.ones((n,n,n), dtype=x.dtype))
        ynd = tvm.nd.array(np.zeros((n,n,n), dtype=y.dtype))
        znd = tvm.nd.array(np.zeros((n,n,n), dtype=z.dtype))
        f(xnd, ynd, znd)

        assert(np.array_equal(xnd.asnumpy(), np.ones((n,n,n))))
        assert(np.array_equal(ynd.asnumpy(), np.ones((n,n,n))))
        assert(np.array_equal(znd.asnumpy(), np.ones((n,n,n))))

    for t in ["float64", "float32", "int64", "int32"]:
        check_assign(t)

def test_trace_expr_sum_generated():
    @tvm.register_func("tvm.trace_callback3")
    def trace_buffer(x):
        return

    def check_expr_sum(dtype):
        n = 4
        a = tvm.placeholder((n,n,n), name="a", dtype=dtype)
        b = tvm.placeholder((n,n,n), name="b", dtype=dtype)
        c = tvm.compute(a.shape, lambda i, j, k: tvm.trace([a[i][j][k]],"tvm.trace_callback3")
                                         + tvm.trace([b[i][j][k]],"tvm.trace_callback3"))
        s = tvm.create_schedule(c.op)
        f = tvm.build(s, [a, b, c])
        xnd = tvm.nd.array(np.array(np.ones((n,n,n), dtype=a.dtype)))
        ynd = tvm.nd.array(np.array(np.ones((n,n,n), dtype=b.dtype)))
        znd = tvm.nd.array(np.zeros((n,n,n), dtype=c.dtype))
        f(xnd, ynd, znd)
        assert(np.array_equal(znd.asnumpy(), xnd.asnumpy() + ynd.asnumpy()))

    for t in ["float64", "float32", "int64", "int32"]:
        check_expr_sum(t)

def test_trace_expr_sum_args():
    @tvm.register_func("tvm.trace_silent")
    def silent(*args):
      return

    def check_expr_sum(dtype):
        n = 4
        a = tvm.placeholder((n,n,n), name="a", dtype=dtype)
        b = tvm.placeholder((n,n,n), name="b", dtype=dtype)
        e = tvm.placeholder((n,n,n), name="e", dtype=dtype)
        d = tvm.placeholder((n,n,n), name="d", dtype=dtype)

        c = tvm.compute(a.shape, lambda i, j, k: tvm.trace([i, j, k, a[i][j][k]], "tvm.trace_silent")
                                               + tvm.trace([i, j, k, b[i][j][k]], "tvm.trace_silent")
                                               + tvm.trace([i, j, k, d[i][j][k]], "tvm.trace_silent")
                                               + tvm.trace([i, j, k, e[i][j][k]], "tvm.trace_silent"))
        s = tvm.create_schedule(c.op)
        f = tvm.build(s, [a, b, d, e, c])
        a_nd = tvm.nd.array(np.array(np.ones((n,n,n), dtype=a.dtype)))
        b_nd = tvm.nd.array(np.array(np.ones((n,n,n), dtype=b.dtype)))
        d_nd = tvm.nd.array(np.array(np.ones((n,n,n), dtype=d.dtype)))
        e_nd = tvm.nd.array(np.array(np.ones((n,n,n), dtype=e.dtype)))
        c_nd = tvm.nd.array(np.zeros((n,n,n), dtype=c.dtype))
        f(a_nd, b_nd, d_nd, e_nd, c_nd)
        assert(np.array_equal(c_nd.asnumpy(), a_nd.asnumpy()
                                            + b_nd.asnumpy()
                                            + d_nd.asnumpy()
                                            + e_nd.asnumpy()))

    for t in ["float64", "float32", "int64", "int32"]:
        check_expr_sum(t)

def test_trace_expr_sum_custom():
    @tvm.register_func("tvm.trace_callback4")
    def trace_buffer(x):
        return

    def check_expr_sum_custom(dtype):
        n = 4
        a = tvm.placeholder((n,n), name="a", dtype=dtype)
        b = tvm.placeholder((n,n), name="b", dtype=dtype)
        c = tvm.compute(a.shape, lambda i,j: tvm.trace([a[i][j]], "tvm.trace_callback4")
                                         + tvm.trace([b[i][j]], "tvm.trace_callback4"))
        s = tvm.create_schedule(c.op)
        f = tvm.build(s, [a, b, c])
        npa = np.array([[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=a.dtype)
        npb = np.array([[1,0,0,0], [0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype=a.dtype)
        xnd = tvm.nd.array(npa)
        ynd = tvm.nd.array(npb)
        znd = tvm.nd.array(np.zeros((n,n), dtype=c.dtype))
        f(xnd, ynd, znd)
        assert(np.array_equal(znd.asnumpy(), npa + npb))

    for t in ["float64", "float32", "int64", "int32"]:
        check_expr_sum_custom(t)

def test_trace_can_change_traced_value_int():
    @tvm.register_func("tvm.trace_change_int_first")
    def trace_buffer(x):
        return 13

    @tvm.register_func("tvm.trace_change_int_second")
    def trace_buffer(x):
        return 14

    def check_assign(dtype):
        n = 4
        x = tvm.placeholder((n,), name="X", dtype=dtype)
        y = tvm.compute(x.shape, lambda i: tvm.trace([x[i]], "tvm.trace_change_int_first"))
        z = tvm.compute(x.shape, lambda i: tvm.trace([y[i]], "tvm.trace_change_int_second"))
        s = tvm.create_schedule(z.op)
        f = tvm.build(s, [x, y, z], "llvm")

        xnd = tvm.nd.array(np.ones((n,), dtype=x.dtype))
        ynd = tvm.nd.array(np.zeros((n,), dtype=y.dtype))
        znd = tvm.nd.array(np.zeros((n,), dtype=z.dtype))
        f(xnd, ynd, znd)
        check_array_first = np.array([13, 13, 13, 13])
        check_array_second = np.array([14, 14, 14, 14])
        assert(np.array_equal(ynd.asnumpy(), check_array_first))
        assert(np.array_equal(znd.asnumpy(), check_array_second))

    for t in ["int64", "int32"]:
        check_assign(t)

def test_trace_can_change_traced_value_float():
    @tvm.register_func("tvm.trace_change_float_first")
    def trace_buffer(x):
        return 13.0

    @tvm.register_func("tvm.trace_change_float_second")
    def trace_buffer(x):
        return 14.0

    def check_assign(dtype):
        n = 4
        x = tvm.placeholder((n,), name="X", dtype=dtype)
        y = tvm.compute(x.shape, lambda i: tvm.trace([x[i]], "tvm.trace_change_float_first"))
        z = tvm.compute(x.shape, lambda i: tvm.trace([y[i]], "tvm.trace_change_float_second"))
        s = tvm.create_schedule(z.op)
        f = tvm.build(s, [x, y, z], "llvm")

        xnd = tvm.nd.array(np.ones((n,), dtype=x.dtype))
        ynd = tvm.nd.array(np.zeros((n,), dtype=y.dtype))
        znd = tvm.nd.array(np.zeros((n,), dtype=z.dtype))
        f(xnd, ynd, znd)
        check_array_first = np.array([13.0, 13.0, 13.0, 13.0])
        check_array_second = np.array([14.0, 14.0, 14.0, 14.0])
        assert(np.array_equal(ynd.asnumpy(), check_array_first))
        assert(np.array_equal(znd.asnumpy(), check_array_second))

    for t in ["float64", "float32"]:
        check_assign(t)

if __name__ == "__main__":
    test_empty_array()
    test_get_global()
    test_get_callback_with_node()
    test_convert()
    test_return_func()
    test_byte_array()
    test_ctx()
    test_trace_expr_assign()
    test_trace_expr_sum_generated()
    test_trace_expr_sum_custom()
    test_trace_expr_sum_args()
    test_trace_default_action()
    test_trace_can_change_traced_value_int()
    test_trace_can_change_traced_value_float()

