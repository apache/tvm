import numpy as np
import tvm


def test_bop_without_topi():
    n = tvm.var('n')
    m = tvm.var('m')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((m,), name='B')
    """
    If topi, which provides overloaded operator support, is not imported,
    ValueError should be thrown.
    """
    try:
        C = A + B
        assert False
    except (ValueError, tvm.TVMError):
        pass


def test_operator_type():
    import topi

    k = 1
    n = tvm.var('n')
    A = tvm.placeholder((), name='A')
    B = tvm.placeholder((10, 5), name='B')
    B1 = B[0]
    B2 = B[0,0]

    assert isinstance(k + n, tvm.expr.Expr)
    assert isinstance(n + n, tvm.expr.Expr)
    assert isinstance(k + A, tvm.expr.Expr)
    assert isinstance(A + k, tvm.expr.Expr)
    assert isinstance(n + A, tvm.expr.Expr)
    assert isinstance(A + n, tvm.expr.Expr)
    assert isinstance(A + A, tvm.expr.Expr)

    assert isinstance(k + B, tvm.tensor.Tensor)
    assert isinstance(B + k, tvm.tensor.Tensor)
    assert isinstance(n + B, tvm.tensor.Tensor)
    assert isinstance(B + n, tvm.tensor.Tensor)
    assert isinstance(A + B, tvm.tensor.Tensor)
    assert isinstance(B + A, tvm.tensor.Tensor)
    assert isinstance(B + B, tvm.tensor.Tensor)

    assert isinstance(k + B2, tvm.expr.Expr)
    assert isinstance(B2 + k, tvm.expr.Expr)
    assert isinstance(n + B2, tvm.expr.Expr)
    assert isinstance(B2 + n, tvm.expr.Expr)
    assert isinstance(B2 + B2, tvm.expr.Expr)
    assert isinstance(B2 + A, tvm.expr.Expr)
    assert isinstance(A + B2, tvm.expr.Expr)
    assert isinstance(B2 + B, tvm.tensor.Tensor)
    assert isinstance(B + B2, tvm.tensor.Tensor)

    try:
        B1 + n
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + A
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + B
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + B1
        assert False
    except (ValueError, tvm.TVMError):
        pass

    try:
        B1 + B2
        assert False
    except (ValueError, tvm.TVMError):
        pass


def test_tensor_scalar_bop():
    import topi

    def tensor_scalar_bop(op):
        n = tvm.var('n')
        k = tvm.var('k')
        A = tvm.placeholder((n,), name='A')
        if op == "add":
            B = A + k
        elif op == "sub":
            B = A - k
        elif op == "mul":
            B = A * k
        elif op == "div":
            B = A / k
        s = tvm.create_schedule(B.op)
        f = tvm.build(s, [n, k, A, B], "llvm")

        ctx = tvm.cpu(0)
        n = 10
        k = 5
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        f(n, k, a, b)

        if op == "add":
            output = a.asnumpy() + k
        elif op == "sub":
            output = a.asnumpy() - k
        elif op == "mul":
            output = a.asnumpy() * k
        elif op == "div":
            output = a.asnumpy() / k
        np.testing.assert_allclose(b.asnumpy(), output)

    tensor_scalar_bop("add")
    tensor_scalar_bop("sub")
    tensor_scalar_bop("mul")
    tensor_scalar_bop("div")


def test_broadcast_bop():
    import topi

    def broadcast_bop(op):
        shape_A = (5, 2, 3)
        shape_B = (2, 1)
        A = tvm.placeholder(shape_A, name='A')
        B = tvm.placeholder(shape_B, name='B')
        if op == "add":
            C = A + B
        elif op == "sub":
            C = A - B
        elif op == "mul":
            C = A * B
        elif op == "div":
            C = A / B
        s = tvm.create_schedule(C.op)
        f = tvm.build(s, [A, B, C], "llvm")

        ctx = tvm.cpu(0)
        n = 10
        m = 20
        npy_A = np.random.uniform(size=shape_A).astype(A.dtype)
        npy_B = np.random.uniform(size=shape_B).astype(B.dtype)
        if op == "add":
            output = npy_A + npy_B
        elif op == "sub":
            output = npy_A - npy_B
        elif op == "mul":
            output = npy_A * npy_B
        elif op == "div":
            npy_B = np.abs(npy_B) + 0.001
            output = npy_A / npy_B
        a = tvm.nd.array(npy_A, ctx)
        b = tvm.nd.array(npy_B, ctx)
        c = tvm.nd.array(np.empty(output.shape).astype(C.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(c.asnumpy(), output, rtol=1E-4, atol=1E-4)

    broadcast_bop("add")
    broadcast_bop("sub")
    broadcast_bop("mul")
    broadcast_bop("div")


def test_combination():
    import topi

    k = 3
    n = 5
    m = 10
    x = tvm.var('x')
    A = tvm.placeholder((n, m), name='A')
    B = tvm.placeholder((n, m), name='B')
    C = tvm.placeholder((n, m), name='C')
    D = k + A - B * C / x
    s = tvm.create_schedule(D.op)
    f = tvm.build(s, [x, A, B, C, D], "llvm")
    ctx = tvm.cpu(0)
    x = 2
    a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(n, m)).astype(B.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=(n, m)).astype(C.dtype), ctx)
    d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
    f(x, a, b, c, d)
    np.testing.assert_allclose(d.asnumpy(), k + a.asnumpy() - b.asnumpy() * c.asnumpy() / x)


if __name__ == "__main__":
    test_bop_without_topi()
    test_operator_type()
    test_tensor_scalar_bop()
    test_broadcast_bop()
    test_combination()
