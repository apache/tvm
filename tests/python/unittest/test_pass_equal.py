import tvm

def test_equal_expr():
    x = tvm.var('x')
    y = tvm.var('y')

    def func1():
        return x + y + 1

    def func2():
        return tvm.exp((x + y + 1) * y / 4)

    assert tvm.ir_pass.Equal(func1(), func1())
    assert tvm.ir_pass.Equal(func2(), func2())
    assert not tvm.ir_pass.Equal(func2(), func1())


def test_equal_compute():
    x = tvm.var('x')
    y = tvm.var('y')
    n = 128
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    ii = tvm.var('i')
    jj = tvm.var('j')

    def func1():
        k = tvm.reduce_axis((0, n), name='k')
        return tvm.sum(A[ii, k] * B[jj, k], axis=k)

    Ab = tvm.decl_buffer((n,), name='A')
    n = tvm.var("n")
    def func2():
        ib = tvm.ir_builder.create()
        A = ib.buffer_ptr(Ab)
        with ib.for_range(0, n, name="i") as i:
            A[i] = A[i] + 1
            with ib.for_range(0, 10, name="j") as j:
                A[j] = A[j] + 2
        return ib.get()

    assert tvm.ir_pass.Equal(func1(), func1())
    assert tvm.ir_pass.Equal(func2(), func2())


if __name__ == "__main__":
    test_equal_expr()
    test_equal_compute()
