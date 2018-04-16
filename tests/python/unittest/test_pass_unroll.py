import tvm
import os


def test_unroll_loop():
    ib = tvm.ir_builder.create()
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    Aptr = ib.buffer_ptr(Ab)
    # for i in 0 to n-1:
    with ib.for_range(n, n + 2, name="i") as i:
        with ib.for_range(0, 8, name="i", for_type="unroll") as j:
            Aptr[j + 1] = Aptr[i] + 1

    stmt = ib.get()
    assert isinstance(stmt, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 16, 8, 0, True)
    assert not isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 15, 8, 0, True)
    assert isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 16, 8, 0, False)
    assert isinstance(ret, tvm.stmt.For)
    assert ret.for_type == tvm.stmt.For.Unrolled

    ib = tvm.ir_builder.create()
    ib.scope_attr(tvm.const(0), "pragma_auto_unroll_max_step", 16)
    ib.emit(stmt)
    wrapped = ib.get()
    wrapped = tvm.make.Block(wrapped, stmt)
    assert isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(wrapped, 0, 8, 0, False)
    assert isinstance(ret.first, tvm.stmt.For)
    assert ret.first.for_type == tvm.stmt.For.Unrolled
    assert isinstance(ret.rest, tvm.stmt.For)
    assert ret.rest.for_type != tvm.stmt.For.Unrolled


if __name__ == "__main__":
    test_unroll_loop()
