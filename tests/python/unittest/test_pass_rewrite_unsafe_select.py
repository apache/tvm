import tvm


def test_rewrite_select():
    ib = tvm.ir_builder.create()
    A = ib.allocate("float32", 100, name="A", scope="global")
    i = tvm.var("i")
    y = tvm.select(i > 1, A[i-1], 1.0)
    yy = tvm.ir_pass.RewriteUnsafeSelect(tvm.make.Evaluate(y)).value

    z = tvm.select(tvm.select(i > 1, A[i-1], 1.0) > 0.0, A[i], 0.1)
    zz = tvm.ir_pass.RewriteUnsafeSelect(tvm.make.Evaluate(z)).value

    a = tvm.select(i>10, y, z)
    aa = tvm.ir_pass.RewriteUnsafeSelect(tvm.make.Evaluate(a)).value
    assert yy.name == "tvm_if_then_else"
    assert zz.name == "tvm_if_then_else"
    assert isinstance(aa, tvm.expr.Select)


if __name__ == "__main__":
    test_rewrite_select()
