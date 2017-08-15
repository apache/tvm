import tvm

def test_coproc_lift():
    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    cp = tvm.thread_axis((0, 1), "cop")
    value = tvm.make.StringImm("xxx")

    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, 10, name="j") as j:
            ib.scope_attr(cp, "coproc_uop_scope", value)
            A[i] = A[i] + 1
        with ib.if_scope(i.equal(0)):
            with ib.for_range(0, 10, name="j") as j:
                ib.scope_attr(cp, "coproc_uop_scope", value)
                A[j] = A[j] + 2
    body = ib.get()
    body = tvm.ir_pass.LiftAttrScope(body, "coproc_uop_scope")
    assert body.body.body.node == cp


if __name__ == "__main__":
    test_coproc_lift()
