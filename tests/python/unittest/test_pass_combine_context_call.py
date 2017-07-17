import tvm

def test_for():
    dev_type = tvm.var("dev_type")
    def device_context(dev_id):
        ctx = tvm.call_extern("handle", "device_context", dev_type, dev_id)
        return tvm.make.Call(
            "handle", "tvm_thread_context", [ctx], tvm.expr.Call.Intrinsic, None, 0)

    ib = tvm.ir_builder.create()
    n = tvm.var("n")
    A = ib.allocate("float32", n, name="A", scope="global")
    with ib.for_range(0, n, name="i") as i:
        ib.emit(tvm.call_extern
                ("int32", "fadd", device_context(0), A))
        with ib.for_range(0, 10, name="j") as j:
            ib.emit(tvm.call_extern
                    ("int32", "fadd", device_context(1), A))
            ib.emit(tvm.call_extern
                    ("int32", "fadd", device_context(0), A))
    body = ib.get()
    f = tvm.ir_pass.MakeAPI(body, "func", [dev_type, n], 2, True)
    f = tvm.ir_pass.CombineContextCall(f)
    assert f.body.value.dtype == "handle"
    assert f.body.body.value.dtype == "handle"


if __name__ == "__main__":
    test_for()
