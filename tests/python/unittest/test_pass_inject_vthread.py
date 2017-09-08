import tvm

def test_vthread():
    dtype = 'int64'
    n = 100
    m = 4
    nthread = 2
    def get_vthread(name):
        tx = tvm.thread_axis(name)
        ty = tvm.thread_axis(name)
        ib = tvm.ir_builder.create()
        A = ib.pointer("float32", name="A")
        C = ib.pointer("float32", name="C")
        with ib.for_range(0, n) as i:
            ib.scope_attr(tx, "virtual_thread", nthread)
            ib.scope_attr(ty, "virtual_thread", nthread)
            B = ib.allocate("float32", m, name="B", scope="shared")
            B[i] = A[i * nthread + tx]
            bbuffer = tvm.decl_buffer((m,), dtype=B.dtype, data=B.asnode())
            ib.emit(tvm.call_extern("int32", "Run",
                                    bbuffer.access_ptr("r"),
                                    tvm.call_pure_intrin("int32", "tvm_context_id")))
            C[i * nthread + tx] = B[i] + 1
        return ib.get()

    stmt = tvm.ir_pass.InjectVirtualThread(get_vthread("vthread"))
    assert stmt.body.body.extents[0].value == 2
    stmt = tvm.ir_pass.InjectVirtualThread(get_vthread("cthread"))
    assert len(stmt.body.body.extents) == 3

if __name__ == "__main__":
    test_vthread()
