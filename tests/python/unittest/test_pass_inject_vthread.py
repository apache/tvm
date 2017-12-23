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


def test_vthread_extern():
    dtype = 'int64'
    n = 100
    m = 4
    nthread = 2
    def get_vthread(name):
        tx = tvm.thread_axis(name)
        ty = tvm.thread_axis(name)
        ib = tvm.ir_builder.create()
        with ib.for_range(0, n) as i:
            ib.scope_attr(tx, "virtual_thread", nthread)
            ib.scope_attr(ty, "virtual_thread", nthread)
            A = ib.allocate("float32", m, name="A", scope="shared")
            B = ib.allocate("float32", m, name="B", scope="shared")
            C = ib.allocate("float32", m, name="C", scope="shared")
            cbuffer = tvm.decl_buffer((m,), dtype=C.dtype, data=C.asnode())
            abuffer = tvm.decl_buffer((m,), dtype=A.dtype, data=A.asnode())
            bbuffer = tvm.decl_buffer((m,), dtype=B.dtype, data=B.asnode())
            A[tx] = tx + 1.0
            B[ty] = ty + 1.0
            ib.emit(tvm.call_extern("int32", "Run",
                                    abuffer.access_ptr("r"),
                                    bbuffer.access_ptr("r"),
                                    cbuffer.access_ptr("rw")))
        return ib.get()

    stmt = tvm.ir_pass.InjectVirtualThread(get_vthread("vthread"))
    assert stmt.body.body.extents[0].value == 2
    assert stmt.body.body.body.body.body.body.extents[0].value == 2
    assert len(stmt.body.body.body.body.body.body.extents) == 3


if __name__ == "__main__":
    test_vthread_extern()
    test_vthread()
