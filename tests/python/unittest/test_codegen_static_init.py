import tvm
import numpy as np

def test_static_init():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    ib = tvm.ir_builder.create()
    A = ib.buffer_ptr(Ab)
    cp = tvm.thread_axis((0, 1), "cop")
    finit = tvm.make.StringImm("TVMBackendRunOnce")
    ib.scope_attr(cp, "coproc_uop_scope", finit)
    with ib.for_range(0, n, "i", for_type="parallel") as i:
        A[i] = A[i] + 1
    stmt = ib.get()
    fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 0, True)
    fapi = tvm.ir_pass.LowerTVMBuiltin(fapi)
    f = tvm.codegen.build_module(fapi, "llvm")
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)
    f(a)
    np.testing.assert_equal(a.asnumpy(), np.ones(a.shape[0]))


if __name__ == "__main__":
    test_static_init()
