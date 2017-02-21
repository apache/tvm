import tvm
import numpy as np

def tvm_call_global(*args):
    args = tvm.convert(args)
    return tvm.make.Call("int32", "tvm_call_global", args, 4, None, 0)


def run_jit(fapi, check):
    for target in ["stackvm"]:
        if target == "llvm":
            f = tvm.codegen.BuildLLVM(fapi)
        else:
            f = tvm.codegen.BuildStackVM(fapi)
        check(f)



def test_stack_vm_basic():
    a = tvm.nd.array(np.zeros(10, dtype='float32'))
    @tvm.register_func
    def tvm_call_back_get_shape(shape0):
        print(shape0)
        assert shape0 == a.shape[0]

    n = tvm.Var('n')
    Ab = tvm.Buffer((n, ), tvm.float32)
    stmt = tvm.make.Evaluate(tvm_call_global("tvm_call_back_get_shape", Ab.shape[0]))
    fapi = tvm.ir_pass.MakeAPI(stmt, "print_shape", [Ab], 1)
    run_jit(fapi, lambda f: f(a))


@tvm.register_func
def tvm_stack_vm_print(*x):
    print(x)


def test_stack_vm_loop():
    dtype = 'int64'
    n = tvm.Var('n')
    Ab = tvm.Buffer((n, ), dtype)
    i = tvm.Var('i')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 0, n - 1, 0, 0,
        tvm.make.Block(
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1,
                           i + 1),
            tvm.make.Evaluate(tvm_call_global("tvm_stack_vm_print", i))))
    fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 1)
    f = tvm.codegen.BuildStackVM(fapi)
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    def check(f):
        f(a)
        np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
    run_jit(fapi, check)


def test_stack_vm_cond():
    dtype = 'int64'
    n = tvm.Var('n')
    Ab = tvm.Buffer((n, ), dtype)
    i = tvm.Var('i')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 0, n - 1, 0, 0,
        tvm.make.IfThenElse(
            tvm.make.EQ(i, 4),
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1, i + 1),
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 2, i + 1)))
    fapi = tvm.ir_pass.MakeAPI(stmt, "test", [Ab], 1)
    def check(f):
        a = tvm.nd.array(np.zeros(10, dtype=dtype))
        f(a)
        y = np.arange(a.shape[0]) * 2
        y[5:] -= 1
        np.testing.assert_equal(a.asnumpy(), y)
    run_jit(fapi, check)


def test_llvm_add_pipeline():
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    s = tvm.Schedule(C.op)
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.Buffer(A.shape, A.dtype, name='A')
    Bb = tvm.Buffer(B.shape, B.dtype, name='B')
    Cb = tvm.Buffer(C.shape, C.dtype, name='C')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, B:Bb, C:Cb})
    stmt = tvm.ir_pass.Simplify(stmt)
    fapi = tvm.ir_pass.MakeAPI(stmt, "myadd", [Ab, Bb, Cb], 3)

    def check_llvm():
        # build and invoke the kernel.
        f = tvm.codegen.BuildLLVM(fapi)
        ctx = tvm.cpu(0)
        # launch the kernel.
        n = 1027
        a = tvm.nd.array(np.random.uniform(size=n).astype(Ab.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(Bb.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=Cb.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
    #check_llvm()

if __name__ == "__main__":
    test_stack_vm_cond()
    test_stack_vm_basic()
    test_stack_vm_loop()
    test_llvm_add_pipeline()
