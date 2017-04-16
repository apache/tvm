import tvm
import numpy as np

def run_jit(fapi, check):
    for target in ["llvm", "stackvm"]:
        if not tvm.codegen.enabled(target):
            continue
        f = tvm.codegen.build_module(fapi, target)
        s = f.get_source()
        check(f)


def test_stack_vm_basic():
    a = tvm.nd.array(np.zeros(10, dtype='float32'))
    @tvm.register_func
    def tvm_call_back_get_shape(shape0):
        print(shape0)
        assert shape0 == a.shape[0]

    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), tvm.float32)
    stmt = tvm.make.Evaluate(tvm.call_packed("tvm_call_back_get_shape", Ab.shape[0]))
    fapi = tvm.ir_pass.MakeAPI(stmt, "print_shape", [Ab], 0)
    run_jit(fapi, lambda f: f(a))


@tvm.register_func
def tvm_stack_vm_print(*x):
    print(x)


def test_stack_vm_loop():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 0, n - 1, 0, 0,
        tvm.make.Block(
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1,
                           i + 1),
            tvm.make.Evaluate(tvm.call_packed("tvm_stack_vm_print", i))))
    fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 0)
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    def check(f):
        f(a)
        np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
    run_jit(fapi, check)


def test_stack_vm_cond():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, 0, n - 1, 0, 0,
        tvm.make.IfThenElse(
            tvm.make.EQ(i, 4),
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1, i + 1),
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 2, i + 1)))
    fapi = tvm.ir_pass.MakeAPI(stmt, "test", [Ab], 0)
    def check(f):
        a = tvm.nd.array(np.zeros(10, dtype=dtype))
        f(a)
        y = np.arange(a.shape[0]) * 2
        y[5:] -= 1
        np.testing.assert_equal(a.asnumpy(), y)
    run_jit(fapi, check)

if __name__ == "__main__":
    test_stack_vm_basic()
    test_stack_vm_cond()
    test_stack_vm_loop()
