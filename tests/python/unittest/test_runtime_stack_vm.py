import tvm
import numpy as np

def tvm_call_global(*args):
    args = tvm.convert(args)
    return tvm.make.Call("int32", "tvm_call_global", args, 4, None, 0)


def test_stack_vm_basic():
    a = tvm.nd.array(np.zeros(10, dtype='float32'))
    @tvm.register_func
    def tvm_call_back_get_shape(shape0):
        print(shape0)
        assert shape0 == a.shape[0]

    n = tvm.Var('n')
    Ab = tvm.Buffer((n, ), tvm.float32)
    stmt = tvm.make.Evaluate(tvm_call_global("tvm_call_back_get_shape", Ab.shape[0]))
    print(stmt)
    fapi = tvm.codegen.MakeAPI(stmt, "print_shape", [Ab], 1)
    print(fapi.body)
    f = tvm.codegen.BuildStackVM(fapi)
    f(a)


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
    print(stmt)
    fapi = tvm.codegen.MakeAPI(stmt, "ramp", [Ab], 1)
    f = tvm.codegen.BuildStackVM(fapi)
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))


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
    print(stmt)
    fapi = tvm.codegen.MakeAPI(stmt, "test", [Ab], 1)
    f = tvm.codegen.BuildStackVM(fapi)
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f(a)
    y = np.arange(a.shape[0]) * 2
    y[5:] -= 1
    np.testing.assert_equal(a.asnumpy(), y)


if __name__ == "__main__":
    test_stack_vm_cond()
    test_stack_vm_loop()
    test_stack_vm_basic()
