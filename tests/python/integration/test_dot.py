import tvm
import numpy as np

def lower(s, args, name="mydot"):
    binds = {}
    arg_list = []

    for x in args:
        assert isinstance(x, tvm.tensor.Tensor)
        buf = tvm.Buffer(x.shape, dtype=x.dtype, name=x.op.name)
        binds[x] = buf
        arg_list.append(buf)
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.StorageFlatten(stmt, binds)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    fapi = tvm.ir_pass.MakeAPI(stmt, name, arg_list, 0)
    return fapi


def mybuild(fapi, target="llvm"):
    return


def test_dot():
    nn = 12
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    k = tvm.IterVar((0, n), name='k')
    C = tvm.compute((1,), lambda _: tvm.sum(A[k] * B[k], axis=k), name='C')
    s = tvm.Schedule(C.op)
    fapi = lower(s, [A, B, C])

    def verify(target):
        if not tvm.codegen.enabled(target):
            print("Target %s is not enabled" % target)
            return
        f = tvm.codegen.build(fapi, target)
        # verify
        ctx = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(nn,)).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=(nn,)).astype(B.dtype), ctx)
        c  = tvm.nd.array(np.zeros((1,), dtype=C.dtype), ctx)
        f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-4)

    verify("llvm")

if __name__ == "__main__":
    test_dot()
