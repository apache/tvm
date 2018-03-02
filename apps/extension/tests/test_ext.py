import tvm_ext
import tvm
import numpy as np

def test_bind_add():
    def add(a, b):
        return a + b
    f = tvm_ext.bind_add(add, 1)
    assert f(2)  == 3

def test_ext_dev():
    n = 10
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute((n,), lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)
    def check_llvm():
        if not tvm.module.enabled("llvm"):
            return
        f = tvm.build(s, [A, B], "ext_dev", "llvm")
        ctx = tvm.ext_dev(0)
        # launch the kernel.
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.zeros(n, dtype=B.dtype), ctx)
        f(a, b)
        np.testing.assert_allclose(b.asnumpy(), a.asnumpy() + 1)
    check_llvm()


def test_sym_add():
    a = tvm.var('a')
    b = tvm.var('b')
    c = tvm_ext.sym_add(a, b)
    assert c.a == a and c.b == b

def test_ext_vec():
    ivec = tvm_ext.ivec_create(1, 2, 3)
    assert(isinstance(ivec, tvm_ext.IntVec))
    assert ivec[0] == 1
    assert ivec[1] == 2

    def ivec_cb(v2):
        assert(isinstance(v2, tvm_ext.IntVec))
        assert v2[2] == 3

    tvm.convert(ivec_cb)(ivec)

def test_extract_ext():
    fdict = tvm.extract_ext_funcs(tvm_ext._LIB.TVMExtDeclare)
    assert fdict["mul"](3, 4) == 12


if __name__ == "__main__":
    test_ext_dev()
    test_ext_vec()
    test_bind_add()
    test_sym_add()
    test_extract_ext()
