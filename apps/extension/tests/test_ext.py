import tvm_ext
import tvm

def test_bind_add():
    def add(a, b):
        return a + b
    f = tvm_ext.bind_add(add, 1)
    assert f(2)  == 3

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

if __name__ == "__main__":
    test_ext_vec()
    test_bind_add()
    test_sym_add()
