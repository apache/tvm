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

if __name__ == "__main__":
    test_bind_add()
    test_sym_add()
