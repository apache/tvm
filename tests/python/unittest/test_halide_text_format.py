import tvm.contrib.pyfrontend as frontend
import tvm, inspect, sys, traceback, numpy

#tests basic features
def outer_product(n, m, a, b, c):
    for i in range(n):
        for j in range(m):
            c[i, j] = a[i] * b[j]

def test_outer_product():
    n = tvm.var('n')
    m = tvm.var('m')
    a = tvm.placeholder((n, ), name = 'a')
    b = tvm.placeholder((m, ), name = 'b')
    c = tvm.placeholder((n, m), name = 'c')
    ir, _ = frontend.parse(outer_product, [n, m, a, b, c])
    #print(ir)
    #Check for i in (0, n)
    assert isinstance(ir, tvm.stmt.For)
    assert ir.loop_var.name == 'i'
    assert ir.min.value == 0
    assert ir.extent.name == 'n'
    ibody = ir.body
    assert isinstance(ibody, tvm.stmt.For)
    #Check for j in (0, m)
    assert ibody.loop_var.name == 'j'
    assert ibody.min.value == 0
    assert ibody.extent.name == 'm'
    #Check loop body
    jbody = ibody.body
    assert isinstance(jbody, tvm.stmt.Provide)
    assert jbody.func.name == 'c'
    assert len(jbody.args) == 2
    assert jbody.args[0].name == 'i'
    assert jbody.args[1].name == 'j'
    assert isinstance(jbody.value, tvm.expr.Mul)
    mul = jbody.value
    assert isinstance(mul.a, tvm.expr.Call)
    assert mul.a.name == 'a'
    assert mul.b.name == 'b'

    func = frontend.lower(outer_product, [n, m, a, b, c])
    func = tvm.build(func)

    _n = 999
    _m = 1001
    _a = numpy.random.rand(_n).astype('float32')
    _b = numpy.random.rand(_m).astype('float32')
    c_python = numpy.zeros((_n, _m), dtype = 'float32')
    outer_product(_n, _m, _a, _b, c_python)

    tvm_a = tvm.ndarray.array(_a)
    tvm_b = tvm.ndarray.array(_b)
    tvm_c = tvm.ndarray.array(numpy.zeros((_n, _m), dtype = 'float32'))
    func(_n, _m, tvm_a, tvm_b, tvm_c)
    numpy.testing.assert_allclose(tvm_c.asnumpy(), c_python, rtol = 1e-5)
    #print(func)

def fanout(n, a, b):
    three = 3.0
    for i in range(a.shape[0] - 3):
        sigma = 0.0
        for j in range(3):
            sigma = sigma + a[i + j]
        sigma = sigma / three
        b[i] = sigma

def test_fanout():
    n = tvm.var('n')
    a = tvm.placeholder((n, ), name = 'a')
    b = tvm.placeholder((n-3, ), name = 'b')
    ir, _ = frontend.parse(fanout, [n, a, b])
    #print(ir)
    assert isinstance(ir, tvm.stmt.Realize)
    assert ir.bounds[0].min.value == 0
    assert ir.bounds[0].extent.value == 1
    assert ir.func.name == 'sigma'
    rbody = ir.body
    #Check for i in (0, n-3)
    assert isinstance(rbody, tvm.stmt.For)
    assert rbody.loop_var.name == 'i'
    assert rbody.min.value == 0
    assert tvm.ir_pass.Equal(rbody.extent, n - 3)
    #Check i loop body
    ibody = rbody.body
    assert isinstance(ibody.first, tvm.stmt.Provide)
    assert ibody.first.func.name == 'sigma'
    assert len(ibody.first.args) == 1
    assert ibody.first.args[0].value == 0
    #Check fanout loop
    jloop = ibody.rest.first
    assert jloop.loop_var.name == 'j'
    assert jloop.min.value == 0
    assert jloop.extent.value == 3
    jbody = jloop.body
    assert isinstance(jbody, tvm.stmt.Provide)
    assert len(jbody.args) == 1
    assert jbody.args[0].value == 0
    assert jbody.func.name == 'sigma'
    assert isinstance(jbody.value, tvm.expr.Add)
    value = jbody.value
    assert isinstance(value.a, tvm.expr.Call)
    assert value.a.name == 'sigma'
    assert len(value.a.args) == 1
    assert value.a.args[0].value == 0
    assert value.b.name == 'a'
    assert len(value.b.args) == 1
    assert tvm.ir_pass.Equal(value.b.args[0], rbody.loop_var + jloop.loop_var)
    divide= ibody.rest.rest.first
    assert isinstance(divide, tvm.stmt.Provide)
    assert len(divide.args) == 1
    assert divide.args[0].value == 0
    value = divide.value
    assert isinstance(value, tvm.expr.Mul)
    assert value.a.name == 'sigma'
    assert len(value.a.args) == 1
    assert value.a.args[0].value == 0
    assert abs(value.b.value - (1 / 3.0)) < 1e-5
    write = ibody.rest.rest.rest
    assert isinstance(write, tvm.stmt.Provide)
    assert write.func.name == 'b'
    assert write.value.name == 'sigma'
    assert len(write.value.args) == 1
    assert write.value.args[0].value == 0
    #print(write)


def failure():
    for i in range(1, 100):
        i = 0

def test_failure():
    try:
        frontend.parse(failure, [])
    except AssertionError:
        _, _, tb = sys.exc_info()
        _, _, func, text = traceback.extract_tb(tb)[-1]
        assert func == 'visit_Name'
        assert text == 'assert node.id not in self.loops_above'

@tvm.contrib.pyfrontend.py_frontend
def annotation(a):
    with Unrolled() as i:
        for i in range(6):
            a[i] = i

def test_unroll():
    a = tvm.placeholder((6, ), name = 'a')
    ir, _ = tvm.contrib.pyfrontend.parse(annotation, [a])
    assert isinstance(ir, tvm.stmt.For)
    assert ir.for_type == tvm.stmt.For.Unrolled

if __name__ == "__main__":
    test_outer_product()
    test_fanout()
    test_failure()
    test_unroll()

#test_annotation()
#test_outer_product()
#test_blur()
#test_failure()

