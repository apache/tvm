import tvm, inspect, sys, traceback, numpy
from tvm.hybrid import hybrid_script

@hybrid_script
def outer_product(n, m, a, b, c):
    for i in serial(n):
        for j in serial(m):
            c[i, j] = a[i] * b[j]

#Test global function
#Test bridge between frontend and backend
def test_outer_product():
    n = tvm.var('n')
    m = tvm.var('m')
    a = tvm.placeholder((n, ), name='a')
    b = tvm.placeholder((m, ), name='b')
    c = tvm.placeholder((n, m), name='c')
    ir, _ = outer_product(n, m, a, b, c)
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

    func = tvm.lower(ir, [n, m, a, b, c])
    func = tvm.build(func)

    _n = 999
    _m = 1001
    _a = numpy.random.rand(_n).astype('float32')
    _b = numpy.random.rand(_m).astype('float32')
    c_python = numpy.zeros((_n, _m), dtype='float32')
    outer_product(_n, _m, _a, _b, c_python)

    tvm_a = tvm.ndarray.array(_a)
    tvm_b = tvm.ndarray.array(_b)
    tvm_c = tvm.ndarray.array(numpy.zeros((_n, _m), dtype='float32'))
    func(_n, _m, tvm_a, tvm_b, tvm_c)
    numpy.testing.assert_allclose(tvm_c.asnumpy(), c_python, rtol=1e-5)

#Test local function
#Test allocation of local variable
def test_fanout():
    @hybrid_script
    def fanout(n, a, b):
        three = 3.0
        for i in serial(a.shape[0] - 3):
            sigma = 0.0
            for j in serial(3):
                sigma = sigma + a[i + j]
            sigma = sigma / three
            b[i] = sigma

    n = tvm.var('n')
    a = tvm.placeholder((n, ), name='a')
    b = tvm.placeholder((n-3, ), name='b')
    ir, _ = fanout(n, a, b)

    #Check for i in (0, n-3)
    assert isinstance(ir, tvm.stmt.For)
    assert ir.loop_var.name == 'i'
    assert ir.min.value == 0
    assert tvm.ir_pass.Equal(ir.extent, n - 3)
    #Check loopbody
    ibody = ir.body
    assert isinstance(ibody, tvm.stmt.Realize)
    assert ibody.bounds[0].min.value == 0
    assert ibody.bounds[0].extent.value == 1
    assert ibody.func.name == 'sigma'
    #Check i loop body
    rbody = ibody.body
    assert isinstance(rbody.first, tvm.stmt.Provide)
    assert rbody.first.func.name == 'sigma'
    assert len(rbody.first.args) == 1
    assert rbody.first.args[0].value == 0
    #Check fanout loop
    jloop = rbody.rest.first
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
    assert tvm.ir_pass.Equal(value.b.args[0], ir.loop_var + jloop.loop_var)
    divide= rbody.rest.rest.first
    assert isinstance(divide, tvm.stmt.Provide)
    assert len(divide.args) == 1
    assert divide.args[0].value == 0
    value = divide.value
    assert isinstance(value, tvm.expr.Mul)
    assert value.a.name == 'sigma'
    assert len(value.a.args) == 1
    assert value.a.args[0].value == 0
    assert abs(value.b.value - (1 / 3.0)) < 1e-5
    write = rbody.rest.rest.rest
    assert isinstance(write, tvm.stmt.Provide)
    assert write.func.name == 'b'
    assert write.value.name == 'sigma'
    assert len(write.value.args) == 1
    assert write.value.args[0].value == 0

def test_failure():
    @hybrid_script
    def failure():
        for i in serial(1, 100):
            i = 0
    try:
        tvm.hybrid.parse(failure, [])
    except AssertionError:
        _, _, tb = sys.exc_info()
        _, _, func, text = traceback.extract_tb(tb)[-1]
        assert func == 'visit_Assign'
        assert text == 'assert lhs not in self.loops_above.keys()'


def test_looptype():
    @hybrid_script
    def looptype(a):
        for i in parallel(6):
            a[i] = i
        for j in vectorized(6):
            a[j] = j
        for k in unrolled(6):
            a[k] = k
    a = tvm.placeholder((6, ), name='a')
    ir, _ = looptype(a)
    iloop = ir.first
    jloop = ir.rest.first
    kloop = ir.rest.rest
    assert iloop.for_type == tvm.stmt.For.Parallel
    assert jloop.for_type == tvm.stmt.For.Vectorized
    assert kloop.for_type == tvm.stmt.For.Unrolled

def test_if():
    @hybrid_script
    def if_then_else(a, b):
        for i in serial(10):
            if i % 2 == 0:
                a[i] = -1
            else:
                a[i] = 1
        for i in unrolled(10):
            b[i] = -1 if i % 2 == 0 else 1

    a = tvm.placeholder((10, ), dtype='int32', name='a')
    b = tvm.placeholder((10, ), dtype='int32', name='b')
    ir, _ = if_then_else(a, b)
    func = tvm.lower(ir, [a, b])
    func = tvm.build(func)
    assert func

    _a = numpy.zeros((10, ), dtype = 'int32')
    _b = numpy.zeros((10, ), dtype = 'int32')
    if_then_else(_a, _b)

    tvm_a = tvm.ndarray.array(numpy.zeros((10, ), dtype='int32'))
    tvm_b = tvm.ndarray.array(numpy.zeros((10, ), dtype='int32'))
    func(tvm_a, tvm_b)

    numpy.testing.assert_allclose(tvm_a.asnumpy(), _a, rtol=1e-5)
    numpy.testing.assert_allclose(tvm_b.asnumpy(), _b, rtol=1e-5)
    numpy.testing.assert_allclose(tvm_a.asnumpy(), tvm_b.asnumpy(), rtol=1e-5)

if __name__ == "__main__":
    test_outer_product()
    test_fanout()
    test_failure()
    test_looptype()
    test_if()

