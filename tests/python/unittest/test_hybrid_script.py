import tvm, inspect, sys, traceback, numpy
from tvm.hybrid import script
from tvm.hybrid.intrin import HYBRID_GLOBALS


@script
def outer_product(n, m, a, b, c):
    for i in range(n):
        for j in range(m):
            c[i, j] = a[i] * b[j]

#Test global function
#Test bridge between frontend and backend
def test_outer_product():
    n = tvm.var('n')
    m = tvm.var('m')
    a = tvm.placeholder((n, ), name='a')
    b = tvm.placeholder((m, ), name='b')
    c = tvm.placeholder((n, m), name='c')
    ir = outer_product(n, m, a, b, c)
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

    for key, _ in HYBRID_GLOBALS.items():
        assert key not in globals().keys()
        assert key not in outer_product.__globals__.keys()

#Test local function
#Test allocation of local variable
def test_fanout():
    @script
    def fanout(n, a, b):
        three = 3.0
        for i in range(a.shape[0] - 3):
            sigma = 0.0
            for j in range(3):
                sigma = sigma + a[i + j]
            sigma = sigma / three
            b[i] = sigma

    n = tvm.var('n')
    a = tvm.placeholder((n, ), 'float32', name='a')
    b = tvm.placeholder((n-3, ), 'float32', name='b')
    ir = fanout(n, a, b)

    #Check for i in (0, n-3)
    assert isinstance(ir, tvm.stmt.For)
    assert ir.loop_var.name == 'i'
    assert ir.min.value == 0
    assert tvm.ir_pass.Equal(ir.extent, n - 3)
    #Check loopbody
    ibody = ir.body
    assert isinstance(ibody, tvm.stmt.AttrStmt)
    abody = ibody.body
    assert isinstance(abody, tvm.stmt.Realize)
    assert abody.bounds[0].min.value == 0
    assert abody.bounds[0].extent.value == 1
    assert abody.func.name == 'sigma'
    #Check i loop body
    rbody = abody.body
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

    func = tvm.build(tvm.lower(ir, [n, a, b]))
    assert func

    np_a = numpy.random.randn(10).astype('float32')
    np_b = numpy.zeros(7).astype('float32')

    nd_a = tvm.ndarray.array(np_a)
    nd_b = tvm.ndarray.array(np_b)

    fanout(10, np_a, np_b)
    func(10, nd_a, nd_b)

    numpy.testing.assert_allclose(nd_b.asnumpy(), np_b, rtol=1e-5, atol=1e-5)


@script
def failure():
    for i in range(1, 100):
        i = 0

def test_failure():
    try:
        tvm.hybrid.parse(failure, [])
    except IOError as err:
        assert sys.version_info[0] == 2
        print('[Warning] Python2 cannot do the failure case because "%s"' % str(err))
    except Exception as err:
        assert str(err) == 'You CAN NEVER overwrite a loop variable!'


def test_looptype():
    @script
    def looptype(a, b, c):
        for i in parallel(8):
            a[i] = i
        for j in vectorize(8):
            b[j] = j
        for k in unroll(8):
            c[k] = k

    a = tvm.placeholder((8, ), name='a', dtype='int32')
    b = tvm.placeholder((8, ), name='b', dtype='int32')
    c = tvm.placeholder((8, ), name='c', dtype='int32')
    ir = looptype(a, b, c)
    iloop = ir.first
    jloop = ir.rest.first
    kloop = ir.rest.rest
    assert iloop.for_type == tvm.stmt.For.Parallel
    assert jloop.for_type == tvm.stmt.For.Vectorized
    assert kloop.for_type == tvm.stmt.For.Unrolled

    func = tvm.build(tvm.lower(ir, [a, b, c]))

    np_a = numpy.zeros((8, )).astype('int32')
    np_b = numpy.zeros((8, )).astype('int32')
    np_c = numpy.zeros((8, )).astype('int32')

    nd_a = tvm.ndarray.array(np_a)
    nd_b = tvm.ndarray.array(np_b)
    nd_c = tvm.ndarray.array(np_c)

    looptype(np_a, np_b, np_c)
    func(nd_a, nd_b, nd_c)

    numpy.testing.assert_allclose(np_a, nd_a.asnumpy())
    numpy.testing.assert_allclose(np_b, nd_b.asnumpy())
    numpy.testing.assert_allclose(np_c, nd_c.asnumpy())


def test_if():
    @script
    def if_then_else(a, b):
        for i in range(10):
            if i % 2 == 0:
                a[i] = -1
            else:
                a[i] = 1
        for i in unroll(10):
            b[i] = -1 if i % 2 == 0 else 1

    a = tvm.placeholder((10, ), dtype='int32', name='a')
    b = tvm.placeholder((10, ), dtype='int32', name='b')
    ir = if_then_else(a, b)
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

def test_bind():
    if not tvm.gpu(0).exist:
        print('No GPU found! Skip this test!')
        return
    @script
    def vec_add(a, b, c):
        for tx in bind('threadIdx.x', 1000):
            c[tx] = b[tx] + c[tx]

    a = tvm.placeholder((1000, ), dtype='float32', name='a')
    b = tvm.placeholder((1000, ), dtype='float32', name='b')
    c = tvm.placeholder((1000, ), dtype='float32', name='c')
    ir = vec_add(a, b, c)

    func = tvm.lower(ir, [a, b, c])
    func = tvm.build(func, target = 'cuda')

    _a = numpy.random.rand(1000).astype('float32')
    _b = numpy.random.rand(1000).astype('float32')
    _c = numpy.zeros((1000, ), dtype = 'float32')


    tvm_a = tvm.ndarray.array(_a, tvm.gpu(0))
    tvm_b = tvm.ndarray.array(_b, tvm.gpu(0))
    tvm_c = tvm.ndarray.array(_c, tvm.gpu(0))

    func(tvm_a, tvm_b, tvm_c)
    vec_add(_a, _b, _c)

    numpy.testing.assert_allclose(_c, tvm_c.asnumpy(), rtol=1e-5)

def test_math_intrin():
    @script
    def intrin_real(a):
        a[0] = sqrt(a[0])
        a[1] = log(a[1])
        a[2] = exp(a[2])
        a[3] = sigmoid(a[3])
        a[4] = power(a[4], a[5])
        a[5] = tanh(a[5])
        a[6] = min(a[4], a[5])
        a[7] = max(a[5], a[6])

    a6 = tvm.placeholder((8, ), dtype='float32', name='a')
    ir = intrin_real(a6)
    func = tvm.build(tvm.lower(ir, [a6]))
    assert func
    a = numpy.arange(2, 10).astype('float32')
    tvm_a = tvm.ndarray.array(a)
    func(tvm_a)
    intrin_real(a)
    numpy.testing.assert_allclose(a, tvm_a.asnumpy(), rtol=1e-5)

    @script
    def intrin_int(a):
        a[0] = popcount(a[0])

    a1 = tvm.placeholder((1, ), dtype='int32')
    ir = intrin_int(a1)
    func = tvm.build(tvm.lower(ir, [a1]))
    assert func
    a = numpy.array([1234567890]).astype('int32')
    tvm_a = tvm.ndarray.array(a)
    intrin_int(a)
    func(tvm_a)
    assert tvm_a.asnumpy()[0] == a[0]

def test_non_zero():
    @tvm.hybrid.script
    def blur(a, b):
        for i in range(2, 32):
            for j in range(2, 32):
                s = 0.0
                for di in range(3):
                    for dj in range(3):
                        s = s + a[i-di, j-dj]
                b[i-2, j-2] = s / 9.0
    try:
        np_a = numpy.random.randn(32, 32).astype('float32')
        np_b = numpy.zeros((30, 30), dtype='float32')
        blur(np_a, np_b)

        ph_a = tvm.placeholder((32, 32), 'float32', 'a')
        ph_b = tvm.placeholder((30, 30), 'float32', 'b')
        ir = tvm.hybrid.parse(blur, [ph_a, ph_b])
        func = tvm.lower(ir, [ph_a, ph_b])
        func = tvm.build(func)

        nd_a = tvm.ndarray.array(np_a)
        nd_b = tvm.ndarray.array(np_b)
        func(nd_a, nd_b)

        numpy.testing.assert_allclose(np_b, nd_b.asnumpy(), atol=1e-5, rtol=1e-5)
    except IOError:
        print('[Warning] Non-zero first test skipped by Python2')

    @tvm.hybrid.script
    def triangle(a, b, c):
        for i in range(10):
            for j in range(i, 10):
                c[i, j] = a[i] * b[j]

    a = tvm.placeholder((10, ), dtype='float32', name='a')
    b = tvm.placeholder((10, ), dtype='float32', name='b')
    c = tvm.placeholder((10, 10), dtype='float32', name='c')

    np_a = numpy.random.randn(10).astype('float32')
    np_b = numpy.random.randn(10).astype('float32')
    np_c = numpy.zeros((10, 10)).astype('float32')

    nd_a = tvm.ndarray.array(np_a)
    nd_b = tvm.ndarray.array(np_b)
    nd_c = tvm.ndarray.array(np_c)

    triangle(np_a, np_b, np_c)

    func = tvm.build(tvm.lower(triangle(a, b, c), [a, b, c]))
    assert func
    func(nd_a, nd_b, nd_c)
    numpy.testing.assert_allclose(nd_c.asnumpy(), np_c)

def test_allocate():
    @tvm.hybrid.script
    def blur2d(a, b):
        for i in range(30):
            ha = allocate((3, 30), 'float32')
            for j in range(3):
                for k in range(30):
                    ha[j, k] = a[i+j, k] + a[i+j, k+1] + a[i+j, k+2]
            for j in range(30):
                b[i, j] = (ha[0, j] + ha[1, j] + ha[2, j]) / 9.0

    a = tvm.placeholder((32, 32), 'float32', 'a')
    b = tvm.placeholder((30, 30), 'float32', 'b')

    func = tvm.build(tvm.lower(blur2d(a, b), [a, b]))
    assert func

    np_a = numpy.random.randn(32, 32).astype('float32')
    np_b = numpy.zeros((30, 30)).astype('float32')

    nd_a = tvm.ndarray.array(np_a)
    nd_b = tvm.ndarray.array(np_b)

    func(nd_a, nd_b)
    blur2d(np_a, np_b)

    numpy.testing.assert_allclose(nd_b.asnumpy(), np_b, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_outer_product()
    test_fanout()
    test_failure()
    test_looptype()
    test_if()
    test_bind()
    test_math_intrin()
    test_non_zero()
    test_allocate()

