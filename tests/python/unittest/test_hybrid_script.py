import tvm, inspect, sys, traceback, numpy, nose, types
from tvm.hybrid import script
from tvm.hybrid.intrin import HYBRID_GLOBALS

@nose.tools.nottest
def run_and_check(func, args, var_dict={}, target='llvm'):
    def tvm_val_2_py_val(val):
        val = tvm.ir_pass.Substitute(val, var_dict)
        val = tvm.ir_pass.Simplify(val)
        assert isinstance(val, (tvm.expr.IntImm, tvm.expr.UIntImm))
        return val.value

    ctx = tvm.context(target, 0)
    op = None

    outs = func(*args)
    op = outs[0].op if isinstance(outs, list) else outs.op

    emu_args = []
    nd_args = []
    for i in args:
        if isinstance(i, tvm.tensor.Tensor):
            shape = [tvm_val_2_py_val(j) for j in i.shape]
            emu_args.append(numpy.random.randn(*shape).astype(i.dtype))
            nd_args.append(tvm.nd.array(emu_args[-1], ctx))
        else:
            assert isinstance(i, tvm.expr.Var)
            emu_args.append(tvm_val_2_py_val(i))
            nd_args.append(emu_args[-1])

    sch = tvm.create_schedule(op)
    module = tvm.build(sch, args + (outs if isinstance(outs, list) else [outs]), target=target)
    assert module
    
    out_tensors = []
    for i in range(op.num_outputs):
        output = op.output(i)
        shape = [tvm_val_2_py_val(j) for j in output.shape]
        nd_args.append(tvm.nd.array(numpy.zeros(shape).astype(output.dtype), ctx))
        out_tensors.append(nd_args[-1])

    ref_data = func(*emu_args)
    if isinstance(ref_data, numpy.ndarray):
        ref_data = [ref_data]
    
    module(*nd_args)

    for nd, np in zip(out_tensors, ref_data):
        tvm.testing.assert_allclose(nd.asnumpy(), np, rtol=1e-5, atol=1e-5)


@script
def outer_product(n, m, a, b):
    """This is a simple outer product.
    Actually this function is not required to be documented.
    I write this docstring to test skipping docstring functionality.
    """
    c = output_tensor((n, m), a.dtype)
    for i in range(n):
        for j in range(m):
            c[i, j] = a[i] * b[j]
    return c

#Test global function
#Test bridge between frontend and backend
def test_outer_product():
    n = tvm.var('n')
    m = tvm.var('m')
    a = tvm.placeholder((n, ), name='a')
    b = tvm.placeholder((m, ), name='b')

    try:
        c = outer_product(n, m, a, b)
        ir = c.op.body
    except IOError as err:
        assert sys.version_info[0] == 2 and str(err) == 'could not get source code'
        return

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


    run_and_check(outer_product, [n, m, a, b], {n: 99, m: 101})

    for key, _ in HYBRID_GLOBALS.items():
        assert key not in globals().keys()
        assert key not in outer_product.__globals__.keys()

#Test local function
#Test allocation of local variable
def test_fanout():
    @script
    def fanout(n, a):
        three = 3.0
        b = output_tensor((a.shape[0] - 3, ), a.dtype)
        for i in range(a.shape[0] - 3):
            sigma = 0.0
            for j in range(3):
                sigma += a[i + j]
            sigma = sigma / three
            b[i] = sigma
        return b

    n = tvm.var('n')
    a = tvm.placeholder((n, ), 'float32', name='a')
    try:
        b = fanout(n, a)
        ir = b.op.body
    except IOError as err:
        assert sys.version_info[0] == 2 and str(err) == 'could not get source code'
        return

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

    run_and_check(fanout, [n, a], {n: 10})


def test_looptype():
    @script
    def looptype(a, b, c):
        d = output_tensor((8, ), 'int32')
        e = output_tensor((8, ), 'int32')
        f = output_tensor((8, ), 'int32')
        for i in parallel(8):
            d[i] = a[i]
        for j in vectorize(8):
            e[j] = b[j]
        for k in unroll(8):
            f[k] = c[k]
        return d, e, f

    a = tvm.placeholder((8, ), name='a', dtype='int32')
    b = tvm.placeholder((8, ), name='b', dtype='int32')
    c = tvm.placeholder((8, ), name='c', dtype='int32')
    try:
        d, e, f = looptype(a, b, c)
        ir = d.op.body
    except:
        return
    iloop = ir.first
    jloop = ir.rest.first
    kloop = ir.rest.rest
    assert iloop.for_type == tvm.stmt.For.Parallel
    assert jloop.for_type == tvm.stmt.For.Vectorized
    assert kloop.for_type == tvm.stmt.For.Unrolled

    run_and_check(looptype, [a, b, c])


def test_if():
    @script
    def if_then_else(a):
        b = output_tensor((10, ), 'int32')
        c = output_tensor((10, ), 'int32')
        for i in range(10):
            if i % 2 == 0:
                c[i] = a[i]
            else:
                c[i] = b[i]
        for i in unroll(10):
            b[i] = -1 if i % 2 == 0 else 1
        return b, c

    a = tvm.placeholder((10, ), dtype='int32', name='a')

    run_and_check(if_then_else, [a])

    @script
    def if_triple_condition(a):
        b = output_tensor((10, ), 'int32')
        for i in range(10):
            if 0 <= i < 5:
                b[i] = a[i]
            else:
                b[i] = a[i] + 1
        return b

    run_and_check(if_triple_condition, [a])

    @script
    def if_and(a):
        b = output_tensor((10, ), 'int32')
        for i in range(10):
            if i >= 0 and i < 5:
                b[i] = a[i]
            else:
                b[i] = a[i] + 1
        return b

    run_and_check(if_and, [a])


def test_bind():
    if not tvm.gpu(0).exist:
        print('[Warning] No GPU found! Skip bind test!')
        return
    @script
    def vec_add(a, b):
        c = output_tensor((1000, ), dtype='float32')
        for tx in bind('threadIdx.x', 1000):
            c[tx] = a[tx] + b[tx]
        return c

    a = tvm.placeholder((1000, ), dtype='float32', name='a')
    b = tvm.placeholder((1000, ), dtype='float32', name='b')

    run_and_check(vec_add, [a, b], target='cuda')

def test_math_intrin():
    @script
    def intrin_real(a):
        b = output_tensor((8, ), 'float32')
        b[0] = sqrt(a[0])
        b[1] = log(a[1])
        b[2] = exp(a[2])
        b[3] = sigmoid(a[3])
        b[4] = power(a[4], a[5])
        b[5] = tanh(a[5])
        b[6] = min(a[4], a[5])
        b[7] = max(a[5], a[6])
        return b

    a8 = tvm.placeholder((8, ), dtype='float32', name='a')
    b8 = intrin_real(a8)
    sch = tvm.create_schedule(b8.op)
    func = tvm.build(sch, [a8, b8])
    assert func
    a = numpy.arange(2, 10).astype('float32')
    tvm_a = tvm.ndarray.array(a)
    tvm_b = tvm.ndarray.array(numpy.zeros((8, ), dtype='float32'))
    b = intrin_real(a)
    func(tvm_a, tvm_b)
    tvm.testing.assert_allclose(b, tvm_b.asnumpy(), rtol=1e-5)

    @script
    def intrin_int(a):
        b = output_tensor((1, ), 'int32')
        b[0] = popcount(a[0])
        return b

    a1 = tvm.placeholder((1, ), dtype='int32')
    b1 = intrin_int(a1)
    sch = tvm.create_schedule(b1.op)
    func = tvm.build(sch, [a1, b1])
    assert func
    a = numpy.array([114514]).astype('int32')
    tvm_a = tvm.ndarray.array(a)
    tvm_b = tvm.ndarray.array(numpy.array([0]).astype('int32'))
    b = intrin_int(a)
    func(tvm_a, tvm_b)
    assert tvm_b.asnumpy()[0] == b[0]

# test non caconical loops
def test_non_zero():
    @tvm.hybrid.script
    def blur(a):
        b = output_tensor((30, 30), 'float32')
        for i in range(2, 32):
            for j in range(2, 32):
                s = 0.0
                for di in range(3):
                    for dj in range(3):
                        s += a[i-di, j-dj]
                b[i-2, j-2] = s / 9.0
        return b

    a = tvm.placeholder((32, 32), 'float32', 'a')
    run_and_check(blur, [a])

    @tvm.hybrid.script
    def triangle(a, b):
        c = output_tensor((10, 10), dtype='float32')
        for i in range(10):
            for j in range(i, 10):
                c[i, j] = a[i] * b[j]
        return c

    a = tvm.placeholder((10, ), dtype='float32', name='a')
    b = tvm.placeholder((10, ), dtype='float32', name='b')

    run_and_check(triangle, [a, b])

def test_allocate():
    @tvm.hybrid.script
    def blur2d(a):
        b = output_tensor((30, 30), 'float32')
        for i in range(30):
            ha = allocate((3, 30), 'float32')
            for j in range(3):
                for k in range(30):
                    ha[j, k] = a[i+j, k] + a[i+j, k+1] + a[i+j, k+2]
            for j in range(30):
                b[i, j] = (ha[0, j] + ha[1, j] + ha[2, j]) / 9.0
        return b

    a = tvm.placeholder((32, 32), 'float32', 'a')
    run_and_check(blur2d, [a])

    if tvm.gpu().exist:
        @tvm.hybrid.script
        def share_vec_add(a, b):
            c = output_tensor((256, ), 'float32')
            shared = allocate((256, ), 'float32', 'shared')
            for i in bind("threadIdx.x", 256):
                shared[i] = a[i]
            local = allocate((256, ), 'float32', 'local')
            for i in bind("threadIdx.x", 256):
                local[i] = b[i]
            for i in bind("threadIdx.x", 256):
                c[i] = shared[i] + local[i]
            return c

        a = tvm.placeholder((256, ), dtype='float32', name='a')
        b = tvm.placeholder((256, ), dtype='float32', name='b')
        run_and_check(share_vec_add, [a, b], target='cuda')
    else:
        print('[Warning] No GPU found! Skip shared mem test!')

def test_upstream():
    @tvm.hybrid.script
    def upstream(a):
        b = output_tensor((20, ), 'float32')
        for i in range(20):
            b[i] = a[i] * i
        return b

    a = tvm.placeholder((20, ), 'float32')
    b = tvm.placeholder((20, ), 'float32')
    c = tvm.compute((20, ), lambda x: a[x] + b[x])
    d = upstream(c)
    sch = tvm.create_schedule([c.op, d.op])
    ir = tvm.lower(sch, [a, b, d], simple_mode=True)
    func = tvm.build(sch, [a, b, d])
    assert(func)

    a = numpy.random.randn(20).astype('float32')
    b = numpy.random.randn(20).astype('float32')
    ref = numpy.zeros((20, ), 'float32')
    for i in range(20):
        ref[i] = (a[i] + b[i]) * i

    tvm_a = tvm.nd.array(a)
    tvm_b = tvm.nd.array(b)
    tvm_d = tvm.nd.array(numpy.zeros((20, )).astype('float32'))

    func(tvm_a, tvm_b, tvm_d)
    tvm.testing.assert_allclose(tvm_d.asnumpy(), ref, 1e-5, 1e-5)

def test_downstream():
    @tvm.hybrid.script
    def downstream(a):
        b = output_tensor((20, ), 'float32')
        for i in range(20):
            b[i] = a[i] * i
        return b

    
    a = tvm.placeholder((20, ), 'float32')
    b = downstream(a)
    c = tvm.compute((20, ), lambda x: b[x] + 1.0)

    sch = tvm.create_schedule(c.op)
    module = tvm.build(sch, [a, c])
    assert module

    a = numpy.random.randn(20).astype('float32')
    ref = numpy.zeros((20, )).astype('float32')
    for i in range(20):
        ref[i] = (a[i] * i) + 1.0

    tvm_a = tvm.nd.array(a)
    tvm_c = tvm.nd.array(numpy.zeros((20, )).astype('float32'))
    module(tvm_a, tvm_c)
    tvm.testing.assert_allclose(tvm_c.asnumpy(), ref, 1e-5, 1e-5)

def test_const_param():
    @tvm.hybrid.script
    def add_something(a, b):
        c = output_tensor((11, ), 'int32')
        for i in range(11):
            c[i] = a[i] + b
        return c

    a = tvm.placeholder((11, ), dtype='int32', name='a')
    b = tvm.const(11, 'int32')
    c = add_something(a, b)
    sch = tvm.create_schedule(c.op)
    module = tvm.build(sch, [a, c], 'llvm')
    assert(module)

    np_a = numpy.arange(11).astype('int32')
    np_b = 11
    np_c = numpy.zeros((11, )).astype('int32')

    nd_a = tvm.ndarray.array(np_a)
    nd_c = tvm.ndarray.array(numpy.zeros((11, )).astype('int32'))
    module(nd_a, nd_c)
    ref = add_something(np_a, 11)

    tvm.testing.assert_allclose(nd_c.asnumpy(), ref, 1e-5, 1e-5)

def test_value_index():
    @tvm.hybrid.script
    def kernel_a(a):
        b = output_tensor((16, ), 'int32')
        c = output_tensor((4, 4), 'int32')
        for i in range(16):
            b[i] = a[i] + 2
            c[i // 4, i % 4] = a[i] + 1
        return b, c

    @tvm.hybrid.script
    def kernel_b(b, a):
        c = output_tensor((4, 4), 'int32')
        for i in range(4):
            for j in range(4):
                c[i, j] = a[i * 4 + j] * b[i, j]
        return c

    a = tvm.placeholder((16, ), 'int32')
    b, c = kernel_a(a)
    d = kernel_b(c, b)
    sch = tvm.create_schedule(d.op)
    module = tvm.build(sch, [a, d])
    assert module

    np_a = numpy.arange(16).astype('int32')
    np_b, np_c = kernel_a(np_a)
    ref = kernel_b(np_c, np_b)

    res = tvm.ndarray.array(numpy.zeros((4, 4)).astype('int32'))
    module(tvm.ndarray.array(np_a), res)
    tvm.testing.assert_allclose(res.asnumpy(), ref)



if __name__ == "__main__":
    test_outer_product()
    test_fanout()
    test_looptype()
    test_if()
    test_bind()
    test_math_intrin()
    test_non_zero()
    test_allocate()
    test_upstream()
    test_downstream()
    test_const_param()
    test_value_index()
    # TODO:
    # test_inplace()


