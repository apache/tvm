# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm, inspect, sys, traceback, numpy, pytest, types, os

from tvm import te
from tvm.contrib import util
from tvm.te.hybrid import script
from tvm.te.hybrid.runtime import HYBRID_GLOBALS

@pytest.mark.skip
def run_and_check(func, args, var_dict={}, target='llvm', sch=None, outs=None):
    def tvm_val_2_py_val(val):
        val = tvm.tir.ir_pass.Substitute(val, var_dict)
        val = tvm.tir.ir_pass.Simplify(val)
        assert isinstance(val, (tvm.tir.IntImm,))
        return val.value

    ctx = tvm.context(target, 0)
    op = None

    if sch is None:
        outs = func(*tuple(tvm.runtime.convert(i) if isinstance(i, list) else i for i in args))
        op = outs[0].op if isinstance(outs, list) else outs.op
        sch = te.create_schedule(op)
    else:
        assert outs is not None
        assert isinstance(outs, list)
        op = outs[0].op

    emu_args = []
    nd_args = []
    for i in args:
        if isinstance(i, te.tensor.Tensor):
            shape = [tvm_val_2_py_val(j) for j in i.shape]
            emu_args.append(numpy.random.randn(*shape).astype(i.dtype))
            nd_args.append(tvm.nd.array(emu_args[-1], ctx))
        elif isinstance(i, tvm.tir.Var):
            emu_args.append(tvm_val_2_py_val(i))
            nd_args.append(emu_args[-1])
        else:
            assert isinstance(i, list)
            emu_args.append(numpy.array(i))

    compile_args = [i for i in args if isinstance(i, (te.tensor.Tensor, tvm.tir.Var))] + \
                   (outs if isinstance(outs, list) else [outs])
    module = tvm.build(sch,
                       compile_args,
                       target=target)
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

    module_args = [i for i in args if isinstance(i, (te.tensor.Tensor, tvm.tir.Var))]
    module_outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    h_module = te.hybrid.build(sch, module_args, module_outs)

    return h_module, module_args, module_outs

@script
def outer_product(n, m, a, b):
    """This is a simple outer product.
    Actually this function is not required to be documented.
    I write this docstring to test skipping docstring functionality.
    """
    c = output_tensor((n, m), a.dtype)
    for i in range(n):
        for j in range(m):
            assert i < n and j < m, "index out of range!"
            c[i, j] = a[i] * b[j]
    return c

#Test global function
#Test bridge between frontend and backend
def test_outer_product():
    n = te.size_var('n')
    m = te.size_var('m')
    a = te.placeholder((n, ), name='a')
    b = te.placeholder((m, ), name='b')

    try:
        c = outer_product(n, m, a, b)
        ir = c.op.body
    except IOError as err:
        assert sys.version_info[0] == 2 and str(err) == 'could not get source code'
        return

    #Check for i in (0, n)
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'i'
    assert ir.min.value == 0
    assert ir.extent.name == 'n'
    ibody = ir.body
    assert isinstance(ibody, tvm.tir.For)
    #Check for j in (0, m)
    assert ibody.loop_var.name == 'j'
    assert ibody.min.value == 0
    assert ibody.extent.name == 'm'
    #Check loop body
    jblock = ibody.body
    assert isinstance(jblock, tvm.tir.SeqStmt)
    jbody = jblock[0]
    assert isinstance(jbody, tvm.tir.AssertStmt)
    assert isinstance(jbody.message, tvm.tir.StringImm)
    assert jbody.message.value == "index out of range!"
    jbody = jblock[1]
    assert isinstance(jbody, tvm.tir.Provide)
    assert jbody.func.name == 'c'
    assert len(jbody.args) == 2
    assert jbody.args[0].name == 'i'
    assert jbody.args[1].name == 'j'
    assert isinstance(jbody.value, tvm.tir.Mul)
    mul = jbody.value
    assert isinstance(mul.a, tvm.tir.Call)
    assert mul.a.name == 'a'
    assert mul.b.name == 'b'

    func, ins, outs = run_and_check(outer_product, [n, m, a, b], {n: 99, m: 101})
    temp = util.tempdir()
    path = temp.relpath('%s.py' % func.name)
    func.save(path)
    func_ = te.hybrid.HybridModule()
    func_.load(path)
    run_and_check(func_, ins, {n: 99, m: 101}, outs=outs)

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

    n = te.size_var('n')
    a = te.placeholder((n, ), 'float32', name='a')
    try:
        b = fanout(n, a)
        ir = b.op.body
    except IOError as err:
        assert sys.version_info[0] == 2 and str(err) == 'could not get source code'
        return

    #Check for i in (0, n-3)
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'i'
    assert ir.min.value == 0
    assert tvm.ir.structural_equal(ir.extent, n - 3)
    #Check loopbody
    ibody = ir.body
    assert isinstance(ibody, tvm.tir.AttrStmt)
    abody = ibody.body
    assert isinstance(abody, tvm.tir.Realize)
    assert abody.bounds[0].min.value == 0
    assert abody.bounds[0].extent.value == 1
    assert abody.func.name == 'sigma'
    #Check i loop body
    rbody = abody.body
    assert isinstance(rbody[0], tvm.tir.Provide)
    assert rbody[0].func.name == 'sigma'
    assert len(rbody[0].args) == 1
    assert rbody[0].args[0].value == 0
    #Check fanout loop
    jloop = rbody[1]
    assert jloop.loop_var.name == 'j'
    assert jloop.min.value == 0
    assert jloop.extent.value == 3
    jbody = jloop.body
    assert isinstance(jbody, tvm.tir.Provide)
    assert len(jbody.args) == 1
    assert jbody.args[0].value == 0
    assert jbody.func.name == 'sigma'
    assert isinstance(jbody.value, tvm.tir.Add)
    value = jbody.value
    assert isinstance(value.a, tvm.tir.Call)
    assert value.a.name == 'sigma'
    assert len(value.a.args) == 1
    assert value.a.args[0].value == 0
    assert value.b.name == 'a'
    assert len(value.b.args) == 1
    assert tvm.ir.structural_equal(value.b.args[0], ir.loop_var + jloop.loop_var)
    divide= rbody[2]
    assert isinstance(divide, tvm.tir.Provide)
    assert len(divide.args) == 1
    assert divide.args[0].value == 0
    value = divide.value
    assert isinstance(value, tvm.tir.Mul)
    assert value.a.name == 'sigma'
    assert len(value.a.args) == 1
    assert value.a.args[0].value == 0
    assert abs(value.b.value - (1 / 3.0)) < 1e-5
    write = rbody[3]
    assert isinstance(write, tvm.tir.Provide)
    assert write.func.name == 'b'
    assert write.value.name == 'sigma'
    assert len(write.value.args) == 1
    assert write.value.args[0].value == 0

    func, ins, outs = run_and_check(fanout, [n, a], {n: 10})
    run_and_check(func, ins, {n: 10}, outs=outs)


def test_looptype():
    @script
    def looptype(a, b, c):
        d = output_tensor((16, ), 'int32')
        e = output_tensor((16, ), 'int32')
        f = output_tensor((16, ), 'int32')
        for i in parallel(16):
            d[i] = a[i]
        for j in vectorize(16):
            e[j] = b[j]
        for k in unroll(16):
            f[k] = c[k]
        return d, e, f

    a = te.placeholder((16, ), name='a', dtype='int32')
    b = te.placeholder((16, ), name='b', dtype='int32')
    c = te.placeholder((16, ), name='c', dtype='int32')
    try:
        d, e, f = looptype(a, b, c)
        ir = d.op.body
    except:
        return
    iloop = ir[0]
    jloop = ir[1]
    kloop = ir[2]
    assert iloop.for_type == tvm.tir.For.Parallel
    assert jloop.for_type == tvm.tir.For.Vectorized
    assert kloop.for_type == tvm.tir.For.Unrolled

    func, ins, outs = run_and_check(looptype, [a, b, c])
    run_and_check(func, ins, outs=outs)


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

    a = te.placeholder((10, ), dtype='int32', name='a')

    func, ins, outs = run_and_check(if_then_else, [a])
    run_and_check(func, ins, outs=outs)

    @script
    def if_triple_condition(a):
        b = output_tensor((10, ), 'int32')
        for i in range(10):
            if 0 <= i < 5:
                b[i] = a[i]
            else:
                b[i] = a[i] + 1
        return b

    func, ins, outs = run_and_check(if_triple_condition, [a])
    run_and_check(func, ins, outs=outs)

    @script
    def if_and(a):
        b = output_tensor((10, ), 'int32')
        for i in range(10):
            if i >= 0 and i < 5:
                b[i] = a[i]
            else:
                b[i] = a[i] + 1
        return b

    func, ins, outs = run_and_check(if_and, [a])
    run_and_check(func, ins, outs=outs)


def test_bind():
    if not tvm.gpu(0).exist:
        print('[Warning] No GPU found! Skip bind test!')
        return

    @script
    def vec_add(a, b):
        c = output_tensor((1000, ), 'float32')
        for tx in bind('threadIdx.x', 1000):
            c[tx] = a[tx] + b[tx]
        return c

    a = te.placeholder((1000, ), dtype='float32', name='a')
    b = te.placeholder((1000, ), dtype='float32', name='b')
    func, ins, outs = run_and_check(vec_add, [a, b], target='cuda')
    run_and_check(func, ins, outs=outs, target='cuda')

    @script
    def raw(a, b):
        c = output_tensor((1000, ), 'float32')
        for i in range(1000):
            c[i] = a[i] + b[i]
        return c

    c = raw(a, b)
    sch = te.create_schedule(c.op)
    x = te.thread_axis('threadIdx.x')
    sch[c].bind(c.op.axis[0], x)
    func, ins, outs = run_and_check(raw, [a, b], sch=sch, outs=[c], target='cuda')
    run_and_check(func, ins, outs=outs, target='cuda')


    @te.hybrid.script
    def foo(a):
        c = output_tensor((a.shape[0],), a.dtype)
        total = allocate((1,), a.dtype, 'local')
        len_i = a.shape[0]
        len_j = a.shape[1]
        for i in bind('threadIdx.x', len_i):
            total[0] = 0.
            for k in const_range(len_j):
                total[0] += a[i, k]
            c[i] = total[0]

        return c

    a = te.placeholder((8, 4), 'float32')
    c = foo(a)
    s = te.create_schedule(c.op)
    ir = tvm.lower(s, [a, c], simple_mode=True)

    func, ins, outs = run_and_check(foo, [a], target='cuda')
    run_and_check(func, ins, outs=outs, target='cuda')

    @te.hybrid.script
    def max_threads(a):
        b = output_tensor(a.shape, a.dtype)
        n = a.shape[0]
        m = max_num_threads(True)
        for i in bind('threadIdx.x', m):
            for j in bind('blockIdx.x', ceil_div(n, m)):
                if i * m + j < n:
                    b[i * m + j] = a[i * m + j] + a[i * m + j]
        return b

    a = te.placeholder((10000, ), 'float32')
    with tvm.target.create('cuda'):
        func, ins, outs = run_and_check(max_threads, [a], target='cuda')
        run_and_check(func, ins, outs=outs, target='cuda')


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

    a8 = te.placeholder((8, ), dtype='float32', name='a')
    b8 = intrin_real(a8)
    sch = te.create_schedule(b8.op)
    func = tvm.build(sch, [a8, b8])
    assert func
    a = numpy.arange(2, 10).astype('float32')
    tvm_a = tvm.nd.array(a)
    tvm_b = tvm.nd.array(numpy.zeros((8, ), dtype='float32'))
    b = intrin_real(a)
    func(tvm_a, tvm_b)
    tvm.testing.assert_allclose(b, tvm_b.asnumpy(), rtol=1e-5)

    @script
    def intrin_int(a):
        b = output_tensor((1, ), 'int32')
        b[0] = popcount(a[0])
        return b

    a1 = te.placeholder((1, ), dtype='int32')
    b1 = intrin_int(a1)
    sch = te.create_schedule(b1.op)
    func = tvm.build(sch, [a1, b1])
    assert func
    a = numpy.array([114514]).astype('int32')
    tvm_a = tvm.nd.array(a)
    tvm_b = tvm.nd.array(numpy.array([0]).astype('int32'))
    b = intrin_int(a)
    func(tvm_a, tvm_b)
    assert tvm_b.asnumpy()[0] == b[0]

# test non caconical loops
def test_non_zero():
    @te.hybrid.script
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

    a = te.placeholder((32, 32), 'float32', 'a')
    func, ins, outs = run_and_check(blur, [a])
    run_and_check(func, ins, outs=outs)

    @te.hybrid.script
    def triangle(a, b):
        c = output_tensor((10, 10), dtype='float32')
        for i in range(10):
            for j in range(i, 10):
                c[i, j] = a[i] * b[j]
        return c

    a = te.placeholder((10, ), dtype='float32', name='a')
    b = te.placeholder((10, ), dtype='float32', name='b')

    func, ins, outs = run_and_check(triangle, [a, b])
    run_and_check(func, ins, outs=outs)

def test_allocate():
    @te.hybrid.script
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

    a = te.placeholder((32, 32), 'float32', 'a')
    b = blur2d(a)
    sch = te.create_schedule(b.op)
    func, ins, outs = run_and_check(blur2d, [a])
    run_and_check(func, ins, outs=outs)

    if tvm.gpu().exist:
        @te.hybrid.script
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

        a = te.placeholder((256, ), dtype='float32', name='a')
        b = te.placeholder((256, ), dtype='float32', name='b')
        c = share_vec_add(a, b)
        func, ins, outs = run_and_check(share_vec_add, [a, b], target='cuda')
        run_and_check(func, ins, outs=outs, target='cuda')
    else:
        print('[Warning] No GPU found! Skip shared mem test!')

def test_upstream():
    @te.hybrid.script
    def upstream(a):
        b = output_tensor((20, ), 'float32')
        for i in range(20):
            b[i] = a[i] * i
        return b

    a = te.placeholder((20, ), 'float32')
    b = te.placeholder((20, ), 'float32')
    c = te.compute((20, ), lambda x: a[x] + b[x])
    d = upstream(c)
    sch = te.create_schedule([c.op, d.op])
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
    @te.hybrid.script
    def downstream(a):
        b = output_tensor((20, ), 'float32')
        for i in range(20):
            b[i] = a[i] * i
        return b


    a = te.placeholder((20, ), 'float32')
    b = downstream(a)
    c = te.compute((20, ), lambda x: b[x] + 1.0)

    sch = te.create_schedule(c.op)
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
    @te.hybrid.script
    def add_something(a, b):
        c = output_tensor((11, ), 'int32')
        for i in range(11):
            c[i] = a[i] + b
        return c

    a = te.placeholder((11, ), dtype='int32', name='a')
    b = tvm.tir.const(11, 'int32')
    c = add_something(a, b)
    sch = te.create_schedule(c.op)
    module = tvm.build(sch, [a, c], 'llvm')
    assert(module)

    np_a = numpy.arange(11).astype('int32')
    np_b = 11
    np_c = numpy.zeros((11, )).astype('int32')

    nd_a = tvm.nd.array(np_a)
    nd_c = tvm.nd.array(numpy.zeros((11, )).astype('int32'))
    module(nd_a, nd_c)
    ref = add_something(np_a, 11)

    tvm.testing.assert_allclose(nd_c.asnumpy(), ref, 1e-5, 1e-5)

def test_value_index():
    @te.hybrid.script
    def kernel_a(a):
        b = output_tensor((16, ), 'int32')
        c = output_tensor((4, 4), 'int32')
        for i in range(16):
            b[i] = a[i] + 2
            c[i // 4, i % 4] = a[i] + 1
        return b, c

    @te.hybrid.script
    def kernel_b(b, a):
        c = output_tensor((4, 4), 'int32')
        for i in range(4):
            for j in range(4):
                c[i, j] = a[i * 4 + j] * b[i, j]
        return c

    a = te.placeholder((16, ), 'int32')
    b, c = kernel_a(a)
    d = kernel_b(c, b)
    sch = te.create_schedule(d.op)
    module = tvm.build(sch, [a, d])
    assert module

    np_a = numpy.arange(16).astype('int32')
    np_b, np_c = kernel_a(np_a)
    ref = kernel_b(np_c, np_b)

    res = tvm.nd.array(numpy.zeros((4, 4)).astype('int32'))
    module(tvm.nd.array(np_a), res)
    tvm.testing.assert_allclose(res.asnumpy(), ref)

def test_func_call():
    @te.hybrid.script
    def foo(a, b):
        for i in range(len(a)):
            a[i] = i + 1.0
        for i in range(len(a)):
            b[i] = i + 1.0
        c = outer_product(10, 10, a, b)
        d = output_tensor(c.shape, c.dtype)
        for i in range(10):
            for j in range(10):
                d[i, j] = c[i, j] + i * j
        return d

    a = te.placeholder((10, ), name='a')
    b = te.placeholder((10, ), name='b')
    func, ins, outs = run_and_check(foo, [a, b])
    run_and_check(func, ins, outs=outs)

def test_bool():
    @te.hybrid.script
    def foo(a):
        b = output_tensor(a.shape, a.dtype)
        b[0] = 1.2
        for i in range(1, a.shape[0] - 1):
            if a[i] * a[i - 1] < a[i] or a[i] * a[i - 1] < a[i - 1] or i * a[i] == a[i]:
                b[i] = a[i]
            else:
                b[i] = 0.0
        return b
    a = te.placeholder((10, ), name='a')
    func, ins, outs = run_and_check(foo, [a])
    run_and_check(func, ins, outs=outs)

def test_const_range():
    @te.hybrid.script
    def foo(a, b):
        c = output_tensor(a.shape, a.dtype)
        d = output_tensor(a.shape, 'int32')

        for i in const_range(2):
            for j in const_range(5):
                c[i, j] = float32(int32(a[i, j]) + b[i, j])

        for i in const_range(len(b)):
            for j in const_range(len(b[0])):
                d[i, j] = int32(a[i, j] + b[i, j])

        return c, d

    a = te.placeholder((2, 5), name='a', dtype='float32')
    b = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
    func, ins, outs = run_and_check(foo, [a, b])
    run_and_check(func, ins, outs=outs)

    @te.hybrid.script
    def goo(a, b):
        c = output_tensor(a.shape, a.dtype)
        len_b = len(b)
        for i in const_range(len_b * 2):
            if i < len_b:
                c[i] = a[i] + b[i]
            else:
                c[i - len_b] = a[i - len_b] + b[i - len_b]
        return c
    a = te.placeholder((5, ), name='a', dtype='int32')
    b = [1, 2, 3, 4, 5]
    c = goo(a, tvm.runtime.convert(b))
    sch = te.create_schedule(c.op)
    func, ins, outs = run_and_check(goo, [a, b])
    run_and_check(func, ins, outs=outs)

    @te.hybrid.script
    def hoo(a, b):
        c = output_tensor(a.shape, a.dtype)
        len_b = len(b)
        for i in range(a.shape[0]):
            for j in const_range(len(b)):
                d = a[i] * b[j]
                d += a[i] + b[j]
                c[i] = d
        return c
    a = te.placeholder((5, ), name='a', dtype='int32')
    b = [1, 2, 3, 4, 5]
    func, ins, outs = run_and_check(hoo, [a, b])
    run_and_check(func, ins, outs=outs)

def test_schedule():
    @script
    def outer_product(a, b):
        c = output_tensor((64, 64), a.dtype)
        for i in range(64):
            for j in range(64):
                c[i, j] = a[i] * b[j]
        return c
    a = te.placeholder((64,), name='a', dtype='float32')
    b = te.placeholder((64,), name='b', dtype='float32')
    c = outer_product(a, b)

    # Test perfect loop split
    # Test loop reorder
    # Test loop annotation
    sch = te.create_schedule(c.op)
    i, j = c.op.axis
    io, ii = sch[c].split(i, 4)
    sch[c].parallel(ii)
    jo, ji = sch[c].split(j, 4)
    joo, joi = sch[c].split(jo, 4)
    sch[c].vectorize(ji)
    sch[c].reorder(ii, io, joo, joi, ji)
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    assert isinstance(ir, tvm.tir.AttrStmt)
    ir = ir.body
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'i.inner'
    ir = ir.body
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'i.outer'
    ir = ir.body
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'j.outer.outer'
    ir = ir.body
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'j.outer.inner'
    ir = ir.body
    func, ins, outs = run_and_check(outer_product, [a, b], sch=sch, outs=[c])
    run_and_check(func, ins, outs=outs)

    # Test fuse
    sch = te.create_schedule(c.op)
    sch[c].fuse(c.op.axis[0], c.op.axis[1])
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    assert isinstance(ir, tvm.tir.AttrStmt)
    ir = ir.body
    assert isinstance(ir, tvm.tir.For)
    assert ir.loop_var.name == 'i.j.fused'
    func, ins, outs = run_and_check(outer_product, [a, b], sch=sch, outs=[c])
    run_and_check(func, ins, outs=outs)

    # Test imperfect loop split
    sch = te.create_schedule(c.op)
    sch[c].split(c.op.axis[0], 3)
    ir = tvm.lower(sch, [a, b, c], simple_mode=True)
    func, ins, outs = run_and_check(outer_product, [a, b], sch=sch, outs=[c])
    run_and_check(func, ins, outs=outs)

    # Test loop binds

def test_capture():
    n = 8

    constant_tuple = (10, n)
    constant_list = [[1, 2], [3, n]]
    const_value = 1

    @te.hybrid.script
    def add_something(a):
        c = output_tensor((constant_tuple[1],), 'int32')
        for i in range(constant_tuple[1]):
            c[i] = a[i] + constant_list[1][const_value]
        return c

    a = te.placeholder((n, ), dtype='int32', name='a')

    func, ins, outs = run_and_check(add_something, [a])
    run_and_check(func, ins, outs=outs)

def test_array_inputs():
    @script
    def sum_array(inputs):
        out = output_tensor((10,), inputs[0].dtype)
        n = len(inputs)
        for i in range(10):
            for j in const_range(n):
                out[i] += inputs[j][i]
        return out
    n = 5
    inputs = []
    for i in range(n):
        inputs.append(te.placeholder((10,), name='t%s' % i, dtype='float32'))

    out = sum_array(tvm.runtime.convert(inputs))
    assert len(out.op.inputs) == n

    sch = te.create_schedule(out.op)
    mod = tvm.build(sch, inputs + [out], target='llvm')
    assert mod

    input_nd = []
    out_ref = numpy.zeros((10,))
    for _ in range(n):
        arr = numpy.random.uniform(size=(10,)).astype('float32')
        input_nd.append(tvm.nd.array(arr))
        out_ref += arr
    out_nd = tvm.nd.array(numpy.zeros((10,), 'float32'))
    mod(*input_nd, out_nd)
    tvm.testing.assert_allclose(out_nd.asnumpy(), out_ref)

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
    test_func_call()
    test_bool()
    test_const_range()
    test_schedule()
    test_capture()
    test_array_inputs()
    # TODO:
    # test_inplace()
