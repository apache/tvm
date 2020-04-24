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
import numpy as np
import pytest

import tvm
from tvm import runtime
from tvm import relay
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing.config import ctx_list
from tvm.relay.prelude import Prelude
from tvm.relay.loops import while_loop
from tvm.relay import testing

def check_result(args, expected_result, mod=None):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on Relay VM.

    Parameters
    ----------
    args: list of Expr
        The arguments to supply the expr.

    expected_result:
        The expected result of running the expression.
    """
    for target, ctx in ctx_list():
        vm = relay.create_executor('vm', ctx=ctx, target=target, mod=mod)

        rts_result = vm.evaluate()(*args)
        tvm.testing.assert_allclose(expected_result, rts_result.asnumpy())

def veval(f, *args, ctx=tvm.cpu(), target="llvm"):
    if isinstance(f, relay.Expr):
        mod = tvm.IRModule()
        mod["main"] = f
    else:
        assert isinstance(f, tvm.IRModule), "expected expression or module"
        mod = f
    exe = relay.vm.compile(mod, target)
    vm = runtime.vm.VirtualMachine(exe)
    vm.init(ctx)
    return vm.invoke("main", *args)

def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy().tolist()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))

def test_split():
    x = relay.var('x', shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    f = relay.Function([x], y)

    x_data = np.random.rand(12,).astype('float32')
    res = veval(f, x_data)
    ref_res = np.split(x_data, 3, axis=0)
    for i in range(3):
        tvm.testing.assert_allclose(res[i].asnumpy(), ref_res[i])

def test_split_no_fuse():
    x = relay.var('x', shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    z = relay.concatenate([relay.TupleGetItem(y, 0)], axis=0)
    z = relay.annotation.stop_fusion(z)
    f = relay.Function([x], z)
    x_data = np.random.rand(12,).astype('float32')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), np.split(x_data, 3, axis=0)[0])

def test_id():
    x = relay.var('x', shape=(10, 10), dtype='float64')
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype('float64')
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([x_data], x_data, mod=mod)

def test_op():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([x_data], 2 * x_data, mod=mod)

def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])

def test_cond():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    mod = tvm.IRModule()
    mod["main"] = f
    # same
    check_result([x_data, x_data], True, mod=mod)

    # diff
    check_result([x_data, y_data], False, mod=mod)

def test_simple_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    f = relay.Function([x, y],
        relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    mod = tvm.IRModule()
    mod["main"] = f
    # same
    check_result([x_data, x_data], x_data, mod=mod)

    # diff
    check_result([x_data, y_data], y_data, mod=mod)

def test_multiple_ifs():
    mod = tvm.IRModule({})
    b = relay.var('b')
    v0 = relay.var('v0')
    v1 = relay.var('v1')
    v2 = relay.var('v2')
    v3 = relay.var('v3')
    out = relay.Tuple([v2, v3])
    out = relay.Let(v3, relay.If(b, v1, v0), out)
    out = relay.Let(v2, relay.If(b, v0, v1), out)
    out = relay.Let(v1, relay.Tuple([relay.const(1)]), out)
    out = relay.Let(v0, relay.Tuple([relay.const(0)]), out)
    fn = relay.Function([b], out)
    mod['main'] = fn
    ctx = tvm.runtime.ndarray.context('llvm', 0)
    vm = relay.create_executor(ctx=ctx, mod=mod, kind='vm')
    res = vmobj_to_list(vm.evaluate()(False))
    assert(res == [1, 0])

def test_simple_call():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    sb.ret(i)
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    iarg = relay.var('iarg', shape=[], dtype='int32')
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    check_result([i_data], i_data, mod=mod)

def test_count_loop():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    result = veval(mod, i_data)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)
    check_result([i_data], i_data, mod=mod)

def test_sum_loop():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, 'int32'))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, 'int32'))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    loop_bound = 0
    i_data = np.array(loop_bound, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    aarg = relay.var('accum', shape=[], dtype='int32')
    mod["main"] = relay.Function([iarg, aarg], sum_up(iarg, aarg))
    check_result([i_data, accum_data], sum(range(1, loop_bound + 1)), mod=mod)

def test_tuple_fst():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([(i_data, j_data)], i_data, mod=mod)

def test_tuple_second():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([(i_data, j_data)], j_data, mod=mod)

def test_list_constructor():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    l = p.l

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)
    f = relay.Function([], one4)

    mod["main"] = f

    result = veval(mod)
    assert len(result) == 2
    assert len(result[1]) == 2

    obj = vmobj_to_list(result)
    tvm.testing.assert_allclose(obj, np.array([3,2,1]))

def test_let_tensor():
    sb = relay.ScopeBuilder()
    shape = (1,)
    x = relay.var('x', shape=shape, dtype='float32')
    x1 = relay.var('x1', shape=shape, dtype='float32')

    x1 = sb.let(x1, x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.random.rand(*shape).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([x_data], x_data + 42.0, mod=mod)

def test_let_scalar():
    sb = relay.ScopeBuilder()

    x = relay.var('x', 'float32')
    x1 = sb.let('x1', x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.array(np.random.rand()).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([x_data], x_data + 42.0, mod=mod)

def test_compose():
    mod = tvm.IRModule()
    p = Prelude(mod)

    compose = p.compose

    # add_one = fun x -> x + 1
    sb = relay.ScopeBuilder()
    x = relay.var('x', 'float32')
    x1 = sb.let('x1', x)
    xplusone = x1 + relay.const(1.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()
    add_one = relay.GlobalVar("add_one")
    add_one_func = relay.Function([x], body)

    # add_two = compose(add_one, add_one)
    sb = relay.ScopeBuilder()
    y = relay.var('y', 'float32')
    add_two_func = sb.let('add_two', compose(add_one_func, add_one_func))
    add_two_res = add_two_func(y)
    sb.ret(add_two_res)
    add_two_body = sb.get()

    mod[add_one] = add_one_func

    f = relay.Function([y], add_two_body)
    mod["main"] = f

    x_data = np.array(np.random.rand()).astype('float32')
    result = veval(mod, [x_data])
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 2.0)

def test_list_hd():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    l = p.l
    hd = p.hd

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)
    three = hd(one4)
    f = relay.Function([], three)

    mod["main"] = f

    result = veval(mod)
    tvm.testing.assert_allclose(result.asnumpy(), 3)

@pytest.mark.xfail
def test_list_tl_empty_list():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    l = p.l
    tl = p.tl

    f = relay.Function([], tl(nil()))

    mod["main"] = f

    result = veval(mod)
    print(result)

def test_list_tl():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    l = p.l
    tl = p.tl

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)

    f = relay.Function([], tl(one4))

    mod["main"] = f

    result = veval(mod)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([2,1]))

def test_list_nth():
    expected = list(range(10))

    for i in range(len(expected)):
        mod = tvm.IRModule()
        p = Prelude(mod)

        nil = p.nil
        cons = p.cons
        nth = p.nth
        l = nil()
        for i in reversed(expected):
            l = cons(relay.const(i), l)

        f = relay.Function([], nth(l, relay.const(i)))
        mod["main"] = f
        result = veval(mod)
        tvm.testing.assert_allclose(result.asnumpy(), expected[i])

def test_list_update():
    expected = list(range(10))

    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    update = p.update

    l = nil()
    # create zero initialized list
    for i in range(len(expected)):
        l = cons(relay.const(0), l)

    # set value
    for i, v in enumerate(expected):
        l = update(l, relay.const(i), relay.const(v))

    f = relay.Function([], l)
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array(expected))

def test_list_length():
    expected = list(range(10))

    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    length = p.length

    l = nil()
    # create zero initialized list
    for i in range(len(expected)):
        l = cons(relay.const(0), l)

    l = length(l)

    f = relay.Function([], l)
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(result.asnumpy(), 10)

def test_list_map():
    mod = tvm.IRModule()
    p = Prelude(mod)

    x = relay.var('x', 'int32')
    add_one_func = relay.Function([x], relay.const(1) + x)

    nil = p.nil
    cons = p.cons
    map = p.map

    l = cons(relay.const(2), cons(relay.const(1), nil()))

    f = relay.Function([], map(add_one_func, l))
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 2]))

def test_list_foldl():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    foldl = p.foldl

    x = relay.var("x")
    y = relay.var("y")
    rev_dup_func = relay.Function([y, x], cons(x, cons(x, y)))

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], foldl(rev_dup_func, nil(), l))
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 3, 2, 2, 1, 1]))

def test_list_foldr():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    foldr = p.foldr

    x = relay.var("x")
    y = relay.var("y")
    identity_func = relay.Function([x, y], cons(x, y))

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], foldr(identity_func, nil(), l))
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([1, 2, 3]))

def test_list_sum():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    sum = p.sum

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], sum(l))
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(result.asnumpy(), 6)

def test_list_filter():
    mod = tvm.IRModule()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    filter = p.filter

    x = relay.var("x", 'int32')
    greater_than_one = relay.Function([x], x > relay.const(1))
    l = cons(relay.const(1),
            cons(relay.const(3),
                cons(relay.const(1),
                    cons(relay.const(5),
                        cons(relay.const(1), nil())))))
    f = relay.Function([], filter(greater_than_one, l))
    mod["main"] = f
    result = veval(mod)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 5]))

def test_closure():
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))
    res = veval(main)
    tvm.testing.assert_allclose(res.asnumpy(), 3.0)

def test_add_op_scalar():
    """
    test_add_op_scalar:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    func = relay.Function([x, y], relay.op.add(x, y))
    x_data = np.array(10.0, dtype='float32')
    y_data = np.array(1.0, dtype='float32')
    mod["main"] = func
    check_result([x_data, y_data], x_data + y_data, mod=mod)

def test_add_op_tensor():
    """
    test_add_op_tensor:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(10, 5))
    func = relay.Function([x, y], relay.op.add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(10, 5).astype('float32')
    mod["main"] = func
    check_result([x_data, y_data], x_data + y_data, mod=mod)

def test_add_op_broadcast():
    """
    test_add_op_broadcast:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var('x', shape=(10, 5))
    y = relay.var('y', shape=(1, 5))
    func = relay.Function([x, y], relay.op.add(x, y))
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(1, 5).astype('float32')
    mod["main"] = func
    check_result([x_data, y_data], x_data + y_data, mod=mod)

def test_vm_optimize():
    mod, params = testing.resnet.get_workload(batch_size=1, num_layers=18)
    comp = relay.vm.VMCompiler()
    opt_mod, _ = comp.optimize(mod, "llvm", params)

def test_loop_free_var():
    x = relay.var('x', shape=(), dtype='int32')
    i = relay.var('i', shape=(), dtype='int32')
    s = relay.var('s', shape=(), dtype='int32')

    def cond(i, _):
        return i < relay.const(10, dtype='int32')

    def body_no_free_var(i, acc):
        incr = relay.const(1, "int32")
        return i + incr, acc + i

    def body_with_free_var(i, acc):
        incr = relay.const(1, "int32")
        return i + incr, acc + x

    for args, body, expected in zip([[], [1]],
                                    [body_no_free_var, body_with_free_var],
                                    [45, 10]):
        loop = while_loop(cond, [i, s], body)
        tup = loop(relay.const(0, dtype='int32'), relay.zeros(shape=(), dtype='int32'))
        ret = relay.TupleGetItem(tup, 1)
        mod = tvm.IRModule()
        mod["main"] = relay.Function(relay.analysis.free_vars(ret), ret)
        check_result(args, expected, mod=mod)

if __name__ == "__main__":
    pytest.main([__file__])
