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
import os
from nose.tools import nottest

import tvm
import numpy as np
from tvm import relay
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude

def veval(f, *args, ctx=tvm.cpu()):
    if isinstance(f, relay.Expr):
        ex = relay.create_executor('vm', mod=relay.Module(), ctx=ctx)
        if len(args) == 0:
            return ex.evaluate(f)
        else:
            return ex.evaluate(f)(*args)
    else:
        assert isinstance(f, relay.Module), "expected expression or module"
        mod = f
        ex = relay.create_executor('vm', mod=mod, ctx=ctx)
        if len(args) == 0:
            return ex.evaluate()
        else:
            return ex.evaluate()(*args)

def test_split():
    x = relay.var('x', shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    z = relay.concatenate([relay.TupleGetItem(y, 0)], axis=0)
    f = relay.Function([x], z)

    x_data = np.random.rand(12,).astype('float32')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), np.split(x_data, 3, axis=0)[0])

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
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype('float64')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

def test_op():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')
    res = veval(f, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data + x_data)

def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])

def test_cond():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('x', shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = veval(f, x_data, x_data)
    np.testing.assert_allclose(res.asnumpy(), True)

    # diff
    res = veval(f, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), False)


def test_simple_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    f = relay.Function([x, y],
        relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    # same
    res = veval(f, x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

    # diff
    res = veval(f, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data)

def test_simple_call():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    sb.ret(i)
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    result = veval(mod, i_data)
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_count_loop():
    mod = relay.module.Module({})
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

def test_sum_loop():
    mod = relay.module.Module({})
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
    result = veval(mod, i_data, accum_data)
    tvm.testing.assert_allclose(result.asnumpy(), sum(range(1, loop_bound + 1)))

def test_tuple_fst():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    result = veval(f, (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), i_data)

def test_tuple_second():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')
    result = veval(f, (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), j_data)

def test_list_constructor():
    def to_list(o):
        if isinstance(o, tvm.relay.backend.interpreter.TensorValue):
            return [o.data.asnumpy().tolist()]
        if isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
            result = []
            for f in o.fields:
                result.extend(to_list(f))
            return result

    mod = relay.Module()
    p = Prelude(mod)

    nil = p.nil
    cons = p.cons
    l = p.l

    # remove all functions to not have pattern match to pass vm compilation
    # TODO(wweic): remove the hack and implement pattern match
    for v, _ in mod.functions.items():
        mod[v] = relay.const(0)

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)
    f = relay.Function([], one4)

    mod["main"] = f

    result = veval(mod)()
    obj = to_list(result)
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
    result = veval(f, x_data)
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 42.0)

def test_let_scalar():
    sb = relay.ScopeBuilder()

    x = relay.var('x', 'float32')
    x1 = sb.let('x1', x)
    xplusone = x1 + relay.const(42.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.array(np.random.rand()).astype('float32')
    result = veval(f, x_data)
    tvm.testing.assert_allclose(result.asnumpy(), x_data + 42.0)

def test_compose():
    mod = relay.Module()
    p = Prelude(mod)

    compose = p.compose

    # remove all functions to not have pattern match to pass vm compilation
    # TODO(wweic): remove the hack and implement pattern match
    for v, _ in mod.functions.items():
        if v.name_hint == 'compose':
            continue
        mod[v] = relay.const(0)

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
    result = veval(mod)(x_data)

    tvm.testing.assert_allclose(result.asnumpy(), x_data + 2.0)

def test_closure():
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))
    res = veval(main)
    tvm.testing.assert_allclose(res.asnumpy(), 3.0)

if __name__ == "__main__":
    test_id()
    test_op()
    test_cond()
    test_simple_if()
    test_simple_call()
    test_count_loop()
    test_sum_loop()
    test_tuple_fst()
    test_tuple_second()
    test_let_scalar()
    test_let_tensor()
    test_split()
    test_split_no_fuse()
    # TODO(@jroesch): restore when match is supported
    # test_list_constructor()
    test_closure()
