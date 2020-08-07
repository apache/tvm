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

import tvm
from tvm import relay
from tvm import IRModule as Module
from tvm.relay import var, Function, op, GlobalVar, TypeVar, FuncType
from tvm.relay.prelude import Prelude
from tvm.relay.testing import add_nat_definitions
from tvm.relay.backend import aot


def compile(f, mod):
    tgt = tvm.target.create('llvm')
    ctx = tvm.context('llvm', 0)
    return aot.compile_prog(f, mod, ctx=ctx, tgt=tgt)


def test_identity():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], x)
    cfunc = compile(func, mod)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), a.asnumpy())


def test_add():
    mod = Module()
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    func = Function([x, y], z)
    cfunc = compile(func, mod)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    c = tvm.nd.array(np.array(2.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), c.asnumpy())


def test_mult_op():
    mod = Module()
    x = var('x', shape=())
    y = var('y', shape=())
    z = x + y
    zz = op.exp(z)
    func = Function([x, y], zz)
    cfunc = compile(func, mod)
    a = tvm.nd.array(np.array(1.0, dtype='float32'))
    b = tvm.nd.array(np.array(1.0, dtype='float32'))
    output = cfunc(a, b)
    np.testing.assert_allclose(output.asnumpy(), np.exp(a.asnumpy() + b.asnumpy()))


def test_double():
    mod = Module()
    x = var('x', shape=())
    double = GlobalVar('double')
    mod[double] = Function([x], x + x)
    x = var('x', shape=())
    cfunc = compile(Function([x], double(double(x))), mod)
    a = tvm.nd.array(np.array(1.5, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(6.0, dtype='float32'))


def test_42():
    mod = Module()
    func = Function([], relay.const(42))
    cfunc = compile(func, mod)
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(42.0, dtype='float32'))


def test_add_42():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], x + relay.const(42.0))
    cfunc = compile(func, mod)
    a = tvm.nd.array(np.array(42.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(84.0, dtype='float32'))


def test_int_mult_3():
    mod = Module()
    x = var('x', dtype='int32', shape=())
    func = Function([x], x * relay.const(3))
    cfunc = compile(func, mod)
    a = tvm.nd.array(np.array(4, dtype='int32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(12, dtype='int32'))


def test_abs():
    mod = Module()
    x = var('x', shape=())
    func = Function([x], relay.If(op.less(x, relay.const(0.0)), relay.const(-1.0) * x, x))
    cfunc = compile(func, mod)
    a = tvm.nd.array(np.array(12.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(12.0, dtype='float32'))
    a = tvm.nd.array(np.array(-34.0, dtype='float32'))
    output = cfunc(a)
    np.testing.assert_allclose(output.asnumpy(), np.array(34.0, dtype='float32'))


def test_recur_sum_global():
    mod = Module()
    x = var('x', dtype='int32', shape=())
    sum = GlobalVar('sum')
    c = relay.const(0)
    mod[sum] = Function([x],
                        relay.If(op.less(x, c), c, x + sum(x - relay.const(1))),
                        relay.TensorType(dtype='int32', shape=()))
    cfunc = compile(Function([], sum(relay.const(10))), mod)
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(55, dtype='int32'))


def nat_to_int(n):
    if n.constructor.tag & 0xff == 1:
        return 1 + nat_to_int(n.fields[0])
    else:
        assert n.constructor.tag & 0xff == 0
        return 0


def int_to_nat(p, i):
    if i > 0:
        return p.s(int_to_nat(p, i - 1))
    else:
        assert i == 0
        return p.z()


def test_nat_3():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    cfunc = compile(Function([], p.s(p.s(p.s(p.z())))), mod)
    output = cfunc()
    assert nat_to_int(output) == 3


def test_nat_add():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    cfunc = compile(Function([], p.add(p.s(p.s(p.s(p.z()))), p.s(p.s(p.s(p.s(p.z())))))), mod)
    output = cfunc()
    assert nat_to_int(output) == 7


def test_add_convert():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    cfunc = compile(p.add, mod)
    output = cfunc(int_to_nat(p, 12), int_to_nat(p, 34))
    assert nat_to_int(output) == 46


def test_ref():
    mod = Module()
    three_with_ref = relay.GlobalVar('three_with_ref')
    i = relay.Var('i')
    iv = relay.Var('iv')
    u = relay.Var('u')
    uv = relay.Var('uv')
    body = relay.add(iv, uv)
    body = relay.Let(uv, relay.RefRead(i), body)
    body = relay.Let(u, relay.RefWrite(i, relay.const(2, dtype='int32')), body)
    body = relay.Let(iv, relay.RefRead(i), body)
    body = relay.Let(i, relay.RefCreate(relay.const(1, dtype='int32')), body)
    mod[three_with_ref] = relay.Function([], body)
    cfunc = compile(three_with_ref, mod)
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(3, dtype='int32'))


def test_tuple():
    mod = Module()
    cfunc = compile(Function([],
                             relay.TupleGetItem(relay.Tuple([relay.const(3, dtype='int32'),
                                                             relay.const(4.0, dtype='float32')]),
                                                1)),
                    mod)
    np.testing.assert_allclose(cfunc().asnumpy(), np.array(4.0, dtype='float32'))


def test_get_valid_counts():
    # Based on test_get_valid_counts in test_op_level5.
    # Tests the case of a packed func returning a Relay tuple.
    # Only checks the shapes of the output because the reference implementation
    # is long and inconvenient.
    shape = (1, 2500, 6)
    score_threshold = 0
    id_index = 0
    score_index = 1
    np_data = np.random.uniform(low=-2, high=2, size=shape).astype("float32")
    mod = Module()
    cfunc = compile(
        relay.Function(
            [],
            relay.vision.get_valid_counts(
                relay.const(np_data), score_threshold, id_index, score_index
            ).astuple()),
        mod)

    relay_out = cfunc()
    out1 = relay_out[0].asnumpy()
    out2 = relay_out[1].asnumpy()
    assert out1.shape == (shape[0],)
    assert out2.shape == shape


def test_compose():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    x = relay.Var('x')
    inc = GlobalVar('inc')
    mod[inc] = Function([x], p.s(x))
    x = relay.Var('x')
    func = GlobalVar('func')
    f = Function([x], relay.Call(p.compose(inc, p.double), [x]))
    mod[func] = f
    cfunc = compile(func, mod)
    assert nat_to_int(cfunc(p.s(p.s(p.z())))) == 5


def test_recur_sum_local():
    mod = Module()
    x = var('x', dtype='int32', shape=())
    t = relay.TensorType(dtype='int32', shape=())
    sum = relay.Var('sum', type_annotation=relay.FuncType([t], t))
    c = relay.const(0)
    func = Function([x],
                    relay.If(op.less(x, c), c, x + sum(x - relay.const(1))),
                    t)
    body = relay.Let(sum, func, sum(relay.const(10)))
    cfunc = compile(Function([], body), mod)
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(55, dtype='int32'))


def test_local_local_rec_outer_scope():
    mod = Module()
    x = var('x', dtype='int32', shape=())
    t = relay.TensorType(dtype='int32', shape=())
    sum = relay.Var('sum', type_annotation=relay.FuncType([t], t))
    c = relay.const(0)

    # we define a locally recursive function inside another function's scope
    # and have that function return the closure of the locally recursive function
    inner_func = Function([x],
                    relay.If(op.less(x, c), c, x + sum(x - relay.const(1))),
                    t)
    outer_func_body = relay.Let(sum, inner_func, sum)
    outer_func = Function([], outer_func_body)
    f = relay.Var('f')
    body = relay.Let(f, outer_func(), f(relay.const(10)))
    cfunc = compile(Function([], body), mod)
    output = cfunc()
    np.testing.assert_allclose(output.asnumpy(), np.array(55, dtype='int32'))


if __name__ == "__main__":
    test_identity()
    test_add()
    test_mult_op()
    test_double()
    test_42()
    test_add_42()
    test_int_mult_3()
    test_abs()
    test_recur_sum_global()
    test_nat_3()
    test_nat_add()
    test_add_convert()
    test_ref()
    test_tuple()
    test_get_valid_counts()
    test_compose()
    test_recur_sum_local()
    test_local_local_rec_outer_scope()
