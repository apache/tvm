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
from tvm.relay.ir_pass import partial_evaluate, dead_code_elimination
from tvm.relay.ir_pass import gradient, alpha_equal, infer_type
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue
from tvm.relay.prelude import Prelude
from tvm.relay import create_executor

from nose.tools import nottest

def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def dcpe(expr):
    return dead_code_elimination(partial_evaluate(expr))


def test_tuple():
    t = relay.TypeVar("t")
    x = relay.Var("x", t)
    body = relay.TupleGetItem(relay.Tuple([relay.const(4.0), x]), 1)
    f = relay.Function([x], body, None, [t])
    assert alpha_equal(dcpe(f), relay.Function([x], x, None, [t]))

@nottest
def test_const_inline():
    # TODO(MK): fix me
    d = relay.Var("d")
    double = relay.Function([d], d + d)
    orig = double(relay.const(4.0))
    assert alpha_equal(dcpe(double(relay.const(4.0))), relay.const(8.0))


def test_ref():
    d = relay.Var("d")
    r = relay.Var("r")
    x = relay.Var("x")
    body = relay.RefRead(r)
    body = relay.Let(x, relay.RefWrite(r, relay.RefRead(r) * relay.RefRead(r)), body)
    body = relay.Let(r, relay.RefCreate(d), body)
    square = relay.Function([d], body)
    assert alpha_equal(dcpe(square), relay.Function([d], d * d))

@nottest
def test_ad():
    # TODO(MK): fix me
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    d = relay.Var("d", t)
    f = relay.Function([d], d * d)
    g = dcpe(gradient(f))
    m = d * d
    o = relay.op.ones_like(m)
    grad = relay.op.zeros_like(d) + relay.op.collapse_sum_like(o * d, d) + relay.op.collapse_sum_like(o * d, d)
    expected = relay.Function([d], relay.Tuple([m, relay.Tuple([grad])]))
    assert alpha_equal(g, expected)


def test_if_ref():
    shape = ()
    dtype = "bool"
    t = relay.TensorType(shape, dtype)
    d = relay.Var("d", t)
    r = relay.Var("r")
    update = relay.Function([], relay.RefWrite(r, relay.RefRead(r) + relay.RefRead(r)))
    u = relay.Var("u")
    body = relay.If(d, u(), u())
    eff = relay.Var("eff")
    body = relay.Let(eff, body, relay.RefRead(r))
    f = relay.Function([d], relay.Let(r, relay.RefCreate(relay.const(1)), relay.Let(u, update, body)))
    f = infer_type(f)
    pe_f = infer_type(partial_evaluate(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(relay.const(True))
    pe_f_res = ex.evaluate(pe_f)(relay.const(True))
    np.testing.assert_allclose(f_res.asnumpy(), 2 * np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), 2 * np.ones_like(pe_f_res.asnumpy()))


def test_function_invalidate():
    shape = ()
    dtype = "bool"
    t = relay.TensorType(shape, dtype)
    d = relay.Var("d", t)
    r = relay.Var("r")
    fetch = relay.Function([], relay.RefRead(r))
    fet = relay.Var("fetch")
    fet_obscured = relay.Var("fetch_obscured")
    u = relay.Var("u")
    body = relay.If(d, fet_obscured(), fet_obscured())
    body = relay.Let(u, relay.RefWrite(r, relay.const(1)), body)
    body = relay.Let(fet_obscured, relay.If(d, fet, fet), body)
    body = relay.Let(fet, fetch, body)
    body = relay.Let(r, relay.RefCreate(relay.const(0)), body)
    f = relay.Function([d], body)
    f = infer_type(f)
    pe_f = infer_type(partial_evaluate(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(relay.const(True))
    pe_f_res = ex.evaluate(pe_f)(relay.const(True))
    np.testing.assert_allclose(f_res.asnumpy(), np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), np.ones_like(pe_f_res.asnumpy()))


def test_head_cons():
    mod = relay.Module()
    p = Prelude(mod)
    def hd_impl():
        a = relay.TypeVar("a")
        x = relay.Var("x", p.l(a))
        y = relay.Var("y")
        z = relay.Var("z")
        cons_case = relay.Clause(relay.PatternConstructor(p.cons,
                                                          [relay.PatternVar(y),
                                                           relay.PatternVar(z)]),
                                 y)
        return relay.Function([x], relay.Match(x, [cons_case]), a, [a])
    t = relay.TypeVar("t")
    x = relay.Var("x", t)
    hd = relay.Var("hd")
    body = relay.Let(hd, hd_impl(), hd(p.cons(x, p.nil())))
    f = relay.Function([x], body, None, [t])
    f = infer_type(f, mod=mod)
    res = dcpe(f)
    assert alpha_equal(res, relay.Function([x], x, t, [t]))


if __name__ == '__main__':
    test_tuple()
    test_const_inline()
    test_ref()
    test_ad()
    test_if_ref()
    test_function_invalidate()
    test_head_cons()
