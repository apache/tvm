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
from tvm.relay.ir_pass import partial_evaluate, alpha_equal, infer_type, dead_code_elimination
from tvm.relay.ir_pass import gradient
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue
from tvm.relay.prelude import Prelude
from tvm.relay import create_executor
from nose.tools import nottest
from tvm.relay import Var, TypeVar, TupleGetItem, Let, Function, const, RefRead, RefWrite, RefCreate
from tvm.relay import TensorType, Tuple, If, Module, Clause, PatternConstructor, PatternVar, Match
from tvm.relay import GlobalVar, Call, Type
from tvm.relay.testing import add_nat_definitions, count, make_nat_value, make_nat_expr

def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def dcpe(expr, mod=None):
    return dead_code_elimination(partial_evaluate(expr, mod=mod), inline_once=True)


def test_tuple():
    t = TypeVar("t")
    x = Var("x", t)
    body = TupleGetItem(relay.Tuple([relay.const(4.0), x]), 1)
    f = Function([x], body, None, [t])
    assert alpha_equal(dcpe(f), relay.Function([x], x, None, [t]))

def test_const_inline():
    d = Var("d")
    double = Function([d], d + d)
    orig = double(const(4.0))
    assert alpha_equal(dcpe(orig), const(8.0))


def test_ref():
    d = relay.Var("d")
    r = relay.Var("r")
    x = relay.Var("x")
    body = relay.RefRead(r)
    body = Let(x, RefWrite(r, RefRead(r) * RefRead(r)), body)
    body = Let(r, RefCreate(d), body)
    square = Function([d], body)
    assert alpha_equal(dcpe(square), Function([d], d * d))


def test_empty_ad():
    shape = (10, 10)
    dtype = "float32"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    f = Function([d], d)
    g = dcpe(gradient(f))
    expected = Function([d], Tuple([d, Tuple([op.ones_like(d)])]))
    assert alpha_equal(g, expected)

def test_ad():
    shape = (10, 10)
    dtype = "float32"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    f = Function([d], d * d)
    g = dcpe(gradient(f))
    m = d * d
    x = relay.Var("x")
    o = op.ones_like(x)
    x1 = relay.Var("x1")
    grad = op.zeros_like(d) + op.collapse_sum_like(x1 * d, d) + op.collapse_sum_like(x1 * d, d)
    body = Tuple([x, Tuple([grad])])
    body = relay.Let(x1, o, body)
    expected = Function([d], relay.Let(x, m, body))
    assert alpha_equal(g, expected)


def test_if_ref():
    shape = ()
    dtype = "bool"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    r = Var("r")
    update = Function([], RefWrite(r, RefRead(r) + RefRead(r)))
    u = Var("u")
    body = If(d, u(), u())
    eff = Var("eff")
    body = Let(eff, body, RefRead(r))
    f = Function([d], Let(r, RefCreate(const(1)), Let(u, update, body)))
    f = infer_type(f)
    pe_f = infer_type(partial_evaluate(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(const(True))
    pe_f_res = ex.evaluate(pe_f)(const(True))
    np.testing.assert_allclose(f_res.asnumpy(), 2 * np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), 2 * np.ones_like(pe_f_res.asnumpy()))


def test_function_invalidate():
    shape = ()
    dtype = "bool"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    r = Var("r")
    fetch = Function([], RefRead(r))
    fet = Var("fetch")
    fet_obscured = Var("fetch_obscured")
    u = Var("u")
    body = If(d, fet_obscured(), fet_obscured())
    body = Let(u, RefWrite(r, const(1)), body)
    body = Let(fet_obscured, If(d, fet, fet), body)
    body = Let(fet, fetch, body)
    body = Let(r, RefCreate(const(0)), body)
    f = Function([d], body)
    f = infer_type(f)
    pe_f = infer_type(partial_evaluate(f))
    ex = create_executor()
    f_res = ex.evaluate(f)(const(True))
    pe_f_res = ex.evaluate(pe_f)(const(True))
    np.testing.assert_allclose(f_res.asnumpy(), np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), np.ones_like(pe_f_res.asnumpy()))


def test_head_cons():
    mod = Module()
    p = Prelude(mod)
    def hd_impl():
        a = TypeVar("a")
        x = Var("x", p.l(a))
        y = Var("y")
        z = Var("z")
        cons_case = Clause(PatternConstructor(p.cons,
                                              [PatternVar(y),
                                               PatternVar(z)]),
                           y)
        y = Var("y")
        z = Var("z")
        return Function([x], Match(x, [cons_case]), a, [a])
    t = TypeVar("t")
    x = Var("x", t)
    hd = Var("hd")
    body = Let(hd, hd_impl(), hd(p.cons(x, p.nil())))
    f = Function([x], body, None, [t])
    f = infer_type(f, mod=mod)
    res = dcpe(f)
    assert alpha_equal(res, Function([x], x, t, [t]))


def test_map():
    mod = Module()
    p = Prelude(mod)
    f = Var("f")
    orig = p.map(f, p.cons(const(1), p.cons(const(2), p.cons(const(3), p.nil()))))
    expected = p.cons(f(const(1)), p.cons(f(const(2)), p.cons(f(const(3)), p.nil())))
    assert alpha_equal(dcpe(orig, mod=mod), expected)


def test_loop():
    mod = Module()
    t = TypeVar("t")
    x = Var("x", t)
    loop = GlobalVar("loop")
    mod[loop] = Function([x], loop(x), t, [t])
    res = dcpe(loop(const(1)), mod=mod)
    expected = Call(loop, [const(1)], None, [None])
    assert alpha_equal(res, expected)


def test_swap_loop():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    y = Var("y", nat)
    loop = GlobalVar("loop")
    mod[loop] = Function([x, y], loop(y, x), nat)
    prog = loop(make_nat_expr(p, 1), make_nat_expr(p, 2))
    res = dcpe(prog, mod=mod)
    assert alpha_equal(prog, res)


def test_abs_diff():
    # TODO(@M.K.): refactor using tuple pattern (not yet implemented)
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    y = Var("y", nat)
    xp = Var("x'", nat)
    yp = Var("y'", nat)
    diff = GlobalVar("diff")
    y_z_case = Clause(PatternConstructor(p.z, []), x)
    y_s_case = Clause(PatternConstructor(p.s, [PatternVar(yp)]), diff(yp, xp))
    x_z_case = Clause(PatternConstructor(p.z, []), y)
    x_s_case = Clause(PatternConstructor(p.s, [PatternVar(xp)]), Match(y, [y_z_case, y_s_case]))
    mod[diff] = Function([x, y], Match(x, [x_z_case, x_s_case]))
    orig = diff(make_nat_expr(p, 7), make_nat_expr(p, 3))
    res = dcpe(orig, mod=mod)
    assert alpha_equal(res, make_nat_expr(p, 4))


def test_match_nat_id():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    y = Var("y", nat)
    nat_id = GlobalVar("nat_id")
    z_case = Clause(PatternConstructor(p.z, []), p.z())
    s_case = Clause(PatternConstructor(p.s, [PatternVar(y)]), p.s(y))
    mod[nat_id] = Function([x], Match(x, [z_case, s_case]))
    orig = nat_id(make_nat_expr(p, 3))
    res = dcpe(orig, mod=mod)
    assert alpha_equal(res, make_nat_expr(p, 3))


def test_nat_id():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    y = Var("y", nat)
    nat_id = GlobalVar("nat_id")
    mod[nat_id] = Function([x], x)
    orig = nat_id(make_nat_expr(p, 3))
    res = dcpe(orig, mod=mod)
    assert alpha_equal(res, make_nat_expr(p, 3))


def test_global_match_nat_id():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    z_case = Clause(PatternConstructor(p.z, []), p.z())
    s_case = Clause(PatternConstructor(p.s, [PatternVar(x)]), p.s(x))
    orig = Match(make_nat_expr(p, 3), [z_case, s_case])
    res = dcpe(orig, mod=mod)
    assert alpha_equal(res, make_nat_expr(p, 3))


def test_double():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)
    orig = p.double(make_nat_expr(p, 3))
    res = dcpe(orig, mod=mod)
    assert alpha_equal(res, make_nat_expr(p, 6))


if __name__ == '__main__':
    test_empty_ad()
    test_tuple()
    test_const_inline()
    test_ref()
    test_ad()
    test_if_ref()
    test_function_invalidate()
    test_head_cons()
    test_map()
    test_loop()
    test_swap_loop()
    test_abs_diff()
    test_double()
    test_nat_id()
    test_global_match_nat_id()
    test_match_nat_id()
