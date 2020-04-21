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
from tvm import te
from tvm import relay
from tvm.relay.prelude import Prelude
from tvm.relay import op, create_executor, transform
from tvm.relay import Var, TypeVar, TupleGetItem, Let, Function, const, RefRead, RefWrite, RefCreate
from tvm.relay import TensorType, Tuple, If, Clause, PatternConstructor, PatternVar, Match
from tvm.relay import GlobalVar, Call
from tvm.relay.transform import gradient
from tvm.relay.testing import add_nat_definitions, make_nat_expr, run_infer_type

def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
       mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def tipe(expr):
    return run_opt_pass(expr, [transform.PartialEvaluate(),
                               transform.InferType()])


def dcpe(expr, mod=None, grad=False):
    passes = [transform.PartialEvaluate(),
              transform.DeadCodeElimination(inline_once=True)]
    if grad:
        expr = gradient(run_infer_type(expr))
    if mod:
        assert isinstance(expr, Function)
        mod["main"] = expr
        seq = tvm.transform.Sequential(passes)
        mod = seq(mod)
        return mod["main"]
    return run_opt_pass(expr, passes)


def test_tuple():
    t = TypeVar("t")
    x = Var("x", t)
    body = TupleGetItem(relay.Tuple([relay.const(4.0), x]), 1)
    f = Function([x], body, None, [t])
    expected = relay.Function([x], x, None, [t])
    expected = run_opt_pass(expected, transform.InferType())
    assert tvm.ir.structural_equal(dcpe(f), expected)


def test_const_inline():
    t = relay.TensorType([], "float32")
    d = Var("d", t)
    double = Function([d], d + d)
    orig = double(const(4.0))
    assert tvm.ir.structural_equal(dcpe(orig), const(8.0))


def test_ref():
    t = relay.TensorType([], "float32")
    d = relay.Var("d", t)
    r = relay.Var("r", relay.RefType(t))
    x = relay.Var("x")
    body = relay.RefRead(r)
    body = Let(x, RefWrite(r, RefRead(r) * RefRead(r)), body)
    body = Let(r, RefCreate(d), body)
    square = Function([d], body)
    expected = run_opt_pass(Function([d], d * d), transform.InferType())
    assert tvm.ir.structural_equal(dcpe(square), expected)


def test_empty_ad():
    shape = (10, 10)
    dtype = "float32"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    f = Function([d], d)
    g = dcpe(f, grad=True)
    expected = Function([d], Tuple([d, Tuple([op.ones_like(d)])]))
    expected = run_opt_pass(expected, transform.InferType())
    assert tvm.ir.structural_equal(g, expected)


def test_ad():
    shape = (10, 10)
    dtype = "float32"
    t = TensorType(shape, dtype)
    d = Var("d", t)
    f = Function([d], d * d)
    g = dcpe(f, grad=True)
    m = d * d
    x = relay.Var("x")
    o = op.ones_like(x)
    x1 = relay.Var("x1")
    grad = op.zeros_like(d) + op.collapse_sum_like(x1 * d, d) + op.collapse_sum_like(x1 * d, d)
    body = Tuple([x, Tuple([grad])])
    body = relay.Let(x1, o, body)
    expected = Function([d], relay.Let(x, m, body))
    expected = run_opt_pass(expected, transform.InferType())
    tvm.ir.assert_structural_equal(g, expected)


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
    pe_f = tipe(f)
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
    pe_f = tipe(f)
    ex = create_executor()
    f_res = ex.evaluate(f)(const(True))
    pe_f_res = ex.evaluate(pe_f)(const(True))
    np.testing.assert_allclose(f_res.asnumpy(), np.ones_like(f_res.asnumpy()))
    np.testing.assert_allclose(pe_f_res.asnumpy(), np.ones_like(pe_f_res.asnumpy()))


def test_head_cons():
    mod = tvm.IRModule()
    p = Prelude(mod)
    hd = p.hd
    t = TypeVar("t")
    x = Var("x", t)
    body = hd(p.cons(x, p.nil()))
    f = Function([x], body, None, [t])
    res = dcpe(f, mod)
    assert tvm.ir.structural_equal(res, Function([x], x, t, [t]))


def test_map():
    mod = tvm.IRModule()
    p = Prelude(mod)
    f = GlobalVar("f")
    t = TypeVar("t")
    a = Var("a", t)
    mod[f] = Function([a], a, t, [t])
    orig = p.map(f, p.cons(const(1), p.cons(const(2), p.cons(const(3), p.nil()))))
    expected = p.cons((const(1)), p.cons((const(2)), p.cons((const(3)), p.nil())))
    expected = Function([], expected)
    mod["main"] = expected
    expected = mod["main"]
    orig = Function([], orig)
    res = dcpe(orig, mod=mod)
    assert tvm.ir.structural_equal(res.body, expected.body)


def test_loop():
    mod = tvm.IRModule()
    t = TypeVar("t")
    x = Var("x", t)
    loop = GlobalVar("loop")
    mod[loop] = Function([x], loop(x), t, [t])
    expected = Call(loop, [const(1)])
    mod["main"] = Function([], expected)
    expected = mod["main"].body
    call = Function([], loop(const(1)))
    res = dcpe(call, mod=mod)
    assert tvm.ir.structural_equal(res.body, expected)


def test_swap_loop():
    mod = tvm.IRModule()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    y = Var("y", nat)
    loop = GlobalVar("loop")
    mod[loop] = Function([x, y], loop(y, x), nat)
    prog = loop(make_nat_expr(p, 1), make_nat_expr(p, 2))
    res = Function([], prog)
    res = dcpe(res, mod=mod)
    assert tvm.ir.structural_equal(prog, res.body)


def test_abs_diff():
    # TODO(@M.K.): refactor using tuple pattern (not yet implemented)
    mod = tvm.IRModule()
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
    orig = Function([], orig)
    res = dcpe(orig, mod=mod)
    assert tvm.ir.structural_equal(res.body, make_nat_expr(p, 4))


def test_match_nat_id():
    mod = tvm.IRModule()
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
    orig = Function([], orig)
    res = dcpe(orig, mod=mod)
    assert tvm.ir.structural_equal(res.body, make_nat_expr(p, 3))


def test_nat_id():
    mod = tvm.IRModule()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    y = Var("y", nat)
    nat_id = GlobalVar("nat_id")
    mod[nat_id] = Function([x], x)
    orig = nat_id(make_nat_expr(p, 3))
    orig = Function([], orig)
    res = dcpe(orig, mod=mod)
    assert tvm.ir.structural_equal(res.body, make_nat_expr(p, 3))


def test_global_match_nat_id():
    mod = tvm.IRModule()
    p = Prelude(mod)
    add_nat_definitions(p)
    nat = p.nat()
    x = Var("x", nat)
    z_case = Clause(PatternConstructor(p.z, []), p.z())
    s_case = Clause(PatternConstructor(p.s, [PatternVar(x)]), p.s(x))
    orig = Match(make_nat_expr(p, 3), [z_case, s_case])
    orig = Function([], orig)
    res = dcpe(orig, mod=mod)
    assert tvm.ir.structural_equal(res.body, make_nat_expr(p, 3))


def test_double():
    mod = tvm.IRModule()
    p = Prelude(mod)
    add_nat_definitions(p)
    orig = p.double(make_nat_expr(p, 3))
    orig = Function([], orig)
    res = dcpe(orig, mod=mod)
    assert tvm.ir.structural_equal(res.body, make_nat_expr(p, 6))


def test_concat():
    t = relay.TensorType([10], "float32")
    x = Var("x", t)
    y = Var("x", t)
    orig = run_infer_type(Function([x, y], op.concatenate([x, y], axis=0)))
    tvm.ir.assert_structural_equal(dcpe(orig), orig)


def test_triangle_number():
    t = relay.TensorType([], "int32")
    x = Var("x", t)
    f_var = Var("f")
    f = Function([x], If(op.equal(x, const(0)), const(0), x + f_var(x - const(1))))
    orig = run_infer_type(Let(f_var, f, f_var(const(10))))
    tvm.ir.assert_structural_equal(dcpe(orig), const(55))


def test_nat_update():
    m = tvm.IRModule()
    p = Prelude(m)
    add_nat_definitions(p)
    m = transform.ToANormalForm()(m)
    transform.PartialEvaluate()(m)


def test_tuple_match():
    a = relay.Var("a")
    b = relay.Var("b")
    clause = relay.Clause(relay.PatternTuple([relay.PatternVar(a), relay.PatternVar(b)]), a + b)
    x = relay.Match(relay.Tuple([relay.const(1), relay.const(1)]), [clause])
    tvm.ir.assert_structural_equal(dcpe(x), const(2))


if __name__ == '__main__':
    test_nat_update()
    test_ref()
    test_tuple()
    test_empty_ad()
    test_const_inline()
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
    test_concat()
    test_triangle_number()
    test_tuple_match()
