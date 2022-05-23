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
import pytest
import sys
import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm import relay
from tvm.relay.analysis import detect_feature
from tvm.relay import op, create_executor, transform
from tvm.relay.prelude import Prelude
from tvm.relay.testing import count
from tvm.relay.analysis import Feature


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    dev = tvm.device("llvm", 0)
    result = create_executor(mod=mod, device=dev, target="llvm").evaluate(expr)
    np.testing.assert_allclose(result.numpy(), expected_result, rtol=rtol)


def test_explicit_bound():
    x = relay.const(1)
    y = op.add(x, x)
    z = op.add(y, y)
    f = relay.Function([], op.add(z, z))
    assert not Feature.fLet in detect_feature(f)
    anf = run_opt_pass(f, transform.ToANormalForm())
    assert Feature.fLet in detect_feature(anf)
    check_eval(f(), 8.0)
    check_eval(anf(), 8.0)


# test that the construction order does not matter,
# and is instead ordered by the scope and by post-dfs ordering.
def test_order():
    z = relay.const(3)
    y = relay.const(2)
    x = relay.const(1)
    val = x + y * z
    check_eval(val, 7.0)
    anf = run_opt_pass(val, [transform.ToANormalForm(), transform.InferType()])
    a = relay.Var("a", relay.IncompleteType())
    b = relay.Var("b", relay.IncompleteType())
    c = relay.Var("c", relay.IncompleteType())
    d = relay.Var("d", relay.IncompleteType())
    e = relay.Var("e", relay.IncompleteType())
    expected_output = e
    expected_output = relay.Let(e, a + d, expected_output)
    expected_output = relay.Let(d, b * c, expected_output)
    expected_output = relay.Let(c, z, expected_output)
    expected_output = relay.Let(b, y, expected_output)
    expected_output = relay.Let(a, x, expected_output)
    expected_output = run_opt_pass(expected_output, transform.InferType())
    assert tvm.ir.structural_equal(anf, expected_output)


def test_if():
    cond = relay.const(True)
    x = relay.If(cond, relay.const(2), relay.const(3))
    anf = run_opt_pass(x, [transform.ToANormalForm(), transform.InferType()])
    a = relay.Var("a", relay.IncompleteType())
    b = relay.Var("b", relay.IncompleteType())
    c = relay.Var("c", relay.IncompleteType())
    d = relay.Var("d", relay.IncompleteType())
    true_branch = relay.Let(a, relay.const(2), a)
    false_branch = relay.Let(b, relay.const(3), b)
    expected_output = relay.If(c, true_branch, false_branch)
    expected_output = relay.Let(d, expected_output, d)
    expected_output = relay.Let(c, cond, expected_output)
    expected_output = run_opt_pass(expected_output, transform.InferType())
    assert tvm.ir.structural_equal(anf, expected_output)


def test_let_as_subexpr():
    def on_cpu(x):
        return relay.annotation.on_device(x, tvm.device("cpu"), constrain_result=True)

    x = relay.Var("x", relay.IncompleteType())
    c = relay.const(1)
    l = relay.Let(x, on_cpu(c + c), x)
    body = l * l

    anf = run_opt_pass(body, [transform.ToANormalForm(), transform.InferType()])

    v0 = relay.Var("v0", relay.IncompleteType())
    v1 = relay.Var("v1", relay.IncompleteType())
    v2 = relay.Var("v2", relay.IncompleteType())
    expected_output = relay.Let(
        v0,
        on_cpu(c),
        relay.Let(
            x,
            on_cpu(v0 + v0),
            relay.Let(v1, x, relay.Let(v2, v1 * v1, v2)),
        ),
    )
    expected_output = run_opt_pass(expected_output, transform.InferType())

    tvm.ir.assert_structural_equal(anf, expected_output)


# make sure we dont infinite loop.
# it is too large so we wont check for the exact program.
def test_recursion():
    """
    Program:
       let f(n: i32) -> i32 = {
          m = (n * 2)
          if (n == 0) {
              return m;
          } else {
              return m + f(n - 1);
          }
       }
       f(5);
    """
    mod = tvm.IRModule()
    i64 = relay.TensorType((), "int64")
    f = relay.GlobalVar("f")
    n = relay.Var("n", i64)
    m = n * relay.const(2, "int64")
    funcbody = relay.If(
        relay.equal(n, relay.const(0, "int64")), m, m + f(n - relay.const(1, "int64"))
    )
    value = relay.Function([n], funcbody, i64, [])
    mod[f] = value
    check_eval(f(relay.const(5, "int64")), 30.0, mod=mod)
    old_f = mod[f]
    mod = transform.ToANormalForm()(mod)
    f = mod[f]
    check_eval(f(relay.const(5, "int64")), 30.0, mod=mod)


def test_ref():
    i = relay.Var("i")
    iv = relay.Var("iv")
    u = relay.Var("u")
    uv = relay.Var("uv")
    body = relay.add(iv, uv)
    body = relay.Let(uv, relay.RefRead(i), body)
    body = relay.Let(u, relay.RefWrite(i, relay.const(2)), body)
    body = relay.Let(iv, relay.RefRead(i), body)
    body = relay.Let(i, relay.RefCreate(relay.const(1)), body)
    check_eval(body, 3)
    opt_body = run_opt_pass(body, transform.ToANormalForm())
    check_eval(opt_body, 3)


def test_nat_add():
    mod = tvm.IRModule()
    p = Prelude(mod)
    p.mod.import_from_std("nat.rly")
    nat, z, s = p.mod.get_type("nat")
    add = p.mod.get_global_var("nat_add")
    dev = tvm.device("llvm", 0)
    intrp = create_executor(mod=mod, device=dev, target="llvm")
    # CAUTION: Following calls to intrp.evaluate(...) will re-prepare the prelude.
    assert mod[add].checked_type == relay.FuncType([nat(), nat()], nat())
    assert count(p, intrp.evaluate(add(s(z()), s(z())))) == 2
    expr = add(s(z()), s(z()))
    f = relay.GlobalVar("f")
    mod[f] = relay.Function([], expr)
    mod = transform.ToANormalForm()(mod)
    expr = mod["f"]
    assert count(p, intrp.evaluate(expr.body)) == 2
    assert Feature.fLet in detect_feature(mod[add])


def test_let():
    x = relay.Var("x")
    y = relay.Var("y")
    d = relay.const(4.0, "float32")
    body = relay.Let(y, x, x + y)
    body = relay.Let(x, d, body)
    check_eval(body, 8)
    opt_body = run_opt_pass(body, transform.ToANormalForm())
    check_eval(opt_body, 8)


def test_function():
    t = relay.TensorType((), "float32")
    x = relay.Var("x", t)
    f = relay.Function([x], x + x)
    d = relay.const(4.0, "float32")
    anf_f = run_opt_pass(f, transform.ToANormalForm())
    assert isinstance(anf_f, relay.Function)
    check_eval(f(d), 8)
    check_eval(anf_f(d), 8)


def test_gradient_if():
    x = relay.var("a", shape=(1, 16))
    y = relay.var("y", shape=(1, 16))
    cond = relay.var("cond", shape=(), dtype="uint1")
    net = relay.If(cond, x, x)
    net = relay.add(x, net)
    net = relay.Function([cond, x, y], net)
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.ToANormalForm()(mod)
    mod = relay.transform.InferType()(mod)
    mod["main"] = relay.transform.gradient(mod["main"], mode="higher_order")
    mod = relay.transform.ToANormalForm()(mod)


if __name__ == "__main__":
    tvm.testing.main()
