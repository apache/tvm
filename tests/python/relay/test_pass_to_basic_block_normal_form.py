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
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay.analysis import detect_feature
from tvm.relay import op, create_executor, transform
from tvm.relay.prelude import Prelude
from tvm.relay.testing import count, create_workload
from tvm.relay.analysis import Feature
from tvm.relay.analysis import check_basic_block_normal_form


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


def test_no_explicit_bind():
    x = relay.const(1)
    y = op.add(x, x)
    z = op.add(y, y)
    f = relay.Function([], op.add(z, z))
    """
    fn () {
      %0 = add(1, 1);
      %1 = add(%0, %0);
      add(%1, %1)
    }
    """
    assert not Feature.fLet in detect_feature(f)
    bblock = run_opt_pass(f, transform.ToBasicBlockNormalForm())
    assert Feature.fLet not in detect_feature(bblock)
    check_eval(f(), 8.0)
    check_eval(bblock(), 8.0)
    check_basic_block_normal_form(bblock)


def test_top_level_nested_if():
    x = relay.var("x", shape=(), dtype="bool")
    y = relay.var("y", shape=(), dtype="float32")
    z = relay.var("z", shape=(), dtype="float32")
    cond_t = relay.const(True)
    cond_f = relay.const(False)
    one = relay.const(1, dtype="float32")
    three = relay.const(3, dtype="float32")
    y2 = relay.add(y, y)
    z2 = relay.add(z, z)
    true_branch = relay.If(cond_t, relay.add(z2, y2), relay.add(three, y2))
    false_branch = relay.If(cond_f, z2, one)
    body = relay.If(x, true_branch, false_branch)
    """
    free_var %x: bool
    if (%x) {
      if (True) {
        free_var %z: float32
        %0 = add(%z, %z);
        free_var %y: float32
        %1 = add(%y, %y);
        add(%0, %1)
      } else {
        add(3f, %1)
      }
    } else {
      if (False) {
        %0
      } else {
        1f
      }
    }
    """

    def expected():
        x = relay.var("x", shape=(), dtype="bool")
        y = relay.var("y", shape=(), dtype="float32")
        z = relay.var("z", shape=(), dtype="float32")
        cond_t = relay.const(True)
        cond_f = relay.const(False)
        one = relay.const(1, dtype="float32")
        three = relay.const(3, dtype="float32")
        y2 = relay.var("y2")
        z2 = relay.var("z2")
        true_branch = relay.If(cond_t, relay.add(z2, y2), relay.add(three, y2))
        true_branch = relay.Let(y2, relay.add(y, y), true_branch)
        false_branch = relay.If(cond_f, z2, one)
        body = relay.If(x, true_branch, false_branch)
        body = relay.Let(z2, relay.add(z, z), body)
        return body

    bblock = run_opt_pass(body, [transform.ToBasicBlockNormalForm(), transform.InferType()])
    """
    free_var %z: float32
    let %x: float32 = add(%z, %z) /* ty=float32 */;
    free_var %x1: bool
    if (%x1) {
      free_var %y: float32
      let %x2: float32 = add(%y, %y) /* ty=float32 */;
      if (True /* ty=bool */) {
        add(%x, %x2) /* ty=float32 */
      } else {
        add(3f /* ty=float32 */, %x2) /* ty=float32 */
      }
    } else {
      if (False /* ty=bool */) {
        %x
      } else {
        1f /* ty=float32 */
      }
    }
    """
    expected_output = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(bblock, expected_output, map_free_vars=True)


def test_nested_if():
    x = relay.var("x", shape=(), dtype="bool")
    y = relay.var("y", shape=(), dtype="float32")
    cond_t = relay.const(True)
    cond_f = relay.const(False)
    one = relay.const(1, dtype="float32")
    two = relay.const(2, dtype="float32")
    three = relay.const(3, dtype="float32")
    y2 = relay.add(y, y)
    true_branch = relay.If(cond_t, y2, relay.add(three, y2))
    false_branch = relay.If(cond_f, two, one)
    body = relay.If(x, true_branch, false_branch)
    """
    free_var %x: bool
    if (%x) {
      if (True) {
        free_var %y: float32
        %0 = add(%y, %y);
        %0
      } else {
        add(3f, %0)
      }
    } else {
      if (False) {
        2f
      } else {
        1f
      }
    }
    """

    def expected():
        x = relay.var("x", shape=(), dtype="bool")
        y = relay.var("y", shape=(), dtype="float32")
        cond_t = relay.const(True)
        cond_f = relay.const(False)
        one = relay.const(1, dtype="float32")
        two = relay.const(2, dtype="float32")
        three = relay.const(3, dtype="float32")
        y2 = relay.var("y2")
        true_branch = relay.If(cond_t, y2, relay.add(three, y2))
        true_branch = relay.Let(y2, relay.add(y, y), true_branch)
        false_branch = relay.If(cond_f, two, one)
        body = relay.If(x, true_branch, false_branch)
        return body

    bblock = run_opt_pass(body, [transform.ToBasicBlockNormalForm(), transform.InferType()])
    """
    free_var %x: bool
    if (%x) {
      free_var %y: float32
      let %x1: float32 = add(%y, %y) /* ty=float32 */;
      if (True /* ty=bool */) {
        %x1
      } else {
        add(3f /* ty=float32 */, %x1) /* ty=float32 */
      }
    } else {
      if (False /* ty=bool */) {
        2f /* ty=float32 */
      } else {
        1f /* ty=float32 */
      }
    }
    """
    expected_output = run_opt_pass(expected(), transform.InferType())
    tvm.ir.assert_structural_equal(bblock, expected_output, map_free_vars=True)
    check_basic_block_normal_form(bblock)


# make sure we do not infinite loop.
# it is too large so we won't check for the exact program.
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
    cond = relay.equal(n, relay.const(0, "int64"))
    false_branch = m + f(n - relay.const(1, "int64"))
    funcbody = relay.If(cond, m, false_branch)
    value = relay.Function([n], funcbody, i64, [])
    mod[f] = value
    check_eval(f(relay.const(5, "int64")), 30.0, mod=mod)
    old_f = mod[f]
    mod = transform.ToBasicBlockNormalForm()(mod)
    f = mod[f]
    check_eval(f(relay.const(5, "int64")), 30.0, mod=mod)
    check_basic_block_normal_form(f)


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
    opt_body = run_opt_pass(body, transform.ToBasicBlockNormalForm())
    check_eval(opt_body, 3)
    check_basic_block_normal_form(opt_body)


def test_nat_add():
    mod = tvm.IRModule()
    p = Prelude(mod)
    p.mod.import_from_std("nat.rly")
    nat, z, s = p.mod.get_type("nat")
    add = p.mod.get_global_var("nat_add")
    dev = tvm.device("llvm", 0)
    assert mod[add].checked_type == relay.FuncType([nat(), nat()], nat())
    assert (
        count(p, create_executor(mod=mod, device=dev, target="llvm").evaluate(add(s(z()), s(z()))))
        == 2
    )
    expr = add(s(z()), s(z()))
    f = relay.GlobalVar("f")
    mod[f] = relay.Function([], expr)
    mod = transform.InferType()(mod)
    mod = transform.ToBasicBlockNormalForm()(mod)
    opt_expr = mod["f"]
    assert (
        count(p, create_executor(mod=mod, device=dev, target="llvm").evaluate(opt_expr.body)) == 2
    )
    assert not Feature.fLet in detect_feature(mod[add])
    check_basic_block_normal_form(opt_expr)


def test_let():
    def test_let1():
        x = relay.Var("x")
        c = relay.const(4.0, "float32")
        body = relay.Let(x, c, x)
        body = run_opt_pass(body, transform.InferType())
        """
        let %x: float32 = 4f /* ty=float32 */;
        %x
        """
        opt_body = run_opt_pass(body, transform.ToBasicBlockNormalForm())
        tvm.ir.assert_structural_equal(body, opt_body)
        check_basic_block_normal_form(opt_body)

    def test_let1_1():
        x = relay.Var("y")
        d = relay.const(4.0, "float32")
        body = relay.Let(x, d, relay.add(x, x))
        body = run_opt_pass(body, transform.InferType())
        opt_body = run_opt_pass(body, transform.ToBasicBlockNormalForm())
        tvm.ir.assert_structural_equal(body, opt_body)
        check_basic_block_normal_form(opt_body)

    def test_let2():
        x = relay.Var("x")
        y = relay.Var("y")
        d = relay.const(4.0, "float32")
        body = relay.Let(y, x, x)
        body = relay.Let(x, d, body)
        body = run_opt_pass(body, transform.InferType())
        check_eval(body, 4)

        def expected():
            x = relay.Var("x")
            y = relay.Var("y")
            d = relay.const(4.0, "float32")
            body = relay.Let(y, x, y)
            body = relay.Let(x, d, body)
            return body

        opt_body = run_opt_pass(body, transform.ToBasicBlockNormalForm())
        expected_body = run_opt_pass(expected(), transform.InferType())
        tvm.ir.assert_structural_equal(opt_body, expected_body)
        check_basic_block_normal_form(opt_body)

    def test_let3():
        x = relay.Var("x")
        y = relay.Var("y")
        z = relay.Var("z")
        c = relay.const(3.0, "float32")
        d = relay.const(4.0, "float32")
        body = relay.Let(z, x + y, x + z)
        body = relay.Let(x, d, body)
        body = relay.Let(y, c, body)
        body = run_opt_pass(body, transform.InferType())
        opt_body = run_opt_pass(body, transform.ToBasicBlockNormalForm())
        tvm.ir.assert_structural_equal(body, opt_body)
        check_basic_block_normal_form(opt_body)

    test_let1()
    test_let1_1()
    test_let2()
    test_let3()


def test_function():
    t = relay.TensorType((), "float32")
    x = relay.Var("x", t)
    f = relay.Function([x], x + x)
    d = relay.const(4.0, "float32")
    bblock = run_opt_pass(f, transform.ToBasicBlockNormalForm())
    assert isinstance(bblock, relay.Function)
    check_eval(f(d), 8)
    check_eval(bblock(d), 8)
    check_basic_block_normal_form(bblock)


def test_gradient_if():
    x = relay.var("a", shape=(1, 16))
    y = relay.var("y", shape=(1, 16))
    cond = relay.var("cond", shape=(), dtype="uint1")
    net = relay.If(cond, x, x)
    net = relay.add(x, net)
    net = relay.Function([cond, x, y], net)
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.ToBasicBlockNormalForm()(mod)
    mod = relay.transform.InferType()(mod)
    net_grad = relay.transform.gradient(mod["main"], mode="higher_order")
    mod["main"] = net_grad
    mod_grad = relay.transform.ToBasicBlockNormalForm()(mod)
    check_basic_block_normal_form(mod_grad["main"])
    check_basic_block_normal_form(mod["main"])


def test_if():
    def if_expr(x):
        """
        free_var %x: float32
        %0 = equal(%x, 2f);
        if (%0) {
          %1 = add(%x, 1f);
          multiply(%1, 2f)
        } else {
          multiply(%1, 1f)
        }
        """
        one = relay.const(1, dtype="float32")
        two = relay.const(2, dtype="float32")
        v1 = relay.add(x, one)
        v2 = relay.equal(x, two)
        true_branch = relay.multiply(v1, two)
        false_branch = relay.multiply(v1, one)
        body = relay.If(v2, true_branch, false_branch)
        return body

    def expected_if_expr(x):
        """
        free_var %x: float32
        let %v1: float32 = add(%x, 1f /* ty=float32 */) /* ty=float32 */;
        %0 = equal(%x, 2f /* ty=float32 */) /* ty=bool */;
        if (%0) {
          multiply(%v1, 2f /* ty=float32 */) /* ty=float32 */
        } else {
          multiply(%v1, 1f /* ty=float32 */) /* ty=float32 */
        }
        """
        one = relay.const(1, dtype="float32")
        two = relay.const(2, dtype="float32")
        v1 = relay.var("v1")
        v2 = relay.equal(x, two)
        true_branch = relay.multiply(v1, two)
        false_branch = relay.multiply(v1, one)
        body = relay.If(v2, true_branch, false_branch)
        body = relay.Let(v1, relay.add(x, one), body)
        return body

    x = relay.var("x", shape=(), dtype="float32")
    body = if_expr(x)
    expected_body = expected_if_expr(x)
    bblock = run_opt_pass(body, [transform.ToBasicBlockNormalForm(), transform.InferType()])
    expected_bblock = run_opt_pass(expected_body, transform.InferType())
    tvm.ir.assert_structural_equal(bblock, expected_bblock, map_free_vars=True)
    check_basic_block_normal_form(bblock)

    func = relay.Function([x], body)
    expected_func = relay.Function([x], expected_body)
    bblock = run_opt_pass(func, [transform.ToBasicBlockNormalForm(), transform.InferType()])
    expected_bblock = run_opt_pass(expected_func, transform.InferType())
    tvm.ir.assert_structural_equal(bblock, expected_bblock)
    check_basic_block_normal_form(bblock)


def test_higher_order_return():
    x = relay.var("x", shape=(1,), dtype="float32")  # , a)
    y = relay.var("y", shape=(1,), dtype="float32")  # , a)
    z = relay.var("z", shape=(1,), dtype="float32")  # , a)
    x2 = relay.add(x, x)
    func_a = relay.Function([y], relay.add(x2, y))  # , a, [a])
    func_b = relay.Function([z], relay.add(x2, z))  # , a, [a])
    body = relay.Tuple([func_a, func_b])
    body = relay.Function([x], body)
    """
    fn (%x: Tensor[(1), float32]) {
      %1 = fn (%y: Tensor[(1), float32]) {
        %0 = add(%x, %x);
        add(%0, %y)
      };
      %2 = fn (%z: Tensor[(1), float32]) {
        add(%0, %z)
      };
      (%1, %2)
    }
    """

    bblock = run_opt_pass(body, transform.ToBasicBlockNormalForm())
    check_basic_block_normal_form(bblock)


def test_higher_order_nested():
    x = relay.var("x", dtype="float32", shape=(1,))
    s = relay.var("s", dtype="float32", shape=(1,))
    shared = relay.add(s, s)
    func_true = relay.Function([x], relay.add(x, shared))
    choice_t = relay.FuncType([], relay.scalar_type("bool"))
    f = relay.Var("f", choice_t)
    z = relay.Var("z")
    body = relay.If(f(), func_true, relay.Function([z], relay.add(z, shared)))
    top = relay.Function([f, s], body)
    """
    fn (%f: fn () -> bool, %s: Tensor[(1), float32]) {
      %0 = %f();
      if (%0) {
        fn (%x: Tensor[(1), float32]) {
          %1 = add(%s, %s);
          add(%x, %1)
        }
      } else {
        fn (%z) {
          add(%z, %1)
        }
      }
    }
    """

    bblock = run_opt_pass(top, transform.ToBasicBlockNormalForm())
    check_basic_block_normal_form(bblock)


def test_immutability():
    simple_net = relay.nn.conv2d(
        data=relay.var("data", relay.TensorType((1, 3, 224, 224), "float32")),
        weight=relay.var("weight"),
        kernel_size=(5, 5),
        channels=3,
        padding=(1, 1),
    )
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    mod, _ = create_workload(simple_net)

    old_mod = mod

    with tvm.transform.PassContext(opt_level=4):
        with tvm.target.Target("llvm"):
            seq = tvm.transform.Sequential(passes=[transform.ToBasicBlockNormalForm()], opt_level=4)
            new_mod = seq(mod)

    assert old_mod.astext() == mod.astext()
    assert old_mod.astext() != new_mod.astext()


if __name__ == "__main__":
    tvm.testing.main()
