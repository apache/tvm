import numpy as np
import tvm
from tvm import relay
from tvm.relay.ir_pass import to_a_normal_form, alpha_equal, infer_type
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue, ConstructorValue
from tvm.relay.prelude import Prelude


def check_eval(expr, expected_result, mod=None, rtol=1e-07):
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def test_explicit_bound():
    x = relay.const(1)
    y = op.add(x, x)
    z = op.add(y, y)
    f = relay.Function([], op.add(z, z))
    assert not "let" in f.astext() # assert the values are implicitly bounded
    anf = to_a_normal_form(f)
    assert "let" in anf.astext() # assert the values are explicitly bounded
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
    anf = infer_type(to_a_normal_form(val))
    a = relay.Var('a', relay.IncompleteType())
    b = relay.Var('b', relay.IncompleteType())
    c = relay.Var('c', relay.IncompleteType())
    d = relay.Var('d', relay.IncompleteType())
    e = relay.Var('e', relay.IncompleteType())
    expected_output = e
    expected_output = relay.Let(e, a + d, expected_output)
    expected_output = relay.Let(d, b * c, expected_output)
    expected_output = relay.Let(c, z, expected_output)
    expected_output = relay.Let(b, y, expected_output)
    expected_output = relay.Let(a, x, expected_output)
    expected_output = infer_type(expected_output)
    assert alpha_equal(anf, expected_output)


def test_if():
    cond = relay.const(True)
    x = relay.If(cond, relay.const(2), relay.const(3))
    anf = infer_type(to_a_normal_form(x))
    a = relay.Var('a', relay.IncompleteType())
    b = relay.Var('b', relay.IncompleteType())
    c = relay.Var('c', relay.IncompleteType())
    d = relay.Var('d', relay.IncompleteType())
    true_branch = relay.Let(a, relay.const(2), a)
    false_branch = relay.Let(b, relay.const(3), b)
    expected_output = relay.If(c, true_branch, false_branch)
    expected_output = relay.Let(d, expected_output, d)
    expected_output = relay.Let(c, cond, expected_output)
    expected_output = infer_type(expected_output)
    assert alpha_equal(anf, expected_output)


# make sure we dont infinite loop.
# it is too large so we wont check for the exact program.
def test_recursion():
    """
    Program:
       let sum_twice(n: i32) -> i32 = {
          m = (n * 2)
          if (n == 0) {
              return m;
          } else {
              return m + sum(n - 1);
          }
       }
       sum_twice(5);
    """
    return # cannot be run as fuse_ops need to recursively visit
    mod = relay.Module()
    i64 = relay.TensorType((), 'int64')
    f = relay.GlobalVar("f")
    n = relay.Var("n", i64)
    m = n * relay.const(2, 'int64')
    funcbody = relay.If(relay.equal(n, relay.const(0, 'int64')),
                        m,
                        m + f(n - relay.const(1, 'int64')))
    value = relay.Function([n], funcbody, i64, [])
    mod[f] = value
    check_eval(f(relay.const(5, 'int64')), 30.0, mod=mod)
    old_f = mod[f]
    f = to_a_normal_form(f, mod=mod)
    check_eval(f(relay.const(5, 'int64')), 30.0, mod=mod)


def test_ref():
    i = relay.Var('i')
    iv = relay.Var('iv')
    u = relay.Var('u')
    uv = relay.Var('uv')
    body = relay.add(iv, uv)
    body = relay.Let(uv, relay.RefRead(i), body)
    body = relay.Let(u, relay.RefWrite(i, relay.const(2)), body)
    body = relay.Let(iv, relay.RefRead(i), body)
    body = relay.Let(i, relay.RefCreate(relay.const(1)), body)
    check_eval(body, 3)
    check_eval(to_a_normal_form(body), 3)


# this is an example of using the adt value in python side
def count(n):
    assert isinstance(n, ConstructorValue)
    if n.constructor.name_hint == 's':
        return 1 + count(n.fields[0])
    else:
        assert n.constructor.name_hint == 'z'
        return 0


def test_add():
    mod = relay.Module()
    p = Prelude(mod)
    nat = p.nat
    add = p.add
    s = p.s
    z = p.z
    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")
    assert mod[add].checked_type == relay.FuncType([nat(), nat()], nat())
    assert count(intrp.evaluate(add(s(z()), s(z())))) == 2
    assert count(intrp.evaluate(to_a_normal_form(add(s(z()), s(z())), mod))) == 2
    assert "let" in mod[add].astext()

def test_let():
    x = relay.Var("x")
    y = relay.Var("y")
    d = relay.const(4.0, 'float32')
    body = relay.Let(y, x, x + y)
    body = relay.Let(x, d, body)
    check_eval(body, 8)
    check_eval(to_a_normal_form(body), 8)

if __name__ == '__main__':
    test_explicit_bound()
    test_order()
    test_if()
    test_recursion()
    test_ref()
    test_add()
    test_let()
