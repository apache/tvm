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
import tvm
from tvm import te
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor


def check_visit(expr):
    try:
        ef = ExprFunctor()
        ef.visit(expr)
        assert False
    except NotImplementedError:
        pass

    ev = ExprVisitor()
    ev.visit(expr)

    em = ExprMutator()
    assert expr == em.visit(expr)


def test_constant():
    check_visit(relay.const(1.0))


def test_tuple():
    t = relay.Tuple([relay.var("x", shape=())])
    check_visit(t)


def test_var():
    v = relay.var("x", shape=())
    check_visit(v)


def test_global():
    v = relay.GlobalVar("f")
    check_visit(v)


def test_function():
    x = relay.var("x", shape=())
    y = relay.var("y", shape=())
    params = [x, y]
    body = x + y
    ret_type = relay.TensorType(())
    type_params = []
    attrs = None  # How to build?
    f = relay.Function(params, body, ret_type, type_params, attrs)
    check_visit(f)


def test_call():
    x = relay.var("x", shape=())
    y = relay.var("y", shape=())
    call = relay.op.add(x, y)
    check_visit(call)


def test_let():
    x = relay.var("x", shape=())
    value = relay.const(2.0)
    body = x + x
    l = relay.Let(x, value, body)
    check_visit(l)


def test_ite():
    cond = relay.var("x", shape=(), dtype="bool")
    ite = relay.If(cond, cond, cond)
    check_visit(ite)


def test_get_item():
    t = relay.Tuple([relay.var("x", shape=())])
    t = relay.TupleGetItem(t, 0)
    check_visit(t)


def test_ref_create():
    r = relay.expr.RefCreate(relay.const(1.0))
    check_visit(r)


def test_ref_read():
    ref = relay.expr.RefCreate(relay.const(1.0))
    r = relay.expr.RefRead(ref)
    check_visit(r)


def test_ref_write():
    ref = relay.expr.RefCreate(relay.const(1.0))
    r = relay.expr.RefWrite(ref, relay.const(2.0))
    check_visit(r)


def test_memo():
    expr = relay.const(1)
    for _ in range(100):
        expr = expr + expr
    check_visit(expr)


def test_match():
    p = relay.prelude.Prelude()
    check_visit(p.mod[p.map])


def test_match_completeness():
    p = relay.prelude.Prelude()
    _, _, nil = p.mod.get_type("List")
    for completeness in [True, False]:
        match_expr = relay.adt.Match(nil, [], complete=completeness)
        result_expr = ExprMutator().visit(match_expr)
        # ensure the mutator doesn't mangle the completeness flag
        assert result_expr.complete == completeness


if __name__ == "__main__":
    test_constant()
    test_tuple()
    test_var()
    test_global()
    test_function()
    test_call()
    test_let()
    test_ite()
    test_ref_create()
    test_ref_read()
    test_ref_write()
    test_memo()
    test_match()
    test_match_completeness()
