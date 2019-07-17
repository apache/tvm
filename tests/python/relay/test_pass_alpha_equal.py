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
from tvm.relay import analysis

def alpha_equal(x, y):
    """
    Wrapper around alpha equality which ensures that
    the hash function respects equality.
    """
    return analysis.alpha_equal(x, y) and analysis.structural_hash(x) == analysis.structural_hash(y)

def test_tensor_type_alpha_equal():
    t1 = relay.TensorType((3, 4), "float32")
    t2 = relay.TensorType((3, 4), "float32")
    t3 = relay.TensorType((3, 4, 5), "float32")
    assert t1 == t2
    assert t1 != t3

    t1 = relay.TensorType((), "float32")
    t2 = relay.TensorType((), "float32")
    assert t1 == t2


def test_incomplete_type_alpha_equal():
    t1 = relay.IncompleteType(relay.Kind.Shape)
    t2 = relay.IncompleteType(relay.Kind.Type)
    t3 = relay.IncompleteType(relay.Kind.Type)

    # only equal when there is pointer equality
    assert t2 == t2
    assert t1 == t1
    assert t1 != t2
    assert t2 != t3


def test_type_param_alpha_equal():
    t1 = relay.TypeVar("v1", relay.Kind.Type)
    t2 = relay.TypeVar("v2", relay.Kind.Shape)
    t3 = relay.TypeVar("v3", relay.Kind.Type)

    # only pointer equality and eq_map allow equal params
    assert t1 == t1
    assert t2 == t2
    assert t1 != t2 # different kind
    assert t1 != t3 # not in eq_map

    # function types are the only way to put type params
    # in eq map
    ft1 = relay.FuncType(tvm.convert([]), t1, tvm.convert([t1]), tvm.convert([]))
    ft2 = relay.FuncType(tvm.convert([]), t3, tvm.convert([t3]), tvm.convert([]))
    # actually an invalid type because t2 is wrong kind
    ft3 = relay.FuncType(tvm.convert([]), t2, tvm.convert([t2]), tvm.convert([]))

    assert ft1 == ft2
    assert ft1 != ft3 # kinds still do not match


def test_func_type_alpha_equal():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")

    tp1 = relay.TypeVar("v1", relay.Kind.Type)
    tp2 = relay.TypeVar("v2", relay.Kind.Type)
    tp3 = relay.TypeVar("v3", relay.Kind.Shape)
    tp4 = relay.TypeVar("v3", relay.Kind.Shape)

    broadcast = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    identity = tvm.get_env_func("tvm.relay.type_relation.Identity")

    tr1 = relay.TypeRelation(broadcast, tvm.convert([tp1, tp3]), 1, None)
    tr2 = relay.TypeRelation(broadcast, tvm.convert([tp2, tp4]), 1, None)
    tr3 = relay.TypeRelation(identity, tvm.convert([tp1, tp3]), 1, None)

    ft = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    translate_vars = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp2, tp4]),
                         tvm.convert([tr2]))
    assert ft == translate_vars

    different_args = relay.FuncType(tvm.convert([t1]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    assert ft != different_args

    different_order = relay.FuncType(tvm.convert([t2, t1]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([tr1]))
    assert ft != different_order

    no_rel = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp3]),
                         tvm.convert([]))
    assert ft != no_rel

    more_vars = relay.FuncType(tvm.convert([t1, t2]), tp2,
                         tvm.convert([tp1, tp2, tp3]),
                         tvm.convert([tr1]))
    assert ft != more_vars

    all_the_vars = relay.FuncType(tvm.convert([t1, t2]), tp1,
                         tvm.convert([tp1, tp2, tp3, tp4]),
                         tvm.convert([tr1, tr2]))
    assert ft != all_the_vars

    different_rel = relay.FuncType(tvm.convert([t1, t2]), tp1,
                                   tvm.convert([tp1, tp3]),
                                   tvm.convert([tr3]))
    assert ft != different_rel

    more_rels = relay.FuncType(tvm.convert([t1, t2]), tp1,
                                   tvm.convert([tp1, tp3]),
                                   tvm.convert([tr1, tr3]))
    assert ft != more_rels


def test_tuple_type_alpha_equal():
    t1 = relay.TensorType((1, 2, 3), "float32")
    t2 = relay.TensorType((1, 2, 3, 4), "float32")
    tp1 = relay.TypeVar("v1", relay.Kind.Type)
    tp2 = relay.TypeVar("v2", relay.Kind.Type)

    tup1 = relay.TupleType(tvm.convert([t1, t2, tp1]))
    tup2 = relay.TupleType(tvm.convert([t1, t2, tp1]))
    tup3 = relay.TupleType(tvm.convert([t2, t1, tp1]))
    tup4 = relay.TupleType(tvm.convert([t1, t2, tp2]))

    # as long as types are alpha-equal and in same order,
    # tuples should be alpha-equal
    assert tup1 == tup2
    assert tup1 != tup3
    assert tup1 != tup4


def test_type_relation_alpha_equal():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")
    t3 = relay.TensorType((1, 2, 3, 4), "float32")

    # functions are compared only by pointer equality so
    # we need to be sure to use the same pointers
    broadcast = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    identity = tvm.get_env_func("tvm.relay.type_relation.Identity")

    attr1 = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr1_same = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr2 = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4,4))

    tr = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr1)
    same = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr1)
    diff_func = relay.TypeRelation(identity, tvm.convert([t1, t2]), 1, attr1)
    diff_order = relay.TypeRelation(broadcast, tvm.convert([t2, t1]), 1, attr1)
    diff_args = relay.TypeRelation(broadcast, tvm.convert([t2, t3]), 1, attr1)
    diff_attr = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr2)
    same_attr = relay.TypeRelation(broadcast, tvm.convert([t1, t2]), 1, attr1_same)

    bigger = relay.TypeRelation(identity, tvm.convert([t1, t3, t2]), 2, attr1)
    diff_num_inputs = relay.TypeRelation(identity, tvm.convert([t1, t3, t2]), 1, attr2)

    # func, number of args, input count, and order should be the same
    assert tr == same
    assert tr != diff_func
    assert tr != diff_order
    assert tr != diff_args
    assert tr != diff_attr
    assert tr == same_attr
    assert tr != bigger

    assert bigger != diff_num_inputs

def test_type_call_alpha_equal():
    h1 = relay.GlobalTypeVar("h1")
    h2 = relay.GlobalTypeVar("h2")
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")
    t3 = relay.TensorType((1, 2, 3, 4), "float32")
    t4 = relay.TensorType((), "float32")

    tc = relay.TypeCall(h1, [t1, t2, t3])
    same = relay.TypeCall(h1, [t1, t2, t3])

    different_func = relay.TypeCall(h2, [t1, t2, t3])
    different_arg = relay.TypeCall(h1, [t1, t2, t4])
    fewer_args = relay.TypeCall(h1, [t1, t2])
    more_args = relay.TypeCall(h1, [t1, t2, t3, t4])
    different_order_args = relay.TypeCall(h1, [t3, t2, t1])

    assert tc == same
    assert tc != different_func
    assert tc != fewer_args
    assert tc != more_args
    assert tc != different_order_args


def test_constant_alpha_equal():
    x = relay.const(1)
    y = relay.const(2)
    assert alpha_equal(x, x)
    assert not alpha_equal(x, y)
    assert alpha_equal(x, relay.const(1))


def test_var_alpha_equal():
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    # normally only pointer equality
    assert alpha_equal(v1, v1)
    assert not alpha_equal(v1, v2)

    # let node allows for setting the eq_map
    l1 = relay.Let(v1, relay.const(1), v1)
    l2 = relay.Let(v2, relay.const(1), v2)
    l3 = relay.Let(v1, relay.const(1), v2)

    assert alpha_equal(l1, l2)
    assert not alpha_equal(l1, l3)

    # type annotations
    tt1 = relay.TensorType([], "int32")
    tt2 = relay.TensorType([], "int32")
    tt3 = relay.TensorType([], "int64")
    v3 = relay.Var("v3", tt1)
    v4 = relay.Var("v4", tt2)
    v5 = relay.Var("v5", tt3)

    l4 = relay.Let(v3, relay.const(1), v3)
    l5 = relay.Let(v4, relay.const(1), v4)
    l6 = relay.Let(v5, relay.const(1), v5)

    # same annotations
    assert alpha_equal(l4, l5)
    # different annotations
    assert not alpha_equal(l4, l6)
    # one null annotation
    assert not alpha_equal(l1, l4)


def test_global_var_alpha_equal():
    v1 = relay.GlobalVar("v1")
    v2 = relay.GlobalVar("v2")

    # only pointer equality suffices (smoke test)
    assert alpha_equal(v1, v1)
    assert not alpha_equal(v1, v2)


def test_tuple_alpha_equal():
    v0 = relay.Var("v0")
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    # unit value is a valid tuple
    assert alpha_equal(relay.Tuple([]), relay.Tuple([]))

    tup = relay.Tuple([v0, relay.const(2), relay.const(3), relay.Tuple([relay.const(4)])])
    same = relay.Tuple([v0, relay.const(2), relay.const(3), relay.Tuple([relay.const(4)])])

    assert alpha_equal(tup, same)

    # use the eq_map


    let_tup = relay.Let(v1, tup, v1)
    let_mapped = relay.Let(v2, relay.Tuple([v0, relay.const(2), relay.const(3),
                                            relay.Tuple([relay.const(4)])]),
                           v2)

    assert alpha_equal(let_tup, let_mapped)

    more_fields = relay.Tuple([v1, relay.const(2), relay.const(3), relay.Tuple([relay.const(4)]), v2])
    assert not alpha_equal(tup, more_fields)

    fewer_fields = relay.Tuple([v1, relay.const(2), relay.const(3)])
    assert not alpha_equal(tup, fewer_fields)

    different_end = relay.Tuple([v1, relay.const(2), relay.const(3),
                           relay.Tuple([relay.const(5)])])
    assert not alpha_equal(tup, different_end)

    different_start = relay.Tuple([v2, relay.const(2), relay.const(3),
                                 relay.Tuple([relay.const(4)])])
    assert not alpha_equal(tup, different_start)

    longer_at_end = relay.Tuple([v1, relay.const(2), relay.const(3),
                                 relay.Tuple([relay.const(4), relay.const(5)])])
    assert not alpha_equal(tup, longer_at_end)


def test_tuple_get_item_alpha_equal():
    x = relay.Var('x')
    y = relay.Var('y')
    assert not alpha_equal(relay.TupleGetItem(x, 1), relay.TupleGetItem(y, 1))
    assert not alpha_equal(relay.TupleGetItem(x, 1), relay.TupleGetItem(x, 2))
    assert alpha_equal(relay.TupleGetItem(x, 1), relay.TupleGetItem(x, 1))


def test_function_alpha_equal():
    tt1 = relay.TensorType((1, 2, 3), "float32")
    tt2 = relay.TensorType((4, 5, 6), "int8")
    tt3 = relay.TupleType([tt1, tt2])

    v1 = relay.Var("v1", tt1)
    v2 = relay.Var("v2", tt2)
    v3 = relay.Var("v3", tt3)
    v4 = relay.Var("v4", tt2)
    vret = relay.Constant(tvm.nd.array(np.ones(1)))

    tp1 = relay.TypeVar("tp1", relay.Kind.Type)
    tp2 = relay.TypeVar("tp2", relay.Kind.Type)
    tp3 = relay.TypeVar("tp3", relay.Kind.Shape)
    tp4 = relay.TypeVar("tp4", relay.Kind.Shape)

    basic_args = [relay.Var("v3", tt1), relay.Var("v4", tt2)]
    basic_tps = [tp1, tp2]

    func = relay.Function([v1, v2], v1,
                          tt2, basic_tps)
    mapped = relay.Function(basic_args, basic_args[0], tt2, basic_tps)
    assert alpha_equal(func, mapped)

    fewer_params = relay.Function([relay.Var("v4", tt2)], v4, tt2, basic_tps)
    assert not alpha_equal(func, fewer_params)

    more_params = relay.Function([relay.Var("v3", tt1),
                                  relay.Var("v4", tt2),
                                  relay.Var("v2", tt2)], v4, tt2, basic_tps)
    assert not alpha_equal(func, more_params)

    params_unordered = relay.Function([v2, v1], v1,
                                      tt2, basic_tps)
    assert not alpha_equal(func, params_unordered)

    params_mismatch = relay.Function([v1, v3], v1,
                                     tt2, basic_tps)
    assert not alpha_equal(func, params_mismatch)

    # also would not typecheck
    ret_type_mismatch = relay.Function(basic_args, v4, tt1, basic_tps)
    assert not alpha_equal(func, ret_type_mismatch)

    # also mis-typed
    different_body = relay.Function(basic_args, v3, tt2, basic_tps)
    assert not alpha_equal(func, different_body)

    fewer_type_params = relay.Function(basic_args, v4, tt2, [tp1])
    assert not alpha_equal(func, fewer_type_params)

    more_type_params = relay.Function(basic_args, v4, tt2, [tp1, tp2, tp3])
    assert not alpha_equal(func, more_type_params)

    type_params_unordered = relay.Function(basic_args, v4, tt2, [tp2, tp1])
    assert not alpha_equal(func, type_params_unordered)

    different_type_params = relay.Function(basic_args, v4, tt2, [tp3, tp4])
    assert not alpha_equal(func, different_type_params)

    # a well-typed example that also differs in body, ret type, and type params
    tupled_example = relay.Function(basic_args, relay.Tuple([v3, v4]), tt3)
    assert not alpha_equal(func, tupled_example)

    # nullable
    no_ret_type = relay.Function(basic_args, v4, None, [tp1, tp2])
    # both null
    assert alpha_equal(no_ret_type, no_ret_type)
    # one null
    assert not alpha_equal(func, no_ret_type)
    assert not alpha_equal(no_ret_type, func)


def test_call_alpha_equal():
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    attr1 = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr1_same = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr2 = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4,4))

    tt1 = relay.TensorType((1, 2, 3), "float32")
    tt2 = relay.TensorType((), "int8")

    basic_args = [relay.const(1), relay.const(2), v2, relay.Tuple([])]

    # manually writing out args to ensure that args does not rely on
    # pointer equality
    call = relay.Call(v1, [relay.const(1), relay.const(2), v2, relay.Tuple([])],
                      attr1, [tt1])
    same = relay.Call(v1, basic_args, attr1, [tt1])
    assert alpha_equal(call, same)

    different_fn = relay.Call(v2, basic_args, attr1, [tt1])
    assert not alpha_equal(call, different_fn)

    fewer_args = relay.Call(v1, [relay.const(1), relay.const(2), v2], attr1, [tt1])
    assert not alpha_equal(call, fewer_args)

    reordered_args = relay.Call(v1, [relay.const(2), relay.const(1),
                                     relay.Tuple([]), v2], attr1, [tt1])
    assert not alpha_equal(call, reordered_args)

    different_args = relay.Call(v1, [relay.const(1), relay.const(2), relay.const(3)],
                                attr1, [tt1])
    assert not alpha_equal(call, different_args)

    more_args = relay.Call(v1, [relay.const(1), relay.const(2), v2, relay.Tuple([]),
                                relay.const(3), relay.const(4)], attr1, [tt1])
    assert not alpha_equal(call, more_args)

    different_attrs = relay.Call(v1, basic_args, attr2, [tt1])
    assert not alpha_equal(call, different_attrs)

    same_attrs = relay.Call(v1, basic_args, attr1_same, [tt1])
    assert alpha_equal(call, same_attrs)

    no_type_args = relay.Call(v1, basic_args, attr1)
    assert not alpha_equal(call, no_type_args)

    more_type_args = relay.Call(v1, basic_args, attr1, [tt1, tt2])
    assert not alpha_equal(call, more_type_args)

    different_type_arg = relay.Call(v1, basic_args, attr1, [tt2])
    assert not alpha_equal(call, different_type_arg)


def test_let_alpha_equal():
    tt1 = relay.TensorType((), "float32")
    tt2 = relay.TensorType((), "int8")
    v1 = relay.Var("v1")
    v1_wtype = relay.Var("v1", tt1)
    v2 = relay.Var("v2")
    v3 = relay.Var("v3")

    let = relay.Let(v1, relay.const(2), v1)
    mapped = relay.Let(v2, relay.const(2), v2)
    assert alpha_equal(let, mapped)

    mismatched_var = relay.Let(v2, relay.const(2), v3)
    assert not alpha_equal(let, mismatched_var)

    different_value = relay.Let(v2, relay.const(3), v2)
    assert not alpha_equal(let, different_value)

    different_body = relay.Let(v2, relay.const(3), relay.const(12))
    assert not alpha_equal(let, different_body)

    # specified types must match

    let_with_type = relay.Let(v1_wtype, relay.const(2), v1_wtype)
    same_type = relay.Let(v1_wtype, relay.const(2), v1_wtype)
    assert alpha_equal(let_with_type, same_type)
    assert not alpha_equal(let, let_with_type)
    v2 = relay.Var("v1", tt2)
    different_type = relay.Let(v2, relay.const(2), v2)
    assert not alpha_equal(let_with_type, different_type)


def test_if_alpha_equal():
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    if_sample = relay.If(v1, relay.const(1), relay.Tuple([relay.const(2), relay.const(3)]))
    same = relay.If(v1, relay.const(1), relay.Tuple([relay.const(2), relay.const(3)]))
    assert alpha_equal(if_sample, same)

    different_cond = relay.If(v2, relay.const(1), relay.Tuple([relay.const(2), relay.const(3)]))
    assert not alpha_equal(if_sample, different_cond)

    different_true = relay.If(v1, relay.const(2), relay.Tuple([relay.const(2), relay.const(3)]))
    assert not alpha_equal(if_sample, different_true)

    different_false = relay.If(v1, relay.const(1), relay.Tuple([]))
    assert not alpha_equal(if_sample, different_false)


def test_constructor_alpha_equal():
    # smoke test: it should be pointer equality
    mod = relay.Module()
    p = relay.prelude.Prelude(mod)

    assert alpha_equal(p.nil, p.nil)
    assert alpha_equal(p.cons, p.cons)
    assert not alpha_equal(p.nil, p.cons)


def test_match_alpha_equal():
    mod = relay.Module()
    p = relay.prelude.Prelude(mod)

    x = relay.Var('x')
    y = relay.Var('y')
    nil_case = relay.Clause(relay.PatternConstructor(p.nil), p.nil())
    cons_case = relay.Clause(relay.PatternConstructor(p.cons,
                                                      [relay.PatternVar(x),
                                                       relay.PatternVar(y)]),
                       p.cons(x, y))

    z = relay.Var('z')
    a = relay.Var('a')
    equivalent_cons = relay.Clause(relay.PatternConstructor(p.cons,
                                                            [relay.PatternVar(z),
                                                             relay.PatternVar(a)]),
                                   p.cons(z, a))

    data = p.cons(relay.const(1), p.cons(relay.const(2), p.nil()))

    match = relay.Match(data, [nil_case, cons_case])
    equivalent = relay.Match(data, [nil_case, equivalent_cons])
    empty = relay.Match(data, [])
    no_cons = relay.Match(data, [nil_case])
    no_nil = relay.Match(data, [cons_case])
    different_data = relay.Match(p.nil(), [nil_case, cons_case])
    different_order = relay.Match(data, [cons_case, nil_case])
    different_nil = relay.Match(data, [
        relay.Clause(relay.PatternConstructor(p.nil), p.cons(p.nil(), p.nil())),
        cons_case
    ])
    different_cons = relay.Match(data, [
        nil_case,
        relay.Clause(relay.PatternConstructor(p.cons,
                                              [relay.PatternWildcard(),
                                               relay.PatternWildcard()]),
                     p.nil())
    ])
    another_case = relay.Match(data, [
        nil_case,
        cons_case,
        relay.Clause(relay.PatternWildcard(), p.nil())
    ])
    wrong_constructors = relay.Match(data, [
        relay.Clause(relay.PatternConstructor(p.none), p.nil()),
        relay.Clause(relay.PatternConstructor(p.some, [relay.PatternVar(x)]),
                     p.cons(x, p.nil()))
    ])

    assert alpha_equal(match, match)
    assert alpha_equal(match, equivalent)
    assert not alpha_equal(match, no_cons)
    assert not alpha_equal(match, no_nil)
    assert not alpha_equal(match, empty)
    assert not alpha_equal(match, different_data)
    assert not alpha_equal(match, different_order)
    assert not alpha_equal(match, different_nil)
    assert not alpha_equal(match, different_cons)
    assert not alpha_equal(match, another_case)
    assert not alpha_equal(match, wrong_constructors)


def test_op_alpha_equal():
    # only checks names
    op1 = relay.op.get("add")
    op2 = relay.op.get("add")
    assert alpha_equal(op1, op2)

    op3 = relay.op.get("take")
    assert not alpha_equal(op1, op3)


def test_graph_equal():
    x = relay.var("x")

    y0 = relay.add(x, x)
    z0 = relay.add(y0, y0)

    y1 = relay.add(x, x)
    z1 = relay.add(y1, y1)

    z3 = relay.add(relay.add(x, x), relay.add(x, x))

    assert alpha_equal(z0, z1)

    # z3's dataflow format is different from z0
    # z0 is computed from a common y0 node
    # Relay view them as different programs
    # Check the difference in the text format.
    assert not alpha_equal(z0, z3)

def test_hash_unequal():
    x1 = relay.var("x1", shape=(10, 10), dtype="float32")
    y1 = relay.var("y1", shape=(10, 10), dtype="float32")
    func1 = relay.Function([x1, y1], relay.add(x1, y1))

    # func2 is exactly same structure with same variables shapes and dtypes
    x2 = relay.var("x2", shape=(10, 10), dtype="float32")
    y2 = relay.var("y2", shape=(10, 10), dtype="float32")
    func2 = relay.Function([x2, y2], relay.add(x2, y2))

    assert analysis.structural_hash(func1) == analysis.structural_hash(func2)

    # func3 is same as func1 but with different var shapes
    x3 = relay.var("x3", shape=(20, 10), dtype="float32")
    y3 = relay.var("y3", shape=(20, 10), dtype="float32")
    func3 = relay.Function([x3, y3], relay.add(x3, y3))

    assert not analysis.structural_hash(func1) == analysis.structural_hash(func3)

if __name__ == "__main__":
    test_tensor_type_alpha_equal()
    test_incomplete_type_alpha_equal()
    test_constant_alpha_equal()
    test_func_type_alpha_equal()
    test_tuple_type_alpha_equal()
    test_type_relation_alpha_equal()
    test_type_call_alpha_equal()
    test_constant_alpha_equal()
    test_global_var_alpha_equal()
    test_tuple_alpha_equal()
    test_tuple_get_item_alpha_equal()
    test_function_alpha_equal()
    test_call_alpha_equal()
    test_let_alpha_equal()
    test_if_alpha_equal()
    test_constructor_alpha_equal()
    test_match_alpha_equal()
    test_op_alpha_equal()
    test_var_alpha_equal()
    test_graph_equal()
    test_hash_unequal()
