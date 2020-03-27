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
from tvm.relay import analysis
from tvm.relay.testing import run_opt_pass

def sequal(x, y):
    """
    Wrapper around alpha equality which ensures that
    the hash function respects equality.
    """
    return (tvm.ir.structural_equal(x, y) and
            analysis.structural_hash(x) == analysis.structural_hash(y))

def sequal_commutative(x, y):
    """
    Check for commutative property of equality
    """
    xy = tvm.ir.structural_equal(x, y)
    yx = tvm.ir.structural_equal(y, x)
    assert xy == yx
    return xy

def test_tensor_type_sequal():
    t1 = relay.TensorType((3, 4), "float32")
    t2 = relay.TensorType((3, 4), "float32")
    t3 = relay.TensorType((3, 4, 5), "float32")
    assert t1 == t2
    assert t1 != t3

    t1 = relay.TensorType((), "float32")
    t2 = relay.TensorType((), "float32")
    assert t1 == t2


def test_incomplete_type_sequal():
    t1 = relay.IncompleteType(relay.TypeKind.ShapeVar)
    t2 = relay.IncompleteType(relay.TypeKind.Type)
    t3 = relay.IncompleteType(relay.TypeKind.Type)

    # only equal when there is pointer equality
    assert t2 == t2
    assert t1 == t1
    assert t1 != t2
    assert t2 != t3


def test_type_param_sequal():
    t1 = relay.TypeVar("v1", relay.TypeKind.Type)
    t2 = relay.TypeVar("v2", relay.TypeKind.ShapeVar)
    t3 = relay.TypeVar("v3", relay.TypeKind.Type)

    # only pointer equality and eq_map allow equal params
    assert t1 == t1
    assert t2 == t2
    assert t1 != t2 # different kind
    assert t1 != t3 # not in eq_map

    # function types are the only way to put type params
    # in eq map
    ft1 = relay.FuncType(tvm.runtime.convert([]), t1, tvm.runtime.convert([t1]), tvm.runtime.convert([]))
    ft2 = relay.FuncType(tvm.runtime.convert([]), t3, tvm.runtime.convert([t3]), tvm.runtime.convert([]))
    # actually an invalid type because t2 is wrong kind
    ft3 = relay.FuncType(tvm.runtime.convert([]), t2, tvm.runtime.convert([t2]), tvm.runtime.convert([]))

    assert ft1 == ft2
    assert ft1 != ft3 # kinds still do not match


def test_func_type_sequal():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")

    tp1 = relay.TypeVar("v1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("v2", relay.TypeKind.Type)
    tp3 = relay.TypeVar("v3", relay.TypeKind.ShapeVar)
    tp4 = relay.TypeVar("v3", relay.TypeKind.ShapeVar)

    broadcast = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Broadcast")
    identity = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Identity")

    tr1 = relay.TypeRelation(broadcast, tvm.runtime.convert([tp1, tp3]), 1, None)
    tr2 = relay.TypeRelation(broadcast, tvm.runtime.convert([tp2, tp4]), 1, None)
    tr3 = relay.TypeRelation(identity, tvm.runtime.convert([tp1, tp3]), 1, None)

    ft = relay.FuncType(tvm.runtime.convert([t1, t2]), tp1,
                         tvm.runtime.convert([tp1, tp3]),
                         tvm.runtime.convert([tr1]))
    translate_vars = relay.FuncType(tvm.runtime.convert([t1, t2]), tp1,
                         tvm.runtime.convert([tp2, tp4]),
                         tvm.runtime.convert([tr2]))
    assert ft == translate_vars

    different_args = relay.FuncType(tvm.runtime.convert([t1]), tp1,
                         tvm.runtime.convert([tp1, tp3]),
                         tvm.runtime.convert([tr1]))
    assert ft != different_args

    different_order = relay.FuncType(tvm.runtime.convert([t2, t1]), tp1,
                         tvm.runtime.convert([tp1, tp3]),
                         tvm.runtime.convert([tr1]))
    assert ft != different_order

    no_rel = relay.FuncType(tvm.runtime.convert([t1, t2]), tp1,
                         tvm.runtime.convert([tp1, tp3]),
                         tvm.runtime.convert([]))
    assert ft != no_rel

    more_vars = relay.FuncType(tvm.runtime.convert([t1, t2]), tp2,
                         tvm.runtime.convert([tp1, tp2, tp3]),
                         tvm.runtime.convert([tr1]))
    assert ft != more_vars

    all_the_vars = relay.FuncType(tvm.runtime.convert([t1, t2]), tp1,
                         tvm.runtime.convert([tp1, tp2, tp3, tp4]),
                         tvm.runtime.convert([tr1, tr2]))
    assert ft != all_the_vars

    different_rel = relay.FuncType(tvm.runtime.convert([t1, t2]), tp1,
                                   tvm.runtime.convert([tp1, tp3]),
                                   tvm.runtime.convert([tr3]))
    assert ft != different_rel

    more_rels = relay.FuncType(tvm.runtime.convert([t1, t2]), tp1,
                                   tvm.runtime.convert([tp1, tp3]),
                                   tvm.runtime.convert([tr1, tr3]))
    assert ft != more_rels


def test_tuple_type_sequal():
    t1 = relay.TensorType((1, 2, 3), "float32")
    t2 = relay.TensorType((1, 2, 3, 4), "float32")
    tp1 = relay.TypeVar("v1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("v2", relay.TypeKind.Type)

    tup1 = relay.TupleType(tvm.runtime.convert([t1, t2, tp1]))
    tup2 = relay.TupleType(tvm.runtime.convert([t1, t2, tp1]))
    tup3 = relay.TupleType(tvm.runtime.convert([t2, t1, tp1]))
    tup4 = relay.TupleType(tvm.runtime.convert([t1, t2, tp2]))

    # as long as types are alpha-equal and in same order,
    # tuples should be alpha-equal
    assert tup1 == tup2
    assert tup1 != tup3
    assert tup1 != tup4


def test_type_relation_sequal():
    t1 = relay.TensorType((1, 2), "float32")
    t2 = relay.TensorType((1, 2, 3), "float32")
    t3 = relay.TensorType((1, 2, 3, 4), "float32")

    # functions are compared only by pointer equality so
    # we need to be sure to use the same pointers
    broadcast = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Broadcast")
    identity = tvm.ir.EnvFunc.get("tvm.relay.type_relation.Identity")

    attr1 = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr1_same = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr2 = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3,4,4))

    tr = relay.TypeRelation(broadcast, tvm.runtime.convert([t1, t2]), 1, attr1)
    same = relay.TypeRelation(broadcast, tvm.runtime.convert([t1, t2]), 1, attr1)
    diff_func = relay.TypeRelation(identity, tvm.runtime.convert([t1, t2]), 1, attr1)
    diff_order = relay.TypeRelation(broadcast, tvm.runtime.convert([t2, t1]), 1, attr1)
    diff_args = relay.TypeRelation(broadcast, tvm.runtime.convert([t2, t3]), 1, attr1)
    diff_attr = relay.TypeRelation(broadcast, tvm.runtime.convert([t1, t2]), 1, attr2)
    same_attr = relay.TypeRelation(broadcast, tvm.runtime.convert([t1, t2]), 1, attr1_same)

    bigger = relay.TypeRelation(identity, tvm.runtime.convert([t1, t3, t2]), 2, attr1)
    diff_num_inputs = relay.TypeRelation(identity, tvm.runtime.convert([t1, t3, t2]), 1, attr2)

    # func, number of args, input count, and order should be the same
    assert tr == same
    assert tr != diff_func
    assert tr != diff_order
    assert tr != diff_args
    assert tr != diff_attr
    assert tr == same_attr
    assert tr != bigger

    assert bigger != diff_num_inputs

def test_type_call_sequal():
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


def test_constant_sequal():
    x = relay.const(1)
    y = relay.const(2)
    assert sequal(x, x)
    assert not sequal(x, y)
    assert sequal(x, relay.const(1))

def test_type_node_sequal():
    v1 = relay.TypeVar('v1', 6)
    v2 = relay.TypeVar('v2', 6)
    assert not sequal(v1, v2)

    v1 = relay.TypeVar('v1', 0)
    v2 = relay.TypeVar('v2', 6)
    assert not sequal(v1, v2)

    assert sequal_commutative(v1, v1)

def test_type_node_incompatible_sequal():
    v1 = relay.TypeVar('v1', 6)
    v2 = relay.Var("v2")
    assert not sequal_commutative(v1, v2)

def test_expr_node_incompatible_sequal():
    v1 = relay.Var("v1")
    v2 = relay.PatternVar(relay.Var("v2"))
    assert not sequal_commutative(v1, v2)

def test_var_sequal():
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    # normally only pointer equality
    assert sequal(v1, v1)
    assert not sequal(v1, v2)

    # let node allows for setting the eq_map
    l1 = relay.Let(v1, relay.const(1), v1)
    l2 = relay.Let(v2, relay.const(1), v2)
    l3 = relay.Let(v1, relay.const(1), v2)

    assert sequal(l1, l2)
    assert not sequal(l1, l3)

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
    assert sequal(l4, l5)
    # different annotations
    assert not sequal(l4, l6)
    # one null annotation
    assert not sequal(l1, l4)


def test_global_var_sequal():
    v1 = relay.GlobalVar("v1")
    v2 = relay.GlobalVar("v2")

    # only pointer equality suffices (smoke test)
    assert sequal(v1, v1)
    assert not sequal(v1, v2)


def test_tuple_sequal():
    v0 = relay.Var("v0")
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    # unit value is a valid tuple
    assert sequal(relay.Tuple([]), relay.Tuple([]))

    tup = relay.Tuple([v0, relay.const(2), relay.const(3), relay.Tuple([relay.const(4)])])
    same = relay.Tuple([v0, relay.const(2), relay.const(3), relay.Tuple([relay.const(4)])])

    assert sequal(tup, same)

    # use the eq_map


    let_tup = relay.Let(v1, tup, v1)
    let_mapped = relay.Let(v2, relay.Tuple([v0, relay.const(2), relay.const(3),
                                            relay.Tuple([relay.const(4)])]),
                           v2)

    assert sequal(let_tup, let_mapped)

    more_fields = relay.Tuple([v1, relay.const(2), relay.const(3), relay.Tuple([relay.const(4)]), v2])
    assert not sequal(tup, more_fields)

    fewer_fields = relay.Tuple([v1, relay.const(2), relay.const(3)])
    assert not sequal(tup, fewer_fields)

    different_end = relay.Tuple([v1, relay.const(2), relay.const(3),
                           relay.Tuple([relay.const(5)])])
    assert not sequal(tup, different_end)

    different_start = relay.Tuple([v2, relay.const(2), relay.const(3),
                                 relay.Tuple([relay.const(4)])])
    assert not sequal(tup, different_start)

    longer_at_end = relay.Tuple([v1, relay.const(2), relay.const(3),
                                 relay.Tuple([relay.const(4), relay.const(5)])])
    assert not sequal(tup, longer_at_end)


def test_tuple_get_item_sequal():
    x = relay.Var('x')
    y = relay.Var('y')
    assert not sequal(relay.TupleGetItem(x, 1), relay.TupleGetItem(y, 1))
    assert not sequal(relay.TupleGetItem(x, 1), relay.TupleGetItem(x, 2))
    assert sequal(relay.TupleGetItem(x, 1), relay.TupleGetItem(x, 1))


def test_function_attr():
    x0 = relay.var('x0', shape=(10, 10))
    w00 = relay.var('w00', shape=(10, 10))
    w01 = relay.var('w01', shape=(10, 10))
    w02 = relay.var('w02', shape=(10, 10))
    z00 = relay.add(x0, w00)
    p00 = relay.subtract(z00, w01)
    q00 = relay.multiply(p00, w02)
    func0 = relay.Function([x0, w00, w01, w02], q00)
    func0 = func0.with_attr("FuncName", tvm.tir.StringImm("a"))

    x1 = relay.var('x1', shape=(10, 10))
    w10 = relay.var('w10', shape=(10, 10))
    w11 = relay.var('w11', shape=(10, 10))
    w12 = relay.var('w12', shape=(10, 10))
    z10 = relay.add(x1, w10)
    p10 = relay.subtract(z10, w11)
    q10 = relay.multiply(p10, w12)
    func1 = relay.Function([x1, w10, w11, w12], q10)
    func1 = func1.with_attr("FuncName", tvm.tir.StringImm("b"))
    assert not sequal(func0, func1)


def test_function_sequal():
    tt1 = relay.TensorType((1, 2, 3), "float32")
    tt2 = relay.TensorType((4, 5, 6), "int8")
    tt3 = relay.TupleType([tt1, tt2])

    v1 = relay.Var("v1", tt1)
    v2 = relay.Var("v2", tt2)
    v3 = relay.Var("v3", tt3)
    v4 = relay.Var("v4", tt2)
    vret = relay.Constant(tvm.nd.array(np.ones(1)))

    tp1 = relay.TypeVar("tp1", relay.TypeKind.Type)
    tp2 = relay.TypeVar("tp2", relay.TypeKind.Type)
    tp3 = relay.TypeVar("tp3", relay.TypeKind.ShapeVar)
    tp4 = relay.TypeVar("tp4", relay.TypeKind.ShapeVar)

    basic_args = [relay.Var("v3", tt1), relay.Var("v4", tt2)]
    basic_tps = [tp1, tp2]

    func = relay.Function([v1, v2], v1,
                          tt2, basic_tps)
    mapped = relay.Function(basic_args, basic_args[0], tt2, basic_tps)
    assert sequal(func, mapped)

    fewer_params = relay.Function([relay.Var("v4", tt2)], v4, tt2, basic_tps)
    assert not sequal(func, fewer_params)

    more_params = relay.Function([relay.Var("v3", tt1),
                                  relay.Var("v4", tt2),
                                  relay.Var("v2", tt2)], v4, tt2, basic_tps)
    assert not sequal(func, more_params)

    params_unordered = relay.Function([v2, v1], v1,
                                      tt2, basic_tps)
    assert not sequal(func, params_unordered)

    params_mismatch = relay.Function([v1, v3], v1,
                                     tt2, basic_tps)
    assert not sequal(func, params_mismatch)

    # also would not typecheck
    ret_type_mismatch = relay.Function(basic_args, v4, tt1, basic_tps)
    assert not sequal(func, ret_type_mismatch)

    # also mis-typed
    different_body = relay.Function(basic_args, v3, tt2, basic_tps)
    assert not sequal(func, different_body)

    fewer_type_params = relay.Function(basic_args, v4, tt2, [tp1])
    assert not sequal(func, fewer_type_params)

    more_type_params = relay.Function(basic_args, v4, tt2, [tp1, tp2, tp3])
    assert not sequal(func, more_type_params)

    type_params_unordered = relay.Function(basic_args, v4, tt2, [tp2, tp1])
    assert not sequal(func, type_params_unordered)

    different_type_params = relay.Function(basic_args, v4, tt2, [tp3, tp4])
    assert not sequal(func, different_type_params)

    # a well-typed example that also differs in body, ret type, and type params
    tupled_example = relay.Function(basic_args, relay.Tuple([v3, v4]), tt3)
    assert not sequal(func, tupled_example)

    # nullable
    no_ret_type = relay.Function(basic_args, v4, None, [tp1, tp2])
    # both null
    assert sequal(no_ret_type, no_ret_type)
    # one null
    assert not sequal(func, no_ret_type)
    assert not sequal(no_ret_type, func)


def test_call_sequal():
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    attr1 = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr1_same = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3,4))
    attr2 = tvm.ir.make_node("attrs.TestAttrs", name="attr", padding=(3,4,4))

    tt1 = relay.TensorType((1, 2, 3), "float32")
    tt2 = relay.TensorType((), "int8")

    basic_args = [relay.const(1), relay.const(2), v2, relay.Tuple([])]

    # manually writing out args to ensure that args does not rely on
    # pointer equality
    call = relay.Call(v1, [relay.const(1), relay.const(2), v2, relay.Tuple([])],
                      attr1, [tt1])
    same = relay.Call(v1, basic_args, attr1, [tt1])
    assert sequal(call, same)

    different_fn = relay.Call(v2, basic_args, attr1, [tt1])
    assert not sequal(call, different_fn)

    fewer_args = relay.Call(v1, [relay.const(1), relay.const(2), v2], attr1, [tt1])
    assert not sequal(call, fewer_args)

    reordered_args = relay.Call(v1, [relay.const(2), relay.const(1),
                                     relay.Tuple([]), v2], attr1, [tt1])
    assert not sequal(call, reordered_args)

    different_args = relay.Call(v1, [relay.const(1), relay.const(2), relay.const(3)],
                                attr1, [tt1])
    assert not sequal(call, different_args)

    more_args = relay.Call(v1, [relay.const(1), relay.const(2), v2, relay.Tuple([]),
                                relay.const(3), relay.const(4)], attr1, [tt1])
    assert not sequal(call, more_args)

    different_attrs = relay.Call(v1, basic_args, attr2, [tt1])
    assert not sequal(call, different_attrs)

    same_attrs = relay.Call(v1, basic_args, attr1_same, [tt1])
    assert sequal(call, same_attrs)

    no_type_args = relay.Call(v1, basic_args, attr1)
    assert not sequal(call, no_type_args)

    more_type_args = relay.Call(v1, basic_args, attr1, [tt1, tt2])
    assert not sequal(call, more_type_args)

    different_type_arg = relay.Call(v1, basic_args, attr1, [tt2])
    assert not sequal(call, different_type_arg)


def test_let_sequal():
    tt1 = relay.TensorType((), "float32")
    tt2 = relay.TensorType((), "int8")
    v1 = relay.Var("v1")
    v1_wtype = relay.Var("v1", tt1)
    v2 = relay.Var("v2")
    v3 = relay.Var("v3")

    let = relay.Let(v1, relay.const(2), v1)
    mapped = relay.Let(v2, relay.const(2), v2)
    assert sequal(let, mapped)

    mismatched_var = relay.Let(v2, relay.const(2), v3)
    assert not sequal(let, mismatched_var)

    different_value = relay.Let(v2, relay.const(3), v2)
    assert not sequal(let, different_value)

    different_body = relay.Let(v2, relay.const(3), relay.const(12))
    assert not sequal(let, different_body)

    # specified types must match

    let_with_type = relay.Let(v1_wtype, relay.const(2), v1_wtype)
    same_type = relay.Let(v1_wtype, relay.const(2), v1_wtype)
    assert sequal(let_with_type, same_type)
    assert not sequal(let, let_with_type)
    v2 = relay.Var("v1", tt2)
    different_type = relay.Let(v2, relay.const(2), v2)
    assert not sequal(let_with_type, different_type)


def test_if_sequal():
    v1 = relay.Var("v1")
    v2 = relay.Var("v2")

    if_sample = relay.If(v1, relay.const(1), relay.Tuple([relay.const(2), relay.const(3)]))
    same = relay.If(v1, relay.const(1), relay.Tuple([relay.const(2), relay.const(3)]))
    assert sequal(if_sample, same)

    different_cond = relay.If(v2, relay.const(1), relay.Tuple([relay.const(2), relay.const(3)]))
    assert not sequal(if_sample, different_cond)

    different_true = relay.If(v1, relay.const(2), relay.Tuple([relay.const(2), relay.const(3)]))
    assert not sequal(if_sample, different_true)

    different_false = relay.If(v1, relay.const(1), relay.Tuple([]))
    assert not sequal(if_sample, different_false)


def test_constructor_sequal():
    # smoke test: it should be pointer equality
    mod = tvm.IRModule()
    p = relay.prelude.Prelude(mod)

    assert sequal(p.nil, p.nil)
    assert sequal(p.cons, p.cons)
    assert not sequal(p.nil, p.cons)


def test_match_sequal():
    mod = tvm.IRModule()
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

    tvm.ir.assert_structural_equal(match, match)
    assert sequal(match, match)
    assert sequal(match, equivalent)
    assert not sequal(match, no_cons)
    assert not sequal(match, no_nil)
    assert not sequal(match, empty)
    assert not sequal(match, different_data)
    assert not sequal(match, different_order)
    assert not sequal(match, different_nil)
    assert not sequal(match, different_cons)
    assert not sequal(match, another_case)
    assert not sequal(match, wrong_constructors)


def test_op_sequal():
    # only checks names
    op1 = relay.op.get("add")
    op2 = relay.op.get("add")
    assert sequal(op1, op2)

    op3 = relay.op.get("take")
    assert not sequal(op1, op3)


def test_graph_equal():
    x = relay.var("x")

    y0 = relay.add(x, x)
    z0 = relay.add(y0, y0)

    y1 = relay.add(x, x)
    z1 = relay.add(y1, y1)

    z3 = relay.add(relay.add(x, x), relay.add(x, x))

    assert sequal(z0, z1)
    assert sequal(z0, z1)

    # z3's dataflow format is different from z0
    # z0 is computed from a common y0 node
    # Relay view them as different programs
    # Check the difference in the text format.
    assert not sequal(z0, z3)

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


def test_tuple_match():
    a = relay.Var("a")
    b = relay.Var("b")
    clause = relay.Clause(relay.PatternTuple([relay.PatternVar(a), relay.PatternVar(b)]), a + b)
    x = relay.Match(relay.Tuple([relay.const(1), relay.const(1)]), [clause])

    a = relay.Var("a")
    b = relay.Var("b")
    clause = relay.Clause(relay.PatternTuple([relay.PatternVar(a), relay.PatternVar(b)]), a + b)
    y = relay.Match(relay.Tuple([relay.const(1), relay.const(1)]), [clause])
    assert sequal(x, y)
    assert analysis.structural_hash(x) == analysis.structural_hash(y)


def test_fn_attribute():
    # create function that performs add
    a = relay.var('a', shape=(10, 10))
    b = relay.var('b', shape=(10, 10))
    add = relay.add(a, b)
    add_fn = relay.Function([a, b], add)
    add_fn = run_opt_pass(add_fn, relay.transform.InferType())

    # create function that performs add with test attribute
    c = relay.var('c', shape=(10, 10))
    d = relay.var('d', shape=(10, 10))
    add_1 = relay.add(c, d)
    add_1_fn = relay.Function([c, d], add_1)
    add_1_fn = add_1_fn.with_attr("TestAttribute", tvm.tir.StringImm("test"))
    add_1_fn = run_opt_pass(add_1_fn, relay.transform.InferType())

    assert not sequal(add_1_fn, add_fn)
    assert not sequal(add_fn, add_1_fn)


if __name__ == "__main__":
    test_tensor_type_sequal()
    test_incomplete_type_sequal()
    test_constant_sequal()
    test_type_node_sequal()
    test_type_node_incompatible_sequal()
    test_expr_node_incompatible_sequal()
    test_func_type_sequal()
    test_tuple_type_sequal()
    test_type_relation_sequal()
    test_type_call_sequal()
    test_constant_sequal()
    test_global_var_sequal()
    test_tuple_sequal()
    test_tuple_get_item_sequal()
    test_function_sequal()
    test_function_attr()
    test_call_sequal()
    test_let_sequal()
    test_if_sequal()
    test_constructor_sequal()
    test_match_sequal()
    test_op_sequal()
    test_var_sequal()
    test_graph_equal()
    test_hash_unequal()
    test_fn_attribute()
