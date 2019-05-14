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
from tvm import relay
from tvm.relay.testing import to_python, run_as_python
from tvm.relay.prelude import Prelude
from tvm.relay.backend.interpreter import TensorValue, TupleValue, RefValue, ConstructorValue

# helper: uses a dummy let binding to sequence a list
# of expressions: expr1; expr2; expr3, etc.
def seq(*exprs):
    ret = exprs[0]
    for expr in exprs[1:]:
        ret = relay.Let(relay.var('_'), ret, expr)
    return ret


# creates a dummy ADT for testing
def init_box_adt(mod):
    box = relay.GlobalTypeVar('box')
    a = relay.TypeVar('a')
    box_ctor = relay.Constructor('box', [a], box)
    mod[box] = relay.TypeData(box, [a], [box_ctor])
    return (box, box_ctor)


# assert that the candidate is a TensorValue with value val
def assert_tensor_value(candidate, val):
    assert isinstance(candidate, TensorValue)
    assert candidate.asnumpy() == val


# assert that the candidate is a TupleValue with the indicate number of fields
def assert_tuple_value(candidate, fields):
    assert isinstance(candidate, TupleValue)
    assert len(candidate.fields) == fields


# assert that the candidate is a ConstructorValue with the approrpaite constructor
# and number of fields
def assert_constructor_value(candidate, constructor, fields):
    assert isinstance(candidate, ConstructorValue)
    assert candidate.constructor == constructor
    assert len(candidate.fields) == fields


def test_create_empty_tuple():
    empty = relay.Tuple([])
    tup_val = run_as_python(empty)
    assert_tuple_value(tup_val, 0)


def test_create_scalar():
    scalar = relay.const(1)
    tensor_val = run_as_python(scalar)
    assert_tensor_value(tensor_val, 1)


def test_create_nested_tuple():
    relay_tup = relay.Tuple([
        relay.const(1), relay.const(2),
        relay.Tuple([
            relay.const(3),
            relay.const(4)
        ])
    ])
    tup_val = run_as_python(relay_tup)
    assert_tuple_value(tup_val, 3)
    for i in range(2):
        assert_tensor_value(tup_val.fields[i], i + 1)
    assert_tuple_value(tup_val.fields[2], 2)
    for i in range(2):
        assert_tensor_value(tup_val.fields[2].fields[i], i + 3)


def test_tuple_get_item():
    relay_tup = relay.Tuple([
        relay.const(1), relay.const(2),
        relay.Tuple([
            relay.const(3),
            relay.const(4)
        ])
    ])
    for i in range(2):
        index = relay.TupleGetItem(relay_tup, i)
        val = run_as_python(index)
        assert_tensor_value(val, i + 1)
    # try the inner value too
    for i in range(2):
        index = relay.TupleGetItem(relay.TupleGetItem(relay_tup, 2), i)
        val = run_as_python(index)
        assert_tensor_value(val, i + 3)


def test_create_let():
    v = relay.Var('v')
    let = relay.Let(v, relay.Tuple([]), relay.Tuple([v, v]))
    tup_val = run_as_python(let)
    assert_tuple_value(tup_val, 2)
    assert_tuple_value(tup_val.fields[0], 0)
    assert_tuple_value(tup_val.fields[1], 0)


def test_create_ref():
    relay_ref = relay.RefCreate(relay.Tuple([]))
    ref_val = run_as_python(relay_ref)
    assert isinstance(ref_val, RefValue)
    assert_tuple_value(ref_val.value, 0)


def test_ref_read():
    v = relay.Var('v')
    assign = relay.Let(v, relay.RefCreate(relay.Tuple([])), relay.RefRead(v))
    read_val = run_as_python(assign)
    assert_tuple_value(read_val, 0)


def test_ref_write():
    # check that the result of a ref write is an empty tuple
    v = relay.Var('v')
    initial_write = relay.Let(v, relay.RefCreate(relay.Tuple([relay.const(1)])),
                              relay.RefWrite(v, relay.Tuple([relay.const(2)])))
    write_val = run_as_python(initial_write)
    assert_tuple_value(write_val, 0)

    # now ensure that the value, once written, can be read back
    read_after_write = relay.Let(v, relay.RefCreate(relay.Tuple([relay.const(1)])),
                                 seq(relay.RefWrite(v, relay.Tuple([relay.const(2)])),
                                     relay.RefRead(v)))
    read_val = run_as_python(read_after_write)
    assert_tuple_value(read_val, 1)
    assert_tensor_value(read_val.fields[0], 2)


def test_if():
    # we will have effects in the blocks to ensure only the intended one is executed
    true_cond = relay.const(True)
    false_cond = relay.const(False)

    v  = relay.Var('v')
    true_branch = seq(relay.RefWrite(v, relay.const(1)), relay.RefRead(v))
    false_branch = seq(relay.RefWrite(v, relay.const(2)), relay.RefRead(v))

    true_expr = relay.Let(v, relay.RefCreate(relay.const(0)),
                          relay.If(true_cond, true_branch, false_branch))
    false_expr = relay.Let(v, relay.RefCreate(relay.const(0)),
                           relay.If(false_cond, true_branch, false_branch))

    true_val = run_as_python(true_expr)
    assert_tensor_value(true_val, 1)

    false_val = run_as_python(false_expr)
    assert_tensor_value(false_val, 2)


def test_local_function():
    v = relay.Var('v')
    ident = relay.Function([v], v)
    f = relay.Var('f')
    call1 = relay.Let(f, ident, f(relay.Tuple([])))
    call2 = relay.Let(f, ident, f(relay.const(2)))

    call_val1 = run_as_python(call1)
    assert_tuple_value(call_val1, 0)

    call_val2 = run_as_python(call2)
    assert_tensor_value(call_val2, 2)


def test_global_function():
    mod = relay.Module()
    ident = relay.GlobalVar('ident')
    a = relay.TypeVar('a')
    v = relay.Var('v', a)
    mod[ident] = relay.Function([v], v, a, [a])

    call1 = ident(relay.const(1))
    call2 = ident(relay.Tuple([relay.const(2), relay.const(2)]))

    call_val1 = run_as_python(call1, mod)
    assert_tensor_value(call_val1, 1)

    call_val2 = run_as_python(call2, mod)
    assert_tuple_value(call_val2, 2)
    assert_tensor_value(call_val2.fields[0], 2)
    assert_tensor_value(call_val2.fields[1], 2)


def test_constructor():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)

    init_box_int = box_ctor(relay.const(1))
    box_val_int = run_as_python(init_box_int, mod)

    assert_constructor_value(box_val_int, box_ctor, 1)
    assert_tensor_value(box_val_int.fields[0], 1)

    init_box_tup = box_ctor(relay.Tuple([]))
    box_val_tup = run_as_python(init_box_tup, mod)

    assert_constructor_value(box_val_tup, box_ctor, 1)
    assert_tuple_value(box_val_tup.fields[0], 0)


def test_match_wildcard():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var('v')
    match = relay.Let(
        v, box_ctor(relay.Tuple([])),
        relay.Match(v, [
            relay.Clause(relay.PatternWildcard(), relay.const(1))
        ]))

    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_match_var():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var('v')
    w = relay.Var('w')
    match = relay.Let(
        v, box_ctor(relay.const(1)),
        relay.Match(v, [
            relay.Clause(relay.PatternVar(w), w)
        ]))

    match_val = run_as_python(match, mod)
    assert_constructor_value(match_val, box_ctor, 1)
    assert_tensor_value(match_val.fields[0], 1)


def test_match_pattern():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var('v')
    w = relay.Var('w')
    match = relay.Let(
        v, box_ctor(relay.const(1)),
        relay.Match(v, [
            relay.Clause(relay.PatternConstructor(box_ctor, [relay.PatternVar(w)]), w)
        ]))
    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_nested_match_pattern():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var('v')
    w = relay.Var('w')
    match = relay.Let(
        v, box_ctor(box_ctor(relay.const(2))),
        relay.Match(v, [
            relay.Clause(
                relay.PatternConstructor(
                    box_ctor, [
                        relay.PatternConstructor(box_ctor, [relay.PatternVar(w)])
                    ]),
                w)]))
    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 2)

def test_match_order():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var('v')
    w = relay.Var('w')
    # wildcard pattern goes first
    match = relay.Let(
        v, box_ctor(box_ctor(relay.const(2))),
        relay.Match(v, [
            relay.Clause(relay.PatternWildcard(), relay.const(1)),
            relay.Clause(
                relay.PatternConstructor(
                    box_ctor, [
                        relay.PatternConstructor(box_ctor, [relay.PatternVar(w)])
                    ]),
                w)]))
    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_local_recursion():
    mod = relay.Module()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var('v')
    w = relay.Var('w')
    f = relay.Var('f')
    # the recursive part never runs and wouldn't terminate if it did but it should
    # type check
    let = relay.Let(f, relay.Function([v], relay.Match(v, [
        relay.Clause(relay.PatternWildcard(), relay.const(0)),
        relay.Clause(relay.PatternConstructor(box_ctor, [relay.PatternVar(w)]), f(w))
    ])),
                    f(box_ctor(relay.const(3))))
    val = run_as_python(let, mod)
    assert_tensor_value(val, 0)
