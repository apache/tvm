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
import tvm.testing
from tvm import relay
from tvm.relay.backend.interpreter import ConstructorValue, RefValue
from tvm.relay.prelude import Prelude
from tvm.relay.testing import run_as_python
from tvm.runtime.container import ADT


# helper: uses a dummy let binding to sequence a list
# of expressions: expr1; expr2; expr3, etc.
def seq(*exprs):
    ret = exprs[0]
    for expr in exprs[1:]:
        ret = relay.Let(relay.var("_"), ret, expr)
    return ret


# creates a dummy ADT for testing
def init_box_adt(mod):
    box = relay.GlobalTypeVar("box")
    a = relay.TypeVar("a")
    box_ctor = relay.Constructor("box", [a], box)
    mod[box] = relay.TypeData(box, [a], [box_ctor])
    return (box, box_ctor)


# assert that the candidate is a NDArray with value val
def assert_tensor_value(candidate, val):
    assert isinstance(candidate, tvm.nd.NDArray)
    assert np.array_equal(candidate.numpy(), np.array(val))


# assert that the candidate is an ADT with the indicated number of fields
def assert_adt_len(candidate, fields):
    assert isinstance(candidate, ADT)
    assert len(candidate) == fields


# assert that the candidate is a ConstructorValue with the approrpaite constructor
# and number of fields
def assert_constructor_value(candidate, constructor, fields):
    assert isinstance(candidate, ConstructorValue)
    assert candidate.tag == constructor.tag
    assert len(candidate.fields) == fields


def test_create_empty_tuple():
    empty = relay.Tuple([])
    tup_val = run_as_python(empty)
    assert_adt_len(tup_val, 0)


def test_create_scalar():
    scalar = relay.const(1)
    tensor_val = run_as_python(scalar)
    assert_tensor_value(tensor_val, 1)


def test_create_tensor():
    tensor = relay.const([[1, 1], [2, 2]])
    tensor_val = run_as_python(tensor)
    assert_tensor_value(tensor_val, [[1, 1], [2, 2]])


def test_create_nested_tuple():
    relay_tup = relay.Tuple(
        [relay.const(1), relay.const(2), relay.Tuple([relay.const(3), relay.const(4)])]
    )
    tup_val = run_as_python(relay_tup)
    assert_adt_len(tup_val, 3)
    for i in range(2):
        assert_tensor_value(tup_val[i], i + 1)
    assert_adt_len(tup_val[2], 2)
    for i in range(2):
        assert_tensor_value(tup_val[2][i], i + 3)


def test_tuple_get_item():
    relay_tup = relay.Tuple(
        [relay.const(1), relay.const(2), relay.Tuple([relay.const(3), relay.const(4)])]
    )
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
    v = relay.Var("v")
    let = relay.Let(v, relay.Tuple([]), relay.Tuple([v, v]))
    tup_val = run_as_python(let)
    assert_adt_len(tup_val, 2)
    assert_adt_len(tup_val[0], 0)
    assert_adt_len(tup_val[1], 0)


def test_create_ref():
    relay_ref = relay.RefCreate(relay.Tuple([]))
    ref_val = run_as_python(relay_ref)
    assert isinstance(ref_val, RefValue)
    assert_adt_len(ref_val.value, 0)


def test_ref_read():
    v = relay.Var("v")
    assign = relay.Let(v, relay.RefCreate(relay.Tuple([])), relay.RefRead(v))
    read_val = run_as_python(assign)
    assert_adt_len(read_val, 0)


def test_ref_write():
    # check that the result of a ref write is an empty tuple
    v = relay.Var("v")
    initial_write = relay.Let(
        v,
        relay.RefCreate(relay.Tuple([relay.const(1)])),
        relay.RefWrite(v, relay.Tuple([relay.const(2)])),
    )
    write_val = run_as_python(initial_write)
    assert_adt_len(write_val, 0)

    # now ensure that the value, once written, can be read back
    # (we read the value before and after mutation)
    w = relay.Var("w")
    read_after_write = relay.Let(
        v,
        relay.RefCreate(relay.Tuple([relay.const(1)])),
        relay.Let(
            w,
            relay.RefCreate(relay.RefRead(v)),
            seq(
                relay.RefWrite(v, relay.Tuple([relay.const(2)])),
                relay.Tuple([relay.RefRead(w), relay.RefRead(v)]),
            ),
        ),
    )
    read_val = run_as_python(read_after_write)
    assert_adt_len(read_val, 2)
    assert_adt_len(read_val[0], 1)
    assert_adt_len(read_val[1], 1)
    assert_tensor_value(read_val[0][0], 1)
    assert_tensor_value(read_val[1][0], 2)


def test_if():
    # we will have effects in the blocks to ensure only the intended one is executed
    true_cond = relay.const(True)
    false_cond = relay.const(False)

    v = relay.Var("v")
    true_branch = seq(relay.RefWrite(v, relay.const(1)), relay.RefRead(v))
    false_branch = seq(relay.RefWrite(v, relay.const(2)), relay.RefRead(v))

    true_expr = relay.Let(
        v, relay.RefCreate(relay.const(0)), relay.If(true_cond, true_branch, false_branch)
    )
    false_expr = relay.Let(
        v, relay.RefCreate(relay.const(0)), relay.If(false_cond, true_branch, false_branch)
    )

    true_val = run_as_python(true_expr)
    assert_tensor_value(true_val, 1)

    false_val = run_as_python(false_expr)
    assert_tensor_value(false_val, 2)


def test_local_function():
    v = relay.Var("v")
    ident = relay.Function([v], v)
    f = relay.Var("f")
    call1 = relay.Let(f, ident, f(relay.Tuple([])))
    call2 = relay.Let(f, ident, f(relay.const(2)))

    call_val1 = run_as_python(call1)
    assert_adt_len(call_val1, 0)

    call_val2 = run_as_python(call2)
    assert_tensor_value(call_val2, 2)


def test_global_function():
    mod = tvm.IRModule()
    ident = relay.GlobalVar("ident")
    a = relay.TypeVar("a")
    v = relay.Var("v", a)
    mod[ident] = relay.Function([v], v, a, [a])

    call1 = ident(relay.const(1))
    call2 = ident(relay.Tuple([relay.const(2), relay.const(2)]))

    call_val1 = run_as_python(call1, mod)
    assert_tensor_value(call_val1, 1)

    call_val2 = run_as_python(call2, mod)
    assert_adt_len(call_val2, 2)
    assert_tensor_value(call_val2[0], 2)
    assert_tensor_value(call_val2[1], 2)


def test_constructor():
    mod = tvm.IRModule()
    box, box_ctor = init_box_adt(mod)

    init_box_int = box_ctor(relay.const(1))
    box_val_int = run_as_python(init_box_int, mod)

    assert_constructor_value(box_val_int, box_ctor, 1)
    assert_tensor_value(box_val_int.fields[0], 1)

    init_box_tup = box_ctor(relay.Tuple([]))
    box_val_tup = run_as_python(init_box_tup, mod)

    assert_constructor_value(box_val_tup, box_ctor, 1)
    assert_adt_len(box_val_tup.fields[0], 0)


def test_match_wildcard():
    mod = tvm.IRModule()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var("v")
    match = relay.Let(
        v,
        box_ctor(relay.Tuple([])),
        relay.Match(v, [relay.Clause(relay.PatternWildcard(), relay.const(1))]),
    )

    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_match_var():
    mod = tvm.IRModule()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var("v")
    w = relay.Var("w")
    match = relay.Let(
        v, box_ctor(relay.const(1)), relay.Match(v, [relay.Clause(relay.PatternVar(w), w)])
    )

    match_val = run_as_python(match, mod)
    assert_constructor_value(match_val, box_ctor, 1)
    assert_tensor_value(match_val.fields[0], 1)


def test_match_pattern():
    mod = tvm.IRModule()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var("v")
    w = relay.Var("w")
    match = relay.Let(
        v,
        box_ctor(relay.const(1)),
        relay.Match(
            v, [relay.Clause(relay.PatternConstructor(box_ctor, [relay.PatternVar(w)]), w)]
        ),
    )
    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_nested_match_pattern():
    mod = tvm.IRModule()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var("v")
    w = relay.Var("w")
    match = relay.Let(
        v,
        box_ctor(box_ctor(relay.const(2))),
        relay.Match(
            v,
            [
                relay.Clause(
                    relay.PatternConstructor(
                        box_ctor, [relay.PatternConstructor(box_ctor, [relay.PatternVar(w)])]
                    ),
                    w,
                )
            ],
        ),
    )
    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 2)


def test_match_order():
    mod = tvm.IRModule()
    box, box_ctor = init_box_adt(mod)
    v = relay.Var("v")
    w = relay.Var("w")
    # wildcard pattern goes first
    match = relay.Let(
        v,
        box_ctor(box_ctor(relay.const(2))),
        relay.Match(
            v,
            [
                relay.Clause(relay.PatternWildcard(), relay.const(1)),
                relay.Clause(
                    relay.PatternConstructor(
                        box_ctor, [relay.PatternConstructor(box_ctor, [relay.PatternVar(w)])]
                    ),
                    w,
                ),
            ],
        ),
    )
    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_local_recursion():
    mod = tvm.IRModule()
    p = Prelude(mod)
    _, cons, nil = p.mod.get_type("List")

    v = relay.Var("v")
    h = relay.Var("h")
    t = relay.Var("t")
    f = relay.Var("f")

    # just returns the same list
    let = relay.Let(
        f,
        relay.Function(
            [v],
            relay.Match(
                v,
                [
                    relay.Clause(
                        relay.PatternConstructor(cons, [relay.PatternVar(h), relay.PatternVar(t)]),
                        cons(h, f(t)),
                    ),
                    relay.Clause(relay.PatternConstructor(nil, []), nil()),
                ],
            ),
        ),
        f(cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))),
    )

    val = run_as_python(let, mod)
    assert_constructor_value(val, cons, 2)
    assert_tensor_value(val.fields[0], 1)
    assert_constructor_value(val.fields[1], cons, 2)
    assert_tensor_value(val.fields[1].fields[0], 2)
    assert_constructor_value(val.fields[1].fields[1], cons, 2)
    assert_tensor_value(val.fields[1].fields[1].fields[0], 3)
    assert_constructor_value(val.fields[1].fields[1].fields[1], nil, 0)


def test_global_recursion():
    mod = tvm.IRModule()
    p = Prelude(mod)
    rlist, cons, nil = p.mod.get_type("List")

    copy = relay.GlobalVar("copy")
    # same as above: it copies the given list
    a = relay.TypeVar("a")
    v = relay.Var("v", rlist(a))
    h = relay.Var("h")
    t = relay.Var("t")
    copy_def = relay.Function(
        [v],
        relay.Match(
            v,
            [
                relay.Clause(
                    relay.PatternConstructor(cons, [relay.PatternVar(h), relay.PatternVar(t)]),
                    cons(h, copy(t)),
                ),
                relay.Clause(relay.PatternConstructor(nil, []), nil()),
            ],
        ),
        rlist(a),
        [a],
    )
    mod[copy] = copy_def

    call1 = copy_def(cons(relay.const(1), cons(relay.const(2), nil())))
    val1 = run_as_python(call1, mod)
    assert_constructor_value(val1, cons, 2)
    assert_tensor_value(val1.fields[0], 1)
    assert_constructor_value(val1.fields[1], cons, 2)
    assert_tensor_value(val1.fields[1].fields[0], 2)
    assert_constructor_value(val1.fields[1].fields[1], nil, 0)

    call2 = copy_def(cons(relay.Tuple([]), nil()))
    val2 = run_as_python(call2, mod)
    assert_constructor_value(val2, cons, 2)
    assert_adt_len(val2.fields[0], 0)
    assert_constructor_value(val2.fields[1], nil, 0)


def test_higher_order_call():
    # test with anon func
    h = relay.Var("h")
    f = relay.Var("f")
    x = relay.Var("x")
    ho_anon = relay.Let(
        h, relay.Function([f], f(relay.Tuple([]))), h(relay.Function([x], relay.const(1)))
    )

    anon_val = run_as_python(ho_anon)
    assert_tensor_value(anon_val, 1)

    # test with named func
    g = relay.Var("g")
    ho_named = relay.Let(
        h,
        relay.Function([f], f(relay.Tuple([]))),
        relay.Let(g, relay.Function([x], relay.const(2)), h(g)),
    )
    named_val = run_as_python(ho_named)
    assert_tensor_value(named_val, 2)


def test_match_effect_exactly_once():
    mod = tvm.IRModule()
    p = Prelude(mod)
    _, cons, nil = p.mod.get_type("List")

    # the list should be of length 1!
    # Unless we mistakenly execute the data clause more than once
    r = relay.Var("r")
    data = seq(relay.RefWrite(r, cons(relay.Tuple([]), relay.RefRead(r))), relay.RefRead(r))
    match = relay.Let(
        r,
        relay.RefCreate(nil()),
        relay.Match(
            data,
            [
                relay.Clause(relay.PatternConstructor(nil, []), relay.const(0)),
                relay.Clause(
                    relay.PatternConstructor(
                        cons, [relay.PatternWildcard(), relay.PatternConstructor(nil, [])]
                    ),
                    relay.const(1),
                ),
                relay.Clause(relay.PatternWildcard(), relay.const(2)),
            ],
        ),
    )

    match_val = run_as_python(match, mod)
    assert_tensor_value(match_val, 1)


def test_arbitrary_let_nesting():
    # something that is tricky to do in Python but comes naturally in Relay
    mod = tvm.IRModule()
    p = Prelude(mod)
    x = relay.Var("x")
    r = relay.Var("r")
    y = relay.Var("y")
    z = relay.Var("z")
    expr = relay.Tuple(
        [
            relay.Let(x, relay.Tuple([relay.const(1), relay.const(2)]), relay.TupleGetItem(x, 1)),
            relay.Let(
                r,
                relay.RefCreate(relay.const(1)),
                seq(relay.RefWrite(r, relay.const(3)), relay.RefRead(r)),
            ),
            relay.Let(y, p.id(relay.Let(z, relay.const(4), z)), y),
        ]
    )

    tup_val = run_as_python(expr, mod)
    assert_adt_len(tup_val, 3)
    assert_tensor_value(tup_val[0], 2)
    assert_tensor_value(tup_val[1], 3)
    assert_tensor_value(tup_val[2], 4)


def test_ref_execution_order():
    # we want to have effects execute from left to right
    x = relay.Var("x")
    y = relay.Var("y")
    f = relay.Var("f")
    r = relay.Var("r")

    expr = relay.Let(
        f,
        relay.Function([x, y], x),
        # r = 1
        relay.Let(
            r,
            relay.RefCreate(relay.const(1)),
            relay.Tuple(
                [
                    # should be 1
                    relay.RefRead(r),
                    # set r to 2 and read back
                    seq(relay.RefWrite(r, relay.const(2)), relay.RefRead(r)),
                    # set r to 3 and read back
                    seq(relay.RefWrite(r, relay.const(3)), relay.RefRead(r)),
                    # set r to 4 and read as first arg to f
                    # set r to 5 and read as second arg to f
                    # f should evaluate to 4
                    f(
                        seq(relay.RefWrite(r, relay.const(4)), relay.RefRead(r)),
                        seq(relay.RefWrite(r, relay.const(5)), relay.RefRead(r)),
                    ),
                    # read back 5
                    relay.RefRead(r),
                ]
            ),
        ),
    )

    tup_val = run_as_python(expr)
    assert_adt_len(tup_val, 5)
    assert_tensor_value(tup_val[0], 1)
    assert_tensor_value(tup_val[1], 2)
    assert_tensor_value(tup_val[2], 3)
    assert_tensor_value(tup_val[3], 4)
    assert_tensor_value(tup_val[4], 5)


def test_op_add():
    add = relay.add(relay.const(1), relay.const(2))
    add_val = run_as_python(add)
    assert_tensor_value(add_val, 3)


# test an op with a tuple input
# adapted from test_stack in test_op_level3
def test_op_stack():
    def verify_stack(dshapes, axis):
        x_data = [np.random.normal(size=shape).astype("int32") for shape in dshapes]
        ref_res = np.stack(x_data, axis=axis)

        args = []
        for data in x_data:
            args.append(relay.const(data))
        call = relay.stack(relay.Tuple(args), axis)
        call_val = run_as_python(call)
        type(call_val)
        assert_tensor_value(call_val, ref_res)

    verify_stack([(2,), (2,), (2,)], -1)
    verify_stack([(2,), (2,), (2,)], 0)
    verify_stack([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1)
    verify_stack([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1)


# test an op with a tuple output
# adapted from test_split_infer_type in test_op_level3
def test_split():
    def verify_split(shape, indices_or_sections, axis=0):
        x = np.random.normal(size=shape).astype("float32")
        ref_res = np.split(x, indices_or_sections, axis=axis)
        call = relay.split(relay.const(x), indices_or_sections, axis=axis)
        call_val = run_as_python(call)
        assert_adt_len(call_val, len(ref_res))
        for i in range(len(ref_res)):
            assert_tensor_value(call_val[i], ref_res[i])

    verify_split((2, 3), 2)
    verify_split((5, 3), [3])
    verify_split((5, 9, 3), [3, 4], 1)
    verify_split((5, 5, 2, 2), 5, 1)
    verify_split((5, 5, 2, 2), 5, 0)


# ensure we can generate code for batch_norm, since it requires simplify_inference
def test_batch_norm():
    def verify_batch_norm(shapes):
        data = [np.absolute(np.random.normal(size=shape).astype("float32")) for shape in shapes]
        relay_args = [relay.const(arg) for arg in data]

        eps = 1e-5

        def reference(x, gamma, beta, moving_mean, moving_var):
            return (x - moving_mean) / np.sqrt(moving_var + eps) * gamma + beta

        ref_res = reference(*data)

        call = relay.nn.batch_norm(*relay_args, epsilon=eps)[0]
        call_val = run_as_python(call)

        # there will be a change in accuracy so we need to check
        # approximate equality
        assert isinstance(call_val, tvm.nd.NDArray)
        tvm.testing.assert_allclose(call_val.numpy(), ref_res, atol=eps, rtol=eps)

    verify_batch_norm([(10, 20), (20,), (20,), (20,), (20,)])
    verify_batch_norm([(20, 10), (10,), (10,), (10,), (10,)])
    verify_batch_norm([(10, 50), (50,), (50,), (50,), (50,)])
    verify_batch_norm([(30, 40), (40,), (40,), (40,), (40,)])


def test_return_global_var():
    tt = relay.TensorType([1], "float32")
    x = relay.Var("x", type_annotation=tt)
    identity = relay.Function([x], x, ret_type=tt)
    mod = tvm.IRModule()
    mod["main"] = identity
    main_var = mod.get_global_var("main")
    main_func = run_as_python(main_var, mod=mod)

    arg = tvm.nd.array(np.array([0.0], dtype="float32"))
    res = main_func(arg)
    assert arg.numpy() == res.numpy()


def test_closure_in_tuple():
    tt = relay.TensorType([1], "float32")
    x = relay.Var("x", type_annotation=tt)
    identity = relay.Function([x], x, ret_type=tt)
    tup = relay.Tuple([identity, identity])
    index = relay.TupleGetItem(tup, 0)

    func = run_as_python(index)
    arg = tvm.nd.array(np.array([0.0], dtype="float32"))
    res = func(arg)
    assert arg.numpy() == res.numpy()


def test_closure_in_ref():
    tt = relay.TensorType([1], "float32")
    x = relay.Var("x", type_annotation=tt)
    identity = relay.Function([x], x, ret_type=tt)
    gv = relay.GlobalVar("id")

    r = relay.Var("r")
    seq = relay.Let(
        r,
        relay.RefCreate(gv),
        relay.Call(relay.RefRead(r), [relay.const(np.array([0.0], dtype="float32"))]),
    )

    mod = tvm.IRModule()
    mod[gv] = identity
    res = run_as_python(seq, mod=mod)
    assert res.numpy() == np.array([0.0], dtype="float32")


def test_compiling_with_main():
    unit_type = relay.TupleType([])
    unit = relay.Function([], relay.Tuple([]), ret_type=unit_type)

    x = relay.Var("x", type_annotation=unit_type)
    identity = relay.Function([x], x, ret_type=unit_type)

    mod = tvm.IRModule()
    mod["unit"] = unit
    mod["main"] = identity
    gv_main = mod.get_global_var("main")
    gv_unit = mod.get_global_var("unit")

    res = run_as_python(gv_main(relay.Call(gv_unit, ())), mod=mod)
    assert isinstance(res, ADT)
    assert len(res) == 0


if __name__ == "__main__":
    tvm.testing.main()
