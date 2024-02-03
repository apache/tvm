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
from unittest.mock import patch

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import IRModule, relay, rpc, runtime
from tvm.contrib import utils
from tvm.relay import testing
from tvm.relay.backend import vm
from tvm.relay.backend.vm import VMCompiler
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.loops import while_loop
from tvm.relay.prelude import Prelude
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.testing import mlp
from tvm.relay.transform import InferType


def check_result(target, dev, args, expected_result, mod):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on Relay VM.

    Parameters
    ----------
    args: list of Expr
        The arguments to supply the expr.

    expected_result:
        The expected result of running the expression.
    """
    rts_result = relay.create_executor("vm", device=dev, target=target, mod=mod).evaluate()(*args)
    tvm.testing.assert_allclose(expected_result, rts_result.numpy())


def veval(f, *args, device=tvm.cpu(), target="llvm"):
    if isinstance(f, relay.Expr):
        mod = tvm.IRModule()
        mod["main"] = f
    else:
        assert isinstance(f, tvm.IRModule), "expected expression or module"
        mod = f
    exe = relay.vm.compile(mod, target)
    vm = runtime.vm.VirtualMachine(exe, device)
    return vm.invoke("main", *args)


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy().tolist()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def test_split(target, dev):
    x = relay.var("x", shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    f = relay.Function([x], y)

    x_data = np.random.rand(
        12,
    ).astype("float32")
    ref_res = np.split(x_data, 3, axis=0)
    res = veval(f, x_data, device=dev, target=target)
    for i in range(3):
        tvm.testing.assert_allclose(res[i].numpy(), ref_res[i])


def test_split_no_fuse(target, dev):
    x = relay.var("x", shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    z = relay.concatenate([relay.TupleGetItem(y, 0)], axis=0)
    z = relay.annotation.stop_fusion(z)
    f = relay.Function([x], z)
    x_data = np.random.rand(
        12,
    ).astype("float32")

    res = veval(f, x_data, device=dev, target=target)
    tvm.testing.assert_allclose(res.numpy(), np.split(x_data, 3, axis=0)[0])


def test_id(target, dev):
    x = relay.var("x", shape=(10, 10), dtype="float64")
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype("float64")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result(target, dev, [x_data], x_data, mod)


def test_op(target, dev):
    x = relay.var("x", shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result(target, dev, [x_data], 2 * x_data, mod)


def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])


@tvm.testing.known_failing_targets("vulkan")
def test_cond(target, dev):
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(10, 10).astype("float32")

    mod = tvm.IRModule()
    mod["main"] = f
    # same
    check_result(target, dev, [x_data, x_data], True, mod)

    # diff
    check_result(target, dev, [x_data, y_data], False, mod)


@tvm.testing.known_failing_targets("vulkan")
def test_simple_if(target, dev):
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(10, 10))
    f = relay.Function([x, y], relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(10, 10).astype("float32")

    mod = tvm.IRModule()
    mod["main"] = f
    # same
    check_result(target, dev, [x_data, x_data], x_data, mod)

    # diff
    check_result(target, dev, [x_data, y_data], y_data, mod)


@tvm.testing.parametrize_targets("llvm")
def test_multiple_ifs(target, dev):
    mod = tvm.IRModule({})
    b = relay.var("b")
    v0 = relay.var("v0")
    v1 = relay.var("v1")
    v2 = relay.var("v2")
    v3 = relay.var("v3")
    out = relay.Tuple([v2, v3])
    out = relay.Let(v3, relay.If(b, v1, v0), out)
    out = relay.Let(v2, relay.If(b, v0, v1), out)
    out = relay.Let(v1, relay.Tuple([relay.const(1)]), out)
    out = relay.Let(v0, relay.Tuple([relay.const(0)]), out)
    fn = relay.Function([b], out)
    mod["main"] = fn
    func = relay.create_executor(device=dev, mod=mod, kind="vm").evaluate()
    res = vmobj_to_list(func(False))
    assert res == [1, 0]


def test_unused_function(target, dev):
    cond = relay.const(True)
    mod = tvm.IRModule()
    then_name = relay.GlobalVar("times_2")
    # define unused function
    else_name = relay.GlobalVar("times_3")
    t1 = relay.TensorType((2, 2), dtype="float32")
    x1 = relay.var("x1", t1, dtype="float32")
    x2 = relay.var("x2", t1, dtype="float32")
    f2 = relay.multiply(x1, relay.const(2.0))
    f3 = relay.multiply(x2, relay.const(3.0))
    mod[then_name] = relay.Function([x1], f2)
    mod[else_name] = relay.Function([x2], f3)
    mod = InferType()(mod)
    x3 = relay.var("x3", t1, dtype="float32")
    # put unused function in else branch
    f = relay.If(cond, then_name(x3), else_name(x3))
    mod["main"] = relay.Function([x3], f)
    x_data = np.random.rand(2, 2).astype("float32")
    y_data = x_data * 2

    check_result(target, dev, [x_data], y_data, mod)


def test_simple_call(target, dev):
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar("sum_up")
    i = relay.var("i", shape=[], dtype="int32")
    sb = ScopeBuilder()
    sb.ret(i)
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], "int32"))
    mod[sum_up] = func
    i_data = np.array(0, dtype="int32")
    iarg = relay.var("iarg", shape=[], dtype="int32")
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    check_result(target, dev, [i_data], i_data, mod)


def test_count_loop(target, dev):
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar("sum_up")
    i = relay.var("i", shape=[], dtype="int32")
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype="int32"))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype="int32"))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], "int32"))
    mod[sum_up] = func
    i_data = np.array(0, dtype="int32")
    iarg = relay.var("i", shape=[], dtype="int32")
    mod["main"] = relay.Function([iarg], sum_up(iarg))
    result = veval(mod, i_data, device=dev, target=target)
    tvm.testing.assert_allclose(result.numpy(), i_data)
    check_result(target, dev, [i_data], i_data, mod)


def test_sum_loop(target, dev):
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar("sum_up")
    i = relay.var("i", shape=[], dtype="int32")
    accum = relay.var("accum", shape=[], dtype="int32")
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, "int32"))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, "int32"))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    mod = relay.transform.InferType()(mod)
    loop_bound = 0
    i_data = np.array(loop_bound, dtype="int32")
    accum_data = np.array(0, dtype="int32")
    iarg = relay.var("i", shape=[], dtype="int32")
    aarg = relay.var("accum", shape=[], dtype="int32")
    mod["main"] = relay.Function([iarg, aarg], sum_up(iarg, aarg))
    check_result(target, dev, [i_data, accum_data], sum(range(1, loop_bound + 1)), mod)


def test_tuple_fst(target, dev):
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var("tup", type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.random.rand(41).astype("float32")
    j_data = np.random.rand(10).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result(target, dev, [(i_data, j_data)], i_data, mod)


def test_tuple_second(target, dev):
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var("tup", type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype("float32")
    j_data = np.random.rand(10).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result(target, dev, [(i_data, j_data)], j_data, mod)


def test_list_constructor(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)
    f = relay.Function([], one4)

    mod["main"] = f

    result = veval(mod, device=dev, target=target)
    assert len(result) == 2
    assert len(result[1]) == 2

    obj = vmobj_to_list(result)
    tvm.testing.assert_allclose(obj, np.array([3, 2, 1]))


def test_let_tensor(target, dev):
    sb = relay.ScopeBuilder()
    shape = (1,)
    x = relay.var("x", shape=shape, dtype="float32")
    x1 = relay.var("x1", shape=shape, dtype="float32")

    x1 = sb.let(x1, x)
    xplusone = x1 + relay.const(42.0, "float32")
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.random.rand(*shape).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result(target, dev, [x_data], x_data + 42.0, mod)


def test_let_scalar(target, dev):
    sb = relay.ScopeBuilder()

    x = relay.var("x", "float32")
    x1 = sb.let("x1", x)
    xplusone = x1 + relay.const(42.0, "float32")
    sb.ret(xplusone)
    body = sb.get()

    f = relay.Function([x], body)

    x_data = np.array(np.random.rand()).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result(target, dev, [x_data], x_data + 42.0, mod)


def test_compose(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    compose = p.compose

    # add_one = fun x -> x + 1
    sb = relay.ScopeBuilder()
    x = relay.var("x", "float32")
    x1 = sb.let("x1", x)
    xplusone = x1 + relay.const(1.0, "float32")
    sb.ret(xplusone)
    body = sb.get()
    add_one = relay.GlobalVar("add_one")
    add_one_func = relay.Function([x], body)

    # add_two = compose(add_one, add_one)
    sb = relay.ScopeBuilder()
    y = relay.var("y", "float32")
    add_two_func = sb.let("add_two", compose(add_one_func, add_one_func))
    add_two_res = add_two_func(y)
    sb.ret(add_two_res)
    add_two_body = sb.get()

    mod[add_one] = add_one_func

    f = relay.Function([y], add_two_body)
    mod["main"] = f

    x_data = np.array(np.random.rand()).astype("float32")
    result = veval(mod, [x_data], device=dev, target=target)
    tvm.testing.assert_allclose(result.numpy(), x_data + 2.0)


def test_list_hd(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")
    hd = mod.get_global_var("hd")

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)
    three = hd(one4)
    f = relay.Function([], three)

    mod["main"] = f

    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(result.numpy(), 3)


def test_list_tl_empty_list(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")
    tl = mod.get_global_var("tl")

    f = relay.Function([], tl(nil()))

    mod["main"] = f

    with pytest.raises(tvm.error.TVMError):
        result = veval(mod, device=dev, target=target)


def test_list_tl(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")
    tl = mod.get_global_var("tl")

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)

    f = relay.Function([], tl(one4))

    mod["main"] = f

    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([2, 1]))


def test_list_nth(target, dev):
    expected = list(range(10))

    for i in range(len(expected)):
        mod = tvm.IRModule()
        p = Prelude(mod)

        _, cons, nil = mod.get_type("List")
        nth = mod.get_global_var("nth")

        l = nil()
        for i in reversed(expected):
            l = cons(relay.const(i), l)

        f = relay.Function([], nth(l, relay.const(i)))
        mod["main"] = f
        result = veval(mod, device=dev, target=target)
        tvm.testing.assert_allclose(result.numpy(), expected[i])


def test_list_update(target, dev):
    expected = list(range(10))

    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    update = mod.get_global_var("update")

    l = nil()
    # create zero initialized list
    for i in range(len(expected)):
        l = cons(relay.const(0), l)

    # set value
    for i, v in enumerate(expected):
        l = update(l, relay.const(i), relay.const(v))

    f = relay.Function([], l)
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array(expected))


def test_list_length(target, dev):
    expected = list(range(10))

    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    length = mod.get_global_var("length")

    l = nil()
    # create zero initialized list
    for _ in range(len(expected)):
        l = cons(relay.const(0), l)

    l = length(l)

    f = relay.Function([], l)
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(result.numpy(), 10)


def test_list_map(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    x = relay.var("x", "int32")
    add_one_func = relay.Function([x], relay.const(1) + x)

    _, cons, nil = mod.get_type("List")
    map = mod.get_global_var("map")

    l = cons(relay.const(2), cons(relay.const(1), nil()))

    f = relay.Function([], map(add_one_func, l))
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 2]))


def test_list_foldl(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    foldl = mod.get_global_var("foldl")

    x = relay.var("x")
    y = relay.var("y")
    rev_dup_func = relay.Function([y, x], cons(x, cons(x, y)))

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], foldl(rev_dup_func, nil(), l))
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 3, 2, 2, 1, 1]))


def test_list_foldr(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    foldr = mod.get_global_var("foldr")

    x = relay.var("x")
    y = relay.var("y")
    identity_func = relay.Function([x, y], cons(x, y))

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], foldr(identity_func, nil(), l))
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([1, 2, 3]))


def test_list_sum(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    sum = mod.get_global_var("sum")

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], sum(l))
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(result.numpy(), 6)


def test_list_filter(target, dev):
    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    filter = mod.get_global_var("filter")

    x = relay.var("x", "int32")
    greater_than_one = relay.Function([x], x > relay.const(1))
    l = cons(
        relay.const(1),
        cons(
            relay.const(3), cons(relay.const(1), cons(relay.const(5), cons(relay.const(1), nil())))
        ),
    )
    f = relay.Function([], filter(greater_than_one, l))
    mod["main"] = f
    result = veval(mod, device=dev, target=target)
    tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 5]))


def test_closure(target, dev):
    x = relay.var("x", shape=())
    y = relay.var("y", shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))
    res = veval(main, device=dev, target=target)
    tvm.testing.assert_allclose(res.numpy(), 3.0)


def test_add_op_scalar(target, dev):
    """
    test_add_op_scalar:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var("x", shape=())  # Default to float32
    y = relay.var("y", shape=())  # Default to float32
    func = relay.Function([x, y], relay.op.add(x, y))
    x_y_data = [
        (np.array(10.0, dtype="float32"), np.array(1.0, dtype="float32")),
        (np.float32(10.0), np.float32(1.0)),
        (10.0, 1.0),
    ]
    for (x_data, y_data) in x_y_data:
        mod["main"] = func
        check_result(target, dev, [x_data, y_data], x_data + y_data, mod)


def test_add_op_scalar_float16(target, dev):
    """
    test_add_op_scalar_float16:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var("x", shape=(), dtype="float16")  # Default to float16
    y = relay.var("y", shape=(), dtype="float16")  # Default to float16
    func = relay.Function([x, y], relay.op.add(x, y))
    x_y_data = [
        (np.array(10.0, dtype="float16"), np.array(1.0, dtype="float16")),
        (np.float16(10.0), np.float16(1.0)),
    ]
    for (x_data, y_data) in x_y_data:
        mod["main"] = func
        check_result(target, dev, [x_data, y_data], x_data + y_data, mod)


def test_add_op_scalar_int(target, dev):
    """
    test_add_op_scalar_int:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var("x", shape=(), dtype="int32")
    y = relay.var("y", shape=(), dtype="int32")
    func = relay.Function([x, y], relay.op.add(x, y))
    x_y_data = [
        (np.array(10.0, dtype="int32"), np.array(1.0, dtype="int32")),
        (np.int32(10), np.int32(1)),
        (10, 1),
    ]
    for (x_data, y_data) in x_y_data:
        mod["main"] = func
        check_result(target, dev, [x_data, y_data], x_data + y_data, mod)


def test_add_op_tensor(target, dev):
    """
    test_add_op_tensor:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(10, 5))
    func = relay.Function([x, y], relay.op.add(x, y))
    x_data = np.random.rand(10, 5).astype("float32")
    y_data = np.random.rand(10, 5).astype("float32")
    mod["main"] = func
    check_result(target, dev, [x_data, y_data], x_data + y_data, mod)


def test_add_op_broadcast(target, dev):
    """
    test_add_op_broadcast:
        fn (x, y) {
            return x + y;
        }
    """
    mod = tvm.IRModule()
    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(1, 5))
    func = relay.Function([x, y], relay.op.add(x, y))
    x_data = np.random.rand(10, 5).astype("float32")
    y_data = np.random.rand(1, 5).astype("float32")
    mod["main"] = func
    check_result(target, dev, [x_data, y_data], x_data + y_data, mod)


def test_vm_optimize_dynamic():
    dtype = "float32"
    x = relay.var("x", shape=(relay.Any(), relay.Any()), dtype=dtype)
    y = relay.var("y", shape=(relay.Any(), relay.Any()), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], relay.add(x, y))
    comp = relay.vm.VMCompiler()
    opt_mod, _ = comp.optimize(mod, target="llvm")
    assert "shape_func" in opt_mod.astext(False)


def test_vm_optimize():
    mod, params = testing.synthetic.get_workload()
    comp = relay.vm.VMCompiler()
    opt_mod, _ = comp.optimize(mod, target="llvm", params=params)

    free_vars = relay.analysis.free_vars(opt_mod["main"].body)
    # Paremeters should all be bound, so the only free var is data
    assert len(free_vars) == 1


def test_loop_free_var(target, dev):
    x = relay.var("x", shape=(), dtype="int32")
    i = relay.var("i", shape=(), dtype="int32")
    s = relay.var("s", shape=(), dtype="int32")

    def cond(i, _):
        return i < relay.const(10, dtype="int32")

    def body_no_free_var(i, acc):
        incr = relay.const(1, "int32")
        return i + incr, acc + i

    def body_with_free_var(i, acc):
        incr = relay.const(1, "int32")
        return i + incr, acc + x

    for args, body, expected in zip([[], [1]], [body_no_free_var, body_with_free_var], [45, 10]):
        loop = while_loop(cond, [i, s], body)
        tup = loop(relay.const(0, dtype="int32"), relay.zeros(shape=(), dtype="int32"))
        ret = relay.TupleGetItem(tup, 1)
        mod = tvm.IRModule()
        mod["main"] = relay.Function(relay.analysis.free_vars(ret), ret)
        check_result(target, dev, args, expected, mod)


def test_vm_reshape_tensor(target, dev):
    x_np = np.random.uniform(size=(8, 16)).astype("float32")
    x = relay.var("x", shape=(8, 16), dtype="float32")
    y = relay.reshape(x, [-1, 4, 8])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert "reshape_tensor" in exec.bytecode
    check_result(target, dev, [x_np], x_np.reshape([4, 4, 8]), mod)

    x = relay.var("x", shape=(8, 16), dtype="float32")
    y = relay.reshape(x, [16, -1])
    y = relay.reverse_reshape(y, [-1, 4, 0])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert exec.bytecode.count("reshape_tensor") == 1
    check_result(target, dev, [x_np], x_np.reshape([4, 4, 8]), mod)

    # reshape with symbolic/any shape
    for n in [tvm.tir.Any(), tvm.te.size_var("n")]:
        x = relay.var("x", shape=(n, 16), dtype="float32")
        y = relay.reshape(x, [-1, 4])
        y = relay.reshape(y, [0, 2, -1])
        mod = tvm.IRModule()
        mod["main"] = relay.Function([x], y)
        with tvm.transform.PassContext(opt_level=3):
            exec = relay.vm.compile(mod, "llvm")
        assert exec.bytecode.count("reshape_tensor") == 1
        check_result(target, dev, [x_np], x_np.reshape([32, 2, 2]), mod)

    # dyn.reshape
    x = relay.var("x", shape=(8, 16), dtype="float32")
    y = relay.var("y", shape=(3,), dtype="int32")
    z = relay.reshape(x, [-1, 4, 8])
    z = relay.reshape(z, y)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], z)
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert exec.bytecode.count("reshape_tensor") == 2
    assert "reshape_tensor" in exec.bytecode
    y_np = np.array([8, 2, 8]).astype("int32")
    check_result(target, dev, [x_np, y_np], x_np.reshape([8, 2, 8]), mod)


def test_vm_reshape_and_copy(target, dev):
    """Make sure the compiler notices the reshape result shape is a literal and can use
    the immediate-mode alloc_tensor instruction instead of alloc_tensor_reg."""
    x_np = np.random.uniform(size=(1, 1)).astype("float32")
    x = relay.var("x", shape=(1, 1), dtype="float32")
    mod = tvm.IRModule.from_expr(relay.Function([x], relay.copy(relay.reshape(x, [0, 1]))))
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert "alloc_tensor" in exec.bytecode
    assert not "alloc_tensor_reg" in exec.bytecode
    check_result(target, dev, [x_np], x_np.reshape([1, 1]), mod)


def test_vm_reshape_tuple(target, dev, x_shape=(1, 4, 2), y_shape=(1, 2, 10)):
    tup = relay.var(
        "tup",
        type_annotation=relay.TupleType([relay.TensorType(x_shape), relay.TensorType(y_shape)]),
    )
    out = relay.reshape(relay.TupleGetItem(tup, 0), (1, -1))
    f = relay.Function([tup], out)

    x_data = np.random.uniform(size=x_shape).astype("float32")
    y_data = np.random.uniform(size=y_shape).astype("float32")

    res = veval(f, (x_data, y_data), device=dev, target=target)
    tvm.testing.assert_allclose(res.numpy(), np.reshape(x_data, (1, -1)))


def test_constant_shape_with_external_codegen():
    @tvm.register_func("relay.ext.test1")
    def relay_ext_test(func):
        return None

    mod = tvm.IRModule()
    shape = (relay.Any(), 25)
    dtype = "float32"

    # external function
    x = relay.var("x", shape=shape, dtype=dtype)
    weight = relay.const(np.random.rand(5, 25).astype("float32"), dtype="float32")
    out = relay.nn.dense(x, weight)
    f1 = relay.Function([x], out)
    f1 = f1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    f1 = f1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    f1 = f1.with_attr("Compiler", "test1")
    f1 = f1.with_attr("global_symbol", "f1")
    glb_f1 = relay.GlobalVar("f1")
    mod[glb_f1] = f1
    mod = relay.transform.InferType()(mod)

    # Main function
    x = relay.var("x", shape=shape, dtype=dtype)
    mod["main"] = relay.Function([x], glb_f1(x))
    comp = relay.vm.VMCompiler()
    opt_mod, _ = comp.optimize(mod, target="llvm")
    assert "shape_func" in opt_mod.astext(False)


def prepare_vm_model(path, tensor_shape):
    """
    Virtual Machine is compiled for simple topology and
    exported as library to given path
    """
    target = tvm.target.Target("llvm --host=llvm")

    # Build a IRModule.
    x = relay.var("x", shape=tensor_shape)
    f = relay.Function([x], x + x)
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)

    # Export to Disk
    vm_exec.mod.export_library(path)


def test_vm_rpc():
    """
    This test checks to make sure you can export a VMExecutable,
    upload it to a remote machine using RPC and then execute it
    on the other machine.
    """
    # Shape for input and output tensors
    shape = (10, 1)

    # Export to Disk
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    prepare_vm_model(path, shape)

    # Use local rpc server for testing.
    # Server must use popen so it doesn't inherit the current process state. It
    # will crash otherwise.
    def check_remote(server):
        remote = rpc.connect(server.host, server.port, session_timeout=10)

        # Upload the serialized Executable.
        remote.upload(path)
        # Get a handle to remote Executable.
        rexec = remote.load_module("vm_library.so")

        device = remote.cpu()
        # Build a VM out of the executable and context.
        vm_factory = runtime.vm.VirtualMachine(rexec, device)
        np_input = np.random.uniform(size=shape).astype("float32")
        input_tensor = tvm.nd.array(np_input, device)
        # Invoke its "main" function.
        out = vm_factory.invoke("main", input_tensor)
        # Check the result.
        np.testing.assert_allclose(out.numpy(), np_input + np_input)

    check_remote(rpc.Server("127.0.0.1"))


def test_vm_invoke_with_outputs_rpc():
    """
    This test checks to make sure you can export a VMExecutable,
    upload it to a remote machine using RPC and then execute it
    on the other machine with preallocated outputs.
    """
    # Shape for input and output tensors
    shape = (3, 2)

    # Export to Disk
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    prepare_vm_model(path, shape)

    # Use local rpc server for testing.
    # Server must use popen so it doesn't inherit the current process state. It
    # will crash otherwise.
    def check_remote_invoke_with_outputs(server):
        remote = rpc.connect(server.host, server.port, session_timeout=10)

        # Upload the serialized Executable.
        remote.upload(path)
        # Get a handle to remote Executable.
        rexec = remote.load_module("vm_library.so")

        device = remote.cpu()
        # Build a VM out of the executable and context.
        vm_factory = runtime.vm.VirtualMachine(rexec, device)
        np_input = np.random.uniform(size=shape).astype("float32")
        input_tensor = tvm.nd.array(np_input, device)
        np_output = np.empty(shape, dtype="float32")
        output_tensor = tvm.nd.array(np_output, device)
        # Invoke its "main" function.
        vm_factory.invoke_with_outputs(
            "main", input_args={"x": input_tensor}, output_args=[output_tensor]
        )
        # Check the result.
        np.testing.assert_allclose(output_tensor.numpy(), np_input + np_input)

    check_remote_invoke_with_outputs(rpc.Server("127.0.0.1"))


def test_vm_invoke_with_outputs():
    target = tvm.target.Target("llvm")
    shape = (3, 2)

    # Build a IRModule.
    x = relay.var("x", shape=shape)
    f = relay.Function([x], x + x)
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    np_input = np.random.uniform(size=shape).astype("float32")
    input_tensor = tvm.nd.array(np_input)
    np_output = np.empty(shape, dtype="float32")
    output_tensor = tvm.nd.array(np_output)
    # Invoke
    vm_factory.invoke_with_outputs(
        "main", input_args={"x": input_tensor}, output_args=[output_tensor]
    )
    # Check the result.
    np.testing.assert_allclose(output_tensor.numpy(), np_input + np_input)


def test_get_output_single():
    target = tvm.target.Target("llvm")

    # Build a IRModule.
    x = relay.var("x", shape=(10,))
    f = relay.Function([x], x + x)
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    inp = np.ones(10, dtype="float32")
    vm_factory.invoke_stateful("main", inp)
    outputs = vm_factory.get_outputs()
    assert len(outputs) == 1
    np.testing.assert_allclose(outputs[0].numpy(), inp + inp)


@tvm.testing.parametrize_targets("llvm")
def test_get_output_multiple(target, dev):
    # Build a IRModule.
    x = relay.var("x", shape=(10,))
    f = relay.Function([x], relay.Tuple([x + x, x]))
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, dev)
    inp = np.ones(10, dtype="float32")
    vm_factory.invoke_stateful("main", inp)
    outputs = vm_factory.get_outputs()
    assert len(outputs) == 2
    np.testing.assert_allclose(outputs[0].numpy(), inp + inp)
    np.testing.assert_allclose(outputs[1].numpy(), inp)


@tvm.testing.parametrize_targets("llvm")
def test_get_input_index(target, dev):
    # Build a IRModule.
    data_0, data_1 = ["d1", "d2"]
    x, y = [relay.var(c, shape=(10,)) for c in [data_0, data_1]]
    f = relay.Function([x, y], x + y)
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, dev)
    assert vm_factory.get_input_index(data_1) == 1
    assert vm_factory.get_input_index(data_0) == 0
    assert vm_factory.get_input_index("invalid") == -1


def get_one_input_relay_mod(tensor_type, shape, data_name):
    x = relay.var(data_name, shape=shape, dtype=tensor_type)
    y = relay.exp(x)
    f = relay.Function([x], y)
    return IRModule.from_expr(f)


@tvm.testing.parametrize_targets("llvm")
def test_one_set_input(target, dev):
    dtype = "float32"
    in_shape = [1, 2, 3, 3]
    in_data_name_0 = "d0"

    mod = get_one_input_relay_mod(dtype, in_shape, in_data_name_0)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    exe = runtime.vm.VirtualMachine(vm_exec, dev)

    data0_core = np.random.uniform(size=in_shape).astype(dtype)
    data0 = tvm.nd.array(data0_core)
    ref_res_core = np.exp(data0_core)
    ref_res = tvm.nd.array(ref_res_core)

    exe.set_input("main", data0)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())

    data_dict = {in_data_name_0: data0}
    exe.set_input("main", **data_dict)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())


def get_multiple_input_relay_mod(tensor_type, shape, data_name0, data_name1):
    x, y = [relay.var(c, shape=shape, dtype=tensor_type) for c in [data_name0, data_name1]]
    f = relay.Function([x, y], x + y)
    return IRModule.from_expr(f)


@tvm.testing.parametrize_targets("llvm")
def test_multiple_set_input(target, dev):
    dtype = "float32"
    in_shape = [1, 2, 3, 3]
    in_data_name_0 = "d0"
    in_data_name_1 = "d1"

    mod = get_multiple_input_relay_mod(dtype, in_shape, in_data_name_0, in_data_name_1)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    exe = runtime.vm.VirtualMachine(vm_exec, dev)

    data0_core = np.random.uniform(size=in_shape).astype(dtype)
    data0 = tvm.nd.array(data0_core)
    data1_core = np.random.uniform(size=in_shape).astype(dtype)
    data1 = tvm.nd.array(data1_core)
    ref_res_core = data0_core + data1_core
    ref_res = tvm.nd.array(ref_res_core)

    exe.set_input("main", data0, data1)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())

    data_dict = {in_data_name_1: data1, in_data_name_0: data0}
    exe.set_input("main", **data_dict)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())


@tvm.testing.parametrize_targets("llvm")
def test_one_set_one_input(target, dev):
    dtype = "float32"
    in_shape = [1, 2, 3, 3]
    in_data_name_0 = "d0"

    mod = get_one_input_relay_mod(dtype, in_shape, in_data_name_0)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    exe = runtime.vm.VirtualMachine(vm_exec, dev)

    data0_core = np.random.uniform(size=in_shape).astype(dtype)
    data0 = tvm.nd.array(data0_core)
    ref_res_core = np.exp(data0_core)
    ref_res = tvm.nd.array(ref_res_core)

    exe.set_one_input("main", 0, data0)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())

    exe.set_one_input("main", in_data_name_0, data0)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())

    data_dict = {in_data_name_0: data0}
    exe.set_one_input("main", **data_dict)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())


@tvm.testing.parametrize_targets("llvm")
def test_multiple_set_one_input(target, dev):
    dtype = "float32"
    in_shape = [1, 2, 3, 3]
    in_data_name_0 = "d0"
    in_data_name_1 = "d1"

    mod = get_multiple_input_relay_mod(dtype, in_shape, in_data_name_0, in_data_name_1)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    exe = runtime.vm.VirtualMachine(vm_exec, dev)

    data0_core = np.random.uniform(size=in_shape).astype(dtype)
    data0 = tvm.nd.array(data0_core)
    data1_core = np.random.uniform(size=in_shape).astype(dtype)
    data1 = tvm.nd.array(data1_core)
    ref_res_core = data0_core + data1_core
    ref_res = tvm.nd.array(ref_res_core)

    exe.set_one_input("main", 1, data1)
    exe.set_one_input("main", 0, data0)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())

    exe.set_one_input("main", in_data_name_1, data1)
    exe.set_one_input("main", in_data_name_0, data0)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())

    data_dict = {in_data_name_1: data1}
    exe.set_one_input("main", **data_dict)
    data_dict = {in_data_name_0: data0}
    exe.set_one_input("main", **data_dict)
    output = exe.invoke("main")
    assert output.dtype == ref_res.dtype
    tvm.testing.assert_allclose(ref_res_core, output.numpy())


@tvm.testing.parametrize_targets("llvm")
def test_benchmark(target, dev):
    mod, params = mlp.get_workload(1)
    lib = vm.compile(mod, target=target, params=params)
    exe = runtime.vm.VirtualMachine(lib, tvm.cpu())
    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"))
    result = exe.benchmark(tvm.cpu(), data, func_name="main", repeat=2, number=1)
    assert result.mean == result.median
    assert result.mean > 0
    assert len(result.results) == 2

    with patch.object(
        tvm.runtime.module.Module,
        "time_evaluator",
        return_value=lambda x: tvm.runtime.module.BenchmarkResult([1, 2, 2, 5]),
    ) as method:
        result = exe.benchmark(dev, data, func_name="main", repeat=2, number=1)
        assert result.mean == 2.5
        assert result.median == 2.0
        assert result.max == 5
        assert result.min == 1
        assert result.std == 1.5


def test_benchmark_end_to_end(target, dev):
    mod, params = mlp.get_workload(1)
    lib = vm.compile(mod, target=target, params=params)
    exe = runtime.vm.VirtualMachine(lib, dev)
    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
    result = exe.benchmark(dev, data, func_name="main", repeat=2, number=1, end_to_end=True)
    assert result.mean > 0


@tvm.testing.requires_cuda
def test_benchmark_end_to_end_rpc():
    server = rpc.Server("127.0.0.1")
    remote = rpc.connect(server.host, server.port)

    mod, params = mlp.get_workload(1)
    lib = vm.compile(mod, target="cuda", params=params)

    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    lib.mod.export_library(path)
    remote.upload(path)
    rlib = remote.load_module("vm_library.so")

    exe = runtime.vm.VirtualMachine(rlib, remote.device("cuda"))
    data = tvm.nd.array(
        np.random.rand(1, 1, 28, 28).astype("float32"), device=remote.device("cuda")
    )
    result = exe.benchmark(
        remote.device("cuda"), data=data, func_name="main", repeat=2, number=1, end_to_end=True
    )
    assert result.mean > 0


def test_shape_func_nested_function():
    @tvm.register_func("relay.ext.test2")
    def relay_ext_test(func):
        return None

    data_shape = (relay.Any(), 16)
    weight_shape = (relay.Any(), 16)

    dense = relay.nn.dense(
        relay.var("data", shape=data_shape), relay.var("weight", shape=weight_shape)
    )
    mod = tvm.IRModule.from_expr(dense)

    patterns = [("test.dense", is_op("nn.dense")(wildcard(), wildcard()))]
    passes = tvm.transform.Sequential(
        [
            relay.transform.MergeComposite(patterns),
            relay.transform.AnnotateTarget(["test2"]),
            relay.transform.PartitionGraph(),
        ]
    )

    mod = passes(mod)

    compiler = VMCompiler()
    compiler.lower(mod, "llvm")


@tvm.testing.requires_cuda
def test_storage_size_and_offset_on_cpu():
    """Tests allocations place sizes and offsets on the CPU host even if the rest
    of the computation is on a different device type."""

    # TODO(mbs): Better would be to test ManifestAlloc independently.
    # And/or move this to C++ and test the VM executable in it's C++ instead of
    # pretty-printed form.

    # CPU = device type 1
    # GPU = device type 2
    def input():
        return tvm.relay.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32],
                      param_device_types=[2], result_device_type=2) {
              add(%a, %a)
            }
        """
        )

    exe = relay.vm.compile(
        input(),
        tvm.target.Target("cuda"),
    )

    # This program needs two constants:
    # - The size of the tensor's storage (first arg) to alloc_storage
    # - The offset of the tensor within the storage (second arg) to alloc_tensor
    # Both should be on the CPU
    assert "VirtualDevice[0]: device type 1" in exe.virtual_devices
    assert "VM Const[0]: NDArray[(),int64,(1,0)]=[140] on device index 0" in exe.constants
    assert "VM Const[1]: NDArray[(),int64,(1,0)]=[0] on device index 0" in exe.constants


@tvm.testing.requires_cuda
def test_reshape_shape_on_cpu():
    """Tests the argument to a reshape places the shape on the CPU host even if the rest
    of the computation is on a different device type."""

    # TODO(mbs): Better would be to test ManifestAlloc independently.
    # And/or move this to C++ and test the VM executable in it's C++ instead of
    # pretty-printed form.

    # CPU = device type 1
    # GPU = device type 2
    def input():
        return tvm.relay.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(2, 8), float32],
                      param_device_types=[2], result_device_type=2) {
              reshape(%x, newshape=[2, 4, 2])
            }
        """
        )

    exe = relay.vm.compile(
        input(),
        tvm.target.Target("cuda"),
    )

    # The newshape annotation should have been turned into a constant on the CPU.
    assert "VirtualDevice[0]: device type 1" in exe.virtual_devices
    assert "VM Const[0]: NDArray[(3),int64,(1,0)]=[2,4,2] on device index 0" in exe.constants


@tvm.testing.requires_cuda
def test_multi_targets():
    # Build an IRModule.
    n = 10
    x = relay.var("x", shape=(n,))
    y = relay.var("y", shape=(n,))
    z = relay.var("z", shape=(n,))
    f = relay.Function([x, y, z], x + relay.op.annotation.on_device(y + z, tvm.cpu()))
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    with tvm.transform.PassContext(
        opt_level=3, config={"relay.fallback_device_type": tvm.cuda().device_type}
    ):
        exe = relay.vm.compile(
            mod, target={"cpu": tvm.target.Target("llvm"), "cuda": tvm.target.Target("cuda")}
        )

    # Run
    vm = runtime.vm.VirtualMachine(exe, [tvm.cuda(), tvm.cpu()])
    x_data = np.random.rand(
        n,
    ).astype("float32")
    y_data = np.random.rand(
        n,
    ).astype("float32")
    z_data = np.random.rand(
        n,
    ).astype("float32")
    actual_result = vm.invoke("main", x_data, y_data, z_data)

    # Test
    expected_result = x_data + y_data + z_data
    tvm.testing.assert_allclose(actual_result.numpy(), expected_result)


def test_let_bound_constants():
    """This tests for an ICHECK failure for ill-formed IR with let-bound constants"""

    x = relay.var("x", shape=(3,), dtype="int32")
    y = relay.take(x, relay.const(0))
    z = relay.const(1)

    f = relay.Function([x], relay.stack((z, y), axis=0))
    mod = IRModule.from_expr(f)

    compiler = VMCompiler()
    compiler.optimize(mod, target="llvm")


def test_large_constants():
    """Large constants can be serialized outside of executable"""
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()

    # fn(x) { add(x, <large constant>) }
    x = relay.var("x", shape=(1000, 1000))
    const_data = np.random.rand(1000, 1000).astype("float32")
    const = relay.const(const_data, dtype="float32")
    func = relay.Function([x], relay.op.add(x, const))
    mod = tvm.IRModule.from_expr(func)

    # Compile to executable.
    vm_exec = vm.compile(mod, target=target)

    # Save to constants and library files
    temp = utils.tempdir()
    path_consts = temp.relpath("consts")
    vm_exec.move_late_bound_consts(path_consts, byte_limit=256)
    path_dso = temp.relpath("lib.so")
    vm_exec.mod.export_library(path_dso)

    # Load library files and constants
    mod = runtime.load_module(path_dso)
    mod["load_late_bound_consts"](path_consts)

    # Test main
    x_data = np.random.rand(1000, 1000).astype("float32")
    the_vm = runtime.vm.VirtualMachine(mod, dev)
    actual = the_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())

    # We load the mod again so it's missing the consts.
    mod = runtime.load_module(path_dso)
    exe = runtime.vm.Executable(mod)

    # Also test loading consts via the VM's wrapper API.
    exe.load_late_bound_consts(path_consts)

    # Test main again with consts now loaded via the above API.
    x_data = np.random.rand(1000, 1000).astype("float32")
    the_vm = runtime.vm.VirtualMachine(exe, dev)
    actual = the_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())


def test_load_late_bound_consts_with_no_late_bound_consts():
    """Check that load_late_bound_consts handles a model with no late bound consts."""
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()

    const_data = np.random.rand(1).astype("float64")
    x = relay.var("x", shape=(1,), dtype="float64")
    const = relay.const(const_data, dtype="float64")

    func = relay.Function([x], relay.op.add(x, const))
    mod = tvm.IRModule.from_expr(func)

    vm_exec = vm.compile(mod, target=target)

    temp = utils.tempdir()
    path_consts = temp.relpath("consts")
    path_dso = temp.relpath("lib.so")

    # Ensure const_data is below the byte threshold for a late-bound const.
    byte_limit = len(const_data.tobytes()) + 1
    vm_exec.move_late_bound_consts(path_consts, byte_limit=byte_limit)
    vm_exec.mod.export_library(path_dso)

    mod = runtime.load_module(path_dso)
    mod["load_late_bound_consts"](path_consts)

    x_data = np.random.rand(1).astype("float64")
    loaded_vm = runtime.vm.VirtualMachine(mod, dev)
    actual = loaded_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())


def test_vm_save_and_load_without_designating_late_bound_consts():
    """Check that a VM can be saved and loaded without late-bound consts in play.

    Specifically, this test ensures that the machinery behind late-bound const
    loading does not assume the need to load late-bound consts (and cause an error)
    when the user did not choose to designate any consts as such.
    """
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()

    const_data = np.random.rand(1).astype("float64")
    x = relay.var("x", shape=(1,), dtype="float64")
    const = relay.const(const_data, dtype="float64")

    func = relay.Function([x], relay.op.add(x, const))
    mod = tvm.IRModule.from_expr(func)

    vm_exec = vm.compile(mod, target=target)

    code, lib = vm_exec.save()
    exe = runtime.vm.Executable.load_exec(code, lib)

    x_data = np.random.rand(1).astype("float64")
    loaded_vm = runtime.vm.VirtualMachine(exe, dev)
    actual = loaded_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())


def test_load_and_save_constants_via_map():
    """Large constants can be serialized outside of executable"""
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()

    # fn(x) { add(x, <large constant>) }
    x = relay.var("x", shape=(1000, 1000))
    const_data = np.random.rand(1000, 1000).astype("float32")
    const = relay.const(const_data, dtype="float32")
    func = relay.Function([x], relay.op.add(x, const))
    mod = tvm.IRModule.from_expr(func)

    # Compile to executable.
    vm_exec = vm.compile(mod, target=target)

    consts_map = vm_exec.get_late_bound_consts(byte_limit=256)

    # Save to constants and library files
    temp = utils.tempdir()
    path_dso = temp.relpath("lib.so")
    vm_exec.mod.export_library(path_dso)

    # Load library files and constants
    mod = runtime.load_module(path_dso)
    mod["load_late_bound_consts_from_map"](consts_map)

    # Test main
    x_data = np.random.rand(1000, 1000).astype("float32")
    the_vm = runtime.vm.VirtualMachine(mod, dev)
    actual = the_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())

    # We load the mod again so it's missing the consts.
    mod = runtime.load_module(path_dso)
    exe = runtime.vm.Executable(mod)

    # Also test loading consts via the VM's wrapper API.
    exe.load_late_bound_consts_from_map(consts_map)

    # Test main again with consts now loaded via the above API.
    x_data = np.random.rand(1000, 1000).astype("float32")
    the_vm = runtime.vm.VirtualMachine(exe, dev)
    actual = the_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())


def test_load_late_bound_consts_via_map_with_no_late_bound_consts():
    """Check that load_late_bound_consts handles a model with no late bound consts."""
    target = tvm.target.Target("llvm")
    dev = tvm.cpu()

    const_data = np.random.rand(1).astype("float64")
    x = relay.var("x", shape=(1,), dtype="float64")
    const = relay.const(const_data, dtype="float64")

    func = relay.Function([x], relay.op.add(x, const))
    mod = tvm.IRModule.from_expr(func)

    vm_exec = vm.compile(mod, target=target)

    temp = utils.tempdir()
    path_dso = temp.relpath("lib.so")

    # Ensure const_data is below the byte threshold for a late-bound const.
    byte_limit = len(const_data.tobytes()) + 1
    consts_map = vm_exec.get_late_bound_consts(byte_limit=byte_limit)
    vm_exec.mod.export_library(path_dso)

    mod = runtime.load_module(path_dso)
    mod["load_late_bound_consts_from_map"](consts_map)

    x_data = np.random.rand(1).astype("float64")
    loaded_vm = runtime.vm.VirtualMachine(mod, dev)
    actual = loaded_vm.invoke("main", x_data)
    expected = x_data + const_data
    tvm.testing.assert_allclose(expected, actual.numpy())


if __name__ == "__main__":
    tvm.testing.main()
