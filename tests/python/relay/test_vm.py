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
import pytest
import time

import tvm
from tvm import runtime
from tvm import relay, IRModule
from tvm.relay.backend import vm
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude
from tvm.relay.loops import while_loop
from tvm.relay import testing
from tvm.contrib import utils
from tvm import rpc
import tvm.testing
from tvm.relay.transform import InferType


def check_result(args, expected_result, mod=None):
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
    for target, dev in tvm.testing.enabled_targets():
        vm = relay.create_executor("vm", device=dev, target=target, mod=mod)
        rts_result = vm.evaluate()(*args)
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


@tvm.testing.uses_gpu
def test_split():
    x = relay.var("x", shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    f = relay.Function([x], y)

    x_data = np.random.rand(
        12,
    ).astype("float32")
    ref_res = np.split(x_data, 3, axis=0)
    for tgt, dev in tvm.testing.enabled_targets():
        res = veval(f, x_data, device=dev, target=tgt)
        for i in range(3):
            tvm.testing.assert_allclose(res[i].numpy(), ref_res[i])


@tvm.testing.uses_gpu
def test_split_no_fuse():
    x = relay.var("x", shape=(12,))
    y = relay.split(x, 3, axis=0).astuple()
    z = relay.concatenate([relay.TupleGetItem(y, 0)], axis=0)
    z = relay.annotation.stop_fusion(z)
    f = relay.Function([x], z)
    x_data = np.random.rand(
        12,
    ).astype("float32")
    for tgt, dev in tvm.testing.enabled_targets():
        res = veval(f, x_data, device=dev, target=tgt)
        tvm.testing.assert_allclose(res.numpy(), np.split(x_data, 3, axis=0)[0])


@tvm.testing.uses_gpu
def test_id():
    x = relay.var("x", shape=(10, 10), dtype="float64")
    f = relay.Function([x], x)
    x_data = np.random.rand(10, 10).astype("float64")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([x_data], x_data, mod=mod)


@tvm.testing.uses_gpu
def test_op():
    x = relay.var("x", shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([x_data], 2 * x_data, mod=mod)


def any(x):
    x = relay.op.nn.batch_flatten(x)
    return relay.op.min(x, axis=[0, 1])


@tvm.testing.uses_gpu
def test_cond():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(10, 10))
    # f = relay.Function([x, y], relay.op.equal(x, y))
    f = relay.Function([x, y], any(relay.op.equal(x, y)))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(10, 10).astype("float32")

    mod = tvm.IRModule()
    mod["main"] = f
    # same
    check_result([x_data, x_data], True, mod=mod)

    # diff
    check_result([x_data, y_data], False, mod=mod)


@tvm.testing.uses_gpu
def test_simple_if():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(10, 10))
    f = relay.Function([x, y], relay.If(any(relay.op.equal(x, y)), x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(10, 10).astype("float32")

    mod = tvm.IRModule()
    mod["main"] = f
    # same
    check_result([x_data, x_data], x_data, mod=mod)

    # diff
    check_result([x_data, y_data], y_data, mod=mod)


@tvm.testing.uses_gpu
def test_multiple_ifs():
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
    dev = tvm.runtime.device("llvm", 0)
    vm = relay.create_executor(device=dev, mod=mod, kind="vm")
    res = vmobj_to_list(vm.evaluate()(False))
    assert res == [1, 0]


@tvm.testing.uses_gpu
def test_unused_function():
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

    check_result([x_data], y_data, mod=mod)


@tvm.testing.uses_gpu
def test_simple_call():
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
    check_result([i_data], i_data, mod=mod)


@tvm.testing.uses_gpu
def test_count_loop():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, i_data, device=dev, target=tgt)
        tvm.testing.assert_allclose(result.numpy(), i_data)
    check_result([i_data], i_data, mod=mod)


@tvm.testing.uses_gpu
def test_sum_loop():
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
    check_result([i_data, accum_data], sum(range(1, loop_bound + 1)), mod=mod)


@tvm.testing.uses_gpu
def test_tuple_fst():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var("tup", type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 0))
    i_data = np.random.rand(41).astype("float32")
    j_data = np.random.rand(10).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([(i_data, j_data)], i_data, mod=mod)


@tvm.testing.uses_gpu
def test_tuple_second():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var("tup", type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype("float32")
    j_data = np.random.rand(10).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    check_result([(i_data, j_data)], j_data, mod=mod)


@tvm.testing.uses_gpu
def test_list_constructor():
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)
    f = relay.Function([], one4)

    mod["main"] = f

    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        assert len(result) == 2
        assert len(result[1]) == 2

        obj = vmobj_to_list(result)
        tvm.testing.assert_allclose(obj, np.array([3, 2, 1]))


@tvm.testing.uses_gpu
def test_let_tensor():
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
    check_result([x_data], x_data + 42.0, mod=mod)


@tvm.testing.uses_gpu
def test_let_scalar():
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
    check_result([x_data], x_data + 42.0, mod=mod)


@tvm.testing.uses_gpu
def test_compose():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, [x_data], device=dev, target=tgt)
        tvm.testing.assert_allclose(result.numpy(), x_data + 2.0)


@tvm.testing.uses_gpu
def test_list_hd():
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

    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(result.numpy(), 3)


@pytest.mark.xfail
def test_list_tl_empty_list():
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")
    tl = mod.get_global_var("tl")

    f = relay.Function([], tl(nil()))

    mod["main"] = f

    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)


@tvm.testing.uses_gpu
def test_list_tl():
    mod = tvm.IRModule()
    p = Prelude(mod)

    l, cons, nil = mod.get_type("List")
    tl = mod.get_global_var("tl")

    one2 = cons(relay.const(1), nil())
    one3 = cons(relay.const(2), one2)
    one4 = cons(relay.const(3), one3)

    f = relay.Function([], tl(one4))

    mod["main"] = f

    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(vmobj_to_list(result), np.array([2, 1]))


@tvm.testing.uses_gpu
def test_list_nth():
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
        for tgt, dev in tvm.testing.enabled_targets():
            result = veval(mod, device=dev, target=tgt)
            tvm.testing.assert_allclose(result.numpy(), expected[i])


@tvm.testing.uses_gpu
def test_list_update():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(vmobj_to_list(result), np.array(expected))


@tvm.testing.uses_gpu
def test_list_length():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(result.numpy(), 10)


@tvm.testing.uses_gpu
def test_list_map():
    mod = tvm.IRModule()
    p = Prelude(mod)

    x = relay.var("x", "int32")
    add_one_func = relay.Function([x], relay.const(1) + x)

    _, cons, nil = mod.get_type("List")
    map = mod.get_global_var("map")

    l = cons(relay.const(2), cons(relay.const(1), nil()))

    f = relay.Function([], map(add_one_func, l))
    mod["main"] = f
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 2]))


@tvm.testing.uses_gpu
def test_list_foldl():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 3, 2, 2, 1, 1]))


@tvm.testing.uses_gpu
def test_list_foldr():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(vmobj_to_list(result), np.array([1, 2, 3]))


@tvm.testing.uses_gpu
def test_list_sum():
    mod = tvm.IRModule()
    p = Prelude(mod)

    _, cons, nil = mod.get_type("List")
    sum = mod.get_global_var("sum")

    l = cons(relay.const(1), cons(relay.const(2), cons(relay.const(3), nil())))
    f = relay.Function([], sum(l))
    mod["main"] = f
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(result.numpy(), 6)


@tvm.testing.uses_gpu
def test_list_filter():
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
    for tgt, dev in tvm.testing.enabled_targets():
        result = veval(mod, device=dev, target=tgt)
        tvm.testing.assert_allclose(vmobj_to_list(result), np.array([3, 5]))


@tvm.testing.uses_gpu
def test_closure():
    x = relay.var("x", shape=())
    y = relay.var("y", shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))
    for tgt, dev in tvm.testing.enabled_targets():
        res = veval(main, device=dev, target=tgt)
        tvm.testing.assert_allclose(res.numpy(), 3.0)


@tvm.testing.uses_gpu
def test_add_op_scalar():
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
        check_result([x_data, y_data], x_data + y_data, mod=mod)


@tvm.testing.uses_gpu
def test_add_op_scalar_int():
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
        check_result([x_data, y_data], x_data + y_data, mod=mod)


@tvm.testing.uses_gpu
def test_add_op_tensor():
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
    check_result([x_data, y_data], x_data + y_data, mod=mod)


@tvm.testing.uses_gpu
def test_add_op_broadcast():
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
    check_result([x_data, y_data], x_data + y_data, mod=mod)


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


@tvm.testing.uses_gpu
def test_loop_free_var():
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
        check_result(args, expected, mod=mod)


@tvm.testing.uses_gpu
def test_vm_reshape_tensor():
    x_np = np.random.uniform(size=(8, 16)).astype("float32")
    x = relay.var("x", shape=(8, 16), dtype="float32")
    y = relay.reshape(x, [-1, 4, 8])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert "reshape_tensor" in exec.bytecode
    check_result([x_np], x_np.reshape([4, 4, 8]), mod)

    x = relay.var("x", shape=(8, 16), dtype="float32")
    y = relay.reshape(x, [16, -1])
    y = relay.reverse_reshape(y, [-1, 4, 0])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    with tvm.transform.PassContext(opt_level=3):
        exec = relay.vm.compile(mod, "llvm")
    assert exec.bytecode.count("reshape_tensor") == 1
    check_result([x_np], x_np.reshape([4, 4, 8]), mod)

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
        check_result([x_np], x_np.reshape([32, 2, 2]), mod)

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
    check_result([x_np, y_np], x_np.reshape([8, 2, 8]), mod)


def test_vm_reshape_tuple(x_shape=(1, 4, 2), y_shape=(1, 2, 10)):
    tup = relay.var(
        "tup",
        type_annotation=relay.TupleType([relay.TensorType(x_shape), relay.TensorType(y_shape)]),
    )
    out = relay.reshape(relay.TupleGetItem(tup, 0), (1, -1))
    f = relay.Function([tup], out)

    x_data = np.random.uniform(size=x_shape).astype("float32")
    y_data = np.random.uniform(size=y_shape).astype("float32")

    for tgt, dev in tvm.testing.enabled_targets():
        res = veval(f, (x_data, y_data), device=dev, target=tgt)
        tvm.testing.assert_allclose(res.numpy(), np.reshape(x_data, (1, -1)))


def test_constant_shape_with_external_codegen():
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
    f1 = f1.with_attr("Compiler", "a")
    glb_f1 = relay.GlobalVar("f1")
    mod[glb_f1] = f1
    mod = relay.transform.InferType()(mod)

    # Main function
    x = relay.var("x", shape=shape, dtype=dtype)
    mod["main"] = relay.Function([x], glb_f1(x))
    comp = relay.vm.VMCompiler()
    opt_mod, _ = comp.optimize(mod, target="llvm")
    assert "shape_func" in opt_mod.astext(False)


def test_vm_rpc():
    """
    This test checks to make sure you can export a VMExecutable,
    upload it to a remote machine using RPC and then execute it
    on the other machine.
    """
    target = tvm.target.Target("llvm --host=llvm")

    # Build a IRModule.
    x = relay.var("x", shape=(10, 1))
    f = relay.Function([x], x + x)
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)

    # Export to Disk
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    vm_exec.mod.export_library(path)

    # Use local rpc server for testing.
    # Server must use popen so it doesn't inherit the current process state. It
    # will crash otherwise.
    server = rpc.Server("localhost", port=9120)
    remote = rpc.connect(server.host, server.port, session_timeout=10)

    # Upload the serialized Executable.
    remote.upload(path)
    # Get a handle to remote Executable.
    rexec = remote.load_module("vm_library.so")

    ctx = remote.cpu()
    # Build a VM out of the executable and context.
    vm_factory = runtime.vm.VirtualMachine(rexec, ctx)
    np_input = np.random.uniform(size=(10, 1)).astype("float32")
    input_tensor = tvm.nd.array(np_input, ctx)
    # Invoke its "main" function.
    out = vm_factory.invoke("main", input_tensor)
    # Check the result.
    np.testing.assert_allclose(out.numpy(), np_input + np_input)

    # delete tensors before the server shuts down so we don't throw errors.
    del input_tensor
    del out

    server.terminate()


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


def test_get_output_multiple():
    target = tvm.target.Target("llvm")

    # Build a IRModule.
    x = relay.var("x", shape=(10,))
    f = relay.Function([x], relay.Tuple([x + x, x]))
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    inp = np.ones(10, dtype="float32")
    vm_factory.invoke_stateful("main", inp)
    outputs = vm_factory.get_outputs()
    assert len(outputs) == 2
    np.testing.assert_allclose(outputs[0].numpy(), inp + inp)
    np.testing.assert_allclose(outputs[1].numpy(), inp)


def test_get_input_index():
    target = tvm.target.Target("llvm")

    # Build a IRModule.
    data_0, data_1 = ["d1", "d2"]
    x, y = [relay.var(c, shape=(10,)) for c in [data_0, data_1]]
    f = relay.Function([x, y], x + y)
    mod = IRModule.from_expr(f)

    # Compile to VMExecutable.
    vm_exec = vm.compile(mod, target=target)
    vm_factory = runtime.vm.VirtualMachine(vm_exec, tvm.cpu())
    assert vm_factory.get_input_index(data_1) == 1
    assert vm_factory.get_input_index(data_0) == 0
    assert vm_factory.get_input_index("invalid") == -1


if __name__ == "__main__":
    pytest.main([__file__])
