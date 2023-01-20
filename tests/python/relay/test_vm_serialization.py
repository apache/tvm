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
# pylint: disable=invalid-name, missing-docstring, no-else-return
"""Unit tests for the Relay VM serialization and deserialization."""
import pytest
import numpy as np

import tvm
from tvm.runtime import vm as _vm
from tvm.relay import vm as rly_vm
from tvm import relay

from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay import transform
from tvm.relay.prelude import Prelude
from tvm.contrib import utils
from tvm.relay import testing


def create_exec(f, target="llvm", params=None):
    if isinstance(f, relay.Expr):
        mod = tvm.IRModule()
        mod["main"] = f
        executable = rly_vm.compile(mod, target=target, params=params)
        return executable
    else:
        assert isinstance(f, tvm.IRModule), "expected mod as tvm.IRModule"
        executable = rly_vm.compile(f, target=target, params=params)
        return executable


def get_serialized_output(mod, *data, params=None, target="llvm", device=tvm.cpu()):
    exe = create_exec(mod, target, params=params)
    code, lib = exe.save()
    des_exec = _vm.Executable.load_exec(code, lib)
    des_vm = _vm.VirtualMachine(des_exec, device)
    result = des_vm.run(*data)
    return result


def run_network(mod, params, dtype="float32"):
    def get_vm_output(mod, data, params, target, device, dtype="float32"):
        result = relay.create_executor("vm", mod=mod, device=device).evaluate()(data, **params)
        return result.numpy().astype(dtype)

    data_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    data = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    dev = tvm.cpu(0)

    tvm_out = get_vm_output(mod, tvm.nd.array(data.astype(dtype)), params, target, dev, dtype)
    vm_out = get_serialized_output(
        mod, tvm.nd.array(data.astype(dtype)), params=params, target=target, device=dev
    )
    tvm.testing.assert_allclose(vm_out.numpy().astype(dtype), tvm_out, rtol=1e-5, atol=1e-5)


def test_serializer():
    mod = tvm.IRModule({})
    a = relay.const(1.0, "float32")
    x = relay.var("x", shape=(10, 10), dtype="float32")
    f1 = relay.Function([x], x + a)
    glb_f1 = relay.GlobalVar("f1")
    mod[glb_f1] = f1

    # TODO(@jroesch): look into optimizing away the need to do this
    mod = transform.InferType()(mod)

    b = relay.const(2.0, "float32")
    y = relay.var("y", shape=(10, 10), dtype="float32")
    f2 = relay.Function([y], y - b)
    glb_f2 = relay.GlobalVar("f2")
    mod[glb_f2] = f2

    # TODO(@jroesch): look into optimizing away the need to do this
    mod = transform.InferType()(mod)

    x1 = relay.var("x1", shape=(10, 10), dtype="float32")
    y1 = relay.var("y1", shape=(10, 10), dtype="float32")
    main = relay.Function([x1, y1], glb_f1(x1) * glb_f2(y1))
    mod["main"] = main

    exe = create_exec(mod)

    glbs = exe.globals
    assert len(glbs) == 3
    assert "f1" in glbs
    assert "f2" in glbs
    assert "main" in glbs

    prim_ops = exe.primitive_ops
    assert any(item.startswith("vm_mod_fused_add") for item in prim_ops)
    assert any(item.startswith("vm_mod_fused_subtract") for item in prim_ops)
    assert any(item.startswith("vm_mod_fused_multiply") for item in prim_ops)

    code = exe.bytecode
    assert "main(x1, y1)" in code
    assert "f1(x)" in code
    assert "f2(y)" in code

    code, lib = exe.save()
    assert isinstance(code, bytearray)
    assert isinstance(lib, tvm.runtime.Module)


def test_save_load():
    x = relay.var("x", shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype("float32")

    # serialize.
    vm = create_exec(f)
    code, lib = vm.save()
    assert isinstance(code, bytearray)

    # save and load the code and lib file.
    tmp = utils.tempdir()
    path_lib = tmp.relpath("lib.so")
    lib.export_library(path_lib)
    with open(tmp.relpath("code.ro"), "wb") as fo:
        fo.write(code)

    loaded_lib = tvm.runtime.load_module(path_lib)
    loaded_code = bytearray(open(tmp.relpath("code.ro"), "rb").read())

    # deserialize.
    des_exec = _vm.Executable.load_exec(loaded_code, loaded_lib)
    des_vm = _vm.VirtualMachine(des_exec, tvm.cpu())

    res = des_vm.run(x_data)
    tvm.testing.assert_allclose(res.numpy(), x_data + x_data)


def test_const():
    c = relay.const(1.0, "float32")
    x = relay.var("x", shape=(10, 10), dtype="float32")
    f = relay.Function([x], x + c)
    x_data = np.random.rand(10, 10).astype("float32")
    res = get_serialized_output(f, x_data)
    tvm.testing.assert_allclose(res.numpy(), x_data + 1)


def test_if():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(10, 10))
    equal = relay.op.equal(x, y)
    equal = relay.op.nn.batch_flatten(equal)
    f = relay.Function([x, y], relay.If(relay.op.min(equal, axis=[0, 1]), x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(10, 10).astype("float32")

    # same
    res = get_serialized_output(f, x_data, x_data)
    tvm.testing.assert_allclose(res.numpy(), x_data)

    # diff
    res = get_serialized_output(f, x_data, y_data)
    tvm.testing.assert_allclose(res.numpy(), y_data)


def test_loop():
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
    mod = transform.InferType()(mod)
    loop_bound = 0
    i_data = np.array(loop_bound, dtype="int32")
    accum_data = np.array(0, dtype="int32")
    iarg = relay.var("i", shape=[], dtype="int32")
    aarg = relay.var("accum", shape=[], dtype="int32")
    mod["main"] = relay.Function([iarg, aarg], sum_up(iarg, aarg))

    result = get_serialized_output(mod, i_data, accum_data)
    tvm.testing.assert_allclose(result.numpy(), sum(range(1, loop_bound + 1)))


def test_tuple():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var("tup", type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype("float32")
    j_data = np.random.rand(10).astype("float32")

    result = get_serialized_output(f, (i_data, j_data))
    tvm.testing.assert_allclose(result.numpy(), j_data)


def test_adt_list():
    mod = tvm.IRModule()
    p = Prelude(mod)
    _, cons, nil = mod.get_type("List")
    l1 = cons(relay.const(1), nil())
    l21 = cons(relay.const(2), l1)
    l321 = cons(relay.const(3), l21)

    f = relay.Function([], l321)
    mod["main"] = f

    result = get_serialized_output(mod)
    assert len(result) == 2
    assert len(result[1]) == 2
    assert len(result[1][1]) == 2
    res = []
    res.append(result[0].numpy().tolist())
    res.append(result[1][0].numpy().tolist())
    res.append(result[1][1][0].numpy().tolist())
    tvm.testing.assert_allclose(res, np.array([3, 2, 1]))


def test_adt_compose():
    mod = tvm.IRModule()
    p = Prelude(mod)

    compose = mod.get_global_var("compose")

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
    result = get_serialized_output(mod, x_data)
    tvm.testing.assert_allclose(result.numpy(), x_data + 2.0)


def test_closure():
    x = relay.var("x", shape=())
    y = relay.var("y", shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))

    res = get_serialized_output(main)
    tvm.testing.assert_allclose(res.numpy(), 3.0)


def test_synthetic():
    mod, params = testing.synthetic.get_workload()
    run_network(mod, params)


def test_mobilenet():
    mod, params = testing.mobilenet.get_workload(batch_size=1)
    run_network(mod, params)


def test_vm_shape_of():
    x = relay.var("x", shape=(relay.Any(), relay.Any(), relay.Any()), dtype="float32")
    relu_x = relay.nn.relu(x)
    data = np.random.uniform(size=(2, 3, 4)).astype("float32")
    args = [data]

    newshape_var = relay.var("newshape", shape=(2,), dtype="int64")
    args.append(np.array((1, -1), dtype="int64"))
    main = relay.Function([x, newshape_var], relay.reshape(relu_x, newshape=newshape_var))

    res = get_serialized_output(main, *args).numpy()
    tvm.testing.assert_allclose(res.flatten(), data.flatten())


def test_dynamic_bcast():
    dtype = "float32"
    x = relay.var("x", shape=(relay.Any(), 2), dtype=dtype)
    y = relay.var("y", shape=(3, 2), dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], relay.add(x, y))
    x_data = np.random.uniform(size=(1, 2)).astype(dtype)
    y_data = np.random.uniform(size=(3, 2)).astype(dtype)
    res_np = np.add(x_data, y_data)
    for target, dev in testing.enabled_targets():
        res = get_serialized_output(mod, *(x_data, y_data), target=target, device=dev)
        tvm.testing.assert_allclose(res.numpy(), res_np)


if __name__ == "__main__":
    tvm.testing.main()
