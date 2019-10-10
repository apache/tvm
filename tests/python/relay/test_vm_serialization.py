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
import numpy as np

import tvm
from tvm import relay
from tvm.relay.module import Module as rly_module
from tvm.relay import vm as _vm
from tvm.relay import serializer, deserializer
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay.prelude import Prelude
from tvm.contrib import util
from tvm.relay import testing

def create_vm(f, ctx=tvm.cpu(), target="llvm", params=None):
    if isinstance(f, relay.Expr):
        mod = relay.Module()
        mod["main"] = f
        vm = _vm.compile(mod, target=target, params=params)
        vm.init(ctx)
        return vm
    else:
        assert isinstance(f, relay.Module), "expected mod as relay.Module"
        vm = _vm.compile(f, target=target, params=params)
        vm.init(ctx)
        return vm


def veval(vm, *args, ctx=tvm.cpu()):
    assert isinstance(vm, _vm.VirtualMachine), "expected VirtualMachine"
    vm.init(ctx)
    ret = vm.run(*args)
    return ret


def run_network(mod,
                params,
                data_shape=(1, 3, 224, 224),
                dtype='float32'):
    def get_vm_output(mod, data, params, target, ctx, dtype='float32'):
        ex = relay.create_executor('vm', mod=mod, ctx=ctx)
        result = ex.evaluate()(data, **params)
        return result.asnumpy().astype(dtype)

    def get_serialized_output(mod, data, params, target, ctx, dtype='float32'):
        vm = create_vm(mod, ctx, target, params=params)
        ser = serializer.Serializer(vm)
        code, lib = ser.serialize()
        deser = deserializer.Deserializer(code, lib)
        des_vm = deser.deserialize()
        des_vm.init(ctx)
        des_vm.load_params(params)
        result = des_vm.run(data)
        return result.asnumpy().astype(dtype)

    data = np.random.uniform(size=data_shape).astype(dtype)
    target = "llvm"
    ctx = tvm.cpu(0)

    tvm_out = get_vm_output(mod, tvm.nd.array(data.astype(dtype)), params,
                            target, ctx, dtype)
    vm_out = get_serialized_output(mod, tvm.nd.array(data.astype(dtype)), params,
                                   target, ctx, dtype)
    tvm.testing.assert_allclose(vm_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_serializer():
    mod = rly_module({})
    a = relay.const(1.0, "float32")
    x = relay.var('x', shape=(10, 10), dtype='float32')
    f1 = relay.Function([x], x + a)
    glb_f1 = relay.GlobalVar("f1")
    mod[glb_f1] = f1

    b = relay.const(2.0, "float32")
    y = relay.var('y', shape=(10, 10), dtype='float32')
    f2 = relay.Function([y], y - b)
    glb_f2 = relay.GlobalVar("f2")
    mod[glb_f2] = f2

    x1 = relay.var('x1', shape=(10, 10), dtype='float32')
    y1 = relay.var('y1', shape=(10, 10), dtype='float32')
    main = relay.Function([x1, y1], glb_f1(x1) * glb_f2(y1))
    mod["main"] = main

    vm = create_vm(mod)
    ser = serializer.Serializer(vm)

    glbs = ser.globals
    assert len(glbs) == 3
    assert "f1" in glbs
    assert "f2" in glbs
    assert "main" in glbs

    prim_ops = ser.primitive_ops
    assert any(item.startswith('fused_add') for item in prim_ops)
    assert any(item.startswith('fused_subtract') for item in prim_ops)
    assert any(item.startswith('fused_multiply') for item in prim_ops)

    code = ser.bytecode
    assert "main 5 2 5" in code
    assert "f1 2 1 3" in code
    assert "f2 2 1 3" in code

    code, lib = ser.serialize()
    assert isinstance(code, bytearray)
    assert isinstance(lib, tvm.module.Module)


def test_save_load():
    x = relay.var('x', shape=(10, 10))
    f = relay.Function([x], x + x)
    x_data = np.random.rand(10, 10).astype('float32')

    # serialize.
    vm = create_vm(f)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    assert isinstance(code, bytearray)

    # save and load the code and lib file.
    tmp = util.tempdir()
    path_lib = tmp.relpath("lib.so")
    lib.export_library(path_lib)
    with open(tmp.relpath("code.bc"), "wb") as fo:
        fo.write(code)

    loaded_lib = tvm.module.load(path_lib)
    loaded_code = bytearray(open(tmp.relpath("code.bc"), "rb").read())

    # deserialize.
    deser = deserializer.Deserializer(loaded_code, loaded_lib)
    des_vm = deser.deserialize()

    res = veval(des_vm, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data + x_data)


def test_const():
    c = relay.const(1.0, "float32")
    x = relay.var('x', shape=(10, 10), dtype='float32')
    f = relay.Function([x], x + c)
    vm = create_vm(f)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    assert isinstance(code, bytearray)
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()
    x_data = np.random.rand(10, 10).astype('float32')
    res = veval(des_vm, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data + 1)


def test_if():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(10, 10))
    equal = relay.op.equal(x, y)
    equal = relay.op.nn.batch_flatten(equal)
    f = relay.Function([x, y], relay.If(relay.op.min(equal, axis=[0, 1]), x,
                                        y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(10, 10).astype('float32')

    vm = create_vm(f)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()

    # same
    res = veval(des_vm, x_data, x_data)
    tvm.testing.assert_allclose(res.asnumpy(), x_data)

    # diff
    res = veval(des_vm, x_data, y_data)
    tvm.testing.assert_allclose(res.asnumpy(), y_data)


def test_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, 'int32'))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, 'int32'))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    loop_bound = 0
    i_data = np.array(loop_bound, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    iarg = relay.var('i', shape=[], dtype='int32')
    aarg = relay.var('accum', shape=[], dtype='int32')
    mod["main"] = relay.Function([iarg, aarg], sum_up(iarg, aarg))

    vm = create_vm(mod)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()

    result = veval(des_vm, i_data, accum_data)
    tvm.testing.assert_allclose(result.asnumpy(), sum(range(1, loop_bound + 1)))


def test_tuple():
    ttype = relay.TupleType([relay.TensorType((1,)), relay.TensorType((10,))])
    tup = relay.var('tup', type_annotation=ttype)
    f = relay.Function([tup], relay.TupleGetItem(tup, 1))
    i_data = np.random.rand(41).astype('float32')
    j_data = np.random.rand(10).astype('float32')

    vm = create_vm(f)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()

    result = veval(des_vm, (i_data, j_data))
    tvm.testing.assert_allclose(result.asnumpy(), j_data)


def test_adt_list():
    mod = relay.Module()
    p = Prelude(mod)

    l1 = p.cons(relay.const(1), p.nil())
    l21 = p.cons(relay.const(2), l1)
    l321 = p.cons(relay.const(3), l21)

    f = relay.Function([], l321)
    mod["main"] = f

    vm = create_vm(mod)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()

    result = veval(des_vm)
    assert len(result) == 2
    assert len(result[1]) == 2
    assert len(result[1][1]) == 2
    res = []
    res.append(result[0].asnumpy().tolist())
    res.append(result[1][0].asnumpy().tolist())
    res.append(result[1][1][0].asnumpy().tolist())
    tvm.testing.assert_allclose(res, np.array([3, 2, 1]))


def test_adt_compose():
    mod = relay.Module()
    p = Prelude(mod)

    compose = p.compose

    # add_one = fun x -> x + 1
    sb = relay.ScopeBuilder()
    x = relay.var('x', 'float32')
    x1 = sb.let('x1', x)
    xplusone = x1 + relay.const(1.0, 'float32')
    sb.ret(xplusone)
    body = sb.get()
    add_one = relay.GlobalVar("add_one")
    add_one_func = relay.Function([x], body)

    # add_two = compose(add_one, add_one)
    sb = relay.ScopeBuilder()
    y = relay.var('y', 'float32')
    add_two_func = sb.let('add_two', compose(add_one_func, add_one_func))
    add_two_res = add_two_func(y)
    sb.ret(add_two_res)
    add_two_body = sb.get()

    mod[add_one] = add_one_func

    f = relay.Function([y], add_two_body)
    mod["main"] = f

    vm = create_vm(mod)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()

    x_data = np.array(np.random.rand()).astype('float32')
    result = veval(des_vm, x_data)

    tvm.testing.assert_allclose(result.asnumpy(), x_data + 2.0)


def test_closure():
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    f = relay.Function([x], x + y)
    ff = relay.Function([y], f)
    clo = ff(relay.const(1.0))
    main = clo(relay.const(2.0))

    vm = create_vm(main)
    ser = serializer.Serializer(vm)
    code, lib = ser.serialize()
    deser = deserializer.Deserializer(code, lib)
    des_vm = deser.deserialize()

    res = veval(des_vm)
    tvm.testing.assert_allclose(res.asnumpy(), 3.0)


def test_resnet():
    mod, params = testing.resnet.get_workload(batch_size=1, num_layers=18)
    run_network(mod, params)


def test_mobilenet():
    mod, params = testing.mobilenet.get_workload(batch_size=1)
    run_network(mod, params)


if __name__ == "__main__":
    test_serializer()
    test_save_load()
    test_const()
    test_if()
    test_loop()
    test_tuple()
    test_adt_list()
    test_adt_compose()
    test_closure()
    test_resnet()
    test_mobilenet()
