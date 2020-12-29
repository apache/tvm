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
# pylint: disable=no-else-return,invalid-name,len-as-condition,too-many-nested-blocks

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay.analysis import context_analysis


def test_device_copy():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    x = relay.var("x", shape=(2, 3))
    copy = relay.op.device_copy(x, tvm.cpu(), tvm.gpu())
    out = copy + relay.const(np.random.rand(2, 3))
    glb_var = relay.GlobalVar("main")
    mod[glb_var] = relay.Function([x], out)
    ca = context_analysis(mod, tvm.cpu())

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.gpu().device_type
    for expr, dev in ca.items():
        if isinstance(expr, _expr.Call):
            assert dev[0].value == gpu_dev
        elif isinstance(expr, _expr.Var):
            assert dev[0].value == cpu_dev
        elif isinstance(expr, _expr.Constant):
            assert dev[0].value == gpu_dev


def test_shape_func():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    data_shape = (relay.Any(),)
    x = relay.var("x", shape=data_shape)
    y = relay.op.vm.shape_of(x)
    z = relay.nn.relu(y)
    p0 = relay.var("p0", shape=data_shape)
    fn = relay.Function([p0], z)
    out = relay.var("out", shape=(1,), dtype="int64")
    ins = relay.Tuple([y])
    outs = relay.Tuple([out])
    is_inputs = [False]
    shape_func = relay.op.vm.shape_func(fn, ins, outs, is_inputs)
    mod["main"] = relay.Function([x, out], shape_func)
    ca = context_analysis(mod, tvm.gpu())
    main = mod["main"]

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.gpu().device_type
    assert main.params[0] in ca and ca[main.params[0]][0].value == gpu_dev
    # The output of shape func should be on cpu.
    assert main.params[1] in ca and ca[main.params[1]][0].value == cpu_dev
    # shape func is the body and it should be on cpu
    assert main.body in ca and ca[main.body][0].value == cpu_dev


def test_vm_shape_of():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    data_shape = (relay.Any(),)
    x = relay.var("x", shape=data_shape)
    y = relay.op.vm.shape_of(x)
    mod["main"] = relay.Function([x], y)
    ca = context_analysis(mod, tvm.gpu())
    main = mod["main"]

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.gpu().device_type
    assert main.params[0] in ca and ca[main.params[0]][0].value == gpu_dev
    assert main.body in ca and ca[main.body][0].value == cpu_dev


def test_alloc_storage():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    mod.import_from_std("core.rly")
    size = relay.Var("size", relay.scalar_type("int64"))
    alignment = relay.Var("alignment", relay.scalar_type("int64"))
    # allocate a chunk on of memory on gpu.
    sto = relay.op.memory.alloc_storage(size, alignment, tvm.gpu())
    mod["main"] = relay.Function([size, alignment], sto)
    ca = context_analysis(mod, tvm.gpu())
    main = mod["main"]
    body = main.body

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.gpu().device_type
    # Inputs are unified with alloc storage inputs which are on cpu
    assert main.params[0] in ca and ca[main.params[0]][0].value == cpu_dev
    assert main.params[1] in ca and ca[main.params[1]][0].value == cpu_dev

    assert isinstance(body, relay.Call) and len(body.args) == 2
    # size of alloc_storage is on cpu
    assert body.args[0] in ca and ca[body.args[0]][0].value == cpu_dev
    # alignment of alloc_storage is on cpu
    assert body.args[1] in ca and ca[body.args[1]][0].value == cpu_dev
    # alloc_storage is on gpu as specified
    assert body in ca and ca[body][0].value == gpu_dev


def test_alloc_tensor():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    mod.import_from_std("core.rly")
    sto_type = relay.TypeCall(mod.get_global_type_var("Storage"), [])
    sto = relay.Var("x", sto_type)
    sh = relay.const(np.array([3, 2]), dtype="int64")
    at = relay.op.memory.alloc_tensor(sto, relay.const(0, dtype="int64"), sh)
    mod["main"] = relay.Function([sto], at)
    ca = context_analysis(mod, tvm.gpu())
    main = mod["main"]
    body = main.body

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.gpu().device_type
    # Input of the function falls back to the default device gpu
    assert main.params[0] in ca and ca[main.params[0]][0].value == gpu_dev

    assert isinstance(body, relay.Call) and len(body.args) == 3
    # storage of alloc_tensor falls back to the default device gpu
    assert body.args[0] in ca and ca[body.args[0]][0].value == gpu_dev
    # shape of alloc_tensor is on cpu
    assert body.args[1] in ca and ca[body.args[1]][0].value == cpu_dev
    # alloc_tensor keeps the same device context as storage which is is on gpu
    assert body in ca and ca[body][0].value == gpu_dev


def test_vm_reshape_tensor():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    x = relay.var("x", shape=(2, 8), dtype="float32")
    shape = relay.const([-1, 4, 2], dtype="int64")
    y = relay.op.vm.reshape_tensor(x, shape, [2, 4, 2])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    ca = context_analysis(mod, tvm.gpu())
    main = mod["main"]
    body = main.body

    cpu_dev = tvm.cpu().device_type
    gpu_dev = tvm.gpu().device_type
    # Input of the function falls back to the default device gpu
    assert main.params[0] in ca and ca[main.params[0]][0].value == gpu_dev

    # dats of reshape_tensor falls back to the default device gpu
    assert body.args[0] in ca and ca[body.args[0]][0].value == gpu_dev
    # shape of reshape_tensor is on cpu
    assert body.args[1] in ca and ca[body.args[1]][0].value == cpu_dev
    # reshape_tensor sits on the same device as the data
    assert body in ca and ca[body][0].value == gpu_dev


def test_dynamic_input():
    if not tvm.testing.device_enabled("cuda") or not tvm.gpu(0).exist:
        return

    mod = tvm.IRModule()
    data_shape = (relay.Any(), relay.Any())
    x0 = relay.var("x0", shape=data_shape)
    x1 = relay.var("x1", shape=data_shape)
    mod["main"] = relay.Function([x0, x1], x0 + x1)

    compiler = relay.vm.VMCompiler()
    mod, _ = compiler.optimize(mod, target="cuda")
    ca = context_analysis(mod, tvm.cpu())
    main = mod["main"]

    gpu_dev = tvm.gpu().device_type
    assert main.params[0] in ca and ca[main.params[0]][0].value == gpu_dev
    assert main.params[1] in ca and ca[main.params[1]][0].value == gpu_dev
    assert main.body in ca and ca[main.body][0].value == gpu_dev


if __name__ == "__main__":
    pytest.main([__file__])
