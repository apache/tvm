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
"""Unit tests for JSON codegen and runtime."""
import os
import sys

import numpy as np

import tvm
import tvm.relay.op as reg
import tvm.relay.testing
from tvm import relay, runtime
from tvm.contrib import util
from tvm.relay import transform
from tvm.relay.analysis.analysis import to_json
from tvm.relay.backend import compile_engine
from tvm.relay.build_module import bind_params_by_name


def set_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func


def check_result(mod,
                 ref_mod,
                 map_inputs,
                 out_shape,
                 tol=1e-5,
                 target="llvm",
                 ctx=tvm.cpu(),
                 params=None):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    # Run the reference result
    compile_engine.get().clear()
    with relay.build_config(opt_level=3):
        json, lib, param = relay.build(ref_mod, target=target, params=params)
    rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

    for name, data in map_inputs.items():
        rt_mod.set_input(name, data)
    rt_mod.set_input(**param)
    rt_mod.run()
    out = tvm.nd.empty(out_shape, ctx=ctx)
    out = rt_mod.get_output(0, out)
    ref_result = out.asnumpy()

    def check_vm_result():
        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe)
        vm.init(ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), ref_result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            json, lib, param = relay.build(mod, target=target, params=params)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)
        tvm.testing.assert_allclose(out.asnumpy(), ref_result, rtol=tol, atol=tol)

    check_vm_result()
    check_graph_runtime_result()


def test_conv2d():
    if not tvm.get_global_func("runtime.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    def conv2d_direct():
        dtype = 'float32'
        ishape = (1, 32, 14, 14)
        w1shape = (32, 32, 3, 3)

        data0 = relay.var("data", shape=ishape, dtype=dtype)
        weight0 = relay.var("weight", shape=w1shape, dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1))

        func = relay.Function([data0, weight0], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("weight", shape=(w1shape), dtype=dtype)
        main_f = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_f

        data0 = relay.var("data", shape=ishape, dtype=dtype)
        weight0 = relay.var("weight", shape=w1shape, dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1))
        main_f = relay.Function([data0, weight0], out)
        ref_mod = tvm.IRModule()
        ref_mod['main'] = main_f

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

        return mod, ref_mod, {"data": i_data, "weight": w1_data}, (1, 32, 14, 14)

    def group_conv2d():
        dtype = 'float32'
        ishape = (1, 32, 14, 14)
        w2shape = (32, 1, 3, 3)

        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        weight0 = relay.var("weight", shape=(w2shape), dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=32)

        func = relay.Function([data0, weight0], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("weight", shape=(w2shape), dtype=dtype)
        main_f = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_f

        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        weight0 = relay.var("weight", shape=(w2shape), dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=32)
        main_f = relay.Function([data0, weight0], out)
        ref_mod = tvm.IRModule()
        ref_mod['main'] = main_f

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w_data = np.random.uniform(0, 1, w2shape).astype(dtype)

        return mod, ref_mod, {"data": i_data, "weight": w_data}, (1, 32, 14, 14)

    for mod, ref_mod, map_inputs, out_shape in [conv2d_direct(), group_conv2d()]:
        # FIXME: Check accuracy. Current avg error: ~0.03
        check_result(mod, ref_mod, map_inputs, out_shape, tol=1e-1)


def test_add():
    if not tvm.get_global_func("runtime.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    shape = (10, 10)

    def gen_add():
        data0 = relay.var("data0", shape=shape, dtype=dtype)
        data1 = relay.var("data1", shape=shape, dtype=dtype)
        out = relay.add(data0, data1)

        func = relay.Function([data0, data1], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        data1 = relay.var("data1", shape=shape, dtype=dtype)
        main_f = relay.Function([data0, data1], glb_var(data0, data1))
        mod["main"] = main_f

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        data1 = relay.var("data1", shape=shape, dtype=dtype)
        out = relay.add(data0, data1)
        main_f = relay.Function([data0, data1], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f

        return mod, ref_mod

    mod, ref_mod = gen_add()

    data0 = np.random.uniform(0, 1, shape).astype(dtype)
    data1 = np.random.uniform(0, 1, shape).astype(dtype)
    check_result(mod, ref_mod, {"data0": data0, "data1": data1}, shape, tol=1e-5)


def test_relu():
    if not tvm.get_global_func("runtime.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    shape = (1, 32, 14, 14)

    def gen_relu():
        data0 = relay.var("data0", shape=shape, dtype=dtype)
        out = relay.nn.relu(data0)

        func = relay.Function([data0], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        main_f = relay.Function([data0], glb_var(data0))
        mod["main"] = main_f

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        out = relay.nn.relu(data0)
        main_f = relay.Function([data0], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f

        return mod, ref_mod

    mod, ref_mod = gen_relu()

    data0 = np.random.uniform(-1, 1, shape).astype(dtype)
    check_result(mod, ref_mod, {"data0": data0,}, (1, 32, 14, 14), tol=1e-5)


def test_dense():
    if not tvm.get_global_func("runtime.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    a_shape = (1, 512)
    b_shape = (1024, 512)

    def gen_dense():
        a = relay.var("A", shape=a_shape, dtype=dtype)
        b = relay.var("B", shape=b_shape, dtype=dtype)
        out = relay.nn.dense(a, b)

        func = relay.Function([a, b], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        a = relay.var("A", shape=a_shape, dtype=dtype)
        b = relay.var("B", shape=b_shape, dtype=dtype)
        main_f = relay.Function([a, b], glb_var(a, b))
        mod["main"] = main_f

        a = relay.var("A", shape=a_shape, dtype=dtype)
        b = relay.var("B", shape=b_shape, dtype=dtype)
        out = relay.nn.dense(a, b)
        main_f = relay.Function([a, b], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f

        return mod, ref_mod

    mod, ref_mod = gen_dense()

    data_a = np.random.uniform(0, 1, a_shape).astype(dtype)
    data_b = np.random.uniform(0, 1, b_shape).astype(dtype)
    check_result(mod, ref_mod, {"A": data_a, "B": data_b}, (1, 1024), tol=1e-5)


def test_bn():
    if not tvm.get_global_func("runtime.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    d_shape = (1, 8)
    c_shape = (8, )

    def gen_bn():
        data = relay.var('data', shape=d_shape)
        gamma = relay.var("gamma", shape=c_shape)
        beta = relay.var("beta", shape=c_shape)
        moving_mean = relay.var("moving_mean", shape=c_shape)
        moving_var = relay.var("moving_var", shape=c_shape)
        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]

        func = relay.Function([data, gamma, beta, moving_mean, moving_var], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        data = relay.var('data', shape=d_shape)
        gamma = relay.var("gamma", shape=c_shape)
        beta = relay.var("beta", shape=c_shape)
        moving_mean = relay.var("moving_mean", shape=c_shape)
        moving_var = relay.var("moving_var", shape=c_shape)
        main_f = relay.Function([data, gamma, beta, moving_mean, moving_var],
                                glb_var(data, gamma, beta, moving_mean, moving_var))
        mod["main"] = main_f

        data = relay.var('data', shape=d_shape)
        gamma = relay.var("gamma", shape=c_shape)
        beta = relay.var("beta", shape=c_shape)
        moving_mean = relay.var("moving_mean", shape=c_shape)
        moving_var = relay.var("moving_var", shape=c_shape)
        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]
        main_f = relay.Function([data, gamma, beta, moving_mean, moving_var], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f

        return mod, ref_mod

    mod, ref_mod = gen_bn()

    data = np.random.uniform(-1, 1, d_shape).astype(dtype)
    gamma = np.random.uniform(-1, 1, c_shape).astype(dtype)
    beta = np.random.uniform(-1, 1, c_shape).astype(dtype)
    moving_mean = np.random.uniform(-1, 1, c_shape).astype(dtype)
    moving_var = np.random.uniform(-1, 1, c_shape).astype(dtype)
    check_result(mod,
                 ref_mod, {
                     "data": data,
                     "gamma": gamma,
                     "beta": beta,
                     "moving_mean": moving_mean,
                     "moving_var": moving_var
                 },
                 d_shape,
                 tol=1e-5)


def test_composite():
    if not tvm.get_global_func("runtime.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 32, 14, 14)
    w1shape = (32, 32, 3, 3)

    def after_partition():
        # Composite function
        in_1 = relay.var("in_1", shape=ishape, dtype=dtype)
        in_2 = relay.var("in_2", shape=w1shape, dtype=dtype)
        conv2d = relay.nn.conv2d(in_1, in_2, kernel_size=(3, 3), padding=(1, 1))
        relu = relay.nn.relu(conv2d)
        func = relay.Function([in_1, in_2], relu)
        func = func.with_attr('Composite', 'conv2d_relu')
        func = func.with_attr('PartitionedFromPattern', 'nn.conv2d_nn.relu_')

        # Partition function
        arg_1 = relay.var("arg_1", shape=ishape, dtype=dtype)
        arg_2 = relay.var("arg_2", shape=w1shape, dtype=dtype)
        call = relay.Call(func, [arg_1, arg_2])
        p_func = relay.Function([arg_1, arg_2], call)
        p_func = set_func_attr(p_func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = p_func

        # Main function
        data = relay.var("data", shape=ishape, dtype=dtype)
        weight = relay.var("input", shape=w1shape, dtype=dtype)
        main_func = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_func
        return mod

    mod = after_partition()
    for global_var, func in mod.functions.items():
        if global_var.name_hint != 'main':
            print(global_var)
            print(to_json(func))



if __name__ == "__main__":
    test_conv2d()
    test_add()
    test_relu()
    test_dense()
    test_bn()
    #test_composite()
