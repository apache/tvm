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
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np

import tvm
from tvm import te
import tvm.relay.testing
import tvm.relay.transform
from tvm import relay
from tvm import runtime
from tvm.contrib import util

def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu()):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
        tmp_path = util.tempdir()
        lib_name = 'lib.so'
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = tvm.runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe)
        vm.init(ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            json, lib, _ = relay.build(mod, target=target)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    check_vm_result()
    check_graph_runtime_result()


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compiler)
    func = func.with_attr("global_symbol", ext_symbol)
    return func


def test_multi_node_subgraph():
    x = relay.var('x', shape=(10, 10))
    w0 = relay.var('w0', shape=(10, 10))
    w1 = relay.var('w1', shape=(10, 10))
    w2 = relay.var('w2', shape=(10, 10))
    w3 = relay.var('w3', shape=(10, 10))
    w4 = relay.var('w4', shape=(10, 10))
    w5 = relay.var('w5', shape=(10, 10))
    w6 = relay.var('w6', shape=(10, 10))
    w7 = relay.var('w7', shape=(10, 10))

    # subgraph0
    x0 = relay.var('x0', shape=(10, 10))
    w00 = relay.var('w00', shape=(10, 10))
    w01 = relay.var('w01', shape=(10, 10))
    w02 = relay.var('w02', shape=(10, 10))
    z00 = relay.add(x0, w00)
    p00 = relay.subtract(z00, w01)
    q00 = relay.multiply(p00, w02)
    subgraph0 = relay.Function([x0, w00, w01, w02], q00)
    subgraph0 = set_external_func_attr(subgraph0, "ccompiler", "ccompiler_0")
    call0 = relay.Call(subgraph0, [x, w0, w1, w2])

    # subgraph1
    x1 = relay.var('x1', shape=(10, 10))
    w10 = relay.var('w10', shape=(10, 10))
    w11 = relay.var('w11', shape=(10, 10))
    w12 = relay.var('w12', shape=(10, 10))
    z10 = relay.add(x1, w10)
    p10 = relay.subtract(z10, w11)
    q10 = relay.multiply(p10, w12)
    subgraph1 = relay.Function([x1, w10, w11, w12], q10)
    subgraph1 = set_external_func_attr(subgraph1, "ccompiler", "ccompiler_1")
    call1 = relay.Call(subgraph1, [x, w3, w4, w5])


    # Other parts on TVM
    z2 = relay.add(x, w6)
    q2 = relay.subtract(z2, w7)

    r = relay.concatenate((call0, call1, q2), axis=0)
    f = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = relay.transform.InferType()(mod)

    x_data = np.random.rand(10, 10).astype('float32')
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype('float32'))

    map_inputs = {"w{}".format(i): w_data[i] for i in range(8)}
    map_inputs["x"] = x_data
    check_result(
        mod, map_inputs, (30, 10),
        np.concatenate((((x_data + w_data[0]) - w_data[1]) * w_data[2],
                        ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                        x_data + w_data[6] - w_data[7]),
                       axis=0))


def test_extern_gcc_single_op():
    x = relay.var('x', shape=(8, 8))
    y = relay.var('y', shape=(8, 8))

    x0 = relay.var('x0', shape=(8, 8))
    y0 = relay.var('y0', shape=(8, 8))
    z = x0 + y0
    f = relay.Function([x0, y0], z)
    f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
    call = relay.Call(f, [x, y])
    mod = tvm.IRModule.from_expr(call)
    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


def test_extern_gcc_single_op_int():
    x = relay.var('x', shape=(8, 8), dtype="int32")
    y = relay.var('y', shape=(8, 8), dtype="int32")

    x0 = relay.var('x0', shape=(8, 8), dtype="int32")
    y0 = relay.var('y0', shape=(8, 8), dtype="int32")
    z = x0 + y0
    f = relay.Function([x0, y0], z)
    f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
    call = relay.Call(f, [x, y])
    mod = tvm.IRModule.from_expr(call)
    x_data = np.random.rand(8, 8).astype('int32')
    y_data = np.random.rand(8, 8).astype('int32')

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


def test_extern_gcc():
    x = relay.var('x', shape=(2, 2))
    y = relay.var('y', shape=(2, 2))

    # subgraph for mul
    x0 = relay.var('x0', shape=(2, 2))
    y0 = relay.var('y0', shape=(2, 2))
    mul = x0 * y0
    mul = relay.Function([x0, y0], mul)
    mul = set_external_func_attr(mul, "ccompiler", "ccompiler_2")
    call_mul = relay.Call(mul, [y, y])

    # subgraph for add
    x1 = relay.var('x1', shape=(2, 2))
    y1 = relay.var('y1', shape=(2, 2))
    add = x1 + y1
    add = relay.Function([x1, y1], add)
    add = set_external_func_attr(add, "ccompiler", "ccompiler_1")
    call_add = relay.Call(add, [x, x])

    # subgraph for sub
    x2 = relay.var('x2', shape=(2, 2))
    y2 = relay.var('y2', shape=(2, 2))
    sub = x2 - y2
    sub = relay.Function([x2, y2], sub)
    sub = set_external_func_attr(sub, "ccompiler", "ccompiler_0")
    call_sub = relay.Call(sub, [call_mul, call_add])
    mod = tvm.IRModule.from_expr(call_sub)

    x_data = np.random.rand(2, 2).astype('float32')
    y_data = np.random.rand(2, 2).astype('float32')

    check_result(mod, {"x": x_data, "y": y_data}, (2, 2), (y_data * y_data) - (x_data + x_data))


def test_extern_dnnl():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)
    data0 = relay.var('data0', shape=(ishape), dtype=dtype)
    weight0 = relay.var('weight0', shape=(w1shape), dtype=dtype)

    data1 = relay.var('data0', shape=(ishape), dtype=dtype)
    weight1 = relay.var('weight0', shape=(w1shape), dtype=dtype)
    weight2 = relay.var('weight1', shape=(w1shape), dtype=dtype)
    depthwise_conv2d_1 = relay.nn.conv2d(data1,
                                         weight1,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         groups=32)
    depthwise_conv2d_2 = relay.nn.conv2d(depthwise_conv2d_1,
                                         weight2,
                                         kernel_size=(3, 3),
                                         padding=(1, 1),
                                         groups=32)
    out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

    f = relay.Function([data1, weight1, weight2], out)
    ref_mod = tvm.IRModule()
    ref_mod['main'] = f

    f = set_external_func_attr(f, "dnnl", "dnnl_0")
    call = relay.Call(f, [data0, weight0, weight0])
    mod = tvm.IRModule.from_expr(call)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu())
    ref_res = ref_ex.evaluate()(i_data, w_data, w_data)
    check_result(mod, {"data0": i_data, "weight0": w_data},
                 (1, 32, 14, 14), ref_res.asnumpy(), tol=1e-5)


if __name__ == "__main__":
    test_multi_node_subgraph()
    test_extern_gcc_single_op()
    test_extern_gcc_single_op_int()
    test_extern_gcc()
    test_extern_dnnl()
