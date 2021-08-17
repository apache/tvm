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
from collections import OrderedDict
import numpy as np
import pytest

import tvm
from tvm import relay, runtime
from tvm.contrib import utils
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.annotation import compiler_begin, compiler_end


def update_lib(lib):
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(test_dir, "..", "..", "..")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    tmp_path = utils.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = tvm.runtime.load_module(lib_path)

    return lib


def check_vm_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu()):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        exe = relay.vm.compile(mod, target=target)
    code, lib = exe.save()
    lib = update_lib(lib)
    exe = runtime.vm.Executable.load_exec(code, lib)
    vm = runtime.vm.VirtualMachine(exe, device)
    out = vm.run(**map_inputs)
    tvm.testing.assert_allclose(out.numpy(), result, rtol=tol, atol=tol)


def check_graph_executor_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu()
):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        json, lib, _ = relay.build(mod, target=target)
    lib = update_lib(lib)
    rt_mod = tvm.contrib.graph_executor.create(json, lib, device)

    for name, data in map_inputs.items():
        rt_mod.set_input(name, data)
    rt_mod.run()
    out = tvm.nd.empty(out_shape, device=device)
    out = rt_mod.get_output(0, out)

    tvm.testing.assert_allclose(out.numpy(), result, rtol=tol, atol=tol)


def check_aot_executor_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu()
):
    if tvm.support.libinfo().get("USE_MICRO", "OFF") != "ON":
        pytest.skip("MicroTVM support not enabled. Set USE_MICRO=ON in config.cmake to enable.")

    # Late import to avoid breaking test with USE_MICRO=OFF.
    from aot.aot_test_utils import AOTTestModel, compile_and_run

    interface_api = "packed"
    use_unpacked_api = False
    use_calculated_workspaces = True
    compile_and_run(
        AOTTestModel(module=mod, inputs=map_inputs, outputs=[result]),
        interface_api,
        use_unpacked_api,
        use_calculated_workspaces,
    )


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compiler)
    func = func.with_attr("global_symbol", ext_symbol)
    return func


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.parametrize(
    "check_result", [check_vm_result, check_graph_executor_result, check_aot_executor_result]
)
def test_multi_node_subgraph(check_result):
    x = relay.var("x", shape=(10, 10))
    w0 = relay.var("w0", shape=(10, 10))
    w1 = relay.var("w1", shape=(10, 10))
    w2 = relay.var("w2", shape=(10, 10))
    w3 = relay.var("w3", shape=(10, 10))
    w4 = relay.var("w4", shape=(10, 10))
    w5 = relay.var("w5", shape=(10, 10))
    w6 = relay.var("w6", shape=(10, 10))
    w7 = relay.var("w7", shape=(10, 10))

    # subgraph0
    x0 = relay.var("x0", shape=(10, 10))
    w00 = relay.var("w00", shape=(10, 10))
    w01 = relay.var("w01", shape=(10, 10))
    w02 = relay.var("w02", shape=(10, 10))
    z00 = relay.add(x0, w00)
    p00 = relay.subtract(z00, w01)
    q00 = relay.multiply(p00, w02)
    subgraph0 = relay.Function([x0, w00, w01, w02], q00)
    subgraph0 = set_external_func_attr(subgraph0, "ccompiler", "ccompiler_0")
    call0 = relay.Call(subgraph0, [x, w0, w1, w2])

    # subgraph1
    x1 = relay.var("x1", shape=(10, 10))
    w10 = relay.var("w10", shape=(10, 10))
    w11 = relay.var("w11", shape=(10, 10))
    w12 = relay.var("w12", shape=(10, 10))
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

    x_data = np.random.rand(10, 10).astype("float32")
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype("float32"))

    map_inputs = OrderedDict([("x", x_data)] + [("w{}".format(i), w_data[i]) for i in range(8)])
    check_result(
        mod,
        map_inputs,
        (30, 10),
        np.concatenate(
            (
                ((x_data + w_data[0]) - w_data[1]) * w_data[2],
                ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                x_data + w_data[6] - w_data[7],
            ),
            axis=0,
        ),
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.parametrize(
    "check_result", [check_vm_result, check_graph_executor_result, check_aot_executor_result]
)
def test_extern_gcc_single_op(check_result):
    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))

    x0 = relay.var("x0", shape=(8, 8))
    y0 = relay.var("y0", shape=(8, 8))
    z = x0 + y0
    f = relay.Function([x0, y0], z)
    f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
    call = relay.Call(f, [x, y])
    mod = tvm.IRModule.from_expr(call)
    x_data = np.random.rand(8, 8).astype("float32")
    y_data = np.random.rand(8, 8).astype("float32")

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.parametrize(
    "check_result", [check_vm_result, check_graph_executor_result, check_aot_executor_result]
)
def test_extern_gcc_single_op_int(check_result):
    x = relay.var("x", shape=(8, 8), dtype="int32")
    y = relay.var("y", shape=(8, 8), dtype="int32")

    x0 = relay.var("x0", shape=(8, 8), dtype="int32")
    y0 = relay.var("y0", shape=(8, 8), dtype="int32")
    z = x0 + y0
    f = relay.Function([x0, y0], z)
    f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
    call = relay.Call(f, [x, y])
    mod = tvm.IRModule.from_expr(call)
    x_data = np.random.rand(8, 8).astype("int32")
    y_data = np.random.rand(8, 8).astype("int32")

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.parametrize(
    "check_result", [check_vm_result, check_graph_executor_result, check_aot_executor_result]
)
def test_extern_gcc(check_result):
    x = relay.var("x", shape=(2, 2))
    y = relay.var("y", shape=(2, 2))

    # subgraph for mul
    x0 = relay.var("x0", shape=(2, 2))
    y0 = relay.var("y0", shape=(2, 2))
    mul = x0 * y0
    mul = relay.Function([x0, y0], mul)
    mul = set_external_func_attr(mul, "ccompiler", "ccompiler_2")
    call_mul = relay.Call(mul, [y, y])

    # subgraph for add
    x1 = relay.var("x1", shape=(2, 2))
    y1 = relay.var("y1", shape=(2, 2))
    add = x1 + y1
    add = relay.Function([x1, y1], add)
    add = set_external_func_attr(add, "ccompiler", "ccompiler_1")
    call_add = relay.Call(add, [x, x])

    # subgraph for sub
    x2 = relay.var("x2", shape=(2, 2))
    y2 = relay.var("y2", shape=(2, 2))
    sub = x2 - y2
    sub = relay.Function([x2, y2], sub)
    sub = set_external_func_attr(sub, "ccompiler", "ccompiler_0")
    call_sub = relay.Call(sub, [call_mul, call_add])
    mod = tvm.IRModule.from_expr(call_sub)

    x_data = np.random.rand(2, 2).astype("float32")
    y_data = np.random.rand(2, 2).astype("float32")

    inputs = OrderedDict(
        [
            ("y", y_data),
            ("x", x_data),
        ]
    )

    check_result(mod, inputs, (2, 2), (y_data * y_data) - (x_data + x_data))


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
def test_extern_gcc_consts():
    @tvm._ffi.register_func("relay.ext.ccompiler.constant_updater")
    def constant_updater(expr, symbol):
        """A dummy constant updater just to test that a custom one works."""
        return {"ccompiler_0_p0": tvm.nd.array(y0_data)}

    x = relay.var("x", shape=(8, 8))
    y0_data = np.random.uniform(0, 1, (8, 8)).astype("float32")

    x0 = relay.var("x0", shape=(8, 8))
    y0_const = relay.const(y0_data, "float32")
    z = x0 + y0_const
    f = relay.Function([x0], z)
    f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
    call = relay.Call(f, [x])
    mod = tvm.IRModule.from_expr(call)

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        compiler = relay.backend.vm.VMCompiler()
        compiler.lower(mod, "llvm")
        compiler.codegen()
        params = compiler.get_params()
        assert len(params) == 1
        assert "ccompiler_0_p0" in params.keys()

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        _, _, params = relay.build(mod, target="llvm")
        assert len(params) == 1
        assert "ccompiler_0_p0" in params.keys()

    tvm._ffi.registry.remove_global_func("relay.ext.ccompiler.constant_updater")


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True),
    reason="skip because DNNL codegen is not available",
)
@pytest.mark.parametrize("check_result", [check_vm_result, check_graph_executor_result])
def test_extern_dnnl(check_result):
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)
    data0 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight0 = relay.var("weight0", shape=(w1shape), dtype=dtype)

    data1 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight1 = relay.var("weight0", shape=(w1shape), dtype=dtype)
    weight2 = relay.var("weight1", shape=(w1shape), dtype=dtype)
    depthwise_conv2d_1 = relay.nn.conv2d(
        data1, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
    )
    depthwise_conv2d_2 = relay.nn.conv2d(
        depthwise_conv2d_1, weight2, kernel_size=(3, 3), padding=(1, 1), groups=32
    )
    out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

    f = relay.Function([data1, weight1, weight2], out)
    ref_mod = tvm.IRModule()
    ref_mod["main"] = f

    f = set_external_func_attr(f, "dnnl", "dnnl_0")
    call = relay.Call(f, [data0, weight0, weight0])
    mod = tvm.IRModule.from_expr(call)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu())
    ref_res = ref_ex.evaluate()(i_data, w_data, w_data)
    check_result(
        mod, {"data0": i_data, "weight0": w_data}, (1, 32, 14, 14), ref_res.numpy(), tol=1e-5
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True),
    reason="skip because DNNL codegen is not available",
)
@pytest.mark.parametrize("check_result", [check_vm_result, check_graph_executor_result])
def test_extern_dnnl_const(check_result):
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)
    data0 = relay.var("data0", shape=(ishape), dtype=dtype)
    w_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    data1 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight1 = relay.const(w_data, dtype=dtype)
    weight2 = relay.const(w_data, dtype=dtype)
    depthwise_conv2d_1 = relay.nn.conv2d(
        data1, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
    )
    depthwise_conv2d_2 = relay.nn.conv2d(
        depthwise_conv2d_1, weight2, kernel_size=(3, 3), padding=(1, 1), groups=32
    )
    out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

    f = relay.Function([data1], out)
    ref_mod = tvm.IRModule()
    ref_mod["main"] = f

    f = set_external_func_attr(f, "dnnl", "dnnl_0")
    call = relay.Call(f, [data0])
    mod = tvm.IRModule.from_expr(call)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu())
    ref_res = ref_ex.evaluate()(i_data)
    check_result(mod, {"data0": i_data}, (1, 32, 14, 14), ref_res.numpy(), tol=1e-5)


def test_load_params_with_constants_in_ext_codegen():
    # After binding params and partitioning graph_module.get_params()
    # might contain parameters that are not an graph executor input but
    # for example constants in external function.
    y_in = np.ones((1,)).astype("float32")
    params = {"y": y_in}
    mod = tvm.IRModule()
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1,))
    xcb = compiler_begin(x, "ccompiler")
    ycb = compiler_begin(y, "ccompiler")
    z = relay.add(xcb, ycb)
    zce = compiler_end(z, "ccompiler")
    mod["main"] = relay.Function([x, y], zce)
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = relay.transform.PartitionGraph()(mod)

    graph_module = relay.build(mod, target="llvm", params=params)
    # Params will be stored in metadata module.
    assert len(graph_module.get_params()) == 0
    lib = update_lib(graph_module.get_lib())
    rt_mod = tvm.contrib.graph_executor.create(graph_module.get_graph_json(), lib, tvm.cpu(0))
    rt_mod.load_params(runtime.save_param_dict(graph_module.get_params()))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
