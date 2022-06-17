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

import sys
from collections import OrderedDict
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relay, runtime
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.annotation import compiler_begin, compiler_end
from utils.external_codegen import (
    update_lib,
    set_external_func_attr,
    parametrize_external_codegen_checks,
    parametrize_external_json_codegen_checks,
    check_graph_executor_result,
    check_vm_result,
)


@parametrize_external_codegen_checks
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


@parametrize_external_codegen_checks
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


@parametrize_external_codegen_checks
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


@parametrize_external_codegen_checks
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


# TODO(mbs): The check_aot_executor_result does not support the list-of-targets, mostly because
# tvm.testing.aot.compile_and_run requires the target to be a kind name string, and
# tvm.testing.aot.compile_models requires a single Target object. However, code outside of
# tvm.testing.aot is ready for this more general form.
@pytest.mark.parametrize("check_result", [check_graph_executor_result, check_vm_result])
def test_extern_gcc_with_target_instance(check_result):
    shape = (8, 8)
    dtype = "int32"

    def make_mod():
        x0 = relay.var("x0", shape=shape, dtype=dtype)
        y0 = relay.var("y0", shape=shape, dtype=dtype)
        z = x0 + y0
        f = relay.Function([x0, y0], z)
        f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
        x = relay.var("x", shape=shape, dtype=dtype)
        y = relay.var("y", shape=shape, dtype=dtype)
        call = relay.Call(f, [x, y])
        return tvm.IRModule.from_expr(call)

    host_target = tvm.target.Target("llvm")
    generic_target = tvm.target.Target("llvm", host=host_target)
    # The header attribute is just whitespace, so compilation is as usual.
    good_extern_codegen_target = tvm.target.Target(
        {"kind": "ccompiler", "header": "// Good"}, host=host_target
    )
    # The header attribute is ill-formed, so compilation is expected to fail.
    bogus_extern_codegen_target = tvm.target.Target(
        {"kind": "ccompiler", "header": "Bogus"}, host=host_target
    )

    mod = make_mod()

    x_data = np.random.rand(*shape).astype(dtype)
    y_data = np.random.rand(*shape).astype(dtype)
    expected_result = x_data + y_data
    inputs = {"x": x_data, "y": y_data}

    check_result(
        mod, inputs, shape, expected_result, target=[generic_target, good_extern_codegen_target]
    )

    with pytest.raises(RuntimeError):
        check_result(
            mod,
            inputs,
            shape,
            expected_result,
            target=[generic_target, bogus_extern_codegen_target],
        )


@pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
@pytest.mark.parametrize("check_result", [check_graph_executor_result, check_vm_result])
def test_extern_gcc_consts(check_result):
    shape = (8, 8)
    dtype = "float32"
    x = relay.var("x", shape=shape)
    y0_data = np.random.uniform(0, 1, shape).astype(dtype)

    x0 = relay.var("x0", shape=shape)
    y0_const = relay.const(y0_data, dtype)
    z = x0 + y0_const
    f = relay.Function([x0], z)
    f = set_external_func_attr(f, "ccompiler", "ccompiler_0")
    call = relay.Call(f, [x])
    mod = tvm.IRModule.from_expr(call)

    # Note that while the VMCompiler get_params() will return all 'parameters' from both
    # TVM and external codegen compiled code, the GraphExecutor.get_params() will return only
    # those from non-external modules. So in the following we'll test by execution rather than
    # test by inspection.
    x_data = np.random.rand(*shape).astype(dtype)
    inputs = {"x": x_data}
    expected_result = x_data + y0_data
    check_result(mod, inputs, shape, expected_result, target="llvm")


@pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True),
    reason="skip because DNNL codegen is not available",
)
@parametrize_external_json_codegen_checks
def test_extern_dnnl_padding(check_result):
    dtype = "float32"
    ishape = (1, 1, 99, 12)
    w1shape = (54, 1, 3, 3)
    data0 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight0 = relay.var("weight0", shape=(w1shape), dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), strides=(2, 2), padding=(1, 0, 1, 1))
    f = relay.Function([data0, weight0], out)
    ref_mod = tvm.IRModule()
    ref_mod["main"] = f

    data1 = relay.var("data0", shape=(ishape), dtype=dtype)
    weight1 = relay.var("weight0", shape=(w1shape), dtype=dtype)
    f = set_external_func_attr(f, "dnnl", "dnnl_0")
    call = relay.Call(f, [data1, weight1])
    mod = tvm.IRModule.from_expr(call)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu()).evaluate()(
        i_data, w_data
    )
    check_result(
        mod, {"data0": i_data, "weight0": w_data}, (1, 54, 50, 6), ref_res.numpy(), tol=1e-5
    )


@pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True),
    reason="skip because DNNL codegen is not available",
)
@parametrize_external_json_codegen_checks
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

    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu()).evaluate()(
        i_data, w_data, w_data
    )
    check_result(
        mod, {"data0": i_data, "weight0": w_data}, (1, 32, 14, 14), ref_res.numpy(), tol=1e-5
    )


@pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True),
    reason="skip because DNNL codegen is not available",
)
@parametrize_external_json_codegen_checks
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

    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu()).evaluate()(i_data)
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
    tvm.testing.main()
