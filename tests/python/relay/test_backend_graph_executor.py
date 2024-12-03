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
from unittest.mock import patch

import tvm
import json
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.op import add
import tvm.testing
from tvm.relay.testing import mlp
from tvm import rpc
from tvm.contrib import utils

# @tq, @jr should we put this in testing ns?
def check_rts(expr, args, expected_result, mod=None):
    """
    Check that evaluating `expr` applied to the arguments produces
    `result` on both the evaluator and TVM runtime.

    Parameters
    ----------
    expr:
        The expression to evaluate

    args: list of Expr
        The arguments to supply the expr.

    expected_result:
        The expected result of running the expression.
    """
    eval_result = relay.create_executor("debug", mod=mod).evaluate(expr)(*args)
    rts_result = relay.create_executor("graph", mod=mod).evaluate(expr)(*args)
    tvm.testing.assert_allclose(eval_result.numpy(), rts_result.numpy())
    tvm.testing.assert_allclose(eval_result.numpy(), expected_result)


def test_add_op_scalar():
    """
    test_add_op_scalar:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var("x", shape=())  # Default to float32
    y = relay.var("y", shape=())  # Default to float32
    func = relay.Function([x, y], add(x, y))
    x_y_data = [
        (np.array(10.0, dtype="float32"), np.array(1.0, dtype="float32")),
        (np.float32(10.0), np.float32(1.0)),
        (10.0, 1.0),
    ]
    for (x_data, y_data) in x_y_data:
        check_rts(func, [x_data, y_data], x_data + y_data)


def test_add_op_scalar_int():
    """
    test_add_op_scalar_int:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var("x", shape=(), dtype="int32")
    y = relay.var("y", shape=(), dtype="int32")
    func = relay.Function([x, y], add(x, y))
    x_y_data = [
        (np.array(10.0, dtype="int32"), np.array(1.0, dtype="int32")),
        (np.int32(10), np.int32(1)),
        (10, 1),
    ]
    for (x_data, y_data) in x_y_data:
        check_rts(func, [x_data, y_data], x_data + y_data)


def test_add_op_tensor():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(10, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype("float32")
    y_data = np.random.rand(10, 5).astype("float32")
    check_rts(func, [x_data, y_data], x_data + y_data)


def test_add_op_broadcast():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(1, 5))
    func = relay.Function([x, y], add(x, y))
    x_data = np.random.rand(10, 5).astype("float32")
    y_data = np.random.rand(1, 5).astype("float32")
    check_rts(func, [x_data, y_data], x_data + y_data)


def test_with_params():
    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(1, 5))
    z = relay.add(x, y)
    z = relay.exp(z)
    func = relay.Function([x, y], z)
    x_data = np.random.rand(10, 5).astype("float32")
    y_data = np.random.rand(1, 5).astype("float32")
    params = {"y": y_data}
    graph, lib, params = relay.build(tvm.IRModule.from_expr(func), "llvm", params=params)
    mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
    mod.set_input(**params)
    mod.set_input(x=x_data)
    mod.run()
    res = mod.get_output(0).numpy()
    ref_res = np.exp(y_data + x_data)
    tvm.testing.assert_allclose(res, ref_res, atol=1e-5, rtol=1e-5)


def test_plan_memory():
    # it is sufficient to cycle through two memories.

    x = relay.var("x", shape=(10,))
    y = relay.var("x", shape=(1,))
    y2 = relay.exp(y)
    z = relay.add(x, y2)
    z = relay.exp(z)
    z = relay.exp(z)
    z = relay.exp(z)
    z = relay.exp(z)
    z = relay.exp(z)
    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.FuseOps(0)(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)
    memory_plan = relay.backend._backend.GraphPlanMemory(func)
    storage_ids = set()
    device_types = set()
    storage_sizes = {}

    for k, v in memory_plan.expr_to_storage_info.items():
        for x in v.storage_ids:
            storage_ids.add(x)
            storage_sizes[x] = v.storage_sizes
        for x in v.device_types:
            device_types.add(x)

    # Current rule requires vars have unique storage id
    # because we don't do inplace, we will need another
    # two alternating temporary space.
    assert len(storage_ids) == 4, f"found storage_ids: {storage_ids}"
    assert len(device_types) == 1
    assert len(storage_sizes) == 4

    # Check the specific size of each sid
    assert (
        storage_sizes[0][0] == 40
        and storage_sizes[1][0] == 4
        and storage_sizes[2][0] == 4
        and storage_sizes[3][0] == 40
    )


def test_plan_2d_memory():
    """Verification if GraphPlanMemory manages 2d memory reffered as
    global.texture* memory scopes in json file."""
    global_virtual_device = tvm.target.VirtualDevice(memory_scope="global")
    texture_virtual_device = tvm.target.VirtualDevice(memory_scope="global.texture")
    metatable = {
        "VirtualDevice": [
            global_virtual_device,
            texture_virtual_device,
        ]
    }

    mod = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%data1: Tensor[(1, 32, 40, 40), float32],
                  %data2: Tensor[(1, 32, 40, 40), float32]) {
          %0 = fn (%a, Primitive=1) {
            layout_transform(%a, src_layout="NCHW", dst_layout="NCHW4c")
          };
          %1 = %0(%data1);
          %3 = %0(%data2);
          %5 = fn (%a {virtual_device=meta[VirtualDevice][0]},  // global
                   %b {virtual_device=meta[VirtualDevice][0]},  // global
                   virtual_device=meta[VirtualDevice][1],       // texture
                   Primitive=1) {
            add(%a, %b)
          };
          %6 = %5(%1, %3);
          %7 = fn (%a {virtual_device=meta[VirtualDevice][1]},  // texture
                   %b {virtual_device=meta[VirtualDevice][0]},  // global
                   virtual_device=meta[VirtualDevice][1],       // texture
                   Primitive=1) {
            add(%a, %b)
          };
          %8 = %7(%6, %3);
          %9 = fn (%a {virtual_device=meta[VirtualDevice][1]},  // texture
                   %b {virtual_device=meta[VirtualDevice][1]},  // texture
                   virtual_device=meta[VirtualDevice][1],       // texture
                   Primitive=1) {
            add(%a, %b)
          };
          %10 = %9(%8, %6);
          %11 = fn (%a,
                    virtual_device=meta[VirtualDevice][0],      // global
                    Primitive=1) {
            layout_transform(%a, src_layout="NCHW4c", dst_layout="NCHW")
          };
          %11(%10)
        }
        """,
        "from_string",
        None,
        metatable,
    )

    GPU_DEVICE = tvm.device("cuda")
    HOST_TARGET = tvm.target.Target("llvm")
    GPU_TARGET = tvm.target.Target("cuda").with_host(HOST_TARGET)
    GPU = tvm.target.VirtualDevice(GPU_DEVICE, GPU_TARGET)  # device_type=2
    CTXT = tvm.transform.PassContext(config={"relay.fallback_device_type": GPU.device_type_int})
    config = tvm.target.make_compilation_config(CTXT, GPU_TARGET)
    mod = relay.transform.InferType()(mod)
    # PlanDevices should succeed.
    mod = relay.transform.PlanDevices(config)(mod)

    func = mod["main"]
    memory_plan = relay.backend._backend.GraphPlanMemory(func)
    virtual_devices = {}

    # We do not have execution ordered information, the only order that we can stick
    # in this place - storage_id
    # for above graph we know that
    # We have
    #  - 8 manageable storages for above graph
    #  - 5 of them are buffers
    #  - 3 of them are textures (2d storages)
    #  - 1 of buffer will be reused, since we have storage id maped data, we will have 4th
    #      storage id reuesed and hidden in virtual_devices map
    #  - no textures are reused so far
    for k, v in memory_plan.expr_to_storage_info.items():
        virtual_devices[v.storage_ids[0]] = v.virtual_devices[0].memory_scope

    # Check the scopes according to abvoce expectaions
    assert (
        virtual_devices[0] == "global"
        and virtual_devices[1] == "global"
        and virtual_devices[2] == "global"
        and virtual_devices[3] == "global"
        and virtual_devices[4] == "global.texture"
        and virtual_devices[5] == "global.texture"
        and virtual_devices[6] == "global.texture"
    )


def test_reshape_nop():
    # test that reshape can be turned into nop
    x = relay.var("x", shape=(10, 4))
    xx = relay.abs(x)
    y = relay.expand_dims(xx, axis=1)
    t0 = relay.reshape(y, (1, 40))
    t1 = relay.abs(y)

    z0 = relay.reshape(t0, (2, 20))
    z1 = relay.sqrt(t1)
    z2 = relay.reshape(t1, (1, 40))

    func = relay.Function([x], relay.Tuple([z0, z1, z2]))
    x_data = np.random.rand(10, 4).astype("float32")
    graph = relay.build(tvm.IRModule.from_expr(func), "llvm")
    graph_json_str = graph.get_graph_json()

    graph_json = json.loads(graph_json_str)

    # reshape must force sharing memory
    storage_ids = graph_json["attrs"]["storage_id"][1]
    assert tuple(storage_ids) == (0, 1, 1, 2, 3, 2)
    assert graph_json["nodes"][2]["attrs"]["func_name"] == "__nop"
    assert graph_json["nodes"][5]["attrs"]["func_name"] == "__nop"

    gmod = graph_executor.GraphModule(graph["default"](tvm.cpu(0)))

    gmod.set_input(x=x_data)
    gmod.run()
    z0_np = x_data.reshape(2, 20)
    z1_np = np.sqrt(
        np.abs(
            x_data.reshape(
                10,
                1,
                4,
            )
        )
    )
    z2_np = np.abs(x_data).reshape(1, 40)
    tvm.testing.assert_allclose(gmod.get_output(0).numpy(), z0_np)
    tvm.testing.assert_allclose(gmod.get_output(1).numpy(), z1_np)
    tvm.testing.assert_allclose(gmod.get_output(2).numpy(), z2_np)


@tvm.testing.uses_gpu
def test_gru_like():
    def unit(rnn_dim):
        X = relay.var("X", shape=(1, rnn_dim))
        W = relay.var("y", shape=(3 * rnn_dim, rnn_dim))
        matmul = relay.nn.dense(X, W)
        splitted = relay.split(matmul, indices_or_sections=3, axis=1)
        out = relay.sigmoid(splitted[0]) + relay.tanh(splitted[1]) * relay.exp(splitted[2])
        return relay.Function([X, W], out)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def unit_numpy(X, W):
        prod = np.dot(X, W.transpose())
        splits = np.split(prod, indices_or_sections=3, axis=1)
        return sigmoid(splits[0]) + np.tanh(splits[1]) * np.exp(splits[2])

    dtype = "float32"
    rnn_dim = 1000
    x = np.random.rand(1, rnn_dim).astype(dtype)
    y = np.random.rand(3 * rnn_dim, rnn_dim).astype(dtype) * 0.01 - 0.005
    out_shape = (1, rnn_dim)
    z = unit(rnn_dim)

    for target, dev in tvm.testing.enabled_targets():
        with tvm.transform.PassContext(opt_level=2):
            graph, lib, params = relay.build(tvm.IRModule.from_expr(z), target)
            m = graph_executor.create(graph, lib, dev)
            m.set_input("X", tvm.nd.array(x.astype(dtype)))
            m.set_input("y", tvm.nd.array(y.astype(dtype)))
            m.set_input(**params)
            m.run()
            out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).numpy()
            ref = unit_numpy(x, y)
            tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_compile_nested_tuples():
    x = relay.var("x", shape=(10,))
    x1 = x + relay.const(1.0)
    x2 = x1 + relay.const(1.0)
    x3 = x2 + relay.const(1.0)
    x4 = x3 + relay.const(1.0)
    out = relay.Tuple([x1, relay.Tuple([relay.Tuple([x2, x3]), x4])])
    func = relay.Function([x], out)

    graph, lib, _ = relay.build(tvm.IRModule.from_expr(func), "llvm")
    mod = graph_executor.create(graph, lib, device=tvm.cpu(0))

    x_data = np.random.uniform(size=(10,)).astype(np.float32)
    mod.set_input(x=x_data)
    mod.run()

    assert mod.get_num_outputs() == 4

    ref = x_data + 1
    for i in range(mod.get_num_outputs()):
        out = mod.get_output(i).numpy()
        tvm.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
        ref = ref + 1


def test_compile_return_empty_tuple():
    x = relay.var("x", shape=[16], dtype="float32")
    mod = tvm.IRModule.from_expr(relay.Function([x], relay.Tuple([])))
    graph, lib, _ = relay.build(mod, "llvm")
    mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
    mod.run()


@tvm.testing.uses_gpu
def test_compile_fused_identity_cast():
    # a fused function that would optimized to identity
    x = relay.var("x", shape=[16], dtype="float32")
    y = relay.cast(x, "float32")
    func1 = relay.Function([x], y).with_attr("Primitive", 1)

    # a fused function with param pass-through
    x = relay.var("x", shape=[16], dtype="float32")
    y = relay.add(x, relay.const(3.14, "float32"))
    func2 = relay.Function([x], relay.Tuple([x, y])).with_attr("Primitive", 1)

    x_global = relay.var("xx", shape=[16], dtype="float32")
    tup = func2(x_global)
    y_global = func1(relay.TupleGetItem(tup, 0) + relay.TupleGetItem(tup, 1))

    mod = tvm.IRModule.from_expr(relay.Function([x_global], y_global))
    for target, device in tvm.testing.enabled_targets():
        with tvm.transform.PassContext(opt_level=2):
            graph, lib, _ = relay.build(mod, target=target)
            executor = graph_executor.create(graph, lib, device=device)
            executor.run()


def test_graph_executor_nested_tuples():
    x, y, z, w = [relay.var(c, shape=(2, 3), dtype="float32") for c in "xyzw"]
    out = relay.Tuple([x, relay.Tuple([y, relay.Tuple([z, w])])])
    func = relay.Function([x, y, z, w], out)

    f = relay.create_executor(
        kind="graph", mod=tvm.IRModule.from_expr(func), device=tvm.cpu(0), target="llvm"
    ).evaluate()

    data = [np.random.uniform(size=(2, 3)).astype("float32") for _ in "xyzw"]
    out = f(*data)
    assert len(out) == 2
    tvm.testing.assert_allclose(out[0].numpy(), data[0])
    assert len(out[1]) == 2
    tvm.testing.assert_allclose(out[1][0].numpy(), data[1])
    assert len(out[1][1]) == 2
    tvm.testing.assert_allclose(out[1][1][0].numpy(), data[2])
    tvm.testing.assert_allclose(out[1][1][1].numpy(), data[3])


def test_graph_executor_api():
    dname_0, dname_1 = "data_0", "data_1"
    data_0, data_1 = [relay.var(c, shape=(1, 1), dtype="float32") for c in [dname_0, dname_1]]
    net = relay.add(data_0, data_1)
    func = relay.Function((data_0, data_1), net)

    lib = relay.build(tvm.IRModule.from_expr(func), "llvm")
    mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))

    assert mod.get_input_index(dname_1) == 1
    assert mod.get_input_index(dname_0) == 0
    assert mod.get_input_index("Invalid") == -1

    shape_dict, dtype_dict = mod.get_input_info()
    assert isinstance(shape_dict, tvm.container.Map)
    assert isinstance(dtype_dict, tvm.container.Map)
    for data in [data_0, data_1]:
        name = data.name_hint
        ty = data.type_annotation
        # verify shape
        assert name in shape_dict
        assert isinstance(shape_dict[name], tvm.runtime.container.ShapeTuple)
        assert shape_dict[name] == tvm.runtime.container.ShapeTuple([i.value for i in ty.shape])
        # verify dtype
        assert name in dtype_dict
        assert isinstance(dtype_dict[name], tvm.runtime.container.String)
        assert dtype_dict[name] == ty.dtype

    shape_dict, dtype_dict = mod.get_output_info()
    assert isinstance(shape_dict, tvm.container.Map)
    assert isinstance(dtype_dict, tvm.container.Map)
    for i, key in enumerate(shape_dict):
        assert mod.get_output_index(key) == i


@tvm.testing.requires_llvm
def test_benchmark():
    mod, params = mlp.get_workload(1)
    lib = relay.build(mod, target="llvm", params=params)
    exe = graph_executor.create(lib.get_graph_json(), lib.lib, tvm.cpu())
    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"))
    result = exe.benchmark(tvm.cpu(), data=data, func_name="run", repeat=2, number=1)
    assert result.mean == result.median
    assert result.mean > 0
    assert len(result.results) == 2

    with patch.object(
        tvm.runtime.module.Module,
        "time_evaluator",
        return_value=lambda: tvm.runtime.module.BenchmarkResult([1, 2, 2, 5]),
    ) as method:
        result = exe.benchmark(tvm.cpu(), data=data, func_name="run", repeat=2, number=1)
        assert result.mean == 2.5
        assert result.median == 2.0
        assert result.max == 5
        assert result.min == 1
        assert result.std == 1.5


@tvm.testing.parametrize_targets("cuda", "llvm")
def test_benchmark_end_to_end(dev, target):
    mod, params = mlp.get_workload(1)
    lib = relay.build(mod, target=target, params=params)
    exe = graph_executor.create(lib.get_graph_json(), lib.lib, dev)
    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"))
    result = exe.benchmark(dev, data=data, func_name="run", repeat=2, number=1, end_to_end=True)
    assert result.mean > 0
    assert len(result.results) == 2


@tvm.testing.requires_cuda
def test_benchmark_end_to_end_rpc():
    server = rpc.Server("127.0.0.1")
    remote = rpc.connect(server.host, server.port)

    mod, params = mlp.get_workload(1)
    lib = relay.build(mod, target="cuda", params=params)

    temp = utils.tempdir()
    path = temp.relpath("library.so")
    lib.export_library(path)
    remote.upload(path)
    rlib = remote.load_module("library.so")

    dev = remote.device("cuda")
    exe = graph_executor.create(lib.get_graph_json(), rlib, dev)

    data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
    result = exe.benchmark(dev, data=data, func_name="run", repeat=2, number=1, end_to_end=True)
    assert result.mean > 0
    assert len(result.results) == 2


if __name__ == "__main__":
    tvm.testing.main()
