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

import tvm
import json
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.op import add
import tvm.testing

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
    intrp = relay.create_executor("debug", mod=mod)
    graph = relay.create_executor("graph", mod=mod)
    eval_result = intrp.evaluate(expr)(*args)
    rts_result = graph.evaluate(expr)(*args)
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


def test_graph_executor_nested_tuples():
    x, y, z, w = [relay.var(c, shape=(2, 3), dtype="float32") for c in "xyzw"]
    out = relay.Tuple([x, relay.Tuple([y, relay.Tuple([z, w])])])
    func = relay.Function([x, y, z, w], out)

    exe = relay.create_executor(
        kind="graph", mod=tvm.IRModule.from_expr(func), device=tvm.cpu(0), target="llvm"
    )
    f = exe.evaluate()

    data = [np.random.uniform(size=(2, 3)).astype("float32") for _ in "xyzw"]
    out = f(*data)
    assert len(out) == 2
    tvm.testing.assert_allclose(out[0].numpy(), data[0])
    assert len(out[1]) == 2
    tvm.testing.assert_allclose(out[1][0].numpy(), data[1])
    assert len(out[1][1]) == 2
    tvm.testing.assert_allclose(out[1][1][0].numpy(), data[2])
    tvm.testing.assert_allclose(out[1][1][1].numpy(), data[3])


if __name__ == "__main__":
    pytest.main([__file__])
