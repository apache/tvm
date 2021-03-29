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

import tvm
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
    tvm.testing.assert_allclose(eval_result.asnumpy(), rts_result.asnumpy())
    tvm.testing.assert_allclose(eval_result.asnumpy(), expected_result)


def test_add_op_scalar():
    """
    Program:
        fn (x, y) {
            return x + y;
        }
    """
    x = relay.var("x", shape=())
    y = relay.var("y", shape=())
    func = relay.Function([x, y], add(x, y))
    x_data = np.array(10.0, dtype="float32")
    y_data = np.array(1.0, dtype="float32")
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
    res = mod.get_output(0).asnumpy()
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
    smap = relay.backend._backend.GraphPlanMemory(func)
    storage_ids = set()
    device_types = set()
    for k, v in smap.items():
        assert len(v) == 2
        for x in v[0]:
            storage_ids.add(x.value)
        for x in v[1]:
            device_types.add(x.value)

    # Current rule requires vars have unique storage id
    # because we don't do inplace, we will need another
    # two alternating temporary space.
    assert len(storage_ids) == 4
    assert len(device_types) == 1


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
            out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
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
        out = mod.get_output(i).asnumpy()
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
    tvm.testing.assert_allclose(out[0].asnumpy(), data[0])
    assert len(out[1]) == 2
    tvm.testing.assert_allclose(out[1][0].asnumpy(), data[1])
    assert len(out[1][1]) == 2
    tvm.testing.assert_allclose(out[1][1][0].asnumpy(), data[2])
    tvm.testing.assert_allclose(out[1][1][1].asnumpy(), data[3])


if __name__ == "__main__":
    test_plan_memory()
    test_with_params()
    test_add_op_scalar()
    test_add_op_tensor()
    test_add_op_broadcast()
    test_gru_like()
    test_compile_nested_tuples()
