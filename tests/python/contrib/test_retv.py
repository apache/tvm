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
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import retv
import re
import numpy as np
import pytest


def test_model_A():
    x = relay.var("x")
    y = relay.const(1)
    z = relay.add(x, y)
    z = relay.multiply(x, z)
    mod = tvm.ir.IRModule().from_expr(z)
    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@main([Var(x)])\n"
        "   `--(call)\n"
        "      |--multiply\n"
        "      |--x\n"
        "      `--(call)\n"
        "         |--add\n"
        "         |--x\n"
        "         `--1\n\n"
    )
    assert res == golden


def test_tuple():
    x = relay.const(1)
    y = relay.const(2)
    z = relay.add(x, y)
    t = relay.Tuple([x, z])
    mod = tvm.ir.IRModule().from_expr(t)
    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@main([])\n"
        "   `--(1, (call))\n"
        "      |--1\n"
        "      `--(call)\n"
        "         |--add\n"
        "         |--1\n"
        "         `--2\n\n"
    )
    assert res == golden


def test_function():
    mod = tvm.IRModule()
    x = relay.var("x", shape=(2,))
    y = relay.var("y", shape=(2,))
    f = relay.Function(relay.analysis.free_vars(x + y), x + y)
    mod["main"] = relay.Function(relay.analysis.free_vars(f), f)
    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@main([])\n"
        "   `--Function_28372544([Var(x, ty=TensorType([2], float32)), Var(y, ty=TensorType([2], float32))])\n"
        "      `--(call)\n"
        "         |--add\n"
        "         |--x\n"
        "         `--y\n\n"
    )
    match = re.search(r"Function_.\d*\(", res)
    if match:
        res = res.replace(match.group(0), "Function_28372544(")
    else:
        assert False
    assert res == golden


def test_if():
    cond = relay.Var("cond")
    left = relay.Var("left")
    right = relay.Var("right")
    ife = relay.If(cond, left, right)
    mod = tvm.ir.IRModule().from_expr(ife)
    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@main([Var(cond), Var(left), Var(right)])\n"
        "   `--if(cond, true, false)\n"
        "      |--cond\n"
        "      |--left\n"
        "      `--right\n\n"
    )
    assert res == golden


def test_global_var():
    x = relay.var("x", shape=(1, 64, 56, 56))
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    y = relay.nn.conv2d(x, weight, channels=64, kernel_size=(3, 3), padding=(1, 1))
    y = relay.nn.relu(y)
    mod = tvm.IRModule()
    foo = relay.GlobalVar("foo")
    mod[foo] = relay.Function([x, weight], y)
    mod = transform.InferType()(mod)
    mod["main"] = relay.Function([x, weight], foo(x, weight))
    mod = transform.InferType()(mod)
    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@foo([Var(x, ty=TensorType([1, 64, 56, 56], float32)), Var(weight, ty=TensorType([64, 64, 3, 3], float32))])\n"
        "   `--(call)\n"
        "      |--nn.relu\n"
        "      `--(call)\n"
        "         |--nn.conv2d\n"
        "         |--x\n"
        "         `--weight\n\n"
        "@main([Var(x, ty=TensorType([1, 64, 56, 56], float32)), Var(weight, ty=TensorType([64, 64, 3, 3], float32))])\n"
        "   `--(call)\n"
        "      |--@foo\n"
        "      |--x\n"
        "      `--weight\n\n"
    )
    assert res == golden


def test_loop():
    from tvm.relay.loops import while_loop

    i = relay.var("i")

    def _cond(i):
        return relay.less(i, relay.const(10))

    def _body(i):
        x = i + relay.const(1)
        return (x,)

    loop = while_loop(_cond, [i], _body)
    body = loop(relay.const(2))
    func = relay.Function([], body)
    mod = tvm.IRModule()
    mod["main"] = func
    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@main([])\n"
        "   `--(call)\n"
        "      |--let(var, val, body)\n"
        "      |  |--while_loop\n"
        "      |  |--Function_34141568([Var(i)])\n"
        "      |  |  `--if(cond, true, false)\n"
        "      |  |     |--(call)\n"
        "      |  |     |  |--less\n"
        "      |  |     |  |--i\n"
        "      |  |     |  `--10\n"
        "      |  |     |--(call)\n"
        "      |  |     |  |--while_loop\n"
        "      |  |     |  `--(call)\n"
        "      |  |     |     |--add\n"
        "      |  |     |     |--i\n"
        "      |  |     |     `--1\n"
        "      |  |     `--()\n"
        "      |  |        `--i\n"
        "      |  `--while_loop\n"
        "      `--2\n\n"
    )
    match = re.search(r"Function_.\d*\(", res)
    if match:
        res = res.replace(match.group(0), "Function_34141568(")
    else:
        assert False
    assert res == golden


def test_where():
    x = relay.const(np.array([[1, 2], [3, 4]]), dtype="int64")
    y = relay.const(np.array([[5, 6], [7, 8]]), dtype="int64")
    condition = relay.const(np.array([[1], [0]]), dtype="int64")
    where = relay.where(condition, x, y)
    mod = tvm.IRModule().from_expr(where)

    viz_res = retv.ASTVisualization()
    mod = viz_res(mod)
    res = viz_res.get_output()
    golden = (
        "== The AST view of the IRModule is ==\n"
        "@main([])\n"
        "   `--(call)\n"
        "      |--where\n"
        "      |--meta[relay.Constant][0]\n\n"
        "      |--meta[relay.Constant][0]\n\n"
        "      `--meta[relay.Constant][0]\n\n\n"
    )
    assert res == golden


if __name__ == "__main__":
    pytest.main([__file__])
