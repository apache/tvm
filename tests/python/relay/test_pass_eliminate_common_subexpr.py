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
"""Test eliminate common subexpr pass"""
import numpy as np
import tvm
from tvm import te

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import transform, analysis


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_simple():
    def before():
        x = relay.var("x", shape=(1, 16))
        y1 = relay.nn.relu(x)
        y2 = relay.nn.relu(x)
        y1 = relay.add(y1, relay.const(1.0, "float32"))
        y2 = relay.add(y2, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16))
        y = relay.nn.relu(x)
        y = relay.add(y, relay.const(1.0, "float32"))
        y = relay.add(y, y)
        f = relay.Function([x], y)
        return run_opt_pass(f, transform.InferType())

    z = before()
    z = run_opt_pass(z, transform.EliminateCommonSubexpr())
    tvm.ir.assert_structural_equal(z, expected())


def test_callback():
    def before():
        x = relay.var("x", shape=(1, 16))
        y1 = relay.nn.relu(x)
        y2 = relay.nn.relu(x)
        y1 = relay.add(y1, relay.const(1.0, "float32"))
        y2 = relay.add(y2, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16))
        y = relay.nn.relu(x)
        y1 = relay.add(y, relay.const(1.0, "float32"))
        y2 = relay.add(y, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return run_opt_pass(f, transform.InferType())

    def fskip(expr):
        if isinstance(expr, relay.expr.Call) and expr.op.name == "add":
            return True
        return False

    z = before()
    z = run_opt_pass(z, transform.EliminateCommonSubexpr(fskip))
    tvm.ir.assert_structural_equal(z, expected())


def test_tuple_get_time():
    def before():
        x = relay.var("x", shape=(1, 16, 1, 1))
        var = relay.var("var", shape=(16,))
        mean = relay.var("mean", shape=(16,))
        beta = relay.var("beta", shape=(16,))
        gamma = relay.var("gamma", shape=(16,))
        BN = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)
        T1 = BN[0]
        T2 = BN[0]
        add = T1 + T2
        f = relay.Function([x, var, mean, beta, gamma], add)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16, 1, 1))
        var = relay.var("var", shape=(16,))
        mean = relay.var("mean", shape=(16,))
        beta = relay.var("beta", shape=(16,))
        gamma = relay.var("gamma", shape=(16,))
        BN = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)
        T1 = BN[0]
        add = T1 + T1
        f = relay.Function([x, var, mean, beta, gamma], add)
        return run_opt_pass(f, transform.InferType())

    z = before()
    z = run_opt_pass(z, transform.EliminateCommonSubexpr())
    tvm.ir.assert_structural_equal(z, expected())


def test_tuple_arg():
    def before():
        x = relay.var("x", shape=(1, 16))
        y1 = relay.nn.relu(x)
        y2 = relay.nn.relu(x)
        y1 = relay.add(y1, relay.const(1.0, "float32"))
        y2 = relay.add(y2, relay.const(1.0, "float32"))
        c0 = relay.const(np.ones((1, 16)), "float32")
        y1 = relay.concatenate([y1, c0], axis=0)
        y2 = relay.concatenate([y2, c0], axis=0)
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16))
        y = relay.nn.relu(x)
        y = relay.add(y, relay.const(1.0, "float32"))
        c0 = relay.const(np.ones((1, 16)), "float32")
        y = relay.concatenate([y, c0], axis=0)
        y = relay.add(y, y)
        f = relay.Function([x], y)
        return run_opt_pass(f, transform.InferType())

    z = before()
    z = run_opt_pass(z, transform.EliminateCommonSubexpr())
    tvm.ir.assert_structural_equal(z, expected())


if __name__ == "__main__":
    tvm.testing.main()
