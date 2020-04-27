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
    assert tvm.ir.structural_equal(z, expected())


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
        if isinstance(expr, relay.expr.Call) and expr.op.name == 'add':
            return True
        return False

    z = before()
    z = run_opt_pass(z, transform.EliminateCommonSubexpr(fskip))
    assert tvm.ir.structural_equal(z, expected())


if __name__ == "__main__":
    test_simple()
    test_callback()
