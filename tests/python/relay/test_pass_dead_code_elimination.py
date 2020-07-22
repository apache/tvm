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
from tvm import te
from tvm import relay
from tvm.relay import Function, transform
from tvm.relay.analysis import free_vars
from tvm.relay.op import log, add, equal, subtract
from tvm.relay.testing import inception_v3

import pytest

class env:
    def __init__(self):
        self.shape = tvm.runtime.convert([1, 2, 3])
        self.tt = relay.TensorType(self.shape, "float32")
        self.int32 = relay.TensorType([], "int32")
        self.float32 = relay.TensorType([], "float32")
        self.one = relay.const(1.0)
        self.two = relay.const(2.0)
        self.three = relay.const(3.0)
        self.a = relay.Var("a", self.float32)
        self.b = relay.Var("b", self.float32)
        self.c = relay.Var("c", self.float32)
        self.d = relay.Var("d", self.float32)
        self.e = relay.Var("e", self.float32)
        self.x = relay.Var("x", self.int32)
        self.y = relay.Var("y", self.int32)
        self.z = relay.Var("z", self.int32)


e = env()


def run_opt_pass(expr, opt_pass):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_let():
    orig = relay.Let(e.x, e.y, e.z)
    orig = run_opt_pass(orig, transform.DeadCodeElimination())
    assert tvm.ir.structural_equal(Function(free_vars(orig), orig), Function([e.z], e.z))


def test_used_let():
    orig = relay.Let(e.c, e.one, e.c + e.c)
    orig = run_opt_pass(orig, transform.DeadCodeElimination())
    expected = relay.Let(e.c, e.one, e.c + e.c)
    assert tvm.ir.structural_equal(Function([], orig), Function([], expected))

def test_inline():
    orig = relay.Let(e.a, e.b, relay.Let(e.c, e.d, e.c))
    orig = run_opt_pass(orig, transform.DeadCodeElimination(True))
    tvm.ir.assert_structural_equal(Function(free_vars(orig), orig), Function([e.d], e.d))


def test_chain_unused_let():
    orig = relay.Let(e.a, e.b, relay.Let(e.c, e.d, e.e))
    orig = run_opt_pass(orig, transform.DeadCodeElimination())
    assert tvm.ir.structural_equal(Function(free_vars(orig), orig), Function([e.e], e.e))


def use_f(func):
    f = relay.Var("f")
    n = relay.Var("n", e.int32)
    data = relay.Var("data", e.float32)
    funcbody = relay.If(equal(n, relay.const(0)),
                        data,
                        relay.Call(f, [subtract(n, relay.const(1)),
                                       log(data)]))
    value = relay.Function([n, data], funcbody, e.float32, [])
    return relay.Let(f, value, func(f))

# make sure we dont infinite loop
def test_recursion():
    """
    Program:
       let f(n: i32, data: f32) -> f32 = {
          if (n == 0) {
              return data;
          } else {
              return f(n - 1, log(data));
          }
       }
       f(2, 10000);
    """
    orig = use_f(lambda f: relay.Call(f, [relay.const(2), relay.const(10000.0)]))
    dced = run_opt_pass(orig, transform.DeadCodeElimination())
    orig = run_opt_pass(orig, transform.InferType())
    tvm.ir.assert_structural_equal(dced, orig)

def test_recursion_dead():
    x = relay.Let(e.a, e.one, e.three)
    dced_f = lambda f: x
    dced = run_opt_pass(use_f(dced_f), transform.DeadCodeElimination())
    assert tvm.ir.structural_equal(dced, e.three)


def test_op_let():
    dced = run_opt_pass(add(relay.Let(e.a, e.one, e.three), e.two),
                        transform.DeadCodeElimination())
    assert tvm.ir.structural_equal(dced, add(e.three, e.two))


def test_tuple_get_item():
    tt = relay.TupleType([e.float32, e.float32])
    t = relay.Var('t', tt)
    a = relay.Var('a')
    g = relay.TupleGetItem(t, 0)
    dced = run_opt_pass(g, transform.DeadCodeElimination())
    assert tvm.ir.structural_equal(Function(free_vars(dced), dced), Function(free_vars(g), g))
    orig = relay.TupleGetItem(relay.Let(a, e.one, t), 0)
    dced = run_opt_pass(orig, transform.DeadCodeElimination())
    assert tvm.ir.structural_equal(Function(free_vars(dced), dced), Function(free_vars(g), g))


@pytest.mark.timeout(timeout=10, method="thread")
def test_complexity():
    g = inception_v3.get_net(1, 1000, (3, 299, 299), 'float32')
    run_opt_pass(g, transform.DeadCodeElimination())


if __name__ == "__main__":
    test_let()
    test_used_let()
    test_inline()
    test_chain_unused_let()
    test_recursion()
    test_recursion_dead()
    test_op_let()
    test_tuple_get_item()
    test_complexity()
