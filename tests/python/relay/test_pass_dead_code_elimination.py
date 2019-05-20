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
from nose.tools import nottest

import tvm
from tvm import relay
from tvm.relay.ir_pass import dead_code_elimination, alpha_equal
from tvm.relay.op import log, add, equal, subtract


class env:
    def __init__(self):
        self.a = relay.Var("a")
        self.b = relay.Var("b")
        self.c = relay.Var("c")
        self.d = relay.Var("d")
        self.e = relay.Var("e")
        self.x = relay.Var("x")
        self.y = relay.Var("y")
        self.z = relay.Var("z")
        self.shape = tvm.convert([1, 2, 3])
        self.tt = relay.TensorType(self.shape, "float32")
        self.int32 = relay.TensorType([], "int32")
        self.float32 = relay.TensorType([], "float32")
        self.one = relay.const(1.0)
        self.two = relay.const(2.0)
        self.three = relay.const(3.0)


e = env()


def test_let():
    orig = relay.Let(e.x, e.y, e.z)
    assert alpha_equal(dead_code_elimination(orig), e.z)


def test_used_let():
    orig = relay.Let(e.c, e.one, e.c + e.c)
    assert alpha_equal(dead_code_elimination(orig), relay.Let(e.c, e.one, e.c + e.c))

@nottest
def test_inline():
    orig = relay.Let(e.a, e.b, relay.Let(e.c, e.d, e.c))
    assert alpha_equal(dead_code_elimination(orig), e.d)


def test_chain_unused_let():
    orig = relay.Let(e.a, e.b, relay.Let(e.c, e.d, e.e))
    assert alpha_equal(dead_code_elimination(orig), e.e)


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
    f = relay.Var("f")
    n = relay.Var("n", e.int32)
    data = relay.Var("data", e.float32)
    funcbody = relay.If(equal(n, relay.const(0)),
                        data,
                        relay.Call(f, [subtract(n, relay.const(1.0)),
                                       log(data)]))
    value = relay.Function([n, data], funcbody, e.float32, [])
    orig = relay.Let(f, value, relay.Call(f, [relay.const(2.0), relay.const(10000.0)]))
    assert alpha_equal(dead_code_elimination(orig), orig)
    assert alpha_equal(dead_code_elimination(relay.Let(f, value, e.three)), e.three)


def test_op_let():
    assert alpha_equal(dead_code_elimination(add(relay.Let(e.a, e.one, e.three), e.two)), add(e.three, e.two))


def test_tuple_get_item():
    t = relay.Var('t')
    g = relay.TupleGetItem(t, 0)
    assert alpha_equal(dead_code_elimination(g), g)
    assert alpha_equal(dead_code_elimination(relay.TupleGetItem(relay.Let(e.a, e.one, t), 0)), g)


if __name__ == "__main__":
    test_let()
    test_used_let()
    test_inline()
    test_chain_unused_let()
    test_recursion()
    test_op_let()
    test_tuple_get_item()
