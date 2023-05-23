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
from tvm import te
from tvm import relay
from tvm.relay import transform


def test_basic():
    mod = tvm.IRModule()
    x2 = relay.var("x2", shape=(10, 5))
    y2 = relay.var("y2", shape=(1, 5))
    level2_func = relay.Function([x2, y2], relay.op.add(x2, y2))

    x1 = relay.var("x1", shape=(10, 5))
    y1 = relay.var("y1", shape=(1, 5))
    level1_func = relay.Function([x1, y1], level2_func(x1, y1))

    mod["main"] = level1_func
    mod = relay.transform.InferType()(mod)
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 2


def test_closure():
    mod = tvm.IRModule()

    x = relay.var("x", shape=(2,))
    y = relay.var("y", shape=(2,))
    inner_func = relay.Function([x], x + y)
    outer_func = relay.Function([y], inner_func)
    clo = outer_func(relay.ones(shape=(2,), dtype="float32"))
    mod["main"] = relay.Function([], relay.Call(clo, [relay.zeros(shape=(2,), dtype="float32")]))

    mod = relay.transform.InferType()(mod)
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 3


def test_recursive():
    mod = tvm.IRModule()

    x = relay.var("x", shape=(2,))
    i = relay.var("i", shape=(), dtype="int32")
    s = relay.var("s", shape=(2,))
    cond = i < relay.const(10, dtype="int32")

    loop = relay.var("while_loop")
    sb = relay.scope_builder.ScopeBuilder()
    with sb.if_scope(cond):
        ii = i + relay.const(1, dtype="int32")
        ss = s + x
        sb.ret(loop(ii, ss))
    with sb.else_scope():
        sb.ret(s)
    func = relay.Function([i, s], sb.get())

    ret = relay.Let(
        loop, func, loop(relay.const(0, dtype="int32"), relay.zeros(shape=(2,), dtype="float32"))
    )
    mod["main"] = relay.Function([x], ret)

    mod = relay.transform.InferType()(mod)
    new_mod = transform.LambdaLift()(mod)
    assert len(new_mod.functions) == 2


if __name__ == "__main__":
    tvm.testing.main()
