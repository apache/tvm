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
import tvm.testing
import numpy as np
from tvm import relay


def test_compile_engine():
    engine = relay.backend.compile_engine.get()
    def get_func(shape):
        x = relay.var("x", shape=shape)
        y = relay.add(x, x)
        z = relay.add(y, x)
        f = relay.Function([x], z)
        mod = relay.Module.from_expr(f)
        mod = relay.transform.InferType()(mod)
        return mod["main"]
    z1 = engine.lower(get_func((10,)), "llvm")
    z2 = engine.lower(get_func((10,)), "llvm")
    z3 = engine.lower(get_func(()), "llvm")
    assert z1.same_as(z2)
    assert not z3.same_as(z1)
    if tvm.context("cuda").exist:
        z4 = engine.lower(get_func(()), "cuda")
        assert not z3.same_as(z4)

    # Test JIT target
    for target in ["llvm"]:
        ctx = tvm.context(target)
        if ctx.exist:
            f = engine.jit(get_func((10,)), target)
            x = tvm.nd.array(np.ones(10).astype("float32"), ctx=ctx)
            y = tvm.nd.empty((10,), ctx=ctx)
            f(x, y)
            tvm.testing.assert_allclose(
                y.asnumpy(), x.asnumpy() * 3)
    engine.dump()

def test_compile_placeholder_bypass():
    engine = relay.backend.compile_engine.get()
    x = relay.var("x", shape=(2, 3))
    y = relay.var("y", shape=(2, 3))
    z = relay.var("z", shape=(2, 3))
    result = relay.Tuple([x, relay.op.concatenate([y, z], axis=0)])
    func = relay.Function(relay.analysis.free_vars(result), result)
    with relay.build_config(opt_level=0):
       graph, lib, params = relay.build(relay.Module.from_expr(func), 'llvm')


def test_compile_injective_with_tuple():
    x = relay.var("x", shape=(2, 3))
    y = relay.var("y", shape=(2, 3))
    x_transpose = relay.transpose(x)
    output = relay.Tuple([x_transpose, y])
    func = relay.Function([x, y], output)
    relay.build(relay.Module.from_expr(func), 'llvm')


def test_compile_tuple_dup():
    x = relay.var("data", shape=(16, 16))
    log = relay.log(x)
    output = relay.Tuple([log, log])
    f = relay.Function([x], output)
    relay.build(relay.Module.from_expr(f), 'llvm')


if __name__ == "__main__":
    test_compile_engine()
    test_compile_placeholder_bypass()
    test_compile_injective_with_tuple()
    test_compile_tuple_dup()
