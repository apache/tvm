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
import tvm.relay
import tvm.testing


@tvm.testing.parametrize_targets
def test_threefry_repeatability(target, ctx):
    target, ctx = "llvm", tvm.cpu(0)
    key1 = tvm.relay.random.threefry_key(1)
    rand1 = tvm.relay.random.threefry_generate(key1, (12,))
    out_key1, out1 = tvm.relay.create_executor(
        "vm", tvm.IRModule.from_expr(tvm.relay.Function([], rand1)), target=target, ctx=ctx
    ).evaluate()()

    key2 = tvm.relay.random.threefry_key(1)
    rand2 = tvm.relay.random.threefry_generate(key2, (12,))
    out_key2, out2 = tvm.relay.create_executor(
        "vm", tvm.IRModule.from_expr(tvm.relay.Function([], rand2)), target=target, ctx=ctx
    ).evaluate()()

    assert (
        out1.asnumpy() == out2.asnumpy()
    ).all(), "Generate on same seed should have the same output random numbers"

    assert (
        out_key1.asnumpy() == out_key2.asnumpy()
    ).all(), "Generate on same seed should have the same next keys"


@tvm.testing.parametrize_targets
def test_threefry_split(target, ctx):
    key = tvm.relay.random.threefry_key(1)
    left, right = tvm.relay.TupleWrapper(tvm.relay.random.threefry_split(key), 2)
    _, rand1 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(left, (12,)), 2)
    _, rand2 = tvm.relay.TupleWrapper(tvm.relay.random.threefry_generate(right, (12,)), 2)
    out1, out2 = tvm.relay.create_executor(
        "vm",
        tvm.IRModule.from_expr(tvm.relay.Function([], tvm.relay.Tuple((rand1, rand2)))),
        target=target,
        ctx=ctx,
    ).evaluate()()

    assert (
        out1.asnumpy() != out2.asnumpy()
    ).any(), "Generate after split should not have the same output"


if __name__ == "__main__":
    test_threefry_repeatability(tvm.target.Target("llvm"), tvm.context("cpu"))
    test_threefry_split(tvm.target.Target("llvm"), tvm.context("cpu"))
