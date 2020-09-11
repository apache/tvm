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
import pytest

from tvm import relay
from tvm.relay.testing import check_grad


def test_cross_entropy_grad():
    for dtype in ("float32", "float64"):
        x = relay.var("x", shape=(2, 5), dtype=dtype)
        y = relay.var("y", shape=(2, 5), dtype=dtype)
        check_grad(
            relay.Function([x, y], relay.op.nn.cross_entropy(x, y)), eps=0.01, scale=0.1, mean=1
        )


def test_cross_entropy_with_logits_grad():
    for dtype in ("float32", "float64"):
        x = relay.var("x", shape=(2, 5), dtype=dtype)
        y = relay.var("y", shape=(2, 5), dtype=dtype)
        check_grad(
            relay.Function([x, y], relay.op.nn.cross_entropy_with_logits(x, y)),
            eps=0.01,
            scale=0.1,
            mean=1,
        )


def test_checkpoint():
    inputs = [relay.var("x{}".format(i), shape=(1,)) for i in range(4)]
    output = relay.multiply(relay.add(inputs[0], inputs[1]), relay.add(inputs[2], inputs[3]))
    check_grad(relay.Function(inputs, relay.annotation.checkpoint(output)))

    scope = relay.ScopeBuilder()
    out_tuple = scope.let(
        "out_tuple",
        relay.Tuple([relay.add(inputs[0], inputs[1]), relay.multiply(inputs[2], inputs[3])]),
    )
    scope.ret(
        relay.subtract(
            relay.annotation.checkpoint(relay.TupleGetItem(out_tuple, 0)),
            relay.TupleGetItem(out_tuple, 1),
        )
    )
    out_single = scope.get()
    check_grad(relay.Function(inputs, out_single))


def test_batch_matmul_grad():
    x = relay.var("x", shape=(2, 3, 5), dtype="float64")
    y = relay.var("y", shape=(2, 4, 5), dtype="float64")
    check_grad(relay.Function([x, y], relay.op.nn.batch_matmul(x, y)))


if __name__ == "__main__":
    pytest.main([__file__])
