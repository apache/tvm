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
import numpy as np

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


def verify_batch_matmul_grad(a_shape, b_shape, transpose_a, transpose_b):
    tensor_a = relay.var("tensor_a", relay.TensorType(a_shape, "float32"))
    tensor_b = relay.var("tensor_b", relay.TensorType(b_shape, "float32"))
    check_grad(
        relay.Function(
            [tensor_a, tensor_b],
            relay.op.nn.batch_matmul(
                tensor_a, tensor_b, transpose_a=transpose_a, transpose_b=transpose_b
            ),
        )
    )


def test_batch_matmul_grad():
    verify_batch_matmul_grad((2, 3, 5), (2, 5, 4), False, False)
    verify_batch_matmul_grad((2, 3, 5), (2, 4, 5), False, True)
    verify_batch_matmul_grad((2, 5, 3), (2, 5, 4), True, False)
    verify_batch_matmul_grad((2, 5, 3), (2, 4, 5), True, True)


def test_reverse_reshape_grad():
    x = relay.var("x", shape=(3, 4, 5), dtype="float64")
    check_grad(relay.Function([x], relay.op.reverse_reshape(x, (-1, 0))))


def test_one_hot_grad():
    indices_shape = (3, 4)
    depth = 5
    axis = -1

    for indices_dtype in ["int32", "int64"]:
        for val_dtype in ["float32", "float64"]:
            inputs = [
                np.random.randint(depth, size=indices_shape, dtype=indices_dtype),
                np.array(np.random.randn() * 1e-5).astype(val_dtype),
                np.array(np.random.randn() * 1e-5).astype(val_dtype),
            ]
            test_inputs = inputs[1:]

            indices = relay.var("indices", shape=indices_shape, dtype=indices_dtype)
            on_val = relay.var("on_val", shape=tuple(), dtype=val_dtype)
            off_val = relay.var("off_val", shape=tuple(), dtype=val_dtype)
            y = relay.one_hot(indices, on_val, off_val, depth, axis, val_dtype)
            f = relay.Function([indices, on_val, off_val], y)

            check_grad(f, inputs=inputs, test_inputs=test_inputs)


if __name__ == "__main__":
    pytest.main([__file__])
