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
"""Test code for space to batch"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing


def verify_space_to_batch_nd(input_shape, block_shape, pad_before, pad_after, pad_value=0):
    out_shape = []
    out_shape.append(int((input_shape[0] * np.prod(block_shape))))
    for i in range(1, len(block_shape) + 1):
        pad = pad_before[i - 1] + pad_after[i - 1]
        out_shape.append(int((input_shape[i] + pad) // block_shape[i - 1]))
    for i in range(len(block_shape) + 1, len(input_shape)):
        out_shape.append(input_shape[i])

    A = te.placeholder(input_shape, name="A", dtype="float32")
    dtype = A.dtype
    a_np = np.random.uniform(size=input_shape).astype(dtype)

    B = topi.nn.space_to_batch_nd(A, block_shape, pad_before, pad_after, pad_value)

    b_np = tvm.topi.testing.space_to_batch_nd_python(
        a_np, block_shape, pad_before, pad_after, pad_value
    )

    def check_target(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.create(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), dev)
        f = tvm.build(s, [A, B], target)
        f(a, b)
        tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3, atol=1e-3)

    for target, dev in tvm.testing.enabled_targets():
        check_target(target, dev)


@tvm.testing.uses_gpu
def test_space_to_batch():
    # Without paddings
    verify_space_to_batch_nd([3, 3, 2, 1], [3], [0], [0])
    # With paddings
    verify_space_to_batch_nd([3, 3, 2, 1], [3], [1], [2])
    # Multiple spatial dims
    verify_space_to_batch_nd([3, 3, 4, 5, 2], [3, 4, 2], [1, 0, 3], [2, 0, 0])
    # No remaining dims
    verify_space_to_batch_nd([3, 3, 4, 5, 2], [3, 4, 2, 2], [1, 4, 0, 0], [2, 0, 1, 0])


if __name__ == "__main__":
    test_space_to_batch()
