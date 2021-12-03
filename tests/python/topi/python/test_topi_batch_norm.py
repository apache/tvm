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
"""Test code for batch_norm operator"""
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing


DEVICE = "llvm"
_BATCH_NORM_IMPLEMENT = {
    "generic": (topi.nn.batch_norm, topi.generic.schedule_batch_norm),
}


@pytest.mark.parametrize(
    "shape, axis, epsilon, center, scale",
    [
        ((1,), 0, 0.1, True, True),
        ((2, 3), 0, 0.1, True, True),
        ((1, 2, 4), 0, 0.1, True, True),
        ((1, 2, 3, 4), 0, 0.001, False, False),
        ((2, 3, 4, 1), 1, 0.01, False, True),
        ((3, 4, 1, 2), 2, 0.1, True, False),
        ((4, 1, 2, 3), 3, 1.0, True, True),
        ((1, 2, 4, 4, 5), 0, 0.1, True, True),
    ],
)
def test_batch_norm(shape, axis, epsilon, center, scale):
    x_np = np.random.random(shape).astype("float32")
    gamma_np = np.random.random((shape[axis],)).astype("float32")
    beta_np = np.random.random((shape[axis],)).astype("float32")

    out_np = tvm.topi.testing.batch_norm(x_np, gamma_np, beta_np, axis, epsilon, center, scale)

    x_te = te.placeholder(shape, name="x", dtype="float32")
    gamma_te = te.placeholder((shape[axis],), name="gamma", dtype="float32")
    beta_te = te.placeholder((shape[axis],), name="beta", dtype="float32")

    with tvm.target.Target(DEVICE):
        fcompute, fschedule = tvm.topi.testing.dispatch(DEVICE, _BATCH_NORM_IMPLEMENT)
        out = fcompute(x_te, gamma_te, beta_te, axis, epsilon, center, scale)
        s = fschedule([out])

        dev = tvm.device(DEVICE, 0)

        x_tvm = tvm.nd.array(x_np, dev)
        gamma_tvm = tvm.nd.array(gamma_np, dev)
        beta_tvm = tvm.nd.array(beta_np, dev)
        out_tvm = tvm.nd.array(np.zeros(shape, dtype=out.dtype), dev)

        f = tvm.build(s, [x_te, gamma_te, beta_te, out], DEVICE)
        f(x_tvm, gamma_tvm, beta_tvm, out_tvm)

        tvm.testing.assert_allclose(out_tvm.numpy(), out_np, rtol=1e-3)


if __name__ == "__main__":
    test_batch_norm()
