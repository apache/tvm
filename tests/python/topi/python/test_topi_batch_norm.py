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
"""Tests for the batch_norm operator."""
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing


_DEVICE = "llvm"
_BATCH_NORM_IMPLEMENT = {
    "generic": (topi.nn.batch_norm, topi.generic.schedule_batch_norm),
    "cpu": (topi.nn.batch_norm, topi.x86.schedule_batch_norm),
}


@pytest.mark.parametrize(
    "shape, axis, epsilon, center, scale, training, momentum",
    [
        ((1,), 0, 0.1, True, True, False, 0.1),
        ((2, 3), 0, 0.1, True, True, False, 0.1),
        ((1, 2, 4), 0, 0.1, True, True, False, 0.1),
        ((1, 2, 3, 4), 0, 0.001, False, False, False, 0.1),
        ((2, 3, 4, 1), 1, 0.01, False, True, False, 0.1),
        ((3, 4, 1, 2), 2, 0.1, True, False, True, 0.1),
        ((4, 1, 2, 3), 3, 1.0, True, True, True, 0.2),
        ((1, 2, 4, 4, 5), 0, 0.1, True, True, True, 0.3),
    ],
)
def test_batch_norm(shape, axis, epsilon, center, scale, training, momentum):
    x_np = np.random.random(shape).astype("float32")
    gamma_np = np.random.random(shape[axis]).astype("float32")
    beta_np = np.random.random(shape[axis]).astype("float32")
    moving_mean_np = np.random.random(shape[axis]).astype("float32")
    moving_var_np = np.random.random(shape[axis]).astype("float32")

    out_x_np, out_moving_mean_np, out_moving_var_np = tvm.topi.testing.batch_norm(
        x_np,
        gamma_np,
        beta_np,
        moving_mean_np,
        moving_var_np,
        axis,
        epsilon,
        center,
        scale,
        training,
        momentum,
    )

    x_te = te.placeholder(shape, name="x", dtype="float32")
    gamma_te = te.placeholder((shape[axis],), name="gamma", dtype="float32")
    beta_te = te.placeholder((shape[axis],), name="beta", dtype="float32")
    moving_mean_te = te.placeholder((shape[axis],), name="moving_mean", dtype="float32")
    moving_var_te = te.placeholder((shape[axis],), name="moving_var", dtype="float32")

    with tvm.target.Target(_DEVICE):
        fcompute, fschedule = tvm.topi.testing.dispatch(_DEVICE, _BATCH_NORM_IMPLEMENT)
        out_x, out_moving_mean, out_moving_var = fcompute(
            x_te,
            gamma_te,
            beta_te,
            moving_mean_te,
            moving_var_te,
            axis,
            epsilon,
            center,
            scale,
            training,
            momentum,
        )
        s = fschedule([out_x, out_moving_mean, out_moving_var])

        dev = tvm.device(_DEVICE, 0)

        x_tvm = tvm.nd.array(x_np, dev)
        gamma_tvm = tvm.nd.array(gamma_np, dev)
        beta_tvm = tvm.nd.array(beta_np, dev)
        moving_mean_tvm = tvm.nd.array(moving_mean_np, dev)
        moving_var_tvm = tvm.nd.array(moving_var_np, dev)
        out_x_tvm = tvm.nd.array(np.zeros(shape, dtype=out_x.dtype), dev)
        out_moving_mean_tvm = tvm.nd.array(
            np.zeros((shape[axis],), dtype=out_moving_mean.dtype), dev
        )
        out_moving_var_tvm = tvm.nd.array(np.zeros((shape[axis],), dtype=out_moving_var.dtype), dev)

        f = tvm.build(
            s,
            [
                x_te,
                gamma_te,
                beta_te,
                moving_mean_te,
                moving_var_te,
                out_x,
                out_moving_mean,
                out_moving_var,
            ],
            _DEVICE,
        )
        f(
            x_tvm,
            gamma_tvm,
            beta_tvm,
            moving_mean_tvm,
            moving_var_tvm,
            out_x_tvm,
            out_moving_mean_tvm,
            out_moving_var_tvm,
        )

        tvm.testing.assert_allclose(out_x_tvm.numpy(), out_x_np, rtol=1e-3)
        tvm.testing.assert_allclose(out_moving_mean_tvm.numpy(), out_moving_mean_np, rtol=1e-3)
        tvm.testing.assert_allclose(out_moving_var_tvm.numpy(), out_moving_var_np, rtol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
