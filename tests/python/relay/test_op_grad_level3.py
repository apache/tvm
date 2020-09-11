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
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
import tvm.testing


@tvm.testing.uses_gpu
def test_clip():
    for dtype in ("float32", "float64"):
        ref = lambda x: np.where(
            x > 10.0, np.zeros_like(x), np.where(x < 1.0, np.zeros_like(x), np.ones_like(x))
        )
        x = relay.var("x", relay.TensorType((10, 4), dtype))
        y = tvm.relay.clip(x, 1.0, 10.0)

        data = np.random.rand(10, 4).astype(dtype) * 11.0
        ref_grad = ref(data)
        fwd_func = relay.Function([x], y)
        fwd_func = run_infer_type(fwd_func)
        bwd_func = run_infer_type(gradient(fwd_func))

        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor(ctx=ctx, target=target)
            op_res, (op_grad,) = intrp.evaluate(bwd_func)(data)
            np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)


def verify_transpose_grad(d_shape, axes=None):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function([data], relay.transpose(data, axes=axes))
    check_grad(fwd_func)


def test_transpose_grad():
    verify_transpose_grad((1, 2, 3, 4))
    verify_transpose_grad((1, 2, 3, 4), axes=(0, 2, 3, 1))


def test_negative_grad():
    data = relay.var("data", relay.TensorType((10, 4), "float32"))
    fwd_func = relay.Function([data], relay.negative(data))
    check_grad(fwd_func)


def test_cast_grad():
    data = relay.var("data", relay.TensorType((10, 4), "float32"))
    fwd_func = relay.Function([data], relay.cast(data, "float64"))
    check_grad(fwd_func)


def test_copy_grad():
    data = relay.var("data", relay.TensorType((10, 4), "float64"))
    fwd_func = relay.Function([data], relay.copy(data))
    check_grad(fwd_func)


if __name__ == "__main__":
    pytest.main()
