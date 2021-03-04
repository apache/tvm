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
"""BNNS integration dense tests."""

import numpy as np
import math
import pytest
import tvm
from tvm import relay
from tvm import testing
from .infrastructure import (
    Device,
    skip_runtime_test,
    skip_codegen_test,
    verify_codegen,
    build_and_run,
    verify,
    generate_trials,
)


def _get_model(a_shape, b_shape, dtype, var_names, is_a_constant=False, is_b_constant=False):
    """Return a model and any parameters it may have"""
    a = relay.var(next(var_names), shape=a_shape, dtype=dtype)
    b = relay.var(next(var_names), shape=b_shape, dtype=dtype)
    params = {}
    if is_b_constant is True:
        b = tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype))
        params["b"] = b
        b = relay.const(b, dtype)
    if is_a_constant is True:
        a = tvm.nd.array(np.random.uniform(-128, 127, a_shape).astype(dtype))
        params["a"] = a
        a = relay.const(a, dtype)
    out = relay.nn.batch_matmul(a, b)
    return out, params


@pytest.mark.skipif(skip_runtime_test(), reason="Skip because BNNS codegen is not available")
def test_matmul():
    device = Device()
    np.random.seed(0)
    dtype = "float32"

    # C[N, I, J] = A[N, I, K] * B[N, J, K]
    shapes_config = [
        # B, I, J, K
        [1, 4, 4, 3],
        [1, 16, 32, 32],
        [2, 1, 1, 3],
        [2, 16, 32, 32],
        [5, 1, 1, 3],
    ]
    data_config = [
        # A_is_constant, B_is_constant
        [False, True],
        [True, False],
        [False, False],
    ]

    for N, I, J, K in shapes_config:
        a_shape = [N, I, K]
        b_shape = [N, J, K]
        for is_a_constant, is_b_constant in data_config:
            outputs = []
            inputs = {
                "a": tvm.nd.array(np.random.uniform(-128, 127, a_shape).astype(dtype)),
                "b": tvm.nd.array(np.random.uniform(-128, 127, b_shape).astype(dtype)),
            }
            func, params = _get_model(
                a_shape,
                b_shape,
                dtype,
                var_names=iter(inputs),
                is_a_constant=is_a_constant,
                is_b_constant=is_b_constant,
            )
            for enable_bnns in [False, True]:
                outputs.append(
                    build_and_run(
                        func,
                        inputs,
                        1,
                        params,
                        device,
                        enable_bnns=enable_bnns,
                    )[0]
                )

            config = {
                "a_shape": a_shape,
                "b_shape": b_shape,
                "dtype": dtype,
            }
            verify(outputs, atol=0.001, rtol=0.01, config=config)


if __name__ == "__main__":
    test_matmul()
