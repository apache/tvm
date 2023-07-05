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

"""Arm(R) Ethos(TM)-N integration reshape tests"""

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.testing import requires_ethosn

from . import infrastructure as tei


def _get_model(input_shape, output_shape, dtype):
    """Return a model and any parameters it may have"""
    a = relay.var("a", shape=input_shape, dtype=dtype)
    req = relay.reshape(a, output_shape)
    return req, {}


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        ((1, 15, 4, 1), (1, 60)),
        ((1, 15, 4, 1), (1, 30, 2)),
        ((1, 15, 4, 1), (1, 4, 15, 1)),
        ((1, 15, 4, 1), (1, 12, 5, 1)),
        ((1, 15, 4, 1), (1, 0, 2, 2)),
        ((1, 15, 4, 1), (1, -1, 2, 1)),
        ((1, 15, 4, 1), (1, -2)),
        ((1, 15, 4, 1), (1, -3, 1, 1)),
        ((1, 15, 4, 1), (1, -4, 3, 5, 4)),
        ((1, 15, 4, 1), (0, -1, -2)),
        ((1, 15, 4, 1), (0, -1, -3, 1)),
        ((1, 15, 4, 1), (1, -4, -1, 5, 4)),
    ],
)
def test_reshape(dtype, input_shape, output_shape):
    """Compare Reshape output with TVM."""

    np.random.seed(0)
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                low=np.iinfo(dtype).min,
                high=np.iinfo(dtype).max + 1,
                size=input_shape,
                dtype=dtype,
            )
        )
    }
    outputs = []
    for npu in [False, True]:
        model, params = _get_model(input_shape, output_shape, dtype)
        mod = tei.make_module(model, params)
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                params,
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "input_shape, output_shape",
    [
        (
            (1, 13, 13, 255),
            (1, 13, 13, 3, 85),
        ),
    ],
)
def test_reshape_failure(input_shape, output_shape):
    """Check Resize is not offloaded."""

    model, params = _get_model(input_shape, output_shape, "int8")
    mod = tei.make_module(model, params)
    tei.build(
        mod,
        params,
        expected_host_ops=1,
        npu_partitions=0,
        additional_config_args={"inline_non_compute_intensive_partitions": False},
    )
