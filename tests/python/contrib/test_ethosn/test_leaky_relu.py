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

"""Integration tests for Leaky ReLU"""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.testing import requires_ethosn

from . import infrastructure as tei


def _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype, alpha):
    x = relay.var("x", shape=shape, dtype=dtype)
    x = relay.qnn.op.dequantize(
        x,
        input_scale=relay.const(input_sc, "float32"),
        input_zero_point=relay.const(input_zp, "int32"),
    )
    x = relay.nn.leaky_relu(x, alpha=alpha)
    return relay.qnn.op.quantize(
        x,
        output_scale=relay.const(output_sc, "float32"),
        output_zero_point=relay.const(output_zp, "int32"),
        out_dtype=dtype,
    )


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 52, 52, 3), (1, 3, 8, 2)])
@pytest.mark.parametrize("alpha", [0.001, 0.5678])
def test_leaky_relu(dtype, shape, alpha):
    """Compare Leaky ReLU output with TVM."""

    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    zp_min = iinfo.min
    zp_max = iinfo.max
    input_zp = zp_min + 128
    input_sc = 0.0068132
    output_zp = zp_min + 126  # values offset more than 126 can cause saturation
    output_sc = 0.0078125

    inputs = {"x": tvm.nd.array(np.random.randint(zp_min, high=zp_max, size=shape, dtype=dtype))}
    outputs = []
    for npu in [False, True]:
        model = _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype, alpha)
        mod = tei.make_module(model, [])
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["int8"])
@pytest.mark.parametrize("shape", [(1, 14, 14, 2)])
@pytest.mark.parametrize("alpha", [-1.34, 2.32, 1, 0])
def test_leaky_relu_unsupported_alpha(dtype, shape, alpha):
    """Test unsupported values of alpha (<= 0, >= 1) in Leaky ReLU."""

    iinfo = np.iinfo(dtype)
    zp_min = iinfo.min

    err_msg = f"leaky relu alpha must be less than 1 and greater than 0, but was {alpha}"

    model = _get_model(shape, zp_min + 120, 0.0068132, zp_min + 128, 0.0078125, dtype, alpha)
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_leaky_relu")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
