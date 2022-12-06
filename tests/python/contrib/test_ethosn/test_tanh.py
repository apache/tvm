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

"""Arm(R) Ethos(TM)-N NPU integration tanh tests"""

import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype):
    a = relay.var("a", shape=shape, dtype=dtype)
    dequantize = relay.qnn.op.dequantize(
        a,
        input_scale=relay.const(input_sc, "float32"),
        input_zero_point=relay.const(input_zp, "int32"),
    )
    tanh = relay.tanh(dequantize)
    model = relay.qnn.op.quantize(
        tanh,
        output_scale=relay.const(output_sc, "float32"),
        output_zero_point=relay.const(output_zp, "int32"),
        out_dtype=dtype,
    )
    return model


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 52, 52, 3)])
def test_tanh(dtype, shape):
    """Compare Tanh output with TVM."""

    zp_min = np.iinfo(dtype).min
    zp_max = np.iinfo(dtype).max

    np.random.seed(0)
    inputs = {
        "a": tvm.nd.array(np.random.randint(zp_min, high=zp_max, size=shape, dtype=dtype)),
    }
    outputs = []
    for npu in [False, True]:
        model = _get_model(shape, zp_min + 120, 0.0250629, zp_min + 128, 0.0078125, dtype)
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
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape, input_zp, input_sc, output_zp, output_sc, err_msg",
    [
        (
            (1, 16, 16, 16),
            120,
            0.0250629,
            64,
            0.0078125,
            "output quantization params=(64, 0.0078125), must = ({test_zp}, 1/256);",
        )
    ],
)
def test_tanh_failure(shape, input_zp, input_sc, output_zp, output_sc, err_msg, dtype):
    """Check Tanh error messages."""

    test_zp = 0 if dtype == "int8" else 128
    model = _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype)
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_tanh")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg.format(test_zp=test_zp))
