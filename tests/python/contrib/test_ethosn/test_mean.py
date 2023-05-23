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

"""Arm(R) Ethos(TM)-N integration mean tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(shape, axis, keepdims, input_zp, input_sc, output_zp, output_sc, dtype):
    a = relay.var("a", shape=shape, dtype=dtype)
    casted = relay.op.cast(a, "int32")
    mean = relay.mean(casted, axis, keepdims)
    model = relay.qnn.op.requantize(
        mean,
        input_scale=relay.const(input_sc, "float32"),
        input_zero_point=relay.const(input_zp, "int32"),
        output_scale=relay.const(output_sc, "float32"),
        output_zero_point=relay.const(output_zp, "int32"),
        out_dtype=dtype,
    )
    return model


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape", [(1, 7, 7, 2048), (1, 8, 8)])
def test_mean(dtype, shape):
    """Compare Mean output with TVM."""

    np.random.seed(0)

    zp_min = np.iinfo(dtype).min
    zp_max = np.iinfo(dtype).max

    inputs = {
        "a": tvm.nd.array(np.random.randint(zp_min, high=zp_max + 1, size=shape, dtype=dtype)),
    }
    outputs = []
    for npu in [False, True]:
        model = _get_model(
            shape, [1, 2], True, zp_min + 128, 0.0784314, zp_min + 128, 0.0784314, dtype=dtype
        )
        mod = tei.make_module(model, [])
        outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_mean_non_equal_quantization(dtype):
    """Test mean is not offloaded when quantization is not equal."""

    np.random.seed(0)

    shape = (1, 7, 7, 2048)
    zp_min = np.iinfo(dtype).min

    model = _get_model(shape, [1, 2], True, zp_min + 120, 0.0068132, zp_min + 128, 0.0078125, dtype)
    mod = tei.make_module(model, [])
    tei.build(mod, {}, npu=True, expected_host_ops=3, npu_partitions=0)
