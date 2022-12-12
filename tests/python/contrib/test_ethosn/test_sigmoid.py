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

"""Arm(R) Ethos(TM)-N integration sigmoid tests"""

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
    sigmoid = relay.sigmoid(dequantize)
    model = relay.qnn.op.quantize(
        sigmoid,
        output_scale=relay.const(output_sc, "float32"),
        output_zero_point=relay.const(output_zp, "int32"),
        out_dtype=dtype,
    )
    return model


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 16, 16),
        (1, 8, 8),
    ],
)
def test_sigmoid(dtype, shape):
    """Compare Sigmoid output with TVM."""
    np.random.seed(0)

    inputs = {
        "a": tvm.nd.array(
            np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
        ),
    }
    outputs = []
    for npu in [False, True]:
        for _ in range(1, 2):
            if dtype == "uint8":
                input_zp = 0
                output_zp = 0
            else:
                input_zp = 127
                output_zp = -128
            model = _get_model(shape, input_zp, 0.02, output_zp, 1.0 / 256.0, dtype)
            mod = tei.make_module(model, [])
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,input_zp,input_sc,output_zp,output_sc,err_msg",
    [
        ((2, 4, 4, 4), 64, 0.2, 0, 1 / 256, "batch size=2, batch size must = 1"),
        (
            (1, 4, 4, 4),
            64,
            0.2,
            3,
            1,
            "output quantization params=(3, 1), must = (0, 1/256)",
        ),
    ],
)
def test_sigmoid_failure(shape, input_zp, input_sc, output_zp, output_sc, err_msg):
    """Check Sigmoid error messages."""

    dtype = "uint8"

    model = _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype)
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_sigmoid")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
