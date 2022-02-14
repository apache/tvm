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
def test_tanh():
    trials = [(1, 512, 512, 3)]

    np.random.seed(0)
    for shape in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
        }
        outputs = []
        for npu in [False, True]:
            model = _get_model(shape, 120, 0.0250629, 128, 0.0078125, "uint8")
            mod = tei.make_module(model, [])
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, "uint8", 1)


@requires_ethosn
def test_tanh_failure():
    trials = [
        (
            (1, 16, 16, 16),
            120,
            0.0250629,
            64,
            0.0078125,
            "uint8",
            "output quantization params=(64, 0.0078125), must = (0, 1/256);",
        ),
    ]

    for shape, input_zp, input_sc, output_zp, output_sc, dtype, err_msg in trials:
        model = _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype)
        model = tei.make_ethosn_composite(model, "ethos-n.qnn_tanh")
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
