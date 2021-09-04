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

"""Ethos-N integration sigmoid tests"""

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei
import numpy as np


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
def test_sigmoid():
    trials = [
        (1, 16, 16, 16),
        (1, 8, 8),
    ]

    np.random.seed(0)
    for shape in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
        }
        outputs = []
        for npu in [False, True]:
            model = _get_model(shape, 64, 0.02, 0, 1 / 256, "uint8")
            mod = tei.make_module(model, [])
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, 1)


@requires_ethosn
def test_sigmoid_failure():
    trials = [
        ((2, 4, 4, 4), 64, 0.2, 0, 1 / 256, "uint8", "batch size=2, batch size must = 1"),
        (
            (1, 4, 4, 4),
            64,
            0.2,
            0,
            1 / 256,
            "int8",
            "dtype='int8', dtype must be either uint8 or int32",
        ),
        (
            (1, 4, 4, 4),
            64,
            0.2,
            0,
            1,
            "uint8",
            "output quantization params=(0, 1), must = (0, 1/256)",
        ),
    ]

    for shape, input_zp, input_sc, output_zp, output_sc, dtype, err_msg in trials:
        model = _get_model(shape, input_zp, input_sc, output_zp, output_sc, dtype)
        model = tei.make_ethosn_composite(model, "ethos-n.qnn_sigmoid")
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
