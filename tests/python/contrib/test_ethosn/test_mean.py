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
def test_mean():
    trials = [(1, 7, 7, 2048), (1, 8, 8)]

    np.random.seed(0)
    for shape in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
        }
        outputs = []
        for npu in [False, True]:
            model = _get_model(shape, [1, 2], True, 128, 0.0784314, 128, 0.0784314, "uint8")
            mod = tei.make_module(model, [])
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, "uint8", 1)
