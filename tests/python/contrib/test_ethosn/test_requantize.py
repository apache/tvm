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

"""Arm(R) Ethos(TM)-N integration requantize tests"""

import pytest
import numpy as np
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(shape, input_zp, input_sc, output_zp, output_sc, in_dtype, out_dtype):
    a = relay.var("a", shape=shape, dtype=in_dtype)
    model = relay.qnn.op.requantize(
        data=a,
        input_scale=relay.const(input_sc, "float32"),
        input_zero_point=relay.const(input_zp, "int32"),
        output_scale=relay.const(output_sc, "float32"),
        output_zero_point=relay.const(output_zp, "int32"),
        out_dtype=out_dtype,
    )
    return model


@requires_ethosn
@pytest.mark.parametrize("in_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize("shape", [(1, 52, 52, 3)])
def test_requantize(in_dtype, out_dtype, shape):
    np.random.seed(0)
    low = 0 if in_dtype == "uint8" else -5
    high = low + 10
    input_zp = (high + low) / 2
    inputs = {
        "a": tvm.nd.array(np.random.randint(low=low, high=high, size=shape, dtype=in_dtype)),
    }
    outputs = []
    for npu in [False, True]:
        model = _get_model(
            shape=shape,
            input_zp=input_zp,
            input_sc=0.002,
            output_zp=10,
            output_sc=0.008,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
        )
        mod = tei.make_module(model, [])
        x = tei.build_and_run(mod, inputs, 1, {}, npu=npu)
        outputs.append(x)

    tei.verify(outputs, out_dtype, 1)


@requires_ethosn
def test_requantize_failure():
    input_sc = 0.8
    output_sc = (input_sc / 128) - 0.0001
    model = _get_model(
        shape=(1, 52, 52, 3),
        input_zp=0,
        input_sc=input_sc,
        output_zp=0,
        output_sc=output_sc,
        in_dtype="int8",
        out_dtype="int8",
    )
    model = tei.make_ethosn_composite(model, "ethos-n.qnn_requantize")
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, "Output scale must be bigger than input scale / 128")
