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

"""Arm(R) Ethos(TM)-N integration addition tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(input_shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, dtype):
    """Return a model and any parameters it may have"""

    a = relay.var("a", shape=input_shape, dtype=dtype)
    b = relay.var("b", shape=input_shape, dtype=dtype)
    model = relay.qnn.op.add(
        lhs=a,
        rhs=b,
        lhs_scale=relay.const(lhs_sc, "float32"),
        lhs_zero_point=relay.const(lhs_zp, "int32"),
        rhs_scale=relay.const(rhs_sc, "float32"),
        rhs_zero_point=relay.const(rhs_zp, "int32"),
        output_scale=relay.const(out_sc, "float32"),
        output_zero_point=relay.const(out_zp, "int32"),
    )
    return model


def _get_addition_qnn_params(dtype, input1_zp, input1_sc, input2_zp, input2_sc):
    input1_max = input1_sc * (255 - input1_zp)
    input1_min = -input1_sc * input1_zp
    input2_max = input2_sc * (255 - input2_zp)
    input2_min = -input2_sc * input2_zp
    output_max = input1_max + input2_max
    output_min = input1_min + input2_min
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
def test_addition(dtype):
    zp_min = np.iinfo(dtype).min
    zp_max = np.iinfo(dtype).max
    trials = [
        ((1, 22, 9, 9), zp_min + 24, 1.057, zp_max - 3, 0.452),
        ((1, 27, 21, 16), zp_min + 79, 0.850, 24, 0.380),
        ((1, 7, 12, 28), zp_min + 125, 1.293, zp_max - 16, 0.320),
        ((1, 14, 9, 6), zp_min + 14, 0.942, zp_max - 28, 1.562),
        ((1, 13, 16, 22), zp_min + 15, 0.727, zp_max - 75, 0.461),
    ]
    np.random.seed(0)
    for shape, rhs_zp, rhs_sc, lhs_zp, lhs_sc in trials:
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.randint(zp_min, zp_max + 1, size=shape, dtype=dtype)),
            "b": tvm.nd.array(np.random.randint(zp_min, zp_max + 1, size=shape, dtype=dtype)),
        }
        out_zp, out_sc = _get_addition_qnn_params(dtype, lhs_zp, lhs_sc, rhs_zp, rhs_sc)
        model = _get_model(shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, dtype)
        for npu in [False, True]:
            mod = tei.make_module(model, [])
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, dtype, 2)


@requires_ethosn
def test_addition_failure():
    trials = [
        (
            (2, 4, 4, 4),
            "uint8",
            0,
            1,
            0,
            1,
            0,
            1,
            "batch size=2, batch size must = 1; batch size=2, batch size must = 1",
        ),
        (
            (1, 4, 4, 4),
            "int16",
            0,
            1,
            0,
            1,
            0,
            1,
            "dtype='int16', dtype must be either uint8, int8 or int32; dtype='int16', dtype must be either uint8, int8 or int32",
        ),
    ]

    for shape, dtype, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, err_msg in trials:
        model = _get_model(shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, dtype)
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
