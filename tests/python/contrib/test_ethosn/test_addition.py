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

"""Ethos-N integration addition tests"""

import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosn import ethosn_available
from . import infrastructure as tei
import numpy as np


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


def _get_addition_qnn_params(input1_zp, input1_sc, input2_zp, input2_sc):
    input1_max = input1_sc * (255 - input1_zp)
    input1_min = -input1_sc * input1_zp
    input2_max = input2_sc * (255 - input2_zp)
    input2_min = -input2_sc * input2_zp
    output_max = input1_max + input2_max
    output_min = input1_min + input2_min
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc


def test_addition():
    if not ethosn_available():
        return

    trials = [
        ((1, 22, 9, 9), 24, 1.057, 253, 0.452),
        ((1, 27, 21, 16), 79, 0.850, 24, 0.380),
        ((1, 7, 12, 28), 125, 1.293, 239, 0.320),
        ((1, 14, 9, 6), 14, 0.942, 227, 1.562),
        ((1, 13, 16, 22), 15, 0.727, 180, 0.461),
    ]
    np.random.seed(0)
    for shape, rhs_zp, rhs_sc, lhs_zp, lhs_sc in trials:
        outputs = []
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
            "b": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
        }
        out_zp, out_sc = _get_addition_qnn_params(lhs_zp, lhs_sc, rhs_zp, rhs_sc)
        model = _get_model(shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, "uint8")
        for npu in [False, True]:
            mod = tei.make_module(model, [])
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, 2)


def test_addition_failure():
    if not ethosn_available():
        return

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
            "int8",
            0,
            1,
            0,
            1,
            0,
            1,
            "dtype='int8', dtype must be either uint8 or int32; dtype='int8', dtype must be either uint8 or int32",
        ),
    ]

    for shape, dtype, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, err_msg in trials:
        model = _get_model(shape, lhs_zp, lhs_sc, rhs_zp, rhs_sc, out_zp, out_sc, dtype)
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
