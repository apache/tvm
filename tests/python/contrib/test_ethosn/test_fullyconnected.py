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

"""Arm(R) Ethos(TM)-N integration fully connected tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(
    shape, weight_shape, input_zp, input_sc, kernel_zp, kernel_sc, output_zp, output_sc, dtype
):
    """Return a model an any parameters it may have"""
    a = relay.var("a", shape=shape, dtype=dtype)
    w = tvm.nd.array(np.ones(weight_shape, dtype))
    weights = relay.const(w, dtype)
    fc = relay.qnn.op.dense(
        a,
        weights,
        input_zero_point=relay.const(input_zp, "int32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        units=weight_shape[0],
        out_dtype="int32",
    )
    b = tvm.nd.array(np.random.randint(0, high=255, size=(shape[0],), dtype="int32"))
    biasc = relay.const(b, "int32")
    bias = relay.nn.bias_add(fc, biasc, axis=0)
    req = relay.qnn.op.requantize(
        bias,
        relay.const(input_sc * kernel_sc, "float32"),  # input zero scale
        relay.const(input_zp * kernel_zp, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output zero scale
        relay.const(output_zp, "int32"),  # output zero point
        out_dtype=dtype,
    )
    params = {"w": w, "b": b}
    return req, params


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8"])
def test_fullyconnected(dtype):
    zp_min = np.iinfo(dtype).min
    zp_max = np.iinfo(dtype).max
    trials = [
        ((1, 1024), zp_min + 71, 0.580, zp_max - 176, 1.498),
        ((1, 4096), zp_min + 166, 1.724, zp_max - 138, 0.180),
        ((1, 16384), zp_min + 101, 1.372, zp_max - 234, 1.346),
    ]
    np.random.seed(0)
    for shape, input_zp, input_sc, kernel_zp, kernel_sc in trials:
        kernel_zp = (
            0
            if dtype == "int8"
            else np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max) + 1
        )
        inputs = {
            "a": tvm.nd.array(
                np.random.randint(
                    np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype
                )
            ),
        }
        outputs = []
        output_zp, output_sc = tei.get_conv2d_qnn_params(
            dtype,
            input_zp,
            input_sc,
            kernel_zp,
            kernel_sc,
            shape[0],
            shape[1],
            1,
        )
        for npu in [False, True]:
            model, params = _get_model(
                shape,
                shape,
                input_zp,
                input_sc,  # input zp, sc
                kernel_zp,
                kernel_sc,  # kernel
                output_zp,
                output_sc,  # output
                dtype,
            )
            mod = tei.make_module(model, params)
            outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))
        tei.verify(outputs, dtype, 1)


@requires_ethosn
def test_fullyconnected_failure():
    trials = [
        (
            (1, 64),
            (1, 64),
            0,
            1024,
            0,
            1024,
            0,
            1,
            "uint8",
            tei.get_overall_scale_range_expected_error_message(),
        ),
        (
            (1, 1, 1, 64),
            (1, 64),
            0,
            1,
            0,
            1,
            0,
            1,
            "uint8",
            "Weights tensor must have I dimension equal to the number of channels of the input tensor.;",
        ),
        ((1024, 64), (1, 64), 0, 1, 0, 1, 0, 1, "uint8", "batch size=1024, batch size must = 1;"),
    ]

    np.random.seed(0)
    for (
        shape,
        weight_shape,
        input_zp,
        input_sc,
        kernel_zp,
        kernel_sc,
        output_zp,
        output_sc,
        dtype,
        err_msg,
    ) in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype=dtype)),
        }
        model, params = _get_model(
            shape,
            weight_shape,
            input_zp,
            input_sc,
            kernel_zp,
            kernel_sc,
            output_zp,
            output_sc,
            dtype,
        )
        model = tei.make_ethosn_composite(model, "ethos-n.qnn_fc")
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
