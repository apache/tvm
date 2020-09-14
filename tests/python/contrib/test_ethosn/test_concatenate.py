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

"""Concatenate tests for Ethos-N"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosn import ethosn_available
from . import infrastructure as tei


def _get_inputs(shapes):
    inputs = {}
    for i, shape in enumerate(shapes):
        inputs["in" + str(i)] = tvm.nd.array(
            np.random.randint(0, high=256, size=shape, dtype="uint8")
        )

    return inputs


def _get_model(shapes, dtype, axis):
    tup = []
    for i, shape in enumerate(shapes):
        a = relay.var("in" + str(i), shape=shape, dtype=dtype)
        tup.append(a)

    zeroi = relay.const(1, "int32")
    zerof = relay.const(0.5, "float32")
    con = relay.qnn.op.concatenate(
        tup,
        input_scales=[zerof] * len(shapes),
        input_zero_points=[zeroi] * len(shapes),
        output_scale=zerof,
        output_zero_point=zeroi,
        axis=axis,
    )
    return con


def test_concatenate():
    if not ethosn_available():
        return

    trials = [
        ([(1, 4), (1, 6)], 1),
        ([(1, 16, 4), (1, 16, 4)], 1),
        ([(1, 25, 4, 16)] * 3, 3),
        ([(1, 25, 4, 16), (1, 25, 5, 16), (1, 25, 6, 16)], 2),
    ]

    for shapes, axis in trials:
        outputs = []
        inputs = _get_inputs(shapes)
        for npu in [False, True]:
            model = _get_model(shapes, "uint8", axis)
            mod = tei.make_module(model, {})
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, 0)


def test_concatenate_failure():
    if not ethosn_available():
        return

    trials = [
        ([(1, 4, 4, 4, 4), (1, 4, 4, 4, 4)], "uint8", 1, "dimensions=5, dimensions must be <= 4;"),
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "uint8",
            3,
            "Concatenation along the channels dimension (axis 3) requires input tensors with a multiple of 16 channels;",
        ),
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "int8",
            2,
            "dtype='int8', dtype must be either uint8 or int32; dtype='int8', dtype must be either uint8 or int32;",
        ),
        (
            [(2, 4, 4, 4), (2, 4, 4, 4)],
            "uint8",
            2,
            "batch size=2, batch size must = 1; batch size=2, batch size must = 1;",
        ),
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "uint8",
            0,
            "Concatenation cannot be performed along batch axis (axis 0);",
        ),
    ]

    for shapes, dtype, axis, err_msg in trials:
        model = _get_model(shapes, dtype, axis)
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
