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

"""Concatenate tests for Arm(R) Ethos(TM)-N"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_inputs(shapes, dtype):
    inputs = {}
    for i, shape in enumerate(shapes):
        inputs["in" + str(i)] = tvm.nd.array(
            np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
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


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shapes,axis",
    [
        ([(1, 4), (1, 6)], 1),
        ([(1, 16, 4), (1, 16, 4)], 1),
        ([(1, 25, 4, 16)] * 3, 3),
        ([(1, 25, 4, 16), (1, 25, 5, 16), (1, 25, 6, 16)], 2),
        ([(1, 4), (1, 6)], -1),
        ([(1, 16, 4), (1, 16, 4)], -2),
    ],
)
def test_concatenate(dtype, shapes, axis):
    """Compare Concatenate output with TVM."""
    np.random.seed(0)

    outputs = []
    inputs = _get_inputs(shapes, dtype)
    for npu in [False, True]:
        model = _get_model(shapes, dtype, axis)
        mod = tei.make_module(model, {})
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                1,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

        tei.verify(outputs, dtype, 0)


@requires_ethosn
@pytest.mark.parametrize(
    "shapes,dtype,axis,err_msg",
    [
        ([(1, 4, 4, 4, 4), (1, 4, 4, 4, 4)], "uint8", 1, "dimensions=5, dimensions must be <= 4;"),
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "uint8",
            3,
            "Concatenation along the channels dimension (axis 3) "
            "requires input tensors with a multiple of 16 channels;",
        ),
        (
            [(1, 4, 4, 4), (1, 4, 4, 4)],
            "int16",
            2,
            "dtype='int16', dtype must be either uint8, int8 or int32; dtype='int16', "
            "dtype must be either uint8, int8 or int32;",
        ),
        (
            [(2, 4, 4, 4), (2, 4, 4, 4)],
            "uint8",
            2,
            "batch size=2, batch size must = 1; batch size=2, batch size must = 1;",
        ),
        (
            [(1, 4, 4, 4)],
            "uint8",
            0,
            "Concatenation cannot be performed along batch axis (axis 0);",
        ),
    ],
)
def test_concatenate_failure(shapes, dtype, axis, err_msg):
    """Check Concatenate error messages."""
    model = _get_model(shapes, dtype, axis)
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
