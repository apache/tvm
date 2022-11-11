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

"""Split tests for Arm(R) Ethos(TM)-N"""

import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.testing import requires_ethosn

from . import infrastructure as tei


def _get_model(shape, dtype, splits, axis):
    a = relay.var("a", shape=shape, dtype=dtype)
    split = relay.op.split(a, indices_or_sections=splits, axis=axis)
    return split.astuple()


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,splits,axis",
    [
        ((1, 16, 16, 32), (2, 7, 10), 2),
        ((1, 12, 8, 16), 3, 1),
    ],
)
def test_split(dtype, shape, splits, axis):
    """Compare Split output with TVM."""
    np.random.seed(0)

    outputs = []
    inputs = {
        "a": tvm.nd.array(
            np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max + 1, size=shape, dtype=dtype)
        )
    }
    for npu in [False, True]:
        model = _get_model(shape, dtype, splits, axis)
        mod = tei.make_module(model, {})
        output_count = splits if isinstance(splits, int) else len(splits) + 1
        outputs.append(
            tei.build_and_run(
                mod,
                inputs,
                output_count,
                {},
                npu=npu,
                additional_config_args={"inline_non_compute_intensive_partitions": False},
            )
        )

        tei.verify(outputs, dtype, 0)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,dtype,splits,axis,err_msg",
    [
        ((1, 4, 4, 4, 4), "uint8", 4, 2, "dimensions=5, dimensions must be <= 4;"),
        ((1, 4, 4, 4), "int16", 4, 2, "dtype='int16', dtype must be either uint8, int8 or int32;"),
        ((2, 4, 4, 4), "uint8", 4, 2, "batch size=2, batch size must = 1;"),
        ((1, 4, 4, 4), "uint8", 1, 0, "Split cannot be performed along batch axis (axis 0);"),
        (
            (1, 4, 4, 4),
            "uint8",
            4,
            3,
            "Split along the channels dimension (axis 3) requires all output sizes "
            "(specified in splitInfo.m_Sizes) to be multiples of 16;",
        ),
    ],
)
def test_split_failure(shape, dtype, splits, axis, err_msg):
    """Check Split error messages."""
    model = _get_model(shape, dtype, splits, axis)
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
