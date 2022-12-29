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

"""Arm(R) Ethos(TM)-N integration relu tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(shape, dtype, a_min, a_max):
    assert a_min >= np.iinfo(dtype).min and a_max <= np.iinfo(dtype).max
    a = relay.var("a", shape=shape, dtype=dtype)
    relu = relay.clip(a, a_min=a_min, a_max=a_max)
    return relu


@requires_ethosn
@pytest.mark.parametrize(
    "shape,a_min,a_max,dtype",
    [
        ((1, 4, 4, 4), 65, 178, "uint8"),
        ((1, 8, 4, 2), 1, 254, "uint8"),
        ((1, 8, 4, 2), -100, 100, "int8"),
        ((1, 16), -120, -20, "int8"),
    ],
)
def test_relu(dtype, shape, a_min, a_max):
    """Compare Relu output with TVM."""
    np.random.seed(0)

    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                low=np.iinfo(dtype).min,
                high=np.iinfo(dtype).max + 1,
                size=shape,
                dtype=dtype,
            )
        ),
    }
    outputs = []
    for npu in [False, True]:
        model = _get_model(inputs["a"].shape, dtype, a_min, a_max)
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

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,dtype,a_min,a_max,err_msg",
    [
        ((1, 4, 4, 4, 4), "uint8", 65, 78, "dimensions=5, dimensions must be <= 4"),
        ((1, 8, 4, 2), "int16", 1, 254, "dtype='int16', dtype must be either uint8, int8 or int32"),
        ((1, 8, 4, 2), "uint8", 254, 1, "Relu has lower bound > upper bound"),
        ((2, 2, 2, 2), "uint8", 1, 63, "batch size=2, batch size must = 1; "),
    ],
)
def test_relu_failure(shape, dtype, a_min, a_max, err_msg):
    """Check Relu error messages."""
    model = _get_model(shape, dtype, a_min, a_max)
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
