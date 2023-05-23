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

"""Arm(R) Ethos(TM)-N integration pooling tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei


def _get_model(shape, typef, sizes, strides, pads, layout, dtype):
    """Return a model and any parameters it may have"""
    req = relay.var("a", shape=shape, dtype=dtype)
    if typef is relay.nn.avg_pool2d:
        req = relay.cast(req, "int32")
    req = typef(req, pool_size=sizes, strides=strides, padding=pads, ceil_mode=True, layout=layout)
    if typef is relay.nn.avg_pool2d:
        req = relay.cast(req, dtype)
    return req


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize(
    "shape,typef,size,stride,pad",
    [
        ((1, 8, 8, 8), relay.nn.max_pool2d, (2, 2), (2, 2), (0, 0, 0, 0)),
        ((1, 9, 9, 9), relay.nn.max_pool2d, (3, 3), (2, 2), (0, 0, 0, 0)),
        ((1, 8, 8, 8), relay.nn.avg_pool2d, (3, 3), (1, 1), (1, 1, 1, 1)),
    ],
)
def test_pooling(dtype, shape, typef, size, stride, pad):
    """Compare Pooling output with TVM."""
    np.random.seed(0)

    layout = "NHWC"

    inputs = {
        "a": tvm.nd.array(
            np.random.randint(
                low=np.iinfo(dtype).min, high=np.iinfo(dtype).max + 1, size=shape, dtype=dtype
            )
        ),
    }
    outputs = []
    model = _get_model(shape, typef, size, stride, pad, layout, dtype)
    for npu in [False, True]:
        mod = tei.make_module(model, {})
        outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

    tei.verify(outputs, dtype, 1)


@requires_ethosn
@pytest.mark.parametrize(
    "shape,size,stride,layout,dtype,err_msg",
    [
        (
            (2, 8, 8, 8),
            (2, 2),
            (2, 2),
            "NHWC",
            "uint8",
            "batch size=2, batch size must = 1",
        ),
        (
            (1, 8, 8, 8),
            (2, 2),
            (2, 2),
            "NHWC",
            "int16",
            "dtype='int16', dtype must be either uint8, int8 or int32",
        ),
        (
            (1, 8, 8, 8),
            (2, 2),
            (2, 2),
            "NCHW",
            "uint8",
            "data format=NCHW, data format must = NHWC",
        ),
        (
            (1, 8, 8, 8),
            (2, 2),
            (2, 2, 2),
            "NHWC",
            "uint8",
            "stride size=3, stride size must = 2",
        ),
        (
            (1, 8, 8, 8),
            (2, 2, 2),
            (2, 2),
            "NHWC",
            "uint8",
            "dimensions=3, dimensions must = 2",
        ),
    ],
)
def test_pooling_failure(shape, size, stride, layout, dtype, err_msg):
    """Check Pooling error messages."""

    typef = relay.nn.max_pool2d
    pad = (0, 0, 0, 0)

    model = _get_model(shape, typef, size, stride, pad, layout, dtype)
    mod = tei.make_ethosn_partition(model)
    tei.test_error(mod, {}, err_msg)
