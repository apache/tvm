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

"""Ethos-N integration reshape tests"""

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from tvm.relay.op.contrib import get_pattern_table
from . import infrastructure as tei
import numpy as np


def _get_model(input_shape, output_shape, dtype):
    """Return a model and any parameters it may have"""
    a = relay.var("a", shape=input_shape, dtype=dtype)
    conv, params = tei.get_conv2d(a, input_shape)
    req = relay.reshape(conv, output_shape)
    return req, params


@requires_ethosn
def test_reshape():
    trials = [
        ((1, 15, 4, 1), (1, 60)),
        ((1, 15, 4, 1), (1, 30, 2)),
        ((1, 15, 4, 1), (1, 4, 15, 1)),
        ((1, 15, 4, 1), (1, 12, 5, 1)),
        ((1, 15, 4, 1), (1, -1, 2, 1)),
    ]

    np.random.seed(0)
    for input_shape, output_shape in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=input_shape, dtype="uint8"))
        }
        outputs = []
        for npu in [False, True]:
            model, params = _get_model(input_shape, output_shape, "uint8")
            mod = tei.make_module(model, params)
            outputs.append(tei.build_and_run(mod, inputs, 1, params, npu=npu))

        tei.verify(outputs, 1)


@requires_ethosn
def test_reshape_failure():
    trials = [
        (
            (1, 15, 4, 1),
            (1, 15, -2),
            "uint8",
            "reshape dimension=-2, reshape dimension must be >= -1",
        ),
    ]

    np.random.seed(0)
    for input_shape, output_shape, dtype, err_msg in trials:
        model, params = _get_model(input_shape, output_shape, dtype)
        mod = tei.make_module(model, params)
        pattern = get_pattern_table("ethos-n")
        mod = tei.make_module(model, params)
        mod = relay.transform.MergeComposite(pattern)(mod)
        mod = tei.make_ethosn_partition(mod["main"].body)
        tei.test_error(mod, {}, err_msg)
