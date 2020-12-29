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

"""Ethos-N integration relu tests"""

import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosn import ethosn_available
from . import infrastructure as tei
import numpy as np


def _get_model(shape, dtype, a_min, a_max):
    a = relay.var("a", shape=shape, dtype=dtype)
    relu = relay.clip(a, a_min=a_min, a_max=a_max)
    return relu


def test_relu():
    if not ethosn_available():
        return

    trials = [
        ((1, 4, 4, 4), 65, 178),
        ((1, 8, 4, 2), 1, 254),
        ((1, 16), 12, 76),
    ]

    for shape, a_min, a_max in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
        }
        outputs = []
        for npu in [False, True]:
            model = _get_model(inputs["a"].shape, "uint8", a_min, a_max)
            mod = tei.make_module(model, {})
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, 1)


def test_relu_failure():
    if not ethosn_available():
        return

    trials = [
        ((1, 4, 4, 4, 4), "uint8", 65, 78, "dimensions=5, dimensions must be <= 4"),
        ((1, 8, 4, 2), "int8", 1, 254, "dtype='int8', dtype must be either uint8 or int32"),
        ((1, 8, 4, 2), "uint8", 254, 1, "Relu has lower bound > upper bound"),
        ((2, 2, 2, 2), "uint8", 1, 63, "batch size=2, batch size must = 1; "),
    ]

    for shape, dtype, a_min, a_max, err_msg in trials:
        model = _get_model(shape, dtype, a_min, a_max)
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
