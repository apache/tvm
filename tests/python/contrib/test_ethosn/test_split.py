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

"""Split tests for Ethos-N"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosn import ethosn_available
from . import infrastructure as tei


def _get_model(shape, dtype, splits, axis):
    a = relay.var("a", shape=shape, dtype=dtype)
    split = relay.op.split(a, indices_or_sections=splits, axis=axis)
    return split.astuple()


def test_split():
    if not ethosn_available():
        return

    trials = [
        ((1, 16, 16, 32), (2, 7, 10), 2),
        ((1, 12, 8, 16), 3, 1),
        ((1, 33), 11, 1),
    ]

    np.random.seed(0)
    for shape, splits, axis in trials:
        outputs = []
        inputs = {"a": tvm.nd.array(np.random.randint(0, high=256, size=shape, dtype="uint8"))}
        for npu in [False, True]:
            model = _get_model(shape, "uint8", splits, axis)
            mod = tei.make_module(model, {})
            output_count = splits if type(splits) == int else len(splits) + 1
            outputs.append(tei.build_and_run(mod, inputs, output_count, {}, npu=npu))

        tei.verify(outputs, 0)


def test_split_failure():
    if not ethosn_available():
        return

    trials = [
        ((1, 4, 4, 4, 4), "uint8", 4, 2, "dimensions=5, dimensions must be <= 4;"),
        ((1, 4, 4, 4), "int8", 4, 2, "dtype='int8', dtype must be either uint8 or int32;"),
        ((2, 4, 4, 4), "uint8", 4, 2, "batch size=2, batch size must = 1;"),
        ((1, 4, 4, 4), "uint8", 1, 0, "Split cannot be performed along batch axis (axis 0);"),
        (
            (1, 4, 4, 4),
            "uint8",
            4,
            3,
            "Split along the channels dimension (axis 3) requires all output sizes (specified in splitInfo.m_Sizes) to be multiples of 16;",
        ),
    ]

    for shape, dtype, splits, axis, err_msg in trials:
        model = _get_model(shape, dtype, splits, axis)
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
