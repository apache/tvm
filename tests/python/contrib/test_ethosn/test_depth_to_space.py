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

"""Ethos-N integration depth-to-space tests"""

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from . import infrastructure as tei
import numpy as np


def _get_model(shape, block, dtype, layout):
    a = relay.var("a", shape=shape, dtype=dtype)
    depth = relay.nn.depth_to_space(a, layout=layout, block_size=block)
    return depth


@requires_ethosn
def test_depth_to_space():
    trials = [
        (1, 16, 16, 16),
        (1, 64, 32, 16),
    ]

    for shape in trials:
        inputs = {
            "a": tvm.nd.array(np.random.randint(0, high=255, size=shape, dtype="uint8")),
        }
        outputs = []
        for npu in [False, True]:
            model = _get_model(shape, 2, "uint8", "NHWC")
            mod = tei.make_module(model, {})
            outputs.append(tei.build_and_run(mod, inputs, 1, {}, npu=npu))

        tei.verify(outputs, 1)


@requires_ethosn
def test_depth_to_space_failure():
    trials = [
        ((2, 16, 16, 16), 2, "uint8", "NHWC", "batch size=2, batch size must = 1"),
        ((1, 16, 16, 16), 2, "int8", "NHWC", "dtype='int8', dtype must be either uint8 or int32"),
        ((1, 16, 16, 16), 4, "uint8", "NHWC", "Only block size of 2 is supported"),
        ((1, 16, 16, 16), 2, "uint8", "NCHW", "layout=NCHW, layout must = NHWC"),
    ]

    for shape, block, dtype, layout, err_msg in trials:
        model = _get_model(shape, block, dtype, layout)
        mod = tei.make_ethosn_partition(model)
        tei.test_error(mod, {}, err_msg)
