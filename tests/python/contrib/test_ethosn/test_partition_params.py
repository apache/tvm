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

"""Arm(R) Ethos(TM)-N partition parameter tests"""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.relay.op.contrib.ethosn import partition_for_ethosn
from tvm.testing import requires_ethosn


@requires_ethosn
def test_ethosn78_partition_no_error():
    """Test Arm(R) Ethos(TM)-N78 partition"""

    a = relay.var("a", shape=[2, 7, 8, 8], dtype="uint8")
    weights = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype("uint8"))
    res = relay.nn.conv2d(
        a, weights, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype="uint8"
    )
    b = relay.var("b", shape=[8], dtype="uint8")
    res = relay.nn.bias_add(res, b, axis=1)

    mod = tvm.IRModule.from_expr(res)
    opts = {"variant": "n78"}
    partition_for_ethosn(mod, **opts)


@requires_ethosn
def test_ethosn78_partition_undefined_variant():
    """Test Arm(R) Ethos(TM)-N78 partition with undefined variant"""

    with pytest.raises(
        ValueError, match=r".*Please specify a variant in the target string, e.g. -variant=n78.*"
    ):
        a = relay.var("a", shape=[2, 7, 8, 8], dtype="uint8")
        weights = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype("uint8"))
        res = relay.nn.conv2d(
            a, weights, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype="uint8"
        )
        b = relay.var("b", shape=[8], dtype="uint8")
        res = relay.nn.bias_add(res, b, axis=1)

        mod = tvm.IRModule.from_expr(res)
        partition_for_ethosn(mod)


@requires_ethosn
def test_ethosn78_partition_invalid_variant():
    """Test Arm(R) Ethos(TM)-N78 partition with invalid variant"""

    with pytest.raises(
        ValueError, match=r".*When targeting Ethos\(TM\)-N78, -variant=n78 should be set.*"
    ):
        a = relay.var("a", shape=[2, 7, 8, 8], dtype="uint8")
        wwights = relay.const(np.random.uniform(-10, 10, (8, 7, 3, 3)).astype("uint8"))
        res = relay.nn.conv2d(
            a, wwights, kernel_size=(3, 3), padding=(1, 1), channels=8, out_dtype="uint8"
        )
        b = relay.var("b", shape=[8], dtype="uint8")
        res = relay.nn.bias_add(res, b, axis=1)

        mod = tvm.IRModule.from_expr(res)
        opts = {"variant": "Ethos-N"}
        partition_for_ethosn(mod, **opts)
