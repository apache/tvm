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
# pylint: disable=wrong-import-position

"""
Tests to check that the NPU partitioning frontend partitions
only supported operations.
"""

import pytest

pytest.importorskip("ethosu.vela")

import tvm
from tvm import relay
from tvm.relay.op.contrib import ethosu


@pytest.mark.parametrize(
    "count_include_pad,pool_shape,padding",
    [
        (True, [2, 2], [0, 0, 0, 0]),
        (False, [2, 2], [4, 4, 5, 5]),
        (False, [9, 9], [1, 1, 1, 1]),
    ],
)
def test_invalid_avg_pool2d(count_include_pad, pool_shape, padding):
    """
    Test unsupported variants of avg_pool2d don't get partitioned.
    """
    ifm_shape = [1, 4, 4, 3]
    strides = [2, 2]

    def get_graph():
        x = relay.var("x", shape=ifm_shape, dtype="int8")
        x = relay.cast(x, dtype="int32")
        x = relay.nn.avg_pool2d(
            x,
            pool_shape,
            strides,
            padding=padding,
            layout="NHWC",
            count_include_pad=count_include_pad,
        )
        x = relay.cast(x, dtype="int8")
        func = relay.Function(relay.analysis.free_vars(x), x)
        return tvm.IRModule.from_expr(func)

    mod = relay.transform.InferType()(get_graph())
    partitioned_mod = ethosu.partition_for_ethosu(mod)
    assert tvm.ir.structural_equal(mod, partitioned_mod)
