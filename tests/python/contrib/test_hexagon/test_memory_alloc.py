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
"""Test memory allocation."""

import numpy as np

import tvm
from tvm.script import tir as T
from tvm.contrib.hexagon import allocate_hexagon_array

from .infrastructure import get_hexagon_target


def generated_func(shape: tuple, dtype: str, axis_separators: list):
    """Generate element wise function."""
    dim0, dim1 = shape

    @T.prim_func
    def elwise(a: T.handle, b: T.handle):
        a_buffer = T.match_buffer(a, shape, dtype=dtype, axis_separators=axis_separators)
        b_buffer = T.match_buffer(b, shape, dtype=dtype, axis_separators=axis_separators)

        for i, j in T.grid(dim0, dim1):
            with T.block("compute"):
                b_buffer[i, j] = a_buffer[i, j] * T.cast(2, dtype=dtype)

    return elwise


class TestMemoryAlloc:
    """Memory allocation test."""

    dtype = tvm.testing.parameter("int8")
    shape = tvm.testing.parameter((128, 128))

    (scope, axis_separators,) = tvm.testing.parameters(
        ("global", []),
        ("global.vtcm", []),
        ("global.vtcm", [1]),
        ("global.ddr", []),
        ("global.ddr", [1]),
    )

    def test_global_axis_separator(self, hexagon_session, shape, dtype, scope, axis_separators):
        """Test with global axis separator."""
        mod1 = tvm.build(
            generated_func(shape, dtype, axis_separators),
            target=get_hexagon_target("v69"),
        )
        mod2 = hexagon_session.load_module(mod1)

        a_np = np.ones(shape=shape, dtype=dtype)
        a = allocate_hexagon_array(
            hexagon_session.device, data=a_np, mem_scope=scope, axis_separators=axis_separators
        )

        b_np = np.zeros(shape=shape, dtype=dtype)
        b = allocate_hexagon_array(
            hexagon_session.device, data=b_np, mem_scope=scope, axis_separators=axis_separators
        )

        mod2(a, b)
        tvm.testing.assert_allclose(a.numpy() * 2, b.numpy(), atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
