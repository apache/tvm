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

""" Test the use of rolling buffer schedule primitive for avoiding repeated loads """

import numpy as np

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from .infrastructure import get_hexagon_target


def generate_conv2d(dtype):
    """Generate a pseudo conv2d with unit weights"""

    @T.prim_func
    def conv2d_2x1_filter(act: T.handle, res: T.handle):
        a_buffer = T.match_buffer(act, (6, 32), dtype)
        b_buffer = T.match_buffer(res, (5, 32), dtype)
        for output_row, i in T.grid(5, 32):
            for reduce_h in T.serial(2):
                with T.block("compute"):
                    vrow = T.axis.spatial(5, output_row)
                    vcol = T.axis.spatial(32, i)
                    vrh = T.axis.reduce(2, reduce_h)
                    with T.init():
                        b_buffer[vrow, vcol] = T.cast(0, dtype=dtype)
                    b_buffer[vrow, vcol] = b_buffer[vrow, vcol] + a_buffer[vrow + vrh, vcol]

    return conv2d_2x1_filter


def test_rolling_buffer_conv2d_2x1(hexagon_session):
    """Test the rolling buffer schedule primitive on a pseudo conv2d"""
    dtype = "float16"
    sch = tir.Schedule(generate_conv2d(dtype).with_attr("global_symbol", "main"))
    compute_block = sch.get_block("compute")
    output_row, _, _ = sch.get_loops(compute_block)
    cache_read_block = sch.cache_read(compute_block, 0, "global.vtcm")
    sch.compute_at(cache_read_block, output_row)
    sch.rolling_buffer(cache_read_block, 0)

    target = get_hexagon_target("v69")
    mod = tvm.build(sch.mod, target=target)
    mod = hexagon_session.load_module(mod)
    a = tvm.nd.array(np.ones((6, 32), dtype=dtype), device=hexagon_session.device)
    b = tvm.nd.array(np.zeros((5, 32), dtype=dtype), device=hexagon_session.device)
    mod(a, b)
    tvm.testing.assert_allclose(b.numpy(), np.full((5, 32), 2, dtype=dtype), atol=1e-3, rtol=1e-3)
