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

""" Compute and schedule for depth to space slice op
"""

from tvm import te, tir, topi
from ..utils import get_layout_transform_fn


def d2s_compute(inp, block_size, layout, mode):
    """depth_to_space compute"""
    return topi.nn.depth_to_space(inp, block_size=block_size, layout=layout, mode=mode)


def d2s_schedule(inp, out, input_layout, output_layout):
    """Schedule for depth to space: top level function"""
    if (input_layout != output_layout) or (
        output_layout not in ("nhwc-8h2w32c2w-2d", "nhwc-8h8w32c-2d")
    ):
        raise RuntimeError(
            f"Unexpected input_layout, output_layout '{input_layout, output_layout}'"
        )
    d2s_func = te.create_prim_func([inp, out])
    sch = tir.Schedule(d2s_func, debug_mask="all")
    compute = sch.get_block("depth_to_space")
    sch.transform_layout(compute, inp.name, get_layout_transform_fn(input_layout))
    sch.transform_layout(compute, out.name, get_layout_transform_fn(output_layout))
    return sch
