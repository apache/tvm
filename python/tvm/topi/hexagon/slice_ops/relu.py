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
# pylint: disable=invalid-name
"""Hexagon slice relu op"""

from tvm import te, tir, topi
from ..utils import get_layout_transform_fn


def relu_compute(Input):
    """Relu topi compute"""
    return topi.nn.relu(Input)


def relu_te_sched(Output, Input, layout):
    """
    Schedule assumes the layout function to be bijective
    """
    s = te.create_schedule(Output.op)
    s[Input].transform_layout(layout)
    out_axes = s[Output].transform_layout(layout)
    fused = s[Output].fuse(out_axes[6], out_axes[7])
    s[Output].vectorize(fused)
    return s


def relu_stir_schedule(Input, Output, input_layout, output_layout):
    """
    Schedule assumes the layout function to be bijective
    """
    if (input_layout != output_layout) or (output_layout != "nhwc-8h2w32c2w-2d"):
        raise RuntimeError(
            f"Unexpected input_layout, output_layout '{input_layout, output_layout}'"
        )
    relu_func = te.create_prim_func([Input, Output])
    sch = tir.Schedule(relu_func, debug_mask="all")
    block = sch.get_block("compute")
    sch.transform_layout(block, Input.name, get_layout_transform_fn(input_layout))
    sch.transform_layout(block, Output.name, get_layout_transform_fn(output_layout))

    n, h, w, c = sch.get_loops(block)
    h_o, h_i = sch.split(h, [None, 8])
    w_o, w_i = sch.split(w, [None, 4])
    c_o, c_i = sch.split(c, [None, 32])
    wio, wii = sch.split(w_i, [None, 2])

    sch.reorder(n, h_o, w_o, c_o, h_i, wio, c_i, wii)

    fused = sch.fuse(c_i, wii)
    sch.vectorize(fused)
    return sch
