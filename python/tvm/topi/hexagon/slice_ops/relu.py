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

import tvm
from tvm import te, tir


def relu_te_compute(Input, dtype):
    """
    Compute assumes the input layout to be NHWC
    """
    x = tvm.tir.const(0, dtype)
    Output = te.compute(
        Input.shape, lambda n, h, w, c: tvm.te.max(Input[n, h, w, c], x), name="reluf16"
    )
    return Output


def reluf16_te_sched(Output, Input, layout):
    """
    Schedule assumes the layout function to be bijective
    """
    s = tvm.te.create_schedule(Output.op)
    s[Input].transform_layout(layout)
    out_axes = s[Output].transform_layout(layout)
    fused = s[Output].fuse(out_axes[6], out_axes[7])
    s[Output].vectorize(fused)
    return s


def reluf16_stir_sched(func, layout):
    """
    Schedule assumes the layout function to be bijective
    """
    sch = tir.Schedule(func, debug_mask="all")
    block = sch.get_block("reluf16")
    n, h, w, c = sch.get_loops(block)
    ho, hi = sch.split(h, [None, 8])
    wo, wi = sch.split(w, [None, 4])
    co, ci = sch.split(c, [None, 32])
    wio, wii = sch.split(wi, [None, 2])
    sch.reorder(n, ho, wo, co, hi, wio, ci, wii)
    sch.transform_layout(block, ("read", 0), layout)
    sch.transform_layout(block, ("write", 0), layout)
    fused = sch.fuse(ci, wii)
    sch.vectorize(fused)
    return sch
