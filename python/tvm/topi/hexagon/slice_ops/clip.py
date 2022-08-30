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

"""
Clip the elements in `A` between `A_min` and `A_max`.
"""

from tvm import te, tir, topi
from ..utils import get_layout_transform_fn


def clip_compute(A, A_min, A_max):
    """
    Use topi clip implementation
    """
    return topi.clip(A, A_min, A_max)


def clip_schedule(outs, ins, output_layout: str, input_layout: str):
    """
    Hexagon clip schedule
    """
    A = ins
    M = outs

    func = te.create_prim_func([A, M])

    s = tir.Schedule(func)

    block = s.get_block("compute")

    input_transformed_layout = get_layout_transform_fn(input_layout)
    s.transform_layout(block, buffer=("read", 0), index_map=input_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=("write", 0), index_map=output_transformed_layout)

    n, h, w, c = s.get_loops(block)

    ho, hi = s.split(h, [None, 8])
    wo, wi = s.split(w, [None, 4])
    co, ci = s.split(c, [None, 32])
    wio, wii = s.split(wi, [None, 2])

    s.reorder(n, ho, wo, co, hi, wio, ci, wii)

    fused = s.fuse(ci, wii)
    s.vectorize(fused)

    return s
