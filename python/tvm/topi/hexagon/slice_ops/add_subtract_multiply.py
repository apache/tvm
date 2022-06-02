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
# pylint: disable=invalid-name, unused-variable, unused-argument, too-many-locals

"""Compute and schedule for add, multiply, subtract slice op

Please note the following assumptions made by the implementation:

1) The inputs will be multiple of crouton layout except for the axis that needs broadcasting."""

from tvm import te
from tvm import tir
from tvm import topi
from ..utils import get_layout_transform_fn


def add_broadcast_compute(A, B):
    """Call the add op from topi"""
    return topi.add(A, B)


def subtract_broadcast_compute(A, B):
    """Call the subtract op from topi"""
    return topi.subtract(A, B)


def multiply_broadcast_compute(A, B):
    """Call the multiply op from topi"""
    return topi.multiply(A, B)


def get_2d_layout(layout):
    """Get the 2d layout for transformation"""
    layout += "-2d"
    return get_layout_transform_fn(layout)


def STIR_broadcast_schedule(
    M, A, B, output_layout: str, input_A_layout: str, input_B_layout: str, op_name: str
):
    """Schedule for input and output layout nhwc-8h2w32c2w considering broadcast"""
    func = te.create_prim_func([A, B, M])

    s = tir.Schedule(func)

    block_dict = {"add": "T_add", "subtract": "T_subtract", "multiply": "T_multiply"}

    block = s.get_block(block_dict[op_name])

    if input_A_layout == "nhwc-8h2w32c2w":
        input_A_transformed_layout = get_2d_layout(input_A_layout)
        s.transform_layout(block, buffer=("read", 0), index_map=input_A_transformed_layout)

    if input_B_layout == "nhwc-8h2w32c2w":
        input_B_transformed_layout = get_2d_layout(input_B_layout)
        s.transform_layout(block, buffer=("read", 1), index_map=input_B_transformed_layout)

    output_transformed_layout = get_2d_layout(output_layout)
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
