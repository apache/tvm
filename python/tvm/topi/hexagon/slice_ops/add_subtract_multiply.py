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

"""Compute and schedule for add, multiply, subtract slice op

Please note the following assumptions made by the implementation:

1) The inputs will be multiple of crouton layout except for the axis that needs broadcasting."""

from tvm import te
from tvm import tir
from tvm import topi
from ..utils import get_layout_transform_fn


def add_broadcast_compute(input_a, input_b):
    """Call the add op from topi"""
    return topi.add(input_a, input_b)


def subtract_broadcast_compute(input_a, input_b):
    """Call the subtract op from topi"""
    return topi.subtract(input_a, input_b)


def multiply_broadcast_compute(input_a, input_b):
    """Call the multiply op from topi"""
    return topi.multiply(input_a, input_b)


def tir_broadcast_schedule(
    out_m,
    input_a,
    input_b,
    output_layout: str,
    input_a_layout: str,
    input_b_layout: str,
    op_name: str,
):
    """Schedule for input and output layout nhwc-8h2w32c2w-2d considering broadcast"""
    func = te.create_prim_func([input_a, input_b, out_m])

    s = tir.Schedule(func)

    block_dict = {"add": "T_add", "subtract": "T_subtract", "multiply": "T_multiply"}

    block = s.get_block(block_dict[op_name])

    if input_a_layout == "nhwc-8h2w32c2w-2d":
        input_a_transformed_layout = get_layout_transform_fn(input_a_layout)
        s.transform_layout(block, buffer=("read", 0), index_map=input_a_transformed_layout)

    if input_b_layout == "nhwc-8h2w32c2w-2d":
        input_b_transformed_layout = get_layout_transform_fn(input_b_layout)
        s.transform_layout(block, buffer=("read", 1), index_map=input_b_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=("write", 0), index_map=output_transformed_layout)

    n, h, w, c = s.get_loops(block)

    h_o, h_i = s.split(h, [None, 8])
    w_o, w_i = s.split(w, [None, 4])
    c_o, c_i = s.split(c, [None, 32])
    wio, wii = s.split(w_i, [None, 2])

    s.reorder(n, h_o, w_o, c_o, h_i, wio, c_i, wii)

    fused = s.fuse(c_i, wii)
    s.vectorize(fused)

    return s
