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

""" Hexagon tanh slice op compute and schedule """
import tvm
from tvm import te, tir
from ..utils import get_layout_transform_fn


def tanh_te_compute(in_tensor):
    out_tensor = te.compute(
        in_tensor.shape, lambda n, h, w, c: tvm.tir.tanh(in_tensor[n, h, w, c]), name="tanhf16"
    )
    return out_tensor


def tanhf16_stir_sched_nhwc(func, in_layout, out_layout, h_split_factor=8):
    """Schedule for nhwc fp16 to nchw fp16 layout"""
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "tanhf16"
    n, h, w, c = sch.get_loops(sch.get_block(block_name))
    h_outer, h_inner = sch.split(h, [None, h_split_factor])
    w_outer, w_inner = sch.split(w, [None, 4])
    c_outer, c_inner = sch.split(c, [None, 32])
    w_inner_o, w_inner_i = sch.split(w_inner, [None, 2])
    sch.reorder(n, h_outer, w_outer, c_outer, h_inner, w_inner_o, c_inner, w_inner_i)
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    fused = sch.fuse(c_inner, w_inner_i)
    sch.vectorize(fused)
    return sch


def tanhf16_schedule(tanh_func, in_layout_str, out_layout_str):
    in_layout_transform_func = get_layout_transform_fn(in_layout_str)
    out_layout_transform_func = get_layout_transform_fn(out_layout_str)
    return tanhf16_stir_sched_nhwc(
        tanh_func,
        in_layout_transform_func,
        out_layout_transform_func,
    )
