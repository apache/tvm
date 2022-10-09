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

""" Hexagon qnn.dequantize slice op compute and schedule"""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn


def dequantize_compute(tensor_A, scale_A, zero_point_A):

    return te.compute(
        tensor_A.shape,
        lambda *indices: (scale_A * (tensor_A[indices] - zero_point_A)).astype("float32"),
        name="dequantize",
    )


def dequantize_stir_schedule_nhwc_8h8w32c(
    _in,
    _out,
    in_layout,
    out_layout,
):
    """Schedule for nhwc int8/uint8 to f32 : nhwc layout"""
    func = te.create_prim_func([_in, _out])
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "dequantize"
    n, h, w, c = sch.get_loops(sch.get_block(block_name))
    ho, hi = sch.split(h, [None, 4])
    wo, wi = sch.split(w, [None, 8])
    wio, wii = sch.split(wi, [None, 4])
    co, ci = sch.split(c, [None, 32])
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    sch.reorder(n, ho, wo, co, hi, wio, wii, ci)
    wii_ci = sch.fuse(wii, ci)
    sch.vectorize(wii_ci)
    return sch


def dequantize_stir_schedule_nc(
    _in,
    _out,
    in_layout,
    out_layout,
):
    """Schedule for nc int8/uint8 to f32 : nc layout"""
    func = te.create_prim_func([_in, _out])
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "dequantize"
    _, c_orig = sch.get_loops(sch.get_block(block_name))
    _, c_inner = sch.split(c_orig, [None, 512])
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    sch.vectorize(c_inner)
    return sch


def dequantize_schedule(_in, _output, in_layout_str, out_layout_str):
    """Schedule for int8/uint8 to f32 : top level function"""
    f32_layout_transform_func = get_layout_transform_fn(out_layout_str)
    in_layout_transform_func = get_layout_transform_fn(in_layout_str)
    if out_layout_str == "nhwc-4h2w32c2w-2d":
        return dequantize_stir_schedule_nhwc_8h8w32c(
            _in,
            _output,
            in_layout_transform_func,
            f32_layout_transform_func,
        )
    if out_layout_str == "nc-512c-2d":
        return dequantize_stir_schedule_nc(
            _in,
            _output,
            in_layout_transform_func,
            f32_layout_transform_func,
        )
    raise RuntimeError(f"Unexpected layout '{layout}'")
