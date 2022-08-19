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
""" Hexagon slice cast op compute and schedule"""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn


def get_layout_transform_for_f32(f32_layout_string):
    """
    Given f32 layout string, return transform_layout function and
    channel/height split factor to be used for scheduling
    """
    layout_transform_fn = get_layout_transform_fn(f32_layout_string)
    if f32_layout_string == "nhwc-8h2w32c2w-2d":
        return [layout_transform_fn, 8]
    if f32_layout_string == "nhwc-4h2w32c2w-2d":
        return [layout_transform_fn, 4]
    if f32_layout_string == "nc-1024c-2d":
        return [layout_transform_fn, 1024]
    if f32_layout_string == "nc-512c-2d":
        return [layout_transform_fn, 512]
    raise RuntimeError(f"Unexpected f32_layout '{f32_layout_string}'")


def cast_f16_f32_compute(in_tensor):
    out_tensor = te.compute(
        in_tensor.shape, lambda *indices: in_tensor[indices].astype("float32"), name="CastF16F32"
    )
    return out_tensor


def cast_f16_f32_stir_schedule_nhwc(func, in_layout, out_layout, h_split_factor):
    """Schedule for nhwc f16 to f32 cast: nhwc layout"""
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "CastF16F32"
    n_orig, h_orig, w_orig, c_orig = sch.get_loops(sch.get_block(block_name))
    h_outer, h_inner = sch.split(h_orig, [None, h_split_factor])
    w_outer, w_inner = sch.split(w_orig, [None, 4])
    c_outer, c_inner = sch.split(c_orig, [None, 32])
    w_inner_o, w_inner_i = sch.split(w_inner, [None, 2])
    sch.reorder(n_orig, h_outer, w_outer, c_outer, h_inner, w_inner_o, c_inner, w_inner_i)
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    fused = sch.fuse(c_inner, w_inner_i)
    sch.vectorize(fused)
    return sch


def cast_f16_f32_stir_schedule_nc(func, in_layout, out_layout, c_split_factor):
    """Schedule for nc f16 to f32 cast: nc layout"""
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "CastF16F32"
    _, c_orig = sch.get_loops(sch.get_block(block_name))
    _, c_inner = sch.split(c_orig, [None, c_split_factor])
    _, c_inner_inner = sch.split(c_inner, [None, 64])
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    sch.vectorize(c_inner_inner)
    return sch


def cast_f16_f32_schedule(cast_func, in_layout_str, out_layout_str):
    """Schedule for f16 to f32 cast: top level function"""
    f32_layout_transform_func, split_factor = get_layout_transform_for_f32(out_layout_str)
    f16_layout_transform_func = get_layout_transform_fn(in_layout_str)
    if in_layout_str == "nhwc-8h2w32c2w-2d":
        return cast_f16_f32_stir_schedule_nhwc(
            cast_func,
            f16_layout_transform_func,
            f32_layout_transform_func,
            split_factor,
        )
    if in_layout_str == "nc-1024c-2d":
        return cast_f16_f32_stir_schedule_nc(
            cast_func, f16_layout_transform_func, f32_layout_transform_func, split_factor
        )
    raise RuntimeError(f"Unexpected input_layout, output_layout '{input_layout, output_layout}'")


def cast_f32_f16_compute(in_tensor):
    out_tensor = te.compute(
        in_tensor.shape, lambda *indices: in_tensor[indices].astype("float16"), name="CastF32F16"
    )
    return out_tensor


def cast_f32_f16_stir_schedule_nhwc(func, in_layout, out_layout, h_split_factor):
    """Schedule for nhwc f32 to f16 cast: nhwc layout"""
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "CastF32F16"
    n_orig, h_orig, w_orig, c_orig = sch.get_loops(sch.get_block(block_name))
    h_outer, h_inner = sch.split(h_orig, [None, h_split_factor])
    w_outer, w_inner = sch.split(w_orig, [None, 4])
    c_outer, c_inner = sch.split(c_orig, [None, 32])
    w_inner_o, w_inner_i = sch.split(w_inner, [None, 2])
    sch.reorder(n_orig, h_outer, w_outer, c_outer, h_inner, w_inner_o, c_inner, w_inner_i)
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    fused = sch.fuse(c_inner, w_inner_i)
    sch.vectorize(fused)
    return sch


def cast_f32_f16_stir_schedule_nc(func, in_layout, out_layout, c_split_factor):
    """Schedule for nc f32 to f16 cast: nc layout"""
    sch = tir.Schedule(func, debug_mask="all")
    block_name = "CastF32F16"
    _, c_orig = sch.get_loops(sch.get_block(block_name))
    _, c_inner = sch.split(c_orig, [None, c_split_factor])
    _, c_inner_inner = sch.split(c_inner, [None, 64])
    sch.transform_layout(block_name, "A", in_layout)
    sch.transform_layout(block_name, block_name, out_layout)
    sch.vectorize(c_inner_inner)
    return sch


def cast_f32_f16_schedule(cast_func, in_layout_str, out_layout_str):
    """Schedule for f32 to f16 cast: top level function"""
    f32_layout_transform_func, split_factor = get_layout_transform_for_f32(in_layout_str)
    f16_layout_transform_func = get_layout_transform_fn(out_layout_str)
    if out_layout_str == "nhwc-8h2w32c2w-2d":
        return cast_f32_f16_stir_schedule_nhwc(
            cast_func, f32_layout_transform_func, f16_layout_transform_func, split_factor
        )
    if out_layout_str == "nc-1024c-2d":
        return cast_f32_f16_stir_schedule_nc(
            cast_func, f32_layout_transform_func, f16_layout_transform_func, split_factor
        )
    raise RuntimeError(f"Unexpected input_layout, output_layout '{in_layout_str, out_layout_str}'")
