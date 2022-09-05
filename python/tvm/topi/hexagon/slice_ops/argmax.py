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
""" Hexagon slice argmax compute and schedule"""

from tvm import tir
from tvm import topi
from ..utils import get_layout_transform_fn


def argmax_compute(in_tensor, axis):
    out_tensor = topi.argmax(in_tensor, axis)
    return out_tensor


def argmax_stir_schedule_nhwc(func, in_layout, out_layout):
    """Schedule for nhwc argmax"""
    sch = tir.Schedule(func, debug_mask="all")
    sch.transform_layout("A_red_temp", "A", in_layout)
    sch.transform_layout("A_red", "A_red", out_layout)
    return sch


def argmax_schedule(argmax_func, in_layout_str, out_layout_str):
    """Schedule for argmax: top level function"""
    if (in_layout_str == "nhwc-8h2w32c2w-2d") and (out_layout_str == "nhw-32h16w-2d"):
        fp16_layout_transform = get_layout_transform_fn(in_layout_str)
        int32_layout_transform = get_layout_transform_fn(out_layout_str)
        tir_s = argmax_stir_schedule_nhwc(
            argmax_func, fp16_layout_transform, int32_layout_transform
        )
        return tir_s
    if (in_layout_str == "nhwc-8h8w32c-2d") and (out_layout_str == "nhw-32h16w-2d"):
        int8_layout_transform = get_layout_transform_fn(in_layout_str)
        int32_layout_transform = get_layout_transform_fn(out_layout_str)
        tir_s = argmax_stir_schedule_nhwc(
            argmax_func, int8_layout_transform, int32_layout_transform
        )
        return tir_s
    raise RuntimeError(f"Unexpected input_layout, output_layout '{in_layout_str, out_layout_str}'")
