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

"""Compute and schedule for resize2d
Please note the following assumptions made by the implementation:
1) The input and output data will be multiple of crouton layout
2) And the supported layout is NHWC"""

from tvm import te
from tvm import tir
from tvm import topi
from .utils import get_layout_transform_fn


def resize2d_compute(
    data,
    roi,
    size,
    layout,
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    bicubic_alpha=-0.5,
    bicubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
    output_shape=None,
):
    """Call resize2d op from topi.image"""
    return topi.image.resize2d(
        data,
        roi,
        size,
        layout,
        method,
        coordinate_transformation_mode,
        rounding_method,
        bicubic_alpha,
        bicubic_exclude,
        extrapolation_value,
        out_dtype,
        output_shape,
    )


def tir_broadcast_schedule(
    out_m,
    input_a,
    input_layout: str,
    output_layout: str,
):
    """Schedule for input and output layout nhwc-8h2w32c2w-2d"""
    func = te.create_prim_func([input_a, out_m])

    s = tir.Schedule(func)

    block = s.get_block("resize")

    if input_layout == "nhwc-8h2w32c2w-2d":
        input_transformed_layout = get_layout_transform_fn(input_layout)
        s.transform_layout(block, buffer=("read", 0), index_map=input_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=("write", 0), index_map=output_transformed_layout)

    return s
