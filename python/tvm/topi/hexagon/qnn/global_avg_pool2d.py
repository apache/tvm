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

"""
Assumptions:
1) The input is in NCHW layout. Squeezenet is the only model that calls
   nn.global_avg_pool2d and the only layout it uses is 'NCHW'.
2) Both input and output dtype is uint8 and
   quantization parameter is provided to the op.
3) Input is assumed to always be multiple of fixed chunk 32c8h8w.
"""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn, get_fixed_point_value, saturate


def global_avg_pool2d_u8(
    data: te.Tensor,
    odtype: str,
    input_zero_point: int,
    input_scale: float,
    output_zero_point: int,
    output_scale: float,
):
    """global_avg_pool2d"""
    input_b, input_c, input_h, input_w = data.shape
    oshape = (input_b, input_c) + (1, 1)

    if input_h * input_w < 256:
        bits = "16"
    else:
        bits = "32"

    if odtype == "uint8":
        temp_dtype = "uint" + bits
    elif odtype == "int8":
        temp_dtype = "int" + bits
    else:
        raise RuntimeError(f"Unsupported output dtype, {odtype}'")

    pool_area = input_h * input_w
    rh_r = te.reduce_axis((0, input_h), name="rh_r")
    rw_r = te.reduce_axis((0, input_w), name="rw_r")

    scale_with_area = input_scale / (output_scale * int(pool_area))
    scale_fixed_point, rsh = get_fixed_point_value(scale_with_area, "int16")
    corr = (output_zero_point << rsh) - input_zero_point * pool_area * scale_fixed_point

    sum_compute = te.compute(
        oshape,
        lambda n, c, h, w: te.sum(
            data[n, c, h + rh_r, w + rw_r].astype(temp_dtype), axis=[rh_r, rw_r]
        ),
        name="sum",
    )

    avg_compute = te.compute(
        oshape,
        lambda n, c, h, w: saturate(
            ((sum_compute[n, c, h, w] * scale_fixed_point) + corr) >> rsh, odtype
        ).astype(odtype),
        name="global_avg_pool2d",
    )

    return avg_compute


def stir_global_avg_pool2d_u8_schedule(outs: te.Tensor, ins: te.Tensor, input_layout: str):
    """Schedule"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)

    sum_block = s.get_block("sum")

    # Input is multiple of fixed chunk but output is NxCx1x1
    # Hence transform_layout is only applied on input
    input_transformed_layout = get_layout_transform_fn(input_layout)
    s.transform_layout(sum_block, buffer=("read", 0), index_map=input_transformed_layout)

    return s
