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

""" Compute and schedule for adaptive_avg_pool1d slice op

Following are few notes and assumptions made by the implementation:

Assumptions:
1) The input is in NCW layout. Distilbert is the only model that calls
   nn.adaptive_avg_pool1d and the only layout it uses is 'NCW'.
2) The op takes output_size as an argument and
   only handles the specialized case where output_size is 1.
   The argument output_size is used as the value of output_width.
3) Both input and output dtype is uint8/int8 and
   quantization parameter is provided to the op.
4) Input is assumed to always be multiple of fixed chunk 32c64w.

Notes:
1) If input width is used as output width, there can be two cases:
    a. If the quantization parameters of input and output are same,
       it can return the input as output so the op will be a no-op.
    b. If the quantization parameters of input and output are different,
       it will essentially be a requantize op.
2) If output_size is a value besides 1 or input_width,
   adaptive_avg_pool1d may use dynamic stride and kernel for each output element.
   When this case occurs, kernel won't be known at compile time. We want to use
   the generic implementation nn.adaptive_avg_pool1d() for this case.
"""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn, get_fixed_point_value, saturate


def adaptive_avg_pool1d(
    data: te.Tensor,
    output_size: list,
    odtype: str,
    input_zero_point: int,
    input_scale: float,
    output_zero_point: int,
    output_scale: float,
):
    """adaptive_avg_pool1d compute"""
    _, _, inw = data.shape

    out_width = output_size[0]

    n, c = data.shape[:2]
    oshape = (n, c) + (out_width,)

    # Kernel is same as input_width since output_width is assumed to be 1
    if out_width == 1:
        kw_r = inw
    else:
        raise RuntimeError(f"Unsupported output_size, {out_width}'")

    if odtype == "uint8":
        temp_dtype = "uint32"
    elif odtype == "int8":
        temp_dtype = "int32"
    else:
        raise RuntimeError(f"Unsupported output dtype, {odtype}'")

    scale_with_area = input_scale / (output_scale * int(kw_r))
    scale_fixed_point, rsh = get_fixed_point_value(scale_with_area, "int16")
    corr = (output_zero_point << rsh) - input_zero_point * kw_r * scale_fixed_point

    rw_r = te.reduce_axis((0, kw_r), name="rw_r")

    sum_compute = te.compute(
        oshape,
        lambda n, c, w: te.sum(data[n, c, w + rw_r].astype(temp_dtype), axis=[rw_r]),
        name="sum",
    )

    avg_compute = te.compute(
        oshape,
        lambda n, c, w: saturate(
            ((sum_compute[n, c, w] * scale_fixed_point) + corr) >> rsh, odtype
        ).astype(odtype),
        name="adaptive_avg_1d",
    )
    return avg_compute


def stir_schedule_ncw_32c64w(outs, ins, input_layout: str):
    """Schedule for input layout ncw-32c64w and output layout ncw"""
    func = te.create_prim_func([ins, outs])
    s = tir.Schedule(func)

    sum_block = s.get_block("sum")

    # Input is multiple of fixed chunk but output is NxCx1
    # Hence transform_layout is only applied on input
    input_transformed_layout = get_layout_transform_fn(input_layout)
    s.transform_layout(sum_block, buffer=("read", 0), index_map=input_transformed_layout)

    return s


def tir_adaptive_avg_pool1d_schedule(outs, ins, output_layout: str, input_layout: str):
    """STIR based schedule"""
    if output_layout == "ncw":
        return stir_schedule_ncw_32c64w(outs, ins, input_layout)
    raise RuntimeError(f"Unexpected layout '{output_layout}'")
