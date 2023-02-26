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
"""Compute and schedule for hexagon quantize
Please note the following assumptions made by the implementation:
1) The input and output data will be multiple of crouton layout
2) And the supported layout is NHWC
3) The input layout will be nhwc-4h2w32c2w-2d and
   output layout will be nhwc-8h8w32c-2d"""


from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn, saturate


def quantize_compute(tensor_A: te.Tensor, scale: float, zero_point: int, dtype: str):
    """Compute for quantize"""
    scale_recip = 1 / scale

    return te.compute(
        tensor_A.shape,
        lambda n, h, w, c: saturate(
            ((tensor_A[n, h, w, c] * scale_recip).astype("int32") + zero_point),
            dtype,
        ).astype(dtype),
        name="quantize",
    )


def tir_quantize_schedule(
    out_M: te.Tensor,
    tensor_A: te.Tensor,
    input_layout: str,
    output_layout: str,
):
    """Schedule for output layout nhwc-8h8w32c-2d"""
    func = te.create_prim_func([tensor_A, out_M])

    s = tir.Schedule(func)

    block = s.get_block("quantize")

    input_transformed_layout = get_layout_transform_fn(input_layout)
    s.transform_layout(block, buffer=tensor_A.name, index_map=input_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=out_M.name, index_map=output_transformed_layout)

    # Fixed chunk size is 2048 byte
    # For uint8 the layout for fixed chunk is 8x8x32
    # where each element is 1 bytes
    # Split and reorder is done to iterate over the fixed chunk
    # Channel is split by a factor of 32
    # Width is split by a factor of 8
    # Height is split by a factor of 8
    n, h, w, c = s.get_loops(block)

    h_o, h_i = s.split(h, [None, 8])
    w_o, w_i = s.split(w, [None, 8])
    c_o, c_i = s.split(c, [None, 32])
    wio, wii = s.split(w_i, [None, 4])

    s.reorder(n, h_o, w_o, c_o, h_i, wio, wii, c_i)

    return s
