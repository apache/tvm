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

"""Compute and schedule for quantized add, multiply, subtract op

Please note the following assumptions made by the implementation:

1) The inputs will be multiple of crouton layout except for the axis that needs broadcasting."""

from tvm import te
from tvm import tir
from ..utils import get_layout_transform_fn, get_fixed_point_value


def broadcast_axis(tensor_A, tensor_B):
    """Find out the indices that will have broadcasting"""
    A_broadcast = []
    B_broadcast = []

    for i in range(len(tensor_A.shape)):
        if tensor_A.shape[i] == tensor_B.shape[i]:
            A_broadcast.append(1)
            B_broadcast.append(1)
        elif tensor_A.shape[i] == 1:
            A_broadcast.append(0)
            B_broadcast.append(1)
        elif tensor_B.shape[i] == 1:
            A_broadcast.append(1)
            B_broadcast.append(0)
    return A_broadcast, B_broadcast


def saturate(x: te.Tensor, dtype: str):
    """Saturate value for the specified data type"""
    return te.max(te.min_value(dtype), te.min(x, te.max_value(dtype)))


def get_int_scale(scale_A, scale_B, scale_M, zero_point_A, zero_point_B, zero_point_M, op):
    """Get fixed-point number"""
    C_recip = 1 / scale_M

    if op == "qmul":
        scale = scale_A * scale_B * C_recip
        scale_fixed_point, rsh = get_fixed_point_value(scale, "int16")
        corr = zero_point_M << rsh

        return scale_fixed_point, rsh, corr
    else:
        a_scale_f = scale_A * C_recip
        b_scale_f = scale_B * C_recip
        scale_fixed_point_a, rsh_a = get_fixed_point_value(a_scale_f, "int16")
        scale_fixed_point_b, rsh_b = get_fixed_point_value(b_scale_f, "int16")

        if rsh_a > rsh_b:
            scale_fixed_point_a = scale_fixed_point_a >> (rsh_a - rsh_b)
            rsh = rsh_b
        else:
            scale_fixed_point_b = scale_fixed_point_b >> (rsh_b - rsh_a)
            rsh = rsh_a

        if op == "qadd":
            corr = (zero_point_M << rsh) - (
                zero_point_A * scale_fixed_point_a + zero_point_B * scale_fixed_point_b
            )
        else:
            corr = (zero_point_M << rsh) - (
                zero_point_A * scale_fixed_point_a - zero_point_B * scale_fixed_point_b
            )

        return scale_fixed_point_a, scale_fixed_point_b, rsh, corr


def qadd_broadcast_compute(
    tensor_A,
    tensor_B,
    output_shape,
    zero_point_A,
    scale_A,
    zero_point_B,
    scale_B,
    zero_point_M,
    scale_M,
    dtype,
):
    """Compute quantized add with broadcasting"""
    A_broadcast, B_broadcast = broadcast_axis(tensor_A, tensor_B)
    n_a, h_a, w_a, c_a = A_broadcast
    n_b, h_b, w_b, c_b = B_broadcast

    scale_a, scale_b, rsh, corr = get_int_scale(
        scale_A, scale_B, scale_M, zero_point_A, zero_point_B, zero_point_M, "qadd"
    )

    return te.compute(
        output_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    (tensor_A[n * n_a, h * h_a, w * w_a, c * c_a] * scale_a)
                    + (tensor_B[n * n_b, h * h_b, w * w_b, c * c_b] * scale_b)
                    + corr
                )
                >> rsh
            ),
            dtype,
        ).astype(dtype),
    )


def qsubtract_broadcast_compute(
    tensor_A,
    tensor_B,
    output_shape,
    zero_point_A,
    scale_A,
    zero_point_B,
    scale_B,
    zero_point_M,
    scale_M,
    dtype,
):
    """Compute quantized subtract with broadcasting"""
    A_broadcast, B_broadcast = broadcast_axis(tensor_A, tensor_B)
    n_a, h_a, w_a, c_a = A_broadcast
    n_b, h_b, w_b, c_b = B_broadcast

    scale_a, scale_b, rsh, corr = get_int_scale(
        scale_A, scale_B, scale_M, zero_point_A, zero_point_B, zero_point_M, "qsub"
    )

    return te.compute(
        output_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    (tensor_A[n * n_a, h * h_a, w * w_a, c * c_a] * scale_a)
                    - (tensor_B[n * n_b, h * h_b, w * w_b, c * c_b] * scale_b)
                    + corr
                )
                >> rsh
            ),
            dtype,
        ).astype(dtype),
    )


def qmultiply_broadcast_compute(
    tensor_A,
    tensor_B,
    output_shape,
    zero_point_A,
    scale_A,
    zero_point_B,
    scale_B,
    zero_point_M,
    scale_M,
    dtype,
):
    """Compute quantized multiply with broadcasting"""
    A_broadcast, B_broadcast = broadcast_axis(tensor_A, tensor_B)
    n_a, h_a, w_a, c_a = A_broadcast
    n_b, h_b, w_b, c_b = B_broadcast

    scale_int, rsh, corr = get_int_scale(
        scale_A, scale_B, scale_M, zero_point_A, zero_point_B, zero_point_M, "qmul"
    )

    return te.compute(
        output_shape,
        lambda n, h, w, c: saturate(
            (
                (
                    scale_int
                    * (tensor_A[n * n_a, h * h_a, w * w_a, c * c_a] - zero_point_A)
                    * (tensor_B[n * n_b, h * h_b, w * w_b, c * c_b] - zero_point_B)
                    + corr
                )
                >> rsh
            ),
            dtype,
        ).astype(dtype),
    )


def tir_schedule_quant(
    out_M,
    tensor_A,
    tensor_B,
    output_layout: str,
    tensor_A_layout: str,
    tensor_B_layout: str,
):
    """Schedule for output layout nhwc-8h8w32c-2d"""
    func = te.create_prim_func([tensor_A, tensor_B, out_M])

    s = tir.Schedule(func)

    block = s.get_block("compute")

    if tensor_A_layout == "nhwc-8h8w32c-2d":
        tensor_A_transformed_layout = get_layout_transform_fn(tensor_A_layout)
        s.transform_layout(block, buffer=tensor_A.name, index_map=tensor_A_transformed_layout)

    if tensor_B_layout == "nhwc-8h8w32c-2d":
        tensor_B_transformed_layout = get_layout_transform_fn(tensor_B_layout)
        s.transform_layout(block, buffer=tensor_B.name, index_map=tensor_B_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=out_M.name, index_map=output_transformed_layout)

    n, h, w, c = s.get_loops(block)

    h_o, h_i = s.split(h, [None, 8])
    w_o, w_i = s.split(w, [None, 8])
    c_o, c_i = s.split(c, [None, 32])
    wio, wii = s.split(w_i, [None, 4])

    s.reorder(n, h_o, w_o, c_o, h_i, wio, wii, c_i)

    return s
