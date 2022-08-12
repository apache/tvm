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

import tvm
from tvm import te
from tvm import tir
from tvm import topi
from ..utils import get_layout_transform_fn
import struct


def broadcast_axis(tensor_A, tensor_B):
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


def flt_getexp(a):
    a_f = struct.pack("f", a)
    a_i = struct.unpack("I", a_f)
    exp_value = ((a_i[0] >> 23) & 0xFF) - 127
    return exp_value


# this function will return x * 2 ^ y
def flt_ldexp(x, y):
    a = ((y + 127) & 0xFF) << 23
    a_i = struct.pack("I", a)
    a_f = struct.unpack("f", a_i)
    return (x * a_f[0]).astype("float32")


def saturate_16(x):
    return max(-32768, min(x, 32767))


def saturate(x, dtype):
    if dtype == "uint8":
        return te.max(0, te.min(x, 255))
    elif dtype == "int8":
        return te.max(-127, te.min(x, 128))
    return x


def get_int_scale(scale_A, scale_B, scale_M, zero_point_A, zero_point_B, zero_point_M, op):
    C_recip = 1 // scale_M

    if op == "qmul":
        scale = scale_A * scale_B * C_recip
        exp = flt_getexp(scale)
        rsh = 14 - exp
        scale_int = int(saturate_16(round(flt_ldexp(scale, rsh))))
        corr = zero_point_M << rsh

        return scale_int, rsh, corr
    else:
        a_scale_f = scale_A * C_recip
        b_scale_f = scale_B * C_recip

        exp_a = flt_getexp(a_scale_f)
        exp_b = flt_getexp(b_scale_f)

        rsh = 14 - max(exp_a, exp_b)

        scale_a = int(saturate_16(round(flt_ldexp(a_scale_f, rsh))))
        scale_b = int(saturate_16(round(flt_ldexp(b_scale_f, rsh))))

        if op == "qadd":
            corr = (zero_point_M << rsh) - (zero_point_A * scale_a + zero_point_B * scale_b)
        else:
            corr = (zero_point_M << rsh) - (zero_point_A * scale_a - zero_point_B * scale_b)

        return scale_a, scale_b, rsh, corr


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
):
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
            "uint8",
        ).astype("uint8"),
    )

    """
    
    return te.compute(
        output_shape,
        lambda n, h, w, c: (((scale_A/scale_M) * (tensor_A[n * n_a, h * h_a, w * w_a, c * c_a] - zero_point_A))
        + ((scale_B/scale_M) * (tensor_B[n * n_b, h * h_b, w * w_b, c * c_b] - zero_point_B))
        + zero_point_M).astype('uint8'),
    )
    """


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
):
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
            "uint8",
        ).astype("uint8"),
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
):
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
            "uint8",
        ).astype("uint8"),
    )


def tir_schedule(
    out_m,
    tensor_A,
    tensor_B,
    output_layout: str,
    tensor_A_layout: str,
    tensor_B_layout: str,
    op_name: str,
):
    """Schedule for input and output layout nhwc-8h8w32c-2d"""
    func = te.create_prim_func([tensor_A, tensor_B, out_m])

    print(func)

    s = tir.Schedule(func)

    block = s.get_block("compute")

    if tensor_A_layout == "nhwc-8h8w32c-2d":
        tensor_A_transformed_layout = get_layout_transform_fn(tensor_A_layout)
        s.transform_layout(block, buffer=("read", 0), index_map=tensor_A_transformed_layout)

    if tensor_B_layout == "nhwc-8h8w32c-2d":
        tensor_B_transformed_layout = get_layout_transform_fn(tensor_B_layout)
        s.transform_layout(block, buffer=("read", 1), index_map=tensor_B_transformed_layout)

    output_transformed_layout = get_layout_transform_fn(output_layout)
    s.transform_layout(block, buffer=("write", 0), index_map=output_transformed_layout)

    return s
