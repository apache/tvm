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

import math

from tvm.script import tir as T


def get_conv_uint8_hvx_intrin(input_shape, kernel_shape, a_offset, w_offset, mem_scope):
    VRMPY_WIDTH = 128

    batches, input_size, _, in_c = input_shape
    w_size, _, _, filters = kernel_shape
    out_size = input_size

    input_padding = w_size // 2

    # For this usage of vrmpy it loads 4 bytes for vv from the kernel. In order
    # for this implementation to not mix output data there will need to be kernel
    # padding to round to the nearest multiple of 4.
    kernel_width_padding = 4 - w_size % 4
    padded_kernel_width = w_size + kernel_width_padding

    # vrmpy buffer loads are always 128B and will go out of bounds for the
    # implementation written here if there is not sufficient padding. This
    # means that for this implementation it must always be a multiple of 128
    # and have the standard padding and the padding needed for the kernel
    # window (4)
    if input_size % VRMPY_WIDTH != 0:
        input_width_padding = (
            (VRMPY_WIDTH - (input_size) % VRMPY_WIDTH) + input_padding + kernel_width_padding
        )
    else:
        input_width_padding = input_padding + kernel_width_padding

    padded_input_height = input_size + 2 * input_padding
    padded_input_width = input_size + input_padding + input_width_padding

    # vrmpy output buffer loads will go out of bounds for this implementation
    # if there is not proper padding.
    padded_output_width = VRMPY_WIDTH * (padded_input_width // VRMPY_WIDTH) + 3

    # The number of vrmpy loads (128B) needed to complete a horizontal frame of the input.
    w_steps = math.ceil(input_size / VRMPY_WIDTH)

    # The number of vrmpy loads (4B) needed to complete a horizontal frame of the kernel.
    kw_steps = math.ceil(w_size / 4)

    @T.prim_func
    def conv2d_vrmpy(a: T.handle, w: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A_local = T.match_buffer(
            a,
            [
                T.cast(padded_input_height, dtype="int32")
                * T.cast(padded_input_width, dtype="int32")
            ],
            dtype="uint8",
            offset_factor=1,
            scope=mem_scope,
        )
        W_local = T.match_buffer(
            w,
            [T.cast(w_size, dtype="int32") * T.cast(padded_kernel_width, dtype="int32")],
            dtype="uint8",
            offset_factor=1,
            scope=mem_scope,
        )
        C_local = T.match_buffer(
            c,
            [T.cast(out_size, dtype="int32") * T.cast(padded_output_width, dtype="int32")],
            dtype="int32",
            offset_factor=1,
            scope=mem_scope,
        )
        with T.block("root"):
            T.reads(
                A_local[
                    0 : T.cast(padded_input_height, dtype="int32")
                    * T.cast(padded_input_width, dtype="int32")
                ],
                W_local[
                    0 : T.cast(w_size, dtype="int32") * T.cast(padded_kernel_width, dtype="int32")
                ],
            )
            T.writes(
                C_local[
                    0 : T.cast(out_size, dtype="int32") * T.cast(padded_output_width, dtype="int32")
                ]
            )
            for y, x_o, x_i, rx_o, ry in T.grid(input_size, w_steps, 4, kw_steps, w_size):
                C_local[
                    T.ramp(y * T.cast(padded_output_width, dtype="int32") + x_o * 128 + x_i, 4, 32)
                ] += T.call_llvm_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.128B"),
                    T.uint32(2),
                    T.reinterpret(
                        A_local[
                            T.ramp(
                                (y + ry) * T.cast(padded_input_width, dtype="int32")
                                + x_o * 128
                                + 4 * rx_o
                                + x_i,
                                1,
                                128,
                            )
                        ],
                        dtype="int32x32",
                    ),
                    T.reinterpret(
                        W_local[
                            T.ramp(ry * T.cast(padded_kernel_width, dtype="int32") + rx_o * 4, 1, 4)
                        ],
                        dtype="int32",
                    ),
                    dtype="int32x32",
                )

    @T.prim_func
    def conv2d_vrmpy_desc(a: T.handle, w: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A_local = T.match_buffer(
            a,
            [padded_input_height, padded_input_width],
            dtype="uint8",
            offset_factor=1,
            scope=mem_scope,
        )
        W_local = T.match_buffer(
            w, [w_size, padded_kernel_width], dtype="uint8", offset_factor=1, scope=mem_scope
        )
        C_local = T.match_buffer(
            c, [out_size, padded_output_width], dtype="int32", offset_factor=1, scope=mem_scope
        )
        with T.block("root"):
            for y, x, ry, rx in T.grid(input_size, input_size, w_size, w_size):
                with T.block("C"):
                    y, x, ry, rx = T.axis.remap("SSRR", [y, x, ry, rx])
                    C_local[y, x] = C_local[y, x] + T.cast(
                        A_local[y + ry, x + rx], "int32"
                    ) * T.cast(W_local[ry, rx], "int32")

    @T.prim_func
    def operator(a: T.handle, w: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(
            a, [batches, input_size, input_size, in_c], dtype="uint8", offset_factor=1
        )
        W = T.match_buffer(w, [w_size, w_size, in_c, filters], dtype="uint8", offset_factor=1)
        C = T.match_buffer(
            c, [batches, out_size, out_size, filters], dtype="int32", offset_factor=1
        )
        A_local = T.alloc_buffer(
            [batches, padded_input_height, padded_input_width, in_c], dtype="uint8", scope=mem_scope
        )
        W_local = T.alloc_buffer(
            [w_size, padded_kernel_width, in_c, filters], dtype="uint8", scope=mem_scope
        )
        C_local = T.alloc_buffer(
            [batches, filters, out_size, padded_output_width], dtype="int32", scope=mem_scope
        )
        with T.block("root"):
            for n, y, x, c in T.grid(batches, padded_input_height, padded_input_width, in_c):
                with T.block("A_local"):
                    nn, yy, xx, cc = T.axis.remap("SSSS", [n, y, x, c])
                    T.reads(
                        A[
                            nn,
                            yy - T.cast(input_padding, dtype="int32"),
                            xx - T.cast(input_padding, dtype="int32"),
                            cc,
                        ]
                    )
                    T.writes(A_local[nn, yy, xx, cc])
                    A_local[nn, yy, xx, cc] = T.if_then_else(
                        T.cast(input_padding, dtype="int32") <= yy
                        and yy
                        < T.cast(padded_input_height, dtype="int32")
                        - T.cast(input_padding, dtype="int32")
                        and T.cast(input_padding, dtype="int32") <= xx
                        and xx
                        < T.cast(padded_input_width, dtype="int32")
                        - T.cast(input_width_padding, dtype="int32"),
                        A[
                            nn,
                            yy - T.cast(input_padding, dtype="int32"),
                            xx - T.cast(input_padding, dtype="int32"),
                            cc,
                        ]
                        - T.cast(a_offset, dtype="uint8"),
                        T.uint8(0),
                        dtype="uint8",
                    )
            for y, x, c, f in T.grid(w_size, padded_kernel_width, in_c, filters):
                with T.block("W_local"):
                    yy, xx, cc, ff = T.axis.remap("SSSS", [y, x, c, f])
                    T.reads(W[yy, xx, cc, ff])
                    T.writes(W_local[yy, xx, cc, ff])
                    W_local[yy, xx, cc, ff] = T.if_then_else(
                        xx
                        < T.cast(padded_kernel_width, dtype="int32")
                        - T.cast(kernel_width_padding, dtype="int32"),
                        W[yy, xx, cc, ff] - T.cast(w_offset, dtype="uint8"),
                        T.uint8(0),
                        dtype="uint8",
                    )
            for n, f, y, x in T.grid(batches, filters, out_size, padded_output_width):
                with T.block("C_local_init"):
                    n, f, y, x = T.axis.remap("SSSS", [n, f, y, x])
                    C_local[n, f, y, x] = 0
            for n, f, y, x, ry, rx, rc in T.grid(
                batches, filters, input_size, input_size, w_size, w_size, in_c
            ):
                with T.block("C"):
                    n, f, y, x, ry, rx, rc = T.axis.remap("SSSSRRR", [n, f, y, x, ry, rx, rc])
                    C_local[n, f, y, x] = C_local[n, f, y, x] + T.cast(
                        A_local[n, y + ry, x + rx, rc], "int32"
                    ) * T.cast(W_local[ry, rx, rc, f], "int32")
            for n, f, y, x in T.grid(batches, filters, out_size, out_size):
                with T.block("C_local"):
                    n, f, y, x = T.axis.remap("SSSS", [n, f, y, x])
                    C[n, y, x, f] = C_local[n, f, y, x]

    return conv2d_vrmpy_desc, conv2d_vrmpy, operator
