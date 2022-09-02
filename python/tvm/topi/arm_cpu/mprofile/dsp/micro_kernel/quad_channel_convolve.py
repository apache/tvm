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
"""This is a special intrinsic used for depthwise convolution using Cortex-M DSP instructions
(v7e-m). It takes as inputs an int8 HWC data tensor and an int8 CHWc kernel. This intrinsic "lays"
the kernel on top of the data tensors starting from a given pointer, performs signed sixteen-bit
multiplies on each pair of values, and sums all the products in an int32 accumlator. This process is
repeated four times giving four int32 outputs - one per channel."""

import textwrap

from tvm import te, tir


def intrin_quad_channel_convolve(tensor_w, channels, kernel_h, kernel_w, suffix):
    """Defines a v7e-m DSP-accelerated four-channel convolution."""
    data_slice = te.placeholder((kernel_h, kernel_w, 4), name="a", dtype="int8")

    if kernel_h * kernel_w % 2 == 1:
        kernel_length = kernel_h * kernel_w + 1
    else:
        kernel_length = kernel_h * kernel_w
    kernel_slice = te.placeholder((kernel_length, 4), name="b", dtype="int8")

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")

    output_slice = te.compute(
        (4,),
        lambda k: te.sum(
            data_slice[kh_i, kw_i, k].astype("int32")
            * kernel_slice[
                (2 * ((3 * kh_i + kw_i) // 2)) + ((k % 4) // 2),
                (2 * ((kh_i + kw_i) % 2)) + (k % 2),
            ].astype("int32"),
            axis=(kh_i, kw_i),
        ),
        name="c",
    )

    data_buf = tir.decl_buffer(
        data_slice.shape,
        data_slice.dtype,
        name="data",
        offset_factor=1,
        strides=[tensor_w * channels, channels, 1],
    )
    kernel_buf = tir.decl_buffer(
        kernel_slice.shape, kernel_slice.dtype, name="kernel", offset_factor=1, strides=[4, 1]
    )
    output_buf = tir.decl_buffer(
        output_slice.shape, output_slice.dtype, name="output", offset_factor=1, strides=[1]
    )

    def intrin_func(ins, outs):
        builder = tir.ir_builder.create()
        builder.emit(
            tir.call_extern(
                "int32",
                f"kernel_convolve_w{tensor_w}_c{channels}_kh{kernel_h}_kw{kernel_w}_{suffix}",
                outs[0].access_ptr("w"),
                ins[0].access_ptr("r"),
                ins[1].access_ptr("r"),
            )
        )
        return builder.get()

    return te.decl_tensor_intrin(
        output_slice.op,
        intrin_func,
        binds={data_slice: data_buf, kernel_slice: kernel_buf, output_slice: output_buf},
    )


def quad_channel_convolve_impl(tensor_w, channels, kernel_h, kernel_w, suffix):
    """Emits C code for quad_channel_convolve. Note that while intrin_quad_channel_convolve supports
    any kernel size, this function only supports 3x3 kernels (this could be fixed with work)."""
    assert kernel_h == kernel_w == 3

    return textwrap.dedent(
        (
            f"""
        #include <stdint.h>
        #include <arm_nnsupportfunctions.h>

        // __SXTB16(_ROR(X, Y)) is combined into one assembly instruction

        #define TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP( \
            arranged_kernel, \
            tensor_v0_c3210, tensor_v1_c3210, \
            sum0, sum1, sum2, sum3) {{ \
          \
          uint32_t tensor_v0_c20 = __SXTB16(tensor_v0_c3210); \
          uint32_t tensor_v0_c31 = __SXTB16(__ROR(tensor_v0_c3210, 8)); \
          uint32_t tensor_v1_c20 = __SXTB16(tensor_v1_c3210); \
          uint32_t tensor_v1_c31 = __SXTB16(__ROR(tensor_v1_c3210, 8)); \
          \
          uint32_t kernel_v1c1_v1c0_v0c1_v0c0 = *arranged_kernel++; \
          uint32_t kernel_v1c3_v1c2_v0c3_v0c2 = *arranged_kernel++; \
          \
          uint32_t kernel_v10_c0 = __SXTB16(kernel_v1c1_v1c0_v0c1_v0c0); \
          uint32_t kernel_v10_c1 = __SXTB16(__ROR(kernel_v1c1_v1c0_v0c1_v0c0, 8)); \
          uint32_t kernel_v10_c2 = __SXTB16(kernel_v1c3_v1c2_v0c3_v0c2); \
          uint32_t kernel_v10_c3 = __SXTB16(__ROR(kernel_v1c3_v1c2_v0c3_v0c2, 8)); \
          \
          uint32_t tensor_v10_c0 = __PKHBT(tensor_v0_c20, tensor_v1_c20, 16); \
          uint32_t tensor_v10_c1 = __PKHBT(tensor_v0_c31, tensor_v1_c31, 16); \
          uint32_t tensor_v10_c2 = __PKHTB(tensor_v1_c20, tensor_v0_c20, 16); \
          uint32_t tensor_v10_c3 = __PKHTB(tensor_v1_c31, tensor_v0_c31, 16); \
          \
          sum_c0 = __SMLAD(tensor_v10_c0, kernel_v10_c0, sum_c0); \
          sum_c1 = __SMLAD(tensor_v10_c1, kernel_v10_c1, sum_c1); \
          sum_c2 = __SMLAD(tensor_v10_c2, kernel_v10_c2, sum_c2); \
          sum_c3 = __SMLAD(tensor_v10_c3, kernel_v10_c3, sum_c3); \
        }}

        /* We do four channels at once to get this speed boost. */
        #ifdef __cplusplus
        extern "C"
        #endif
        int32_t kernel_convolve_w{tensor_w}_c{channels}_kh{kernel_h}_kw{kernel_w}_{suffix}(
            uint32_t *out,
            uint32_t *tensor,
            uint32_t *packed_kernel) {{

          uint32_t sum_c0 = 0;
          uint32_t sum_c1 = 0;
          uint32_t sum_c2 = 0;
          uint32_t sum_c3 = 0;

          TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP(
            packed_kernel,
            *tensor,
            *(tensor + {channels // 4}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP(
            packed_kernel,
            *(tensor + {(2) * channels // 4}),
            *(tensor + {tensor_w * (channels // 4)}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP(
            packed_kernel,
            *(tensor + {(tensor_w + 1) * (channels // 4)}),
            *(tensor + {(tensor_w + 2) * (channels // 4)}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP(
            packed_kernel,
            *(tensor + {(2 * tensor_w) * (channels // 4)}),
            *(tensor + {(2 * tensor_w + 1) * (channels // 4)}),
            sum_c0, sum_c1, sum_c2, sum_c3)
          TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP(
            packed_kernel,
            *(tensor + {(2 * tensor_w + 2) * (channels // 4)}),
            0,
            sum_c0, sum_c1, sum_c2, sum_c3)

          out[0] = sum_c0;
          out[1] = sum_c1;
          out[2] = sum_c2;
          out[3] = sum_c3;
          return 0;
        }}

        #undef TVMGEN_QUAD_CHANNEL_REARRANGE_SUM_DSP
        """
        )
    )
