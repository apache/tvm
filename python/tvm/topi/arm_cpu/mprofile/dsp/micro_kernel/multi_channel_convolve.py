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
from .common import num_simd_lanes_per_word, common_includes


def _get_func_name(in_dtype, tensor_w, channels, kernel_h, kernel_w, suffix):
    """Gets the C function name of the tensorized function."""
    return f"kernel_convolve_{in_dtype}_w{tensor_w}_c{channels}_kh{kernel_h}_kw{kernel_w}_{suffix}"


def intrin_multi_channel_convolve(
    in_dtype, _tensor_h, tensor_w, channels, kernel_h, kernel_w, suffix
):
    """Defines a v7e-m DSP-accelerated multi-channel convolution. Works on two
    channels if in_dtype==int16, and four channels if in_dtype==int8."""
    simd_lanes = num_simd_lanes_per_word(in_dtype)

    overlap_dims = (kernel_h, kernel_w, simd_lanes)
    data_slice = te.placeholder(overlap_dims, name="data_slice", dtype=in_dtype)
    kernel_slice = te.placeholder(overlap_dims, name="kernel_slice", dtype=in_dtype)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")

    output_slice = te.compute(
        (simd_lanes,),
        lambda k: te.sum(
            data_slice[kh_i, kw_i, k].astype("int32") * kernel_slice[kh_i, kw_i, k].astype("int32"),
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
        kernel_slice.shape,
        kernel_slice.dtype,
        name="kernel",
        offset_factor=1,
        strides=[kernel_w * simd_lanes, simd_lanes, 1],
    )
    output_buf = tir.decl_buffer(
        output_slice.shape, output_slice.dtype, name="output", offset_factor=1, strides=[1]
    )

    def intrin_func(ins, outs):
        builder = tir.ir_builder.create()
        builder.emit(
            tir.call_extern(
                "int32",
                _get_func_name(in_dtype, tensor_w, channels, kernel_h, kernel_w, suffix),
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


def multi_channel_convolve_impl(in_dtype, *args) -> str:
    """Generates C code for a fast multi-channel convolution function for ARM Cortex-M. This is done
    by calling a sub-function depending on the input data type, as since v7e-m has no quad multiply
    accumulate instruction, the int8 and int16 cases work differently."""
    if in_dtype == "int8":
        return _quad_int8_channel_convolve_impl(*args)
    if in_dtype == "int16":
        return _dual_int16_channel_convolve_impl(*args)

    raise NotImplementedError(f"No Cortex-M {in_dtype} depthwise_conv2d implementation exists!")


def _quad_int8_channel_convolve_impl(_tensor_h, tensor_w, channels, kernel_h, kernel_w, suffix):
    return textwrap.dedent(
        (
            common_includes
            + f"""
        // __SXTB16(_ROR(X, Y)) is combined into one assembly instruction

        #define TVMGEN_QUAD_INT8_CHANNEL_REARRANGE_SUM_DSP( \
            arranged_kernel, \
            tensor_c3210, \
            sum_c0, sum_c1, sum_c2, sum_c3) {{ \
          \
          int32_t kernel_c3210 = *arranged_kernel++; \
          \
          int32_t tensor_c20 = __sxtb16(tensor_c3210); \
          int32_t kernel_c20 = __sxtb16(kernel_c3210); \
          sum_c0 = __builtin_arm_smlabb(tensor_c20, kernel_c20, sum_c0); \
          sum_c2 = __builtin_arm_smlatt(tensor_c20, kernel_c20, sum_c2); \
          \
          int32_t tensor_c31 = __sxtb16(__ror(tensor_c3210, 8)); \
          int32_t kernel_c31 = __sxtb16(__ror(kernel_c3210, 8)); \
          sum_c1 = __builtin_arm_smlabb(tensor_c31, kernel_c31, sum_c1); \
          sum_c3 = __builtin_arm_smlatt(tensor_c31, kernel_c31, sum_c3); \
        }}

        /* We do four channels at once to get this speed boost. */
        #ifdef __cplusplus
        extern "C"
        #endif
        int32_t {_get_func_name("int8", tensor_w, channels, kernel_h, kernel_w, suffix)}(
            int32_t *out,
            int8_t *tensor,
            int8_t *kernel) {{

          int32_t sum_c0 = 0;
          int32_t sum_c1 = 0;
          int32_t sum_c2 = 0;
          int32_t sum_c3 = 0;

          int32_t kernel_i32[{kernel_h} * {kernel_w}];
          memcpy(kernel_i32, kernel, {kernel_h} * {kernel_w} * sizeof(int32_t));
          int32_t *arranged_kernel = kernel_i32;

          int32_t tensor_length = {((kernel_w - 1) * (channels // 4) + (kernel_h - 1) * tensor_w * (channels // 4)) + 1};
          int32_t tensor_i32[tensor_length];
          memcpy(tensor_i32, tensor, tensor_length * sizeof(int32_t));

          #pragma GCC unroll 3
          for (int i = 0; i < {kernel_h}; i++) {{
            #pragma GCC unroll 3
            for (int j = 0; j < {kernel_w}; j++) {{
              TVMGEN_QUAD_INT8_CHANNEL_REARRANGE_SUM_DSP(
                arranged_kernel,
                *(tensor_i32 + j * {channels // 4} + i * {tensor_w * (channels // 4)}),
                sum_c0, sum_c1, sum_c2, sum_c3)
            }}
          }}

          out[0] = sum_c0;
          out[1] = sum_c1;
          out[2] = sum_c2;
          out[3] = sum_c3;
          return 0;
        }}

        #undef TVMGEN_QUAD_INT8_CHANNEL_REARRANGE_SUM_DSP
        """
        )
    )


def _dual_int16_channel_convolve_impl(_tensor_h, tensor_w, channels, kernel_h, kernel_w, suffix):
    return textwrap.dedent(
        (
            common_includes
            + f"""
        #include <stdint.h>

        /* We do four channels at once to get this speed boost. */
        #ifdef __cplusplus
        extern "C"
        #endif
        int32_t {_get_func_name("int16", tensor_w, channels, kernel_h, kernel_w, suffix)}(
            int32_t *out,
            int16_t *tensor,
            int16_t *kernel) {{

          int32_t sum_c0 = 0;
          int32_t sum_c1 = 0;

          int32_t kernel_i32[{kernel_h} * {kernel_w}];
          memcpy(kernel_i32, kernel, {kernel_h} * {kernel_w} * sizeof(int32_t));

          int32_t tensor_length = {((kernel_w - 1) * (channels // 2) + (kernel_h - 1) * tensor_w * (channels // 2)) + 1};
          int32_t tensor_i32[tensor_length];
          memcpy(tensor_i32, tensor, tensor_length * sizeof(int32_t));

          #pragma GCC unroll 3
          for (int i = 0; i < {kernel_h}; i++) {{
            #pragma GCC unroll 3
            for (int j = 0; j < {kernel_w}; j++) {{
              int32_t tensor_c10 = tensor_i32[j * {channels // 2} + i * {tensor_w * (channels // 2)}];
              int32_t kernel_c10 = kernel_i32[{kernel_w} * i + j];
              sum_c0 = __builtin_arm_smlabb(tensor_c10, kernel_c10, sum_c0);
              sum_c1 = __builtin_arm_smlatt(tensor_c10, kernel_c10, sum_c1);
            }}
          }}

          out[0] = sum_c0;
          out[1] = sum_c1;
          return 0;
        }}

        #undef TVMGEN_DUAL_INT16_CHANNEL_REARRANGE_SUM
        """
        )
    )
