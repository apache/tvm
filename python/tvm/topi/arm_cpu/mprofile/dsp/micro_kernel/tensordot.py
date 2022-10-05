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
"""Computes a "jumpy tensordot" operator, which can be used to tensorize many common operators
including regular conv2d, depthwise conv2d, and grouped conv2d provided the data and kernel layouts
are the optimal ones. When groups=1, the optimal data layout is NHWC and kernel layout is OHWI. When
this is a depthwise convolution, the optimal data layout is NCHW and kernel layout is OIHW."""

import textwrap

from tvm import te, tir

from .common import num_simd_lanes_per_word


def _get_func_name(in_dtype, tensor_h, jump, tensor_w, suffix):
    """Gets the C function name of the tensordot function."""
    return f"tensordot_{in_dtype}_h{tensor_h}_j{jump}_w{tensor_w}_{suffix}"


def make_intrin_tensordot(slices, strides, tensordot_params):
    """Helper function for constructing tensordot intrinsic. We can't construct the whole thing here
    (as multiple schedules use tensordot and each must build the intrinstic differently) but we can
    build part here to simplify the code."""

    # in_dtype, tensor_h, jump, tensor_w, suffix = tensordot_params
    data, kernel, output = slices
    data_strides, kernel_strides = strides

    data_buf = tir.decl_buffer(
        data.shape, data.dtype, name="data", offset_factor=1, strides=data_strides
    )
    kernel_buf = tir.decl_buffer(
        kernel.shape,
        kernel.dtype,
        name="kernel",
        offset_factor=1,
        strides=kernel_strides,
    )
    output_buf = tir.decl_buffer(
        output.shape, output.dtype, name="output", offset_factor=1, strides=[1]
    )

    def intrin_func(ins, outs):
        builder = tir.ir_builder.create()
        builder.emit(
            tir.call_extern(
                "int32",
                _get_func_name(*tensordot_params),
                outs[0].access_ptr("w"),
                ins[0].access_ptr("r"),
                ins[1].access_ptr("r"),
            )
        )
        return builder.get()

    return te.decl_tensor_intrin(
        output.op,
        intrin_func,
        binds={data: data_buf, kernel: kernel_buf, output: output_buf},
    )


def tensordot_impl(in_dtype: str, tensor_h: int, jump: int, tensor_w: int, suffix: str) -> str:
    """Generates C code for taking the dot products of two `tensor_h` * `tensor_w` tensors. Also has
    a `jump` argument that advances the pointer of one tensor by that many words after each row. The
    `jump` and `tensor_w` values must be word-aligned for the input data type, as non-word-aligned
    memory access is slow on the Cortex-M series. Depending on the input datatype, the code may
    contain DSP instructions for Arm v7e-m. C code contains DSP instructions for Arm v7e-m. See
    the below pseudocode for reference:

    tensordot(out_ptr, dat_ptr, ker_ptr) {
        sum = 0;
        for (i = 0; i < tensor_h; i++) {
            for (j = 0; j < tensor_w; j++) {
                sum += (*dat_ptr++) * (*ker_ptr++);
            }
            dat_ptr += jump;
        }
        *out_ptr = sum;
    }
    """

    simd_lanes = num_simd_lanes_per_word(in_dtype)
    assert tensor_w % simd_lanes == 0
    assert jump % simd_lanes == 0

    if in_dtype == "int8":
        inner_loop = """
              uint32_t tensor_c20 = __SXTB16(tensor_batch);
              uint32_t kernel_c20 = __SXTB16(kernel_batch);
              sum = __SMLAD(tensor_c20, kernel_c20, sum);

              uint32_t tensor_c31 = __SXTB16(__ROR(tensor_batch, 8));
              uint32_t kernel_c31 = __SXTB16(__ROR(kernel_batch, 8));
              sum = __SMLAD(tensor_c31, kernel_c31, sum);"""

    elif in_dtype == "int16":
        inner_loop = """
              sum = __SMLAD(tensor_batch, kernel_batch, sum);"""

    elif in_dtype == "int32":
        inner_loop = """
              // Compiles to a single MAC instruction
              sum += tensor_batch * kernel_batch;"""

    else:
        raise ValueError(f"No tensordot implementation exists for dtype '{in_dtype}'!")

    function_name = _get_func_name(in_dtype, tensor_h, jump, tensor_w, suffix)
    return textwrap.dedent(
        (
            f"""
        #include <stdint.h>
        #include <arm_nnsupportfunctions.h>

        #ifdef __cplusplus
        extern "C"
        #endif
        __STATIC_FORCEINLINE int32_t {function_name}(
            uint32_t *out,
            uint32_t *tensor,
            uint32_t *kernel) {{

          uint32_t sum = 0;

          #pragma GCC unroll {tensor_h}
          for (int i = 0; i < {tensor_h}; i++) {{
            #pragma GCC unroll {tensor_w // simd_lanes}
            for (int j = 0; j < {tensor_w // simd_lanes}; j++) {{
              uint32_t tensor_batch = *tensor++;
              uint32_t kernel_batch = *kernel++;
              {inner_loop.strip()}
            }}
            tensor += {jump // simd_lanes};
          }}
          out[0] = sum;
          return 0;
        }}
        """
        )
    )
