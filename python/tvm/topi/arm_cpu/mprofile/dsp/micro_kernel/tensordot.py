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
"""Computes a "jumpy tensordot" operator, which can/should be used to
tensorize ANY aritrarily grouped conv2D, provided the data and kernel
layouts are the optimal ones. When groups=1, the optimal data layout
is NHWC and kernel layout is OHWI. When this is a depthwise convolution,
the optimal data layout is NCHW and kernel layout is OIHW."""

import textwrap

from tvm import te, tir

from .common import num_simd_lanes_per_word

def _get_func_name(in_dtype, tensor_h, jump, tensor_w, suffix):
    """Gets the C function name of the tensorized function."""
    return f"tensordot_{in_dtype}_h{tensor_h}_j{jump}_w{tensor_w}_{suffix}"


def make_intrin_tensordot(operator, binds, tensordot_params):
    #in_dtype, tensor_h, jump, tensor_w, suffix = tensordot_params

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
        operator,
        intrin_func,
        binds=binds,
    )


def intrin_depthwise_conv2d_tensordot(in_dtype, tensor_w, kernel_h, kernel_w, suffix):
    data_slice = te.placeholder((kernel_h, kernel_w), name="a", dtype=in_dtype)
    kernel_slice = te.placeholder((kernel_h, kernel_w), name="b", dtype=in_dtype)

    kh_i = te.reduce_axis((0, kernel_h), name="kh_i")
    kw_i = te.reduce_axis((0, kernel_w), name="kw_i")

    output_slice = te.compute((1,),
        lambda k: te.sum(
            data_slice[kh_i, kw_i].astype("int32")
            * kernel_slice[kh_i, kw_i].astype("int32"),
            axis=[kh_i, kw_i],
        ),
        name="c",
    )

    data_buf = tir.decl_buffer(
        data_slice.shape,
        data_slice.dtype,
        name="data",
        offset_factor=1,
        strides=[tensor_w, 1],
    )
    kernel_buf = tir.decl_buffer(
        kernel_slice.shape, kernel_slice.dtype, name="kernel", offset_factor=1, strides=[kernel_w, 1]
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
        binds=binings,
    )


def tensordot_impl(in_dtype, tensor_h, jump, tensor_w, suffix):
    assert in_dtype in ["int8", "int16", "int32"]
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
