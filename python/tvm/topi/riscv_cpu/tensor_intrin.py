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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D int8 schedule on RISCV"""

import tvm
from tvm import te
from tvm.contrib import clang


def dot_int8_int8_int32():
    int32_lanes = 4  # 4 int32 lanes = 128.
    num_int8_elements = 4  # 4 int8 elements in int32.
    data = te.placeholder((num_int8_elements,), dtype="uint8", name="data")
    kernel = te.placeholder((int32_lanes, num_int8_elements), dtype="int8", name="kernel")
    k = te.reduce_axis((0, num_int8_elements), name="k")
    C = te.compute(
        (int32_lanes,),
        lambda i: te.sum(data[k].astype("int32") * kernel[i, k].astype("int32"), axis=k),
        name="C",
    )

    a_buffer = tvm.tir.decl_buffer(
        data.shape, dtype="uint8", name="a_buffer", offset_factor=1, strides=[1]
    )
    b_buffer = tvm.tir.decl_buffer(
        kernel.shape, dtype="int8", name="b_buffer", offset_factor=1, strides=[te.var("ldw"), 1]
    )

    def _intrin_func(ins, outs):
        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    outs[0].dtype,
                    "dot_uint8_int8_int32_body",
                    ins[0].access_ptr("r"),
                    ins[1].access_ptr("r"),
                    outs[0].access_ptr("w"),
                )
            )
            return ib.get()

        def _reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    outs[0].dtype, "dot_uint8_int8_int32_reset", outs[0].access_ptr("w")
                )
            )
            return ib.get()

        def _reduce_update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    outs[0].dtype,
                    "dot_uint8_int8_int32_update",
                    ins[0].access_ptr("r"),
                    ins[1].access_ptr("r"),
                    outs[0].access_ptr("w"),
                )
            )
            return ib.get()

        return _body(), _reset(), _reduce_update()

    buffer_params = {"offset_factor": 1}
    return te.decl_tensor_intrin(
        C.op,
        _intrin_func,
        binds={data: a_buffer, kernel: b_buffer},
        default_buffer_params=buffer_params,
    )


def int8_conv2d_impl():
    """Emit C or IR code for conv2d impl."""
    cc_code = f"""
#ifndef TVM_RISCV_CONV2D_INT8
#define TVM_RISCV_CONV2D_INT8
#include <riscv_vector.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t dot_uint8_int8_int32_reset(int32_t* res) {{
  // In this function, we set all values in the output array to 0.
  for (uint32_t i = 0; i < 4; ++i)
    res[i] = 0;
  return 0;
}}

// In this function we need to multiply two vectors, then in the resulting vector calculate the sum of
// all its elements and store it to output array.
//
// Example without vectorization:
// for (unsigned i = 0; i < 4; ++i) {{
//   output[i] = 0;
//   for (unsigned j = 0; j < 4; ++j) {{
//     output[i] += data[k] * kernel[i][j];
//   }}
// }}

#ifdef __cplusplus
extern "C"
#endif
int32_t dot_uint8_int8_int32_body(uint8_t* data, int8_t* kernel, int32_t* output) {{
  // Load values from data into a vector.
  vuint8mf2_t v_data = vle8_v_u8mf2(data, -1);

  // Dummy vector for operations.
  vint32m1_t empty;

  // Load values from kernel[i][*],
  // then we multiply two vectors with type extension:
  // v_mul_[i] = v_kernel_[i] * v_data,
  // then we count the sum in resulting vector:
  // v_sum_[i] = sum(v_mul_[i])

  vint8mf2_t v_kernel_1 = vle8_v_i8mf2(&kernel[0], -1);
  vint16m1_t v_mul_1 = vwmulsu_vv_i16m1(v_kernel_1, v_data, -1);
  vint32m1_t v_sum_1 = vwredsum(empty, v_mul_1, empty, 4);

  vint8mf2_t v_kernel_2 = vle8_v_i8mf2(&kernel[4], -1);
  vint16m1_t v_mul_2 = vwmulsu_vv_i16m1(v_kernel_2, v_data, -1);
  vint32m1_t v_sum_2 = vwredsum(empty, v_mul_2, empty, 4);

  vint8mf2_t v_kernel_3 = vle8_v_i8mf2(&kernel[8], -1);
  vint16m1_t v_mul_3 = vwmulsu_vv_i16m1(v_kernel_3, v_data, -1);
  vint32m1_t v_sum_3 = vwredsum(empty, v_mul_3, empty, 4);

  vint8mf2_t v_kernel_4 = vle8_v_i8mf2(&kernel[12], -1);
  vint16m1_t v_mul_4 = vwmulsu_vv_i16m1(v_kernel_4, v_data, -1);
  vint32m1_t v_sum_4 = vwredsum(empty, v_mul_4, empty, 4);

  // Save new values to output.
  output[0] = vmv_x_s_i32m1_i32(v_sum_1);
  output[1] = vmv_x_s_i32m1_i32(v_sum_2);
  output[2] = vmv_x_s_i32m1_i32(v_sum_3);
  output[3] = vmv_x_s_i32m1_i32(v_sum_4);

  return 0;
}}

// In this function we need to multiply two vectors, then in the resulting vector calculate the sum of
// all its elements and add it to the value from output.
//
// Example without vectorization:
// for (unsigned i = 0; i < 4; ++i)
//   for (unsigned j = 0; j < 4; ++j)
//     output[i] += data[k] * kernel[i][j];

#ifdef __cplusplus
extern "C"
#endif
int32_t dot_uint8_int8_int32_update(uint8_t* data, int8_t* kernel, int32_t* output) {{
  // Load values from data into a vector.
  vuint8mf2_t v_data = vle8_v_u8mf2(data, -1);

  // Dummy vector for operations.
  vint32m1_t empty;

  // Load values from output into vectors.
  vint32m1_t v_output_1 = vle32_v_i32m1(&output[0], -1);
  vint32m1_t v_output_2 = vle32_v_i32m1(&output[1], -1);
  vint32m1_t v_output_3 = vle32_v_i32m1(&output[2], -1);
  vint32m1_t v_output_4 = vle32_v_i32m1(&output[3], -1);

  // Load values from kernel[i][*],
  // then we multiply two vectors with type extension:
  // v_mul_[i] = v_kernel_[i] * v_data,
  // then we count the sum in resulting vector and add value from output:
  // v_sum_[i] = v_output_[i] + sum(v_mul_[i])

  vint8mf2_t v_kernel_1 = vle8_v_i8mf2(&kernel[0], -1);
  vint16m1_t v_mul_1 = vwmulsu_vv_i16m1(v_kernel_1, v_data, -1);
  vint32m1_t v_sum_1 = vwredsum(empty, v_mul_1, v_output_1, 4);

  vint8mf2_t v_kernel_2 = vle8_v_i8mf2(&kernel[4], -1);
  vint16m1_t v_mul_2 = vwmulsu_vv_i16m1(v_kernel_2, v_data, -1);
  vint32m1_t v_sum_2 = vwredsum(empty, v_mul_2, v_output_2, 4);

  vint8mf2_t v_kernel_3 = vle8_v_i8mf2(&kernel[8], -1);
  vint16m1_t v_mul_3 = vwmulsu_vv_i16m1(v_kernel_3, v_data, -1);
  vint32m1_t v_sum_3 = vwredsum(empty, v_mul_3, v_output_3, 4);

  vint8mf2_t v_kernel_4 = vle8_v_i8mf2(&kernel[12], -1);
  vint16m1_t v_mul_4 = vwmulsu_vv_i16m1(v_kernel_4, v_data, -1);
  vint32m1_t v_sum_4 = vwredsum(empty, v_mul_4, v_output_4, 4);

  // Save updated values to output.
  output[0] = vmv_x_s_i32m1_i32(v_sum_1);
  output[1] = vmv_x_s_i32m1_i32(v_sum_2);
  output[2] = vmv_x_s_i32m1_i32(v_sum_3);
  output[3] = vmv_x_s_i32m1_i32(v_sum_4);

  return 0;
}}
#endif
    """
    return cc_code
