/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/runtime/micro/utvm_device_lib.h"
extern void* __tvm_module_ctx = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_3( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 3))) {
    TVMAPISetLastError("fused_nn_conv2d_3: num_args should be 3");
    return -1;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (8 == ((int32_t)arg0_strides[2]))) && (64 == ((int32_t)arg0_strides[1]))) && (16384 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -2;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (1 == ((int32_t)arg1_strides[2]))) && (1 == ((int32_t)arg1_strides[1]))) && (256 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -3;
    }
  }
  float* output_unpack = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (4 == ((int32_t)arg2_strides[2]))) && (16 == ((int32_t)arg2_strides[1]))) && (8192 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -4;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_3: Expect arg[0] to be pointer");
    return -5;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_3: Expect arg[1] to be pointer");
    return -6;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_3: Expect arg[2] to be pointer");
    return -7;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -8;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -9;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -10;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -11;
  }
  if (!((((int32_t)arg0_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -12;
  }
  if (!((((int32_t)arg0_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -13;
  }
  if (!((((int32_t)arg0_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -14;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -15;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -16;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -17;
  }
  if (!((((int32_t)arg1_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -18;
  }
  if (!((((int32_t)arg1_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -19;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -20;
  }
  if (!((((int32_t)arg1_shape[3]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -21;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -22;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -23;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -24;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -25;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -26;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -27;
  }
  if (!((((int32_t)arg2_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -28;
  }
  if (!((((int32_t)arg2_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -29;
  }
  if (!((((int32_t)arg2_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -30;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -31;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -32;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -33;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)50176, 2, 32);
  if (data_vec == NULL) {
    return -34;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)524288, 2, 32);
  if (kernel_vec == NULL) {
    return -35;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 224; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 7; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 7) + w)] = placeholder[(((((((C_h_fused / 7) * 8) + c) * 8) + (C_h_fused % 7)) * 8) + w)];
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 64; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 32; ++CI) {
      for (int32_t ci = 0; ci < 8; ++ci) {
        for (int32_t co = 0; co < 8; ++co) {
          (( float*)kernel_vec)[((((((CO_h_fused * 32) + CI) * 8) + ci) * 8) + co)] = placeholder1[((((((CO_h_fused * 8) + co) * 32) + CI) * 8) + ci)];
        }
      }
    }
  }
  for (int32_t c_outer_h_outer_fused = 0; c_outer_h_outer_fused < 64; ++c_outer_h_outer_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 32; ++ic_outer) {
      for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
        for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
          conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((ic_outer * 56) + ic_inner) * 7)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c)]));
        }
        for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
          conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 2)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c1)]));
        }
        for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
          conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 4)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c2)]));
        }
        for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
          conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 6)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c3)]));
        }
        for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
          conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 112)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c4)]));
        }
        for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
          conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 114)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c5)]));
        }
        for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
          conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 116)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c6)]));
        }
        for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
          conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 118)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c7)]));
        }
        for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
          conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 224)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c8)]));
        }
        for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
          conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 226)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c9)]));
        }
        for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
          conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 228)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c10)]));
        }
        for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
          conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 230)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c11)]));
        }
        for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
          conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 336)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c12)]));
        }
        for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
          conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 338)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c13)]));
        }
        for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
          conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 340)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c14)]));
        }
        for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
          conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((ic_outer * 56) + ic_inner) * 7) + 342)] * (( float*)kernel_vec)[((((((c_outer_h_outer_fused * 32) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c15)]));
        }
      }
    }
    for (int32_t h_inner = 0; h_inner < 4; ++h_inner) {
      for (int32_t w_inner = 0; w_inner < 4; ++w_inner) {
        for (int32_t c_inner = 0; c_inner < 8; ++c_inner) {
          output_unpack[((((((c_outer_h_outer_fused * 8) + c_inner) * 4) + h_inner) * 4) + w_inner)] = conv_global[((((h_inner * 4) + w_inner) * 8) + c_inner)];
        }
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -36;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -37;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_2( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 3))) {
    TVMAPISetLastError("fused_nn_conv2d_2: num_args should be 3");
    return -38;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (16 == ((int32_t)arg0_strides[2]))) && (256 == ((int32_t)arg0_strides[1]))) && (32768 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -39;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (1 == ((int32_t)arg1_strides[2]))) && (1 == ((int32_t)arg1_strides[1]))) && (128 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -40;
    }
  }
  float* output_unpack = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (8 == ((int32_t)arg2_strides[2]))) && (64 == ((int32_t)arg2_strides[1]))) && (16384 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -41;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_2: Expect arg[0] to be pointer");
    return -42;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_2: Expect arg[1] to be pointer");
    return -43;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_2: Expect arg[2] to be pointer");
    return -44;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -45;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -46;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -47;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -48;
  }
  if (!((((int32_t)arg0_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -49;
  }
  if (!((((int32_t)arg0_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -50;
  }
  if (!((((int32_t)arg0_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -51;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -52;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -53;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -54;
  }
  if (!((((int32_t)arg1_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -55;
  }
  if (!((((int32_t)arg1_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -56;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -57;
  }
  if (!((((int32_t)arg1_shape[3]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -58;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -59;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -60;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -61;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -62;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -63;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -64;
  }
  if (!((((int32_t)arg2_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -65;
  }
  if (!((((int32_t)arg2_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -66;
  }
  if (!((((int32_t)arg2_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -67;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -68;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -69;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -70;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)115200, 2, 32);
  if (data_vec == NULL) {
    return -71;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)131072, 2, 32);
  if (kernel_vec == NULL) {
    return -72;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 240; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 15; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 15) + w)] = placeholder[(((((((C_h_fused / 15) * 8) + c) * 16) + (C_h_fused % 15)) * 16) + w)];
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 32; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 16; ++CI) {
      for (int32_t ci = 0; ci < 8; ++ci) {
        for (int32_t co = 0; co < 8; ++co) {
          (( float*)kernel_vec)[((((((CO_h_fused * 16) + CI) * 8) + ci) * 8) + co)] = placeholder1[((((((CO_h_fused * 8) + co) * 16) + CI) * 8) + ci)];
        }
      }
    }
  }
  for (int32_t c_outer_h_outer_fused = 0; c_outer_h_outer_fused < 128; ++c_outer_h_outer_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
      for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
        for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
          conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15))] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c)]));
        }
        for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
          conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 2)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c1)]));
        }
        for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
          conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 4)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c2)]));
        }
        for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
          conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 6)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c3)]));
        }
        for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
          conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 8)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c4)]));
        }
        for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
          conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 10)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c5)]));
        }
        for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
          conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 12)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c6)]));
        }
        for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
          conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 14)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c7)]));
        }
        for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
          conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 240)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c8)]));
        }
        for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
          conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 242)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c9)]));
        }
        for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
          conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 244)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c10)]));
        }
        for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
          conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 246)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c11)]));
        }
        for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
          conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 248)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c12)]));
        }
        for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
          conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 250)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c13)]));
        }
        for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
          conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 252)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c14)]));
        }
        for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
          conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((ic_outer * 1800) + ((c_outer_h_outer_fused % 4) * 480)) + (ic_inner * 15)) + 254)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 4) * 16) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c15)]));
        }
      }
    }
    for (int32_t h_inner = 0; h_inner < 2; ++h_inner) {
      for (int32_t w_inner = 0; w_inner < 8; ++w_inner) {
        for (int32_t c_inner = 0; c_inner < 8; ++c_inner) {
          output_unpack[(((((((((c_outer_h_outer_fused / 4) * 8) + c_inner) * 4) + (c_outer_h_outer_fused % 4)) * 2) + h_inner) * 8) + w_inner)] = conv_global[((((h_inner * 8) + w_inner) * 8) + c_inner)];
        }
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -73;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -74;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_1( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 3))) {
    TVMAPISetLastError("fused_nn_conv2d_1: num_args should be 3");
    return -75;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -76;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (1 == ((int32_t)arg1_strides[2]))) && (1 == ((int32_t)arg1_strides[1]))) && (64 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -77;
    }
  }
  float* output_unpack = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (16 == ((int32_t)arg2_strides[2]))) && (256 == ((int32_t)arg2_strides[1]))) && (32768 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -78;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_1: Expect arg[0] to be pointer");
    return -79;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_1: Expect arg[1] to be pointer");
    return -80;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_1: Expect arg[2] to be pointer");
    return -81;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -82;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -83;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -84;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -85;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -86;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -87;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -88;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -89;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -90;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -91;
  }
  if (!((((int32_t)arg1_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -92;
  }
  if (!((((int32_t)arg1_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -93;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -94;
  }
  if (!((((int32_t)arg1_shape[3]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -95;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -96;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -97;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -98;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -99;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -100;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -101;
  }
  if (!((((int32_t)arg2_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -102;
  }
  if (!((((int32_t)arg2_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -103;
  }
  if (!((((int32_t)arg2_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -104;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -105;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -106;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -107;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)246016, 2, 32);
  if (data_vec == NULL) {
    return -108;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)32768, 2, 32);
  if (kernel_vec == NULL) {
    return -109;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 248; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 31; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 31) + w)] = placeholder[(((((((C_h_fused / 31) * 8) + c) * 32) + (C_h_fused % 31)) * 32) + w)];
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 16; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 8; ++CI) {
      for (int32_t ci = 0; ci < 8; ++ci) {
        for (int32_t co = 0; co < 8; ++co) {
          (( float*)kernel_vec)[((((((CO_h_fused * 8) + CI) * 8) + ci) * 8) + co)] = placeholder1[((((((CO_h_fused * 8) + co) * 8) + CI) * 8) + ci)];
        }
      }
    }
  }
  for (int32_t c_outer_h_outer_fused = 0; c_outer_h_outer_fused < 256; ++c_outer_h_outer_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
      for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
        for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
          conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31))] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c)]));
        }
        for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
          conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 2)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c1)]));
        }
        for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
          conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 4)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c2)]));
        }
        for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
          conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 6)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c3)]));
        }
        for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
          conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 8)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c4)]));
        }
        for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
          conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 10)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c5)]));
        }
        for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
          conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 12)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c6)]));
        }
        for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
          conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 14)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c7)]));
        }
        for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
          conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 16)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c8)]));
        }
        for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
          conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 18)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c9)]));
        }
        for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
          conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 20)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c10)]));
        }
        for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
          conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 22)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c11)]));
        }
        for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
          conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 24)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c12)]));
        }
        for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
          conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 26)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c13)]));
        }
        for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
          conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 28)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c14)]));
        }
        for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
          conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((ic_outer * 7688) + ((c_outer_h_outer_fused % 16) * 496)) + (ic_inner * 31)) + 30)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 16) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c15)]));
        }
      }
    }
    for (int32_t w_inner = 0; w_inner < 16; ++w_inner) {
      for (int32_t c_inner = 0; c_inner < 8; ++c_inner) {
        output_unpack[(((((((c_outer_h_outer_fused / 16) * 8) + c_inner) * 16) + (c_outer_h_outer_fused % 16)) * 16) + w_inner)] = conv_global[((w_inner * 8) + c_inner)];
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -110;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -111;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 3))) {
    TVMAPISetLastError("fused_nn_conv2d: num_args should be 3");
    return -112;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -113;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (1 == ((int32_t)arg1_strides[2]))) && (1 == ((int32_t)arg1_strides[1]))) && (64 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -114;
    }
  }
  float* output_unpack = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (32 == ((int32_t)arg2_strides[2]))) && (1024 == ((int32_t)arg2_strides[1]))) && (65536 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -115;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d: Expect arg[0] to be pointer");
    return -116;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d: Expect arg[1] to be pointer");
    return -117;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d: Expect arg[2] to be pointer");
    return -118;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -119;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -120;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -121;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -122;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -123;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -124;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -125;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -126;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -127;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -128;
  }
  if (!((((int32_t)arg1_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -129;
  }
  if (!((((int32_t)arg1_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -130;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -131;
  }
  if (!((((int32_t)arg1_shape[3]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -132;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -133;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -134;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -135;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -136;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -137;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -138;
  }
  if (!((((int32_t)arg2_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -139;
  }
  if (!((((int32_t)arg2_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -140;
  }
  if (!((((int32_t)arg2_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -141;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -142;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -143;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -144;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)262144, 2, 32);
  if (data_vec == NULL) {
    return -145;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)16384, 2, 32);
  if (kernel_vec == NULL) {
    return -146;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 256; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 32; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 32) + w)] = placeholder[(((((((C_h_fused / 32) * 8) + c) * 32) + (C_h_fused % 32)) * 32) + w)];
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 8; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 8; ++CI) {
      for (int32_t ci = 0; ci < 8; ++ci) {
        for (int32_t co = 0; co < 8; ++co) {
          (( float*)kernel_vec)[((((((CO_h_fused * 8) + CI) * 8) + ci) * 8) + co)] = placeholder1[((((((CO_h_fused * 8) + co) * 8) + CI) * 8) + ci)];
        }
      }
    }
  }
  for (int32_t c_outer_h_outer_fused = 0; c_outer_h_outer_fused < 256; ++c_outer_h_outer_fused) {
    void* conv_global = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
    if (conv_global == NULL) {
      return -147;
    }
    for (int32_t ow_c_outer = 0; ow_c_outer < 2; ++ow_c_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
        (( float*)conv_global)[((ow_c_outer * 128) + oc_block_c_init)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init1) + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init2) + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init3) + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init4) + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init5) + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init6) + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init7) + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init8) + 64)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init9) + 72)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init10) + 80)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init11) + 88)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init12) + 96)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init13) + 104)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init14) + 112)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
        (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c_init15) + 120)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
        for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
          for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
            (( float*)conv_global)[((ow_c_outer * 128) + oc_block_c)] = ((( float*)conv_global)[((ow_c_outer * 128) + oc_block_c)] + ((( float*)data_vec)[(((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c)]));
          }
          for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c1) + 8)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c1) + 8)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 1)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c1)]));
          }
          for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c2) + 16)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c2) + 16)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 2)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c2)]));
          }
          for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c3) + 24)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c3) + 24)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 3)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c3)]));
          }
          for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c4) + 32)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c4) + 32)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 4)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c4)]));
          }
          for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c5) + 40)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c5) + 40)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 5)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c5)]));
          }
          for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c6) + 48)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c6) + 48)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 6)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c6)]));
          }
          for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c7) + 56)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c7) + 56)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 7)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c7)]));
          }
          for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c8) + 64)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c8) + 64)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 8)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c8)]));
          }
          for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c9) + 72)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c9) + 72)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 9)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c9)]));
          }
          for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c10) + 80)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c10) + 80)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 10)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c10)]));
          }
          for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c11) + 88)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c11) + 88)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 11)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c11)]));
          }
          for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c12) + 96)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c12) + 96)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 12)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c12)]));
          }
          for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c13) + 104)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c13) + 104)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 13)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c13)]));
          }
          for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c14) + 112)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c14) + 112)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 14)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c14)]));
          }
          for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
            (( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c15) + 120)] = ((( float*)conv_global)[(((ow_c_outer * 128) + oc_block_c15) + 120)] + ((( float*)data_vec)[((((((((ic_outer * 32) + (c_outer_h_outer_fused % 32)) * 8) + ic_inner) * 2) + ow_c_outer) * 16) + 15)] * (( float*)kernel_vec)[(((((((c_outer_h_outer_fused / 32) * 8) + ic_outer) * 8) + ic_inner) * 8) + oc_block_c15)]));
          }
        }
      }
    }
    for (int32_t w_outer = 0; w_outer < 2; ++w_outer) {
      for (int32_t w_inner = 0; w_inner < 16; ++w_inner) {
        for (int32_t c_inner = 0; c_inner < 8; ++c_inner) {
          output_unpack[(((((((((c_outer_h_outer_fused / 32) * 8) + c_inner) * 32) + (c_outer_h_outer_fused % 32)) * 2) + w_outer) * 16) + w_inner)] = (( float*)conv_global)[((((w_outer * 16) + w_inner) * 8) + c_inner)];
        }
      }
    }
    if (TVMBackendFreeWorkspace(1, dev_id, conv_global) != 0) {
      return -148;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -149;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -150;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_7( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_7: num_args should be 5");
    return -151;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (3072 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -152;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (27 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -153;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -154;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -155;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (32 == ((int32_t)arg4_strides[2]))) && (1024 == ((int32_t)arg4_strides[1]))) && (65536 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -156;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_7: Expect arg[0] to be pointer");
    return -157;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_7: Expect arg[1] to be pointer");
    return -158;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_7: Expect arg[2] to be pointer");
    return -159;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_7: Expect arg[3] to be pointer");
    return -160;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_7: Expect arg[4] to be pointer");
    return -161;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -162;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -163;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -164;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -165;
  }
  if (!((((int32_t)arg0_shape[1]) == 3))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -166;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -167;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -168;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -169;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -170;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -171;
  }
  if (!((((int32_t)arg1_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -172;
  }
  if (!((((int32_t)arg1_shape[1]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -173;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -174;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -175;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -176;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -177;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -178;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -179;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -180;
  }
  if (!((((int32_t)arg2_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -181;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -182;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -183;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -184;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -185;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -186;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -187;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -188;
  }
  if (!((((int32_t)arg3_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -189;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -190;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -191;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -192;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -193;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -194;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -195;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -196;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -197;
  }
  if (!((((int32_t)arg4_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -198;
  }
  if (!((((int32_t)arg4_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -199;
  }
  if (!((((int32_t)arg4_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -200;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -201;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -202;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -203;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)13872, 2, 32);
  if (data_vec == NULL) {
    return -204;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)6912, 2, 32);
  if (kernel_vec == NULL) {
    return -205;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 34; ++C_h_fused) {
    for (int32_t c = 0; c < 3; ++c) {
      for (int32_t w = 0; w < 34; ++w) {
        (( float*)data_vec)[((((C_h_fused * 3) + c) * 34) + w)] = (((((1 <= C_h_fused) && (C_h_fused < 33)) && (1 <= w)) && (w < 33)) ? placeholder[(((((c * 32) + C_h_fused) * 32) + w) + -33)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 24; ++CO_h_fused) {
    for (int32_t w1 = 0; w1 < 3; ++w1) {
      for (int32_t ci = 0; ci < 3; ++ci) {
        for (int32_t co = 0; co < 8; ++co) {
          (( float*)kernel_vec)[((((((CO_h_fused * 3) + w1) * 3) + ci) * 8) + co)] = placeholder1[(((((((((CO_h_fused / 3) * 8) + co) * 3) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
    void* conv = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
    if (conv == NULL) {
      return -206;
    }
     float conv_global[128];
    for (int32_t ow_outer = 0; ow_outer < 2; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
        conv_global[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
        conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
        conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
        conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
        conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
        conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
        conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
        conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
        conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
        conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
        conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
        conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
        conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
        conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
        conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
        conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
      }
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 3; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 1)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 3)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 5)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 7)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c7)]));
            }
            for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
              conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 8)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c8)]));
            }
            for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
              conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 9)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c9)]));
            }
            for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
              conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 10)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c10)]));
            }
            for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
              conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 11)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c11)]));
            }
            for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
              conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 12)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c12)]));
            }
            for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
              conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 13)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c13)]));
            }
            for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
              conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 14)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c14)]));
            }
            for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
              conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[(((((((kh + (ax1_outer_ax2_fused % 32)) * 3) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 15)] * (( float*)kernel_vec)[(((((((((ax1_outer_ax2_fused / 32) * 3) + kh) * 3) + kw) * 3) + ic_inner) * 8) + oc_block_c15)]));
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 8; ++oc_block) {
          (( float*)conv)[((((ow_outer * 16) + ow_inner) * 8) + oc_block)] = conv_global[((ow_inner * 8) + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
          T_relu[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)] = ((((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)])) > (0.000000e+00f) ? ((((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)])) : (0.000000e+00f);
        }
      }
    }
    if (TVMBackendFreeWorkspace(1, dev_id, conv) != 0) {
      return -207;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -208;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -209;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_3( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_nn_conv2d_add_3: num_args should be 4");
    return -210;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -211;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (576 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -212;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (32 == ((int32_t)arg2_strides[2]))) && (1024 == ((int32_t)arg2_strides[1]))) && (65536 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -213;
    }
  }
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (32 == ((int32_t)arg3_strides[2]))) && (1024 == ((int32_t)arg3_strides[1]))) && (65536 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -214;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_3: Expect arg[0] to be pointer");
    return -215;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_3: Expect arg[1] to be pointer");
    return -216;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_3: Expect arg[2] to be pointer");
    return -217;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_3: Expect arg[3] to be pointer");
    return -218;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -219;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -220;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -221;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -222;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -223;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -224;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -225;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -226;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -227;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -228;
  }
  if (!((((int32_t)arg1_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -229;
  }
  if (!((((int32_t)arg1_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -230;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -231;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -232;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -233;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -234;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -235;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -236;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -237;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -238;
  }
  if (!((((int32_t)arg2_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -239;
  }
  if (!((((int32_t)arg2_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -240;
  }
  if (!((((int32_t)arg2_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -241;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -242;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -243;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -244;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -245;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -246;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -247;
  }
  if (!((((int32_t)arg3_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -248;
  }
  if (!((((int32_t)arg3_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -249;
  }
  if (!((((int32_t)arg3_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -250;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -251;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -252;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -253;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)295936, 2, 32);
  if (data_vec == NULL) {
    return -254;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)147456, 2, 32);
  if (kernel_vec == NULL) {
    return -255;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 272; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 34; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 34) + w)] = (((((1 <= (C_h_fused % 34)) && ((C_h_fused % 34) < 33)) && (1 <= w)) && (w < 33)) ? placeholder[((((((((C_h_fused / 34) * 8) + c) * 32) + (C_h_fused % 34)) * 32) + w) + -33)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 24; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 8; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 8) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 8) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
    void* conv = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
    if (conv == NULL) {
      return -256;
    }
     float conv_global[128];
    for (int32_t ow_outer = 0; ow_outer < 2; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
        conv_global[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
        conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
        conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
        conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
        conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
        conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
        conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
        conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
        conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
        conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
        conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
        conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
        conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
        conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
        conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
        conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
                conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
              }
              for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
                conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
              }
              for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
                conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
              }
              for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
                conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
              }
              for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
                conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
              }
              for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
                conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
              }
              for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
                conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
              }
              for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
                conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
              }
              for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
                conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
              }
              for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
                conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 9)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
              }
              for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
                conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
              }
              for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
                conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 11)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
              }
              for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
                conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
              }
              for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
                conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 13)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
              }
              for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
                conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
              }
              for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
                conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 15)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 8; ++oc_block) {
          (( float*)conv)[((((ow_outer * 16) + ow_inner) * 8) + oc_block)] = conv_global[((ow_inner * 8) + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
          T_add[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)] = ((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] + placeholder2[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)]);
        }
      }
    }
    if (TVMBackendFreeWorkspace(1, dev_id, conv) != 0) {
      return -257;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -258;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -259;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_6( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_6: num_args should be 5");
    return -260;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -261;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (576 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -262;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -263;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -264;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (32 == ((int32_t)arg4_strides[2]))) && (1024 == ((int32_t)arg4_strides[1]))) && (65536 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -265;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_6: Expect arg[0] to be pointer");
    return -266;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_6: Expect arg[1] to be pointer");
    return -267;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_6: Expect arg[2] to be pointer");
    return -268;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_6: Expect arg[3] to be pointer");
    return -269;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_6: Expect arg[4] to be pointer");
    return -270;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -271;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -272;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -273;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -274;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -275;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -276;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -277;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -278;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -279;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -280;
  }
  if (!((((int32_t)arg1_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -281;
  }
  if (!((((int32_t)arg1_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -282;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -283;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -284;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -285;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -286;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -287;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -288;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -289;
  }
  if (!((((int32_t)arg2_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -290;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -291;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -292;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -293;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -294;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -295;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -296;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -297;
  }
  if (!((((int32_t)arg3_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -298;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -299;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -300;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -301;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -302;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -303;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -304;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -305;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -306;
  }
  if (!((((int32_t)arg4_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -307;
  }
  if (!((((int32_t)arg4_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -308;
  }
  if (!((((int32_t)arg4_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -309;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -310;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -311;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -312;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)295936, 2, 32);
  if (data_vec == NULL) {
    return -313;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)147456, 2, 32);
  if (kernel_vec == NULL) {
    return -314;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 272; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 34; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 34) + w)] = (((((1 <= (C_h_fused % 34)) && ((C_h_fused % 34) < 33)) && (1 <= w)) && (w < 33)) ? placeholder[((((((((C_h_fused / 34) * 8) + c) * 32) + (C_h_fused % 34)) * 32) + w) + -33)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 24; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 8; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 8) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 8) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
    void* conv = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
    if (conv == NULL) {
      return -315;
    }
     float conv_global[128];
    for (int32_t ow_outer = 0; ow_outer < 2; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
        conv_global[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
        conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
        conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
        conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
        conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
        conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
        conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
        conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
        conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
        conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
        conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
        conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
        conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
        conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
        conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
        conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
                conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
              }
              for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
                conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
              }
              for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
                conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
              }
              for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
                conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
              }
              for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
                conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
              }
              for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
                conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
              }
              for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
                conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
              }
              for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
                conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
              }
              for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
                conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
              }
              for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
                conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 9)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
              }
              for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
                conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
              }
              for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
                conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 11)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
              }
              for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
                conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
              }
              for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
                conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 13)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
              }
              for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
                conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
              }
              for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
                conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 15)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 8; ++oc_block) {
          (( float*)conv)[((((ow_outer * 16) + ow_inner) * 8) + oc_block)] = conv_global[((ow_inner * 8) + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
          T_relu[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)] = ((((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)])) > (0.000000e+00f) ? ((((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)])) : (0.000000e+00f);
        }
      }
    }
    if (TVMBackendFreeWorkspace(1, dev_id, conv) != 0) {
      return -316;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -317;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -318;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_multiply_add_nn_relu_3( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_3: num_args should be 4");
    return -319;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -320;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((((1 == ((int32_t)arg1_strides[2])) && (1 == ((int32_t)arg1_strides[1]))) && (1 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -321;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -322;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (32 == ((int32_t)arg3_strides[2]))) && (1024 == ((int32_t)arg3_strides[1]))) && (65536 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -323;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_3: Expect arg[0] to be pointer");
    return -324;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_3: Expect arg[1] to be pointer");
    return -325;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_3: Expect arg[2] to be pointer");
    return -326;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_3: Expect arg[3] to be pointer");
    return -327;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -328;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -329;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -330;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -331;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -332;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -333;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -334;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -335;
  }
  if (!((3 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 3");
    return -336;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -337;
  }
  if (!((((int32_t)arg1_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -338;
  }
  if (!((((int32_t)arg1_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -339;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -340;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -341;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -342;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -343;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -344;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -345;
  }
  if (!((((int32_t)arg2_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -346;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -347;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -348;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -349;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -350;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -351;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -352;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -353;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -354;
  }
  if (!((((int32_t)arg3_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -355;
  }
  if (!((((int32_t)arg3_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -356;
  }
  if (!((((int32_t)arg3_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -357;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -358;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -359;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -360;
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 64; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 32; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        T_relu[((((ax0_ax1_fused * 32) + ax2) * 32) + ax3)] = (((placeholder[((((ax0_ax1_fused * 32) + ax2) * 32) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) > (0.000000e+00f) ? (((placeholder[((((ax0_ax1_fused * 32) + ax2) * 32) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) : (0.000000e+00f);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_2( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_nn_conv2d_add_2: num_args should be 4");
    return -361;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (16 == ((int32_t)arg0_strides[2]))) && (256 == ((int32_t)arg0_strides[1]))) && (32768 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -362;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (1152 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -363;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (16 == ((int32_t)arg2_strides[2]))) && (256 == ((int32_t)arg2_strides[1]))) && (32768 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -364;
    }
  }
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (16 == ((int32_t)arg3_strides[2]))) && (256 == ((int32_t)arg3_strides[1]))) && (32768 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -365;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_2: Expect arg[0] to be pointer");
    return -366;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_2: Expect arg[1] to be pointer");
    return -367;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_2: Expect arg[2] to be pointer");
    return -368;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_2: Expect arg[3] to be pointer");
    return -369;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -370;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -371;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -372;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -373;
  }
  if (!((((int32_t)arg0_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -374;
  }
  if (!((((int32_t)arg0_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -375;
  }
  if (!((((int32_t)arg0_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -376;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -377;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -378;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -379;
  }
  if (!((((int32_t)arg1_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -380;
  }
  if (!((((int32_t)arg1_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -381;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -382;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -383;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -384;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -385;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -386;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -387;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -388;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -389;
  }
  if (!((((int32_t)arg2_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -390;
  }
  if (!((((int32_t)arg2_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -391;
  }
  if (!((((int32_t)arg2_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -392;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -393;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -394;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -395;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -396;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -397;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -398;
  }
  if (!((((int32_t)arg3_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -399;
  }
  if (!((((int32_t)arg3_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -400;
  }
  if (!((((int32_t)arg3_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -401;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -402;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -403;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -404;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)165888, 2, 32);
  if (data_vec == NULL) {
    return -405;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)589824, 2, 32);
  if (kernel_vec == NULL) {
    return -406;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 288; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 18; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 18) + w)] = (((((1 <= (C_h_fused % 18)) && ((C_h_fused % 18) < 17)) && (1 <= w)) && (w < 17)) ? placeholder[((((((((C_h_fused / 18) * 8) + c) * 16) + (C_h_fused % 18)) * 16) + w) + -17)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 48; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 16; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 16) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 16) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
            for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
              conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
            }
            for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
              conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 9)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
            }
            for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
              conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
            }
            for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
              conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 11)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
            }
            for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
              conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
            }
            for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
              conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 13)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
            }
            for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
              conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
            }
            for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
              conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 15)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_add[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)] = (conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)]);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -407;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -408;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_multiply_add_nn_relu( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu: num_args should be 4");
    return -409;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (4 == ((int32_t)arg0_strides[2]))) && (16 == ((int32_t)arg0_strides[1]))) && (8192 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -410;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((((1 == ((int32_t)arg1_strides[2])) && (1 == ((int32_t)arg1_strides[1]))) && (1 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -411;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -412;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (4 == ((int32_t)arg3_strides[2]))) && (16 == ((int32_t)arg3_strides[1]))) && (8192 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -413;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu: Expect arg[0] to be pointer");
    return -414;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu: Expect arg[1] to be pointer");
    return -415;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu: Expect arg[2] to be pointer");
    return -416;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu: Expect arg[3] to be pointer");
    return -417;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -418;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -419;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -420;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -421;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -422;
  }
  if (!((((int32_t)arg0_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -423;
  }
  if (!((((int32_t)arg0_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -424;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -425;
  }
  if (!((3 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 3");
    return -426;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -427;
  }
  if (!((((int32_t)arg1_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -428;
  }
  if (!((((int32_t)arg1_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -429;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -430;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -431;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -432;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -433;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -434;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -435;
  }
  if (!((((int32_t)arg2_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -436;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -437;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -438;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -439;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -440;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -441;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -442;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -443;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -444;
  }
  if (!((((int32_t)arg3_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -445;
  }
  if (!((((int32_t)arg3_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -446;
  }
  if (!((((int32_t)arg3_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -447;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -448;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -449;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -450;
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 512; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 4; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 4; ++ax3) {
        T_relu[((((ax0_ax1_fused * 4) + ax2) * 4) + ax3)] = (((placeholder[((((ax0_ax1_fused * 4) + ax2) * 4) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) > (0.000000e+00f) ? (((placeholder[((((ax0_ax1_fused * 4) + ax2) * 4) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) : (0.000000e+00f);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu: num_args should be 5");
    return -451;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (4 == ((int32_t)arg0_strides[2]))) && (16 == ((int32_t)arg0_strides[1]))) && (8192 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -452;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (4608 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -453;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -454;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -455;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (4 == ((int32_t)arg4_strides[2]))) && (16 == ((int32_t)arg4_strides[1]))) && (8192 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -456;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu: Expect arg[0] to be pointer");
    return -457;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu: Expect arg[1] to be pointer");
    return -458;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu: Expect arg[2] to be pointer");
    return -459;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu: Expect arg[3] to be pointer");
    return -460;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu: Expect arg[4] to be pointer");
    return -461;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -462;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -463;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -464;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -465;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -466;
  }
  if (!((((int32_t)arg0_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -467;
  }
  if (!((((int32_t)arg0_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -468;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -469;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -470;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -471;
  }
  if (!((((int32_t)arg1_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -472;
  }
  if (!((((int32_t)arg1_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -473;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -474;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -475;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -476;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -477;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -478;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -479;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -480;
  }
  if (!((((int32_t)arg2_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -481;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -482;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -483;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -484;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -485;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -486;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -487;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -488;
  }
  if (!((((int32_t)arg3_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -489;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -490;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -491;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -492;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -493;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -494;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -495;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -496;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -497;
  }
  if (!((((int32_t)arg4_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -498;
  }
  if (!((((int32_t)arg4_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -499;
  }
  if (!((((int32_t)arg4_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -500;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -501;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -502;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -503;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)73728, 2, 32);
  if (data_vec == NULL) {
    return -504;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)9437184, 2, 32);
  if (kernel_vec == NULL) {
    return -505;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 384; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 6; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 6) + w)] = (((((1 <= (C_h_fused % 6)) && ((C_h_fused % 6) < 5)) && (1 <= w)) && (w < 5)) ? placeholder[((((((((C_h_fused / 6) * 8) + c) * 4) + (C_h_fused % 6)) * 4) + w) + -5)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 192; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 64; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 64) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 64) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[32];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 64; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)] = (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -506;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -507;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_1( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_nn_conv2d_add_1: num_args should be 4");
    return -508;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (8 == ((int32_t)arg0_strides[2]))) && (64 == ((int32_t)arg0_strides[1]))) && (16384 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -509;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (2304 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -510;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (8 == ((int32_t)arg2_strides[2]))) && (64 == ((int32_t)arg2_strides[1]))) && (16384 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -511;
    }
  }
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (8 == ((int32_t)arg3_strides[2]))) && (64 == ((int32_t)arg3_strides[1]))) && (16384 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -512;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_1: Expect arg[0] to be pointer");
    return -513;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_1: Expect arg[1] to be pointer");
    return -514;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_1: Expect arg[2] to be pointer");
    return -515;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_1: Expect arg[3] to be pointer");
    return -516;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -517;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -518;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -519;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -520;
  }
  if (!((((int32_t)arg0_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -521;
  }
  if (!((((int32_t)arg0_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -522;
  }
  if (!((((int32_t)arg0_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -523;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -524;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -525;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -526;
  }
  if (!((((int32_t)arg1_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -527;
  }
  if (!((((int32_t)arg1_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -528;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -529;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -530;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -531;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -532;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -533;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -534;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -535;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -536;
  }
  if (!((((int32_t)arg2_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -537;
  }
  if (!((((int32_t)arg2_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -538;
  }
  if (!((((int32_t)arg2_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -539;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -540;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -541;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -542;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -543;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -544;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -545;
  }
  if (!((((int32_t)arg3_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -546;
  }
  if (!((((int32_t)arg3_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -547;
  }
  if (!((((int32_t)arg3_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -548;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -549;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -550;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -551;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)102400, 2, 32);
  if (data_vec == NULL) {
    return -552;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)2359296, 2, 32);
  if (kernel_vec == NULL) {
    return -553;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 320; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 10; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 10) + w)] = (((((1 <= (C_h_fused % 10)) && ((C_h_fused % 10) < 9)) && (1 <= w)) && (w < 9)) ? placeholder[((((((((C_h_fused / 10) * 8) + c) * 8) + (C_h_fused % 10)) * 8) + w) + -9)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 96; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 32; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 32) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 32) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[64];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 32; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_add[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)] = (conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)]);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -554;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -555;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_dense_nn_bias_add( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_nn_dense_nn_bias_add: num_args should be 4");
    return -556;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((1 == ((int32_t)arg0_strides[1])) && (512 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -557;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((1 == ((int32_t)arg1_strides[1])) && (512 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -558;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((1 == ((int32_t)arg2_strides[0])))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -559;
    }
  }
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((1 == ((int32_t)arg3_strides[1])) && (10 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -560;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_dense_nn_bias_add: Expect arg[0] to be pointer");
    return -561;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_dense_nn_bias_add: Expect arg[1] to be pointer");
    return -562;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_dense_nn_bias_add: Expect arg[2] to be pointer");
    return -563;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_dense_nn_bias_add: Expect arg[3] to be pointer");
    return -564;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -565;
  }
  if (!((2 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 2");
    return -566;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -567;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -568;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -569;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -570;
  }
  if (!((2 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 2");
    return -571;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -572;
  }
  if (!((((int32_t)arg1_shape[0]) == 10))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -573;
  }
  if (!((((int32_t)arg1_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -574;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -575;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -576;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -577;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 1");
    return -578;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -579;
  }
  if (!((((int32_t)arg2_shape[0]) == 10))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -580;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -581;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -582;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -583;
  }
  if (!((2 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 2");
    return -584;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -585;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -586;
  }
  if (!((((int32_t)arg3_shape[1]) == 10))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -587;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -588;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -589;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -590;
  }
   float compute[10];
  for (int32_t y_outer_x_outer_fused = 0; y_outer_x_outer_fused < 10; ++y_outer_x_outer_fused) {
     float compute1[16];
    for (int32_t x_init = 0; x_init < 16; ++x_init) {
      compute1[x_init] = 0.000000e+00f;
    }
    for (int32_t k = 0; k < 32; ++k) {
      for (int32_t x = 0; x < 16; ++x) {
        compute1[x] = (compute1[x] + (placeholder[((k * 16) + x)] * placeholder1[((((y_outer_x_outer_fused * 32) + k) * 16) + x)]));
      }
    }
    compute[y_outer_x_outer_fused] = 0.000000e+00f;
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[0]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[1]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[2]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[3]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[4]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[5]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[6]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[7]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[8]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[9]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[10]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[11]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[12]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[13]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[14]);
    compute[y_outer_x_outer_fused] = (compute[y_outer_x_outer_fused] + compute1[15]);
  }
  for (int32_t ax1 = 0; ax1 < 10; ++ax1) {
    T_add[ax1] = (compute[ax1] + placeholder2[ax1]);
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_global_avg_pool2d( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 2))) {
    TVMAPISetLastError("fused_nn_global_avg_pool2d: num_args should be 2");
    return -591;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (4 == ((int32_t)arg0_strides[2]))) && (16 == ((int32_t)arg0_strides[1]))) && (8192 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -592;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* tensor = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (1 == ((int32_t)arg1_strides[2]))) && (1 == ((int32_t)arg1_strides[1]))) && (512 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -593;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_global_avg_pool2d: Expect arg[0] to be pointer");
    return -594;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_global_avg_pool2d: Expect arg[1] to be pointer");
    return -595;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -596;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -597;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -598;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -599;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -600;
  }
  if (!((((int32_t)arg0_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -601;
  }
  if (!((((int32_t)arg0_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -602;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -603;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -604;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -605;
  }
  if (!((((int32_t)arg1_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -606;
  }
  if (!((((int32_t)arg1_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -607;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -608;
  }
  if (!((((int32_t)arg1_shape[3]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -609;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -610;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -611;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -612;
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 512; ++ax0_ax1_fused) {
    tensor[ax0_ax1_fused] = 0.000000e+00f;
    for (int32_t rv1 = 0; rv1 < 4; ++rv1) {
      for (int32_t rv2 = 0; rv2 < 4; ++rv2) {
        tensor[ax0_ax1_fused] = (tensor[ax0_ax1_fused] + (placeholder[((((ax0_ax1_fused * 4) + rv1) * 4) + rv2)] * 6.250000e-02f));
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_multiply_add_nn_relu_1( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 6))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: num_args should be 6");
    return -613;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = (( int32_t*)arg_type_ids)[5];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (8 == ((int32_t)arg0_strides[2]))) && (64 == ((int32_t)arg0_strides[1]))) && (16384 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -614;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (2304 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -615;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (8 == ((int32_t)arg2_strides[2]))) && (64 == ((int32_t)arg2_strides[1]))) && (16384 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -616;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -617;
    }
  }
  float* placeholder4 = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!((((1 == ((int32_t)arg4_strides[2])) && (1 == ((int32_t)arg4_strides[1]))) && (1 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -618;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg5)[0].data);
  int64_t* arg5_shape = (int64_t*)(((TVMArray*)arg5)[0].shape);
  int64_t* arg5_strides = (int64_t*)(((TVMArray*)arg5)[0].strides);
  if (!(arg5_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg5_strides[3])) && (8 == ((int32_t)arg5_strides[2]))) && (64 == ((int32_t)arg5_strides[1]))) && (16384 == ((int32_t)arg5_strides[0]))))) {
      TVMAPISetLastError("arg5.strides: expected to be compact array");
      return -619;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: Expect arg[0] to be pointer");
    return -620;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: Expect arg[1] to be pointer");
    return -621;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: Expect arg[2] to be pointer");
    return -622;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: Expect arg[3] to be pointer");
    return -623;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: Expect arg[4] to be pointer");
    return -624;
  }
  if (!(((((arg5_code == 3) || (arg5_code == 13)) || (arg5_code == 7)) || (arg5_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_1: Expect arg[5] to be pointer");
    return -625;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -626;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -627;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -628;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -629;
  }
  if (!((((int32_t)arg0_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -630;
  }
  if (!((((int32_t)arg0_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -631;
  }
  if (!((((int32_t)arg0_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -632;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -633;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -634;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -635;
  }
  if (!((((int32_t)arg1_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -636;
  }
  if (!((((int32_t)arg1_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -637;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -638;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -639;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -640;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -641;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -642;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -643;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -644;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -645;
  }
  if (!((((int32_t)arg2_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -646;
  }
  if (!((((int32_t)arg2_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -647;
  }
  if (!((((int32_t)arg2_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -648;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -649;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -650;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -651;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -652;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -653;
  }
  if (!((((int32_t)arg3_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -654;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -655;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -656;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -657;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -658;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -659;
  }
  if (!((3 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 3");
    return -660;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -661;
  }
  if (!((((int32_t)arg4_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -662;
  }
  if (!((((int32_t)arg4_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -663;
  }
  if (!((((int32_t)arg4_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -664;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -665;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -666;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -667;
  }
  if (!((4 == (((TVMArray*)arg5)[0].ndim)))) {
    TVMAPISetLastError("arg5.ndim is expected to equal 4");
    return -668;
  }
  if (!(((((((TVMArray*)arg5)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg5)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg5)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg5.dtype is expected to be float32");
    return -669;
  }
  if (!((((int32_t)arg5_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg5.shape[0] has an unsatisfied constraint");
    return -670;
  }
  if (!((((int32_t)arg5_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg5.shape[1] has an unsatisfied constraint");
    return -671;
  }
  if (!((((int32_t)arg5_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg5.shape[2] has an unsatisfied constraint");
    return -672;
  }
  if (!((((int32_t)arg5_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg5.shape[3] has an unsatisfied constraint");
    return -673;
  }
  if (!(((((TVMArray*)arg5)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg5.byte_offset has an unsatisfied constraint");
    return -674;
  }
  if (!((1 == (((TVMArray*)arg5)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg5.device_type has an unsatisfied constraint");
    return -675;
  }
  if (!((dev_id == (((TVMArray*)arg5)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg5.device_id has an unsatisfied constraint");
    return -676;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)102400, 2, 32);
  if (data_vec == NULL) {
    return -677;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)2359296, 2, 32);
  if (kernel_vec == NULL) {
    return -678;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 320; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 10; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 10) + w)] = (((((1 <= (C_h_fused % 10)) && ((C_h_fused % 10) < 9)) && (1 <= w)) && (w < 9)) ? placeholder[((((((((C_h_fused / 10) * 8) + c) * 8) + (C_h_fused % 10)) * 8) + w) + -9)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 96; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 32; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 32) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 32) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[64];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 32; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)] = ((((conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)])) > (0.000000e+00f) ? ((((conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -679;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -680;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_multiply_add_nn_relu_2( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 6))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: num_args should be 6");
    return -681;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = (( int32_t*)arg_type_ids)[5];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (16 == ((int32_t)arg0_strides[2]))) && (256 == ((int32_t)arg0_strides[1]))) && (32768 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -682;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (1152 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -683;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (16 == ((int32_t)arg2_strides[2]))) && (256 == ((int32_t)arg2_strides[1]))) && (32768 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -684;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -685;
    }
  }
  float* placeholder4 = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!((((1 == ((int32_t)arg4_strides[2])) && (1 == ((int32_t)arg4_strides[1]))) && (1 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -686;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg5)[0].data);
  int64_t* arg5_shape = (int64_t*)(((TVMArray*)arg5)[0].shape);
  int64_t* arg5_strides = (int64_t*)(((TVMArray*)arg5)[0].strides);
  if (!(arg5_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg5_strides[3])) && (16 == ((int32_t)arg5_strides[2]))) && (256 == ((int32_t)arg5_strides[1]))) && (32768 == ((int32_t)arg5_strides[0]))))) {
      TVMAPISetLastError("arg5.strides: expected to be compact array");
      return -687;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: Expect arg[0] to be pointer");
    return -688;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: Expect arg[1] to be pointer");
    return -689;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: Expect arg[2] to be pointer");
    return -690;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: Expect arg[3] to be pointer");
    return -691;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: Expect arg[4] to be pointer");
    return -692;
  }
  if (!(((((arg5_code == 3) || (arg5_code == 13)) || (arg5_code == 7)) || (arg5_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_2: Expect arg[5] to be pointer");
    return -693;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -694;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -695;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -696;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -697;
  }
  if (!((((int32_t)arg0_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -698;
  }
  if (!((((int32_t)arg0_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -699;
  }
  if (!((((int32_t)arg0_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -700;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -701;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -702;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -703;
  }
  if (!((((int32_t)arg1_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -704;
  }
  if (!((((int32_t)arg1_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -705;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -706;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -707;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -708;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -709;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -710;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -711;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -712;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -713;
  }
  if (!((((int32_t)arg2_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -714;
  }
  if (!((((int32_t)arg2_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -715;
  }
  if (!((((int32_t)arg2_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -716;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -717;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -718;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -719;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -720;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -721;
  }
  if (!((((int32_t)arg3_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -722;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -723;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -724;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -725;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -726;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -727;
  }
  if (!((3 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 3");
    return -728;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -729;
  }
  if (!((((int32_t)arg4_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -730;
  }
  if (!((((int32_t)arg4_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -731;
  }
  if (!((((int32_t)arg4_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -732;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -733;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -734;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -735;
  }
  if (!((4 == (((TVMArray*)arg5)[0].ndim)))) {
    TVMAPISetLastError("arg5.ndim is expected to equal 4");
    return -736;
  }
  if (!(((((((TVMArray*)arg5)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg5)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg5)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg5.dtype is expected to be float32");
    return -737;
  }
  if (!((((int32_t)arg5_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg5.shape[0] has an unsatisfied constraint");
    return -738;
  }
  if (!((((int32_t)arg5_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg5.shape[1] has an unsatisfied constraint");
    return -739;
  }
  if (!((((int32_t)arg5_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg5.shape[2] has an unsatisfied constraint");
    return -740;
  }
  if (!((((int32_t)arg5_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg5.shape[3] has an unsatisfied constraint");
    return -741;
  }
  if (!(((((TVMArray*)arg5)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg5.byte_offset has an unsatisfied constraint");
    return -742;
  }
  if (!((1 == (((TVMArray*)arg5)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg5.device_type has an unsatisfied constraint");
    return -743;
  }
  if (!((dev_id == (((TVMArray*)arg5)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg5.device_id has an unsatisfied constraint");
    return -744;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)165888, 2, 32);
  if (data_vec == NULL) {
    return -745;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)589824, 2, 32);
  if (kernel_vec == NULL) {
    return -746;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 288; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 18; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 18) + w)] = (((((1 <= (C_h_fused % 18)) && ((C_h_fused % 18) < 17)) && (1 <= w)) && (w < 17)) ? placeholder[((((((((C_h_fused / 18) * 8) + c) * 16) + (C_h_fused % 18)) * 16) + w) + -17)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 48; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 16; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 16) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 16) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
            for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
              conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
            }
            for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
              conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 9)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
            }
            for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
              conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
            }
            for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
              conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 11)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
            }
            for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
              conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
            }
            for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
              conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 13)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
            }
            for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
              conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
            }
            for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
              conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 15)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)] = ((((conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)])) > (0.000000e+00f) ? ((((conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -747;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -748;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_multiply_add_nn_relu_3( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 6))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: num_args should be 6");
    return -749;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = (( int32_t*)arg_type_ids)[5];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -750;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (576 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -751;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (32 == ((int32_t)arg2_strides[2]))) && (1024 == ((int32_t)arg2_strides[1]))) && (65536 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -752;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -753;
    }
  }
  float* placeholder4 = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!((((1 == ((int32_t)arg4_strides[2])) && (1 == ((int32_t)arg4_strides[1]))) && (1 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -754;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg5)[0].data);
  int64_t* arg5_shape = (int64_t*)(((TVMArray*)arg5)[0].shape);
  int64_t* arg5_strides = (int64_t*)(((TVMArray*)arg5)[0].strides);
  if (!(arg5_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg5_strides[3])) && (32 == ((int32_t)arg5_strides[2]))) && (1024 == ((int32_t)arg5_strides[1]))) && (65536 == ((int32_t)arg5_strides[0]))))) {
      TVMAPISetLastError("arg5.strides: expected to be compact array");
      return -755;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: Expect arg[0] to be pointer");
    return -756;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: Expect arg[1] to be pointer");
    return -757;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: Expect arg[2] to be pointer");
    return -758;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: Expect arg[3] to be pointer");
    return -759;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: Expect arg[4] to be pointer");
    return -760;
  }
  if (!(((((arg5_code == 3) || (arg5_code == 13)) || (arg5_code == 7)) || (arg5_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu_3: Expect arg[5] to be pointer");
    return -761;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -762;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -763;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -764;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -765;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -766;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -767;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -768;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -769;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -770;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -771;
  }
  if (!((((int32_t)arg1_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -772;
  }
  if (!((((int32_t)arg1_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -773;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -774;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -775;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -776;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -777;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -778;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -779;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -780;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -781;
  }
  if (!((((int32_t)arg2_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -782;
  }
  if (!((((int32_t)arg2_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -783;
  }
  if (!((((int32_t)arg2_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -784;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -785;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -786;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -787;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -788;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -789;
  }
  if (!((((int32_t)arg3_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -790;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -791;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -792;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -793;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -794;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -795;
  }
  if (!((3 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 3");
    return -796;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -797;
  }
  if (!((((int32_t)arg4_shape[0]) == 64))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -798;
  }
  if (!((((int32_t)arg4_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -799;
  }
  if (!((((int32_t)arg4_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -800;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -801;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -802;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -803;
  }
  if (!((4 == (((TVMArray*)arg5)[0].ndim)))) {
    TVMAPISetLastError("arg5.ndim is expected to equal 4");
    return -804;
  }
  if (!(((((((TVMArray*)arg5)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg5)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg5)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg5.dtype is expected to be float32");
    return -805;
  }
  if (!((((int32_t)arg5_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg5.shape[0] has an unsatisfied constraint");
    return -806;
  }
  if (!((((int32_t)arg5_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg5.shape[1] has an unsatisfied constraint");
    return -807;
  }
  if (!((((int32_t)arg5_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg5.shape[2] has an unsatisfied constraint");
    return -808;
  }
  if (!((((int32_t)arg5_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg5.shape[3] has an unsatisfied constraint");
    return -809;
  }
  if (!(((((TVMArray*)arg5)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg5.byte_offset has an unsatisfied constraint");
    return -810;
  }
  if (!((1 == (((TVMArray*)arg5)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg5.device_type has an unsatisfied constraint");
    return -811;
  }
  if (!((dev_id == (((TVMArray*)arg5)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg5.device_id has an unsatisfied constraint");
    return -812;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)295936, 2, 32);
  if (data_vec == NULL) {
    return -813;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)147456, 2, 32);
  if (kernel_vec == NULL) {
    return -814;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 272; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 34; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 34) + w)] = (((((1 <= (C_h_fused % 34)) && ((C_h_fused % 34) < 33)) && (1 <= w)) && (w < 33)) ? placeholder[((((((((C_h_fused / 34) * 8) + c) * 32) + (C_h_fused % 34)) * 32) + w) + -33)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 24; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 8; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 8) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 8) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
    void* conv = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1024, 2, 32);
    if (conv == NULL) {
      return -815;
    }
     float conv_global[128];
    for (int32_t ow_outer = 0; ow_outer < 2; ++ow_outer) {
      for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
        conv_global[oc_block_c_init] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
        conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
        conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
        conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
        conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
        conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
        conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
        conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
        conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
        conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
        conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
        conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
        conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
        conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
        conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
      }
      for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
        conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
      }
      for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
        for (int32_t kh = 0; kh < 3; ++kh) {
          for (int32_t kw = 0; kw < 3; ++kw) {
            for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
              for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
                conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
              }
              for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
                conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
              }
              for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
                conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
              }
              for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
                conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
              }
              for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
                conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
              }
              for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
                conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
              }
              for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
                conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
              }
              for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
                conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
              }
              for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
                conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
              }
              for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
                conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 9)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
              }
              for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
                conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
              }
              for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
                conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 11)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
              }
              for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
                conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
              }
              for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
                conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 13)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
              }
              for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
                conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
              }
              for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
                conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[(((((((((ic_outer * 34) + kh) + (ax1_outer_ax2_fused % 32)) * 8) + ic_inner) * 34) + (ow_outer * 16)) + kw) + 15)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 32) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
              }
            }
          }
        }
      }
      for (int32_t ow_inner = 0; ow_inner < 16; ++ow_inner) {
        for (int32_t oc_block = 0; oc_block < 8; ++oc_block) {
          (( float*)conv)[((((ow_outer * 16) + ow_inner) * 8) + oc_block)] = conv_global[((ow_inner * 8) + oc_block)];
        }
      }
    }
    for (int32_t ax3_outer = 0; ax3_outer < 2; ++ax3_outer) {
      for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
        for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
          T_relu[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)] = (((((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] + placeholder2[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((((( float*)conv)[((((ax3_outer * 16) + ax3_inner) * 8) + ax1_inner)] + placeholder2[(((((((((ax1_outer_ax2_fused / 32) * 8) + ax1_inner) * 32) + (ax1_outer_ax2_fused % 32)) * 2) + ax3_outer) * 16) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 32) * 8) + ax1_inner)])) : (0.000000e+00f);
        }
      }
    }
    if (TVMBackendFreeWorkspace(1, dev_id, conv) != 0) {
      return -816;
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -817;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -818;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_nn_conv2d_add: num_args should be 4");
    return -819;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (4 == ((int32_t)arg0_strides[2]))) && (16 == ((int32_t)arg0_strides[1]))) && (8192 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -820;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (4608 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -821;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (4 == ((int32_t)arg2_strides[2]))) && (16 == ((int32_t)arg2_strides[1]))) && (8192 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -822;
    }
  }
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (4 == ((int32_t)arg3_strides[2]))) && (16 == ((int32_t)arg3_strides[1]))) && (8192 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -823;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add: Expect arg[0] to be pointer");
    return -824;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add: Expect arg[1] to be pointer");
    return -825;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add: Expect arg[2] to be pointer");
    return -826;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add: Expect arg[3] to be pointer");
    return -827;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -828;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -829;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -830;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -831;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -832;
  }
  if (!((((int32_t)arg0_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -833;
  }
  if (!((((int32_t)arg0_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -834;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -835;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -836;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -837;
  }
  if (!((((int32_t)arg1_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -838;
  }
  if (!((((int32_t)arg1_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -839;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -840;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -841;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -842;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -843;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -844;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -845;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -846;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -847;
  }
  if (!((((int32_t)arg2_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -848;
  }
  if (!((((int32_t)arg2_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -849;
  }
  if (!((((int32_t)arg2_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -850;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -851;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -852;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -853;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -854;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -855;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -856;
  }
  if (!((((int32_t)arg3_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -857;
  }
  if (!((((int32_t)arg3_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -858;
  }
  if (!((((int32_t)arg3_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -859;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -860;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -861;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -862;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)73728, 2, 32);
  if (data_vec == NULL) {
    return -863;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)9437184, 2, 32);
  if (kernel_vec == NULL) {
    return -864;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 384; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 6; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 6) + w)] = (((((1 <= (C_h_fused % 6)) && ((C_h_fused % 6) < 5)) && (1 <= w)) && (w < 5)) ? placeholder[((((((((C_h_fused / 6) * 8) + c) * 4) + (C_h_fused % 6)) * 4) + w) + -5)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 192; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 64; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 64) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 64) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[32];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 64; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_add[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)] = (conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)]);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -865;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -866;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_1( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_1: num_args should be 5");
    return -867;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (8 == ((int32_t)arg0_strides[2]))) && (64 == ((int32_t)arg0_strides[1]))) && (16384 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -868;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (2304 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -869;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -870;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -871;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (4 == ((int32_t)arg4_strides[2]))) && (16 == ((int32_t)arg4_strides[1]))) && (8192 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -872;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_1: Expect arg[0] to be pointer");
    return -873;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_1: Expect arg[1] to be pointer");
    return -874;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_1: Expect arg[2] to be pointer");
    return -875;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_1: Expect arg[3] to be pointer");
    return -876;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_1: Expect arg[4] to be pointer");
    return -877;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -878;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -879;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -880;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -881;
  }
  if (!((((int32_t)arg0_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -882;
  }
  if (!((((int32_t)arg0_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -883;
  }
  if (!((((int32_t)arg0_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -884;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -885;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -886;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -887;
  }
  if (!((((int32_t)arg1_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -888;
  }
  if (!((((int32_t)arg1_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -889;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -890;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -891;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -892;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -893;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -894;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -895;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -896;
  }
  if (!((((int32_t)arg2_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -897;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -898;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -899;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -900;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -901;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -902;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -903;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -904;
  }
  if (!((((int32_t)arg3_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -905;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -906;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -907;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -908;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -909;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -910;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -911;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -912;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -913;
  }
  if (!((((int32_t)arg4_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -914;
  }
  if (!((((int32_t)arg4_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -915;
  }
  if (!((((int32_t)arg4_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -916;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -917;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -918;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -919;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)82944, 2, 32);
  if (data_vec == NULL) {
    return -920;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4718592, 2, 32);
  if (kernel_vec == NULL) {
    return -921;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 288; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 9; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 9) + w)] = ((1 <= ((C_h_fused % 9)) < (w) ? ((C_h_fused % 9)) : (w)) ? placeholder[((((((((C_h_fused / 9) * 8) + c) * 8) + (C_h_fused % 9)) * 8) + w) + -9)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 192; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 32; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 32) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 32) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[32];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 32; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((ic_outer * 648) + ((ax1_outer_ax2_fused % 4) * 144)) + (kh * 72)) + (ic_inner * 9)) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((ic_outer * 648) + ((ax1_outer_ax2_fused % 4) * 144)) + (kh * 72)) + (ic_inner * 9)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((ic_outer * 648) + ((ax1_outer_ax2_fused % 4) * 144)) + (kh * 72)) + (ic_inner * 9)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((ic_outer * 648) + ((ax1_outer_ax2_fused % 4) * 144)) + (kh * 72)) + (ic_inner * 9)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)] = (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -922;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -923;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_5( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_5: num_args should be 5");
    return -924;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (65536 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -925;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (576 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -926;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -927;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -928;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (16 == ((int32_t)arg4_strides[2]))) && (256 == ((int32_t)arg4_strides[1]))) && (32768 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -929;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_5: Expect arg[0] to be pointer");
    return -930;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_5: Expect arg[1] to be pointer");
    return -931;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_5: Expect arg[2] to be pointer");
    return -932;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_5: Expect arg[3] to be pointer");
    return -933;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_5: Expect arg[4] to be pointer");
    return -934;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -935;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -936;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -937;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -938;
  }
  if (!((((int32_t)arg0_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -939;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -940;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -941;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -942;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -943;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -944;
  }
  if (!((((int32_t)arg1_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -945;
  }
  if (!((((int32_t)arg1_shape[1]) == 64))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -946;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -947;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -948;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -949;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -950;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -951;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -952;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -953;
  }
  if (!((((int32_t)arg2_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -954;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -955;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -956;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -957;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -958;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -959;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -960;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -961;
  }
  if (!((((int32_t)arg3_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -962;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -963;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -964;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -965;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -966;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -967;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -968;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -969;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -970;
  }
  if (!((((int32_t)arg4_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -971;
  }
  if (!((((int32_t)arg4_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -972;
  }
  if (!((((int32_t)arg4_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -973;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -974;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -975;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -976;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)278784, 2, 32);
  if (data_vec == NULL) {
    return -977;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)294912, 2, 32);
  if (kernel_vec == NULL) {
    return -978;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 264; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 33; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 33) + w)] = ((1 <= ((C_h_fused % 33)) < (w) ? ((C_h_fused % 33)) : (w)) ? placeholder[((((((((C_h_fused / 33) * 8) + c) * 32) + (C_h_fused % 33)) * 32) + w) + -33)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 48; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 8; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 8) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 8) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 8; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
            for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
              conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 16)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
            }
            for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
              conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 18)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
            }
            for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
              conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 20)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
            }
            for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
              conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 22)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
            }
            for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
              conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 24)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
            }
            for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
              conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 26)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
            }
            for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
              conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 28)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
            }
            for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
              conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((((ic_outer * 8712) + ((ax1_outer_ax2_fused % 16) * 528)) + (kh * 264)) + (ic_inner * 33)) + kw) + 30)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 8) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)] = (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -979;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -980;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_3( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_3: num_args should be 5");
    return -981;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (16 == ((int32_t)arg0_strides[2]))) && (256 == ((int32_t)arg0_strides[1]))) && (32768 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -982;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (1152 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -983;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -984;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -985;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (8 == ((int32_t)arg4_strides[2]))) && (64 == ((int32_t)arg4_strides[1]))) && (16384 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -986;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_3: Expect arg[0] to be pointer");
    return -987;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_3: Expect arg[1] to be pointer");
    return -988;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_3: Expect arg[2] to be pointer");
    return -989;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_3: Expect arg[3] to be pointer");
    return -990;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_3: Expect arg[4] to be pointer");
    return -991;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -992;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -993;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -994;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -995;
  }
  if (!((((int32_t)arg0_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -996;
  }
  if (!((((int32_t)arg0_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -997;
  }
  if (!((((int32_t)arg0_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -998;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -999;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -1000;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1001;
  }
  if (!((((int32_t)arg1_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1002;
  }
  if (!((((int32_t)arg1_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1003;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1004;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -1005;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1006;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1007;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1008;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -1009;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1010;
  }
  if (!((((int32_t)arg2_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1011;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1012;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1013;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1014;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1015;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1016;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -1017;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1018;
  }
  if (!((((int32_t)arg3_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1019;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1020;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1021;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1022;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1023;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1024;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -1025;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -1026;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -1027;
  }
  if (!((((int32_t)arg4_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -1028;
  }
  if (!((((int32_t)arg4_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -1029;
  }
  if (!((((int32_t)arg4_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -1030;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -1031;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -1032;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -1033;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)147968, 2, 32);
  if (data_vec == NULL) {
    return -1034;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)1179648, 2, 32);
  if (kernel_vec == NULL) {
    return -1035;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 272; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 17; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 17) + w)] = ((1 <= ((C_h_fused % 17)) < (w) ? ((C_h_fused % 17)) : (w)) ? placeholder[((((((((C_h_fused / 17) * 8) + c) * 16) + (C_h_fused % 17)) * 16) + w) + -17)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 96; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 16; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 16) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 16) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[64];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((ic_outer * 2312) + ((ax1_outer_ax2_fused % 8) * 272)) + (kh * 136)) + (ic_inner * 17)) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)] = (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -1036;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -1037;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_2( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_2: num_args should be 5");
    return -1038;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (8 == ((int32_t)arg0_strides[2]))) && (64 == ((int32_t)arg0_strides[1]))) && (16384 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1039;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (2304 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1040;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -1041;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -1042;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (8 == ((int32_t)arg4_strides[2]))) && (64 == ((int32_t)arg4_strides[1]))) && (16384 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -1043;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_2: Expect arg[0] to be pointer");
    return -1044;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_2: Expect arg[1] to be pointer");
    return -1045;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_2: Expect arg[2] to be pointer");
    return -1046;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_2: Expect arg[3] to be pointer");
    return -1047;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_2: Expect arg[4] to be pointer");
    return -1048;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1049;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1050;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1051;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1052;
  }
  if (!((((int32_t)arg0_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1053;
  }
  if (!((((int32_t)arg0_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1054;
  }
  if (!((((int32_t)arg0_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1055;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1056;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -1057;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1058;
  }
  if (!((((int32_t)arg1_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1059;
  }
  if (!((((int32_t)arg1_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1060;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1061;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -1062;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1063;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1064;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1065;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -1066;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1067;
  }
  if (!((((int32_t)arg2_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1068;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1069;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1070;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1071;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1072;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1073;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -1074;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1075;
  }
  if (!((((int32_t)arg3_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1076;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1077;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1078;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1079;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1080;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1081;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -1082;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -1083;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -1084;
  }
  if (!((((int32_t)arg4_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -1085;
  }
  if (!((((int32_t)arg4_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -1086;
  }
  if (!((((int32_t)arg4_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -1087;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -1088;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -1089;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -1090;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)102400, 2, 32);
  if (data_vec == NULL) {
    return -1091;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)2359296, 2, 32);
  if (kernel_vec == NULL) {
    return -1092;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 320; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 10; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 10) + w)] = (((((1 <= (C_h_fused % 10)) && ((C_h_fused % 10) < 9)) && (1 <= w)) && (w < 9)) ? placeholder[((((((((C_h_fused / 10) * 8) + c) * 8) + (C_h_fused % 10)) * 8) + w) + -9)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 96; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 32; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 32) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 32) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[64];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 32; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((((ic_outer * 10) + kh) + (ax1_outer_ax2_fused % 8)) * 8) + ic_inner) * 10) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 8) * 32) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 8) * 8) + ax1_inner) * 8) + (ax1_outer_ax2_fused % 8)) * 8) + ax3_inner)] = (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 8) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -1093;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -1094;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_batch_flatten( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 2))) {
    TVMAPISetLastError("fused_nn_batch_flatten: num_args should be 2");
    return -1095;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (1 == ((int32_t)arg0_strides[2]))) && (1 == ((int32_t)arg0_strides[1]))) && (512 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1096;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* tensor = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((1 == ((int32_t)arg1_strides[1])) && (512 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1097;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_batch_flatten: Expect arg[0] to be pointer");
    return -1098;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_batch_flatten: Expect arg[1] to be pointer");
    return -1099;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1100;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1101;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1102;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1103;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1104;
  }
  if (!((((int32_t)arg0_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1105;
  }
  if (!((((int32_t)arg0_shape[3]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1106;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1107;
  }
  if (!((2 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 2");
    return -1108;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1109;
  }
  if (!((((int32_t)arg1_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1110;
  }
  if (!((((int32_t)arg1_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1111;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1112;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1113;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1114;
  }
  for (int32_t ax1 = 0; ax1 < 512; ++ax1) {
    tensor[ax1] = placeholder[ax1];
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_multiply_add_nn_relu_1( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_1: num_args should be 4");
    return -1115;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (8 == ((int32_t)arg0_strides[2]))) && (64 == ((int32_t)arg0_strides[1]))) && (16384 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1116;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((((1 == ((int32_t)arg1_strides[2])) && (1 == ((int32_t)arg1_strides[1]))) && (1 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1117;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -1118;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (8 == ((int32_t)arg3_strides[2]))) && (64 == ((int32_t)arg3_strides[1]))) && (16384 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -1119;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_1: Expect arg[0] to be pointer");
    return -1120;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_1: Expect arg[1] to be pointer");
    return -1121;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_1: Expect arg[2] to be pointer");
    return -1122;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_1: Expect arg[3] to be pointer");
    return -1123;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1124;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1125;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1126;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1127;
  }
  if (!((((int32_t)arg0_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1128;
  }
  if (!((((int32_t)arg0_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1129;
  }
  if (!((((int32_t)arg0_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1130;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1131;
  }
  if (!((3 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 3");
    return -1132;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1133;
  }
  if (!((((int32_t)arg1_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1134;
  }
  if (!((((int32_t)arg1_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1135;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1136;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1137;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1138;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1139;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -1140;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1141;
  }
  if (!((((int32_t)arg2_shape[0]) == 256))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1142;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1143;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1144;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1145;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1146;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1147;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -1148;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1149;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1150;
  }
  if (!((((int32_t)arg3_shape[1]) == 256))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1151;
  }
  if (!((((int32_t)arg3_shape[2]) == 8))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1152;
  }
  if (!((((int32_t)arg3_shape[3]) == 8))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -1153;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1154;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1155;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1156;
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 256; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 8; ++ax3) {
        T_relu[((((ax0_ax1_fused * 8) + ax2) * 8) + ax3)] = (((placeholder[((((ax0_ax1_fused * 8) + ax2) * 8) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) > (0.000000e+00f) ? (((placeholder[((((ax0_ax1_fused * 8) + ax2) * 8) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) : (0.000000e+00f);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_add_multiply_add_nn_relu( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 6))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: num_args should be 6");
    return -1157;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  void* arg5 = (((TVMValue*)args)[5].v_handle);
  int32_t arg5_code = (( int32_t*)arg_type_ids)[5];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (4 == ((int32_t)arg0_strides[2]))) && (16 == ((int32_t)arg0_strides[1]))) && (8192 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1158;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (4608 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1159;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg2_strides[3])) && (4 == ((int32_t)arg2_strides[2]))) && (16 == ((int32_t)arg2_strides[1]))) && (8192 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -1160;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -1161;
    }
  }
  float* placeholder4 = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!((((1 == ((int32_t)arg4_strides[2])) && (1 == ((int32_t)arg4_strides[1]))) && (1 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -1162;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg5)[0].data);
  int64_t* arg5_shape = (int64_t*)(((TVMArray*)arg5)[0].shape);
  int64_t* arg5_strides = (int64_t*)(((TVMArray*)arg5)[0].strides);
  if (!(arg5_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg5_strides[3])) && (4 == ((int32_t)arg5_strides[2]))) && (16 == ((int32_t)arg5_strides[1]))) && (8192 == ((int32_t)arg5_strides[0]))))) {
      TVMAPISetLastError("arg5.strides: expected to be compact array");
      return -1163;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: Expect arg[0] to be pointer");
    return -1164;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: Expect arg[1] to be pointer");
    return -1165;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: Expect arg[2] to be pointer");
    return -1166;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: Expect arg[3] to be pointer");
    return -1167;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: Expect arg[4] to be pointer");
    return -1168;
  }
  if (!(((((arg5_code == 3) || (arg5_code == 13)) || (arg5_code == 7)) || (arg5_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_add_multiply_add_nn_relu: Expect arg[5] to be pointer");
    return -1169;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1170;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1171;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1172;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1173;
  }
  if (!((((int32_t)arg0_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1174;
  }
  if (!((((int32_t)arg0_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1175;
  }
  if (!((((int32_t)arg0_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1176;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1177;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -1178;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1179;
  }
  if (!((((int32_t)arg1_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1180;
  }
  if (!((((int32_t)arg1_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1181;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1182;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -1183;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1184;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1185;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1186;
  }
  if (!((4 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 4");
    return -1187;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1188;
  }
  if (!((((int32_t)arg2_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1189;
  }
  if (!((((int32_t)arg2_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1190;
  }
  if (!((((int32_t)arg2_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1191;
  }
  if (!((((int32_t)arg2_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg2.shape[3] has an unsatisfied constraint");
    return -1192;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1193;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1194;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1195;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -1196;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1197;
  }
  if (!((((int32_t)arg3_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1198;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1199;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1200;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1201;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1202;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1203;
  }
  if (!((3 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 3");
    return -1204;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -1205;
  }
  if (!((((int32_t)arg4_shape[0]) == 512))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -1206;
  }
  if (!((((int32_t)arg4_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -1207;
  }
  if (!((((int32_t)arg4_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -1208;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -1209;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -1210;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -1211;
  }
  if (!((4 == (((TVMArray*)arg5)[0].ndim)))) {
    TVMAPISetLastError("arg5.ndim is expected to equal 4");
    return -1212;
  }
  if (!(((((((TVMArray*)arg5)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg5)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg5)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg5.dtype is expected to be float32");
    return -1213;
  }
  if (!((((int32_t)arg5_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg5.shape[0] has an unsatisfied constraint");
    return -1214;
  }
  if (!((((int32_t)arg5_shape[1]) == 512))) {
    TVMAPISetLastError("Argument arg5.shape[1] has an unsatisfied constraint");
    return -1215;
  }
  if (!((((int32_t)arg5_shape[2]) == 4))) {
    TVMAPISetLastError("Argument arg5.shape[2] has an unsatisfied constraint");
    return -1216;
  }
  if (!((((int32_t)arg5_shape[3]) == 4))) {
    TVMAPISetLastError("Argument arg5.shape[3] has an unsatisfied constraint");
    return -1217;
  }
  if (!(((((TVMArray*)arg5)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg5.byte_offset has an unsatisfied constraint");
    return -1218;
  }
  if (!((1 == (((TVMArray*)arg5)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg5.device_type has an unsatisfied constraint");
    return -1219;
  }
  if (!((dev_id == (((TVMArray*)arg5)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg5.device_id has an unsatisfied constraint");
    return -1220;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)73728, 2, 32);
  if (data_vec == NULL) {
    return -1221;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)9437184, 2, 32);
  if (kernel_vec == NULL) {
    return -1222;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 384; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 6; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 6) + w)] = (((((1 <= (C_h_fused % 6)) && ((C_h_fused % 6) < 5)) && (1 <= w)) && (w < 5)) ? placeholder[((((((((C_h_fused / 6) * 8) + c) * 4) + (C_h_fused % 6)) * 4) + w) + -5)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 192; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 64; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 64) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 64) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[32];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 64; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 6) + kh) + (ax1_outer_ax2_fused % 4)) * 8) + ic_inner) * 6) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 4) * 64) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 4; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)] = ((((conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)])) > (0.000000e+00f) ? ((((conv_global[((ax3_inner * 8) + ax1_inner)] + placeholder2[(((((((ax1_outer_ax2_fused / 4) * 8) + ax1_inner) * 4) + (ax1_outer_ax2_fused % 4)) * 4) + ax3_inner)]) * placeholder3[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)]) + placeholder4[(((ax1_outer_ax2_fused / 4) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -1223;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -1224;
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_multiply_add_nn_relu_2( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_2: num_args should be 4");
    return -1225;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (16 == ((int32_t)arg0_strides[2]))) && (256 == ((int32_t)arg0_strides[1]))) && (32768 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1226;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((((1 == ((int32_t)arg1_strides[2])) && (1 == ((int32_t)arg1_strides[1]))) && (1 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1227;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -1228;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (16 == ((int32_t)arg3_strides[2]))) && (256 == ((int32_t)arg3_strides[1]))) && (32768 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -1229;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_2: Expect arg[0] to be pointer");
    return -1230;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_2: Expect arg[1] to be pointer");
    return -1231;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_2: Expect arg[2] to be pointer");
    return -1232;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add_nn_relu_2: Expect arg[3] to be pointer");
    return -1233;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1234;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1235;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1236;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1237;
  }
  if (!((((int32_t)arg0_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1238;
  }
  if (!((((int32_t)arg0_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1239;
  }
  if (!((((int32_t)arg0_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1240;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1241;
  }
  if (!((3 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 3");
    return -1242;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1243;
  }
  if (!((((int32_t)arg1_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1244;
  }
  if (!((((int32_t)arg1_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1245;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1246;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1247;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1248;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1249;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -1250;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1251;
  }
  if (!((((int32_t)arg2_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1252;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1253;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1254;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1255;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1256;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1257;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -1258;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1259;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1260;
  }
  if (!((((int32_t)arg3_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1261;
  }
  if (!((((int32_t)arg3_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1262;
  }
  if (!((((int32_t)arg3_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -1263;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1264;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1265;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1266;
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 128; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 16; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 16; ++ax3) {
        T_relu[((((ax0_ax1_fused * 16) + ax2) * 16) + ax3)] = (((placeholder[((((ax0_ax1_fused * 16) + ax2) * 16) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) > (0.000000e+00f) ? (((placeholder[((((ax0_ax1_fused * 16) + ax2) * 16) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused])) : (0.000000e+00f);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_multiply_add( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 4))) {
    TVMAPISetLastError("fused_multiply_add: num_args should be 4");
    return -1267;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (32 == ((int32_t)arg0_strides[2]))) && (1024 == ((int32_t)arg0_strides[1]))) && (3072 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1268;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((((1 == ((int32_t)arg1_strides[2])) && (1 == ((int32_t)arg1_strides[1]))) && (1 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1269;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -1270;
    }
  }
  float* T_add = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg3_strides[3])) && (32 == ((int32_t)arg3_strides[2]))) && (1024 == ((int32_t)arg3_strides[1]))) && (3072 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -1271;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add: Expect arg[0] to be pointer");
    return -1272;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add: Expect arg[1] to be pointer");
    return -1273;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add: Expect arg[2] to be pointer");
    return -1274;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_multiply_add: Expect arg[3] to be pointer");
    return -1275;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1276;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1277;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1278;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1279;
  }
  if (!((((int32_t)arg0_shape[1]) == 3))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1280;
  }
  if (!((((int32_t)arg0_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1281;
  }
  if (!((((int32_t)arg0_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1282;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1283;
  }
  if (!((3 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 3");
    return -1284;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1285;
  }
  if (!((((int32_t)arg1_shape[0]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1286;
  }
  if (!((((int32_t)arg1_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1287;
  }
  if (!((((int32_t)arg1_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1288;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1289;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1290;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1291;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -1292;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1293;
  }
  if (!((((int32_t)arg2_shape[0]) == 3))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1294;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1295;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1296;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1297;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1298;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1299;
  }
  if (!((4 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 4");
    return -1300;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1301;
  }
  if (!((((int32_t)arg3_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1302;
  }
  if (!((((int32_t)arg3_shape[1]) == 3))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1303;
  }
  if (!((((int32_t)arg3_shape[2]) == 32))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1304;
  }
  if (!((((int32_t)arg3_shape[3]) == 32))) {
    TVMAPISetLastError("Argument arg3.shape[3] has an unsatisfied constraint");
    return -1305;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1306;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1307;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1308;
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 3; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 32; ++ax2) {
      for (int32_t ax3 = 0; ax3 < 32; ++ax3) {
        T_add[((((ax0_ax1_fused * 32) + ax2) * 32) + ax3)] = ((placeholder[((((ax0_ax1_fused * 32) + ax2) * 32) + ax3)] * placeholder1[ax0_ax1_fused]) + placeholder2[ax0_ax1_fused]);
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_conv2d_multiply_add_nn_relu_4( void* args,  void* arg_type_ids, int32_t num_args) {
  if (!((num_args == 5))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_4: num_args should be 5");
    return -1309;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  void* arg3 = (((TVMValue*)args)[3].v_handle);
  int32_t arg3_code = (( int32_t*)arg_type_ids)[3];
  void* arg4 = (((TVMValue*)args)[4].v_handle);
  int32_t arg4_code = (( int32_t*)arg_type_ids)[4];
  float* placeholder = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg0_strides[3])) && (16 == ((int32_t)arg0_strides[2]))) && (256 == ((int32_t)arg0_strides[1]))) && (32768 == ((int32_t)arg0_strides[0]))))) {
      TVMAPISetLastError("arg0.strides: expected to be compact array");
      return -1310;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* placeholder1 = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg1_strides[3])) && (3 == ((int32_t)arg1_strides[2]))) && (9 == ((int32_t)arg1_strides[1]))) && (1152 == ((int32_t)arg1_strides[0]))))) {
      TVMAPISetLastError("arg1.strides: expected to be compact array");
      return -1311;
    }
  }
  float* placeholder2 = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((((1 == ((int32_t)arg2_strides[2])) && (1 == ((int32_t)arg2_strides[1]))) && (1 == ((int32_t)arg2_strides[0]))))) {
      TVMAPISetLastError("arg2.strides: expected to be compact array");
      return -1312;
    }
  }
  float* placeholder3 = (float*)(((TVMArray*)arg3)[0].data);
  int64_t* arg3_shape = (int64_t*)(((TVMArray*)arg3)[0].shape);
  int64_t* arg3_strides = (int64_t*)(((TVMArray*)arg3)[0].strides);
  if (!(arg3_strides == NULL)) {
    if (!((((1 == ((int32_t)arg3_strides[2])) && (1 == ((int32_t)arg3_strides[1]))) && (1 == ((int32_t)arg3_strides[0]))))) {
      TVMAPISetLastError("arg3.strides: expected to be compact array");
      return -1313;
    }
  }
  float* T_relu = (float*)(((TVMArray*)arg4)[0].data);
  int64_t* arg4_shape = (int64_t*)(((TVMArray*)arg4)[0].shape);
  int64_t* arg4_strides = (int64_t*)(((TVMArray*)arg4)[0].strides);
  if (!(arg4_strides == NULL)) {
    if (!(((((1 == ((int32_t)arg4_strides[3])) && (16 == ((int32_t)arg4_strides[2]))) && (256 == ((int32_t)arg4_strides[1]))) && (32768 == ((int32_t)arg4_strides[0]))))) {
      TVMAPISetLastError("arg4.strides: expected to be compact array");
      return -1314;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_4: Expect arg[0] to be pointer");
    return -1315;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_4: Expect arg[1] to be pointer");
    return -1316;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_4: Expect arg[2] to be pointer");
    return -1317;
  }
  if (!(((((arg3_code == 3) || (arg3_code == 13)) || (arg3_code == 7)) || (arg3_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_4: Expect arg[3] to be pointer");
    return -1318;
  }
  if (!(((((arg4_code == 3) || (arg4_code == 13)) || (arg4_code == 7)) || (arg4_code == 4)))) {
    TVMAPISetLastError("fused_nn_conv2d_multiply_add_nn_relu_4: Expect arg[4] to be pointer");
    return -1319;
  }
  if (!((dev_type == 1))) {
    TVMAPISetLastError("device_type need to be 1");
    return -1320;
  }
  if (!((4 == (((TVMArray*)arg0)[0].ndim)))) {
    TVMAPISetLastError("arg0.ndim is expected to equal 4");
    return -1321;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg0.dtype is expected to be float32");
    return -1322;
  }
  if (!((((int32_t)arg0_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg0.shape[0] has an unsatisfied constraint");
    return -1323;
  }
  if (!((((int32_t)arg0_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg0.shape[1] has an unsatisfied constraint");
    return -1324;
  }
  if (!((((int32_t)arg0_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[2] has an unsatisfied constraint");
    return -1325;
  }
  if (!((((int32_t)arg0_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg0.shape[3] has an unsatisfied constraint");
    return -1326;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg0.byte_offset has an unsatisfied constraint");
    return -1327;
  }
  if (!((4 == (((TVMArray*)arg1)[0].ndim)))) {
    TVMAPISetLastError("arg1.ndim is expected to equal 4");
    return -1328;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg1.dtype is expected to be float32");
    return -1329;
  }
  if (!((((int32_t)arg1_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[0] has an unsatisfied constraint");
    return -1330;
  }
  if (!((((int32_t)arg1_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg1.shape[1] has an unsatisfied constraint");
    return -1331;
  }
  if (!((((int32_t)arg1_shape[2]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[2] has an unsatisfied constraint");
    return -1332;
  }
  if (!((((int32_t)arg1_shape[3]) == 3))) {
    TVMAPISetLastError("Argument arg1.shape[3] has an unsatisfied constraint");
    return -1333;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg1.byte_offset has an unsatisfied constraint");
    return -1334;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg1.device_type has an unsatisfied constraint");
    return -1335;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg1.device_id has an unsatisfied constraint");
    return -1336;
  }
  if (!((3 == (((TVMArray*)arg2)[0].ndim)))) {
    TVMAPISetLastError("arg2.ndim is expected to equal 3");
    return -1337;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg2.dtype is expected to be float32");
    return -1338;
  }
  if (!((((int32_t)arg2_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg2.shape[0] has an unsatisfied constraint");
    return -1339;
  }
  if (!((((int32_t)arg2_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[1] has an unsatisfied constraint");
    return -1340;
  }
  if (!((((int32_t)arg2_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg2.shape[2] has an unsatisfied constraint");
    return -1341;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg2.byte_offset has an unsatisfied constraint");
    return -1342;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg2.device_type has an unsatisfied constraint");
    return -1343;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg2.device_id has an unsatisfied constraint");
    return -1344;
  }
  if (!((3 == (((TVMArray*)arg3)[0].ndim)))) {
    TVMAPISetLastError("arg3.ndim is expected to equal 3");
    return -1345;
  }
  if (!(((((((TVMArray*)arg3)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg3)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg3)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg3.dtype is expected to be float32");
    return -1346;
  }
  if (!((((int32_t)arg3_shape[0]) == 128))) {
    TVMAPISetLastError("Argument arg3.shape[0] has an unsatisfied constraint");
    return -1347;
  }
  if (!((((int32_t)arg3_shape[1]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[1] has an unsatisfied constraint");
    return -1348;
  }
  if (!((((int32_t)arg3_shape[2]) == 1))) {
    TVMAPISetLastError("Argument arg3.shape[2] has an unsatisfied constraint");
    return -1349;
  }
  if (!(((((TVMArray*)arg3)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg3.byte_offset has an unsatisfied constraint");
    return -1350;
  }
  if (!((1 == (((TVMArray*)arg3)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg3.device_type has an unsatisfied constraint");
    return -1351;
  }
  if (!((dev_id == (((TVMArray*)arg3)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg3.device_id has an unsatisfied constraint");
    return -1352;
  }
  if (!((4 == (((TVMArray*)arg4)[0].ndim)))) {
    TVMAPISetLastError("arg4.ndim is expected to equal 4");
    return -1353;
  }
  if (!(((((((TVMArray*)arg4)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg4)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg4)[0].dtype.lanes) == (uint16_t)1)))) {
    TVMAPISetLastError("arg4.dtype is expected to be float32");
    return -1354;
  }
  if (!((((int32_t)arg4_shape[0]) == 1))) {
    TVMAPISetLastError("Argument arg4.shape[0] has an unsatisfied constraint");
    return -1355;
  }
  if (!((((int32_t)arg4_shape[1]) == 128))) {
    TVMAPISetLastError("Argument arg4.shape[1] has an unsatisfied constraint");
    return -1356;
  }
  if (!((((int32_t)arg4_shape[2]) == 16))) {
    TVMAPISetLastError("Argument arg4.shape[2] has an unsatisfied constraint");
    return -1357;
  }
  if (!((((int32_t)arg4_shape[3]) == 16))) {
    TVMAPISetLastError("Argument arg4.shape[3] has an unsatisfied constraint");
    return -1358;
  }
  if (!(((((TVMArray*)arg4)[0].byte_offset) == (uint64_t)0))) {
    TVMAPISetLastError("Argument arg4.byte_offset has an unsatisfied constraint");
    return -1359;
  }
  if (!((1 == (((TVMArray*)arg4)[0].ctx.device_type)))) {
    TVMAPISetLastError("Argument arg4.device_type has an unsatisfied constraint");
    return -1360;
  }
  if (!((dev_id == (((TVMArray*)arg4)[0].ctx.device_id)))) {
    TVMAPISetLastError("Argument arg4.device_id has an unsatisfied constraint");
    return -1361;
  }
  void* data_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)165888, 2, 32);
  if (data_vec == NULL) {
    return -1362;
  }
  void* kernel_vec = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)589824, 2, 32);
  if (kernel_vec == NULL) {
    return -1363;
  }
  for (int32_t C_h_fused = 0; C_h_fused < 288; ++C_h_fused) {
    for (int32_t c = 0; c < 8; ++c) {
      for (int32_t w = 0; w < 18; ++w) {
        (( float*)data_vec)[((((C_h_fused * 8) + c) * 18) + w)] = (((((1 <= (C_h_fused % 18)) && ((C_h_fused % 18) < 17)) && (1 <= w)) && (w < 17)) ? placeholder[((((((((C_h_fused / 18) * 8) + c) * 16) + (C_h_fused % 18)) * 16) + w) + -17)] : 0.000000e+00f);
      }
    }
  }
  for (int32_t CO_h_fused = 0; CO_h_fused < 48; ++CO_h_fused) {
    for (int32_t CI = 0; CI < 16; ++CI) {
      for (int32_t w1 = 0; w1 < 3; ++w1) {
        for (int32_t ci = 0; ci < 8; ++ci) {
          for (int32_t co = 0; co < 8; ++co) {
            (( float*)kernel_vec)[(((((((((((CO_h_fused / 3) * 16) + CI) * 3) + (CO_h_fused % 3)) * 3) + w1) * 8) + ci) * 8) + co)] = placeholder1[(((((((((((CO_h_fused / 3) * 8) + co) * 16) + CI) * 8) + ci) * 3) + (CO_h_fused % 3)) * 3) + w1)];
          }
        }
      }
    }
  }
  for (int32_t ax1_outer_ax2_fused = 0; ax1_outer_ax2_fused < 256; ++ax1_outer_ax2_fused) {
     float conv_global[128];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 8; ++oc_block_c_init) {
      conv_global[oc_block_c_init] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 8; ++oc_block_c_init1) {
      conv_global[(oc_block_c_init1 + 8)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 8; ++oc_block_c_init2) {
      conv_global[(oc_block_c_init2 + 16)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 8; ++oc_block_c_init3) {
      conv_global[(oc_block_c_init3 + 24)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 8; ++oc_block_c_init4) {
      conv_global[(oc_block_c_init4 + 32)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 8; ++oc_block_c_init5) {
      conv_global[(oc_block_c_init5 + 40)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 8; ++oc_block_c_init6) {
      conv_global[(oc_block_c_init6 + 48)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 8; ++oc_block_c_init7) {
      conv_global[(oc_block_c_init7 + 56)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 8; ++oc_block_c_init8) {
      conv_global[(oc_block_c_init8 + 64)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 8; ++oc_block_c_init9) {
      conv_global[(oc_block_c_init9 + 72)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 8; ++oc_block_c_init10) {
      conv_global[(oc_block_c_init10 + 80)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 8; ++oc_block_c_init11) {
      conv_global[(oc_block_c_init11 + 88)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 8; ++oc_block_c_init12) {
      conv_global[(oc_block_c_init12 + 96)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 8; ++oc_block_c_init13) {
      conv_global[(oc_block_c_init13 + 104)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init14 = 0; oc_block_c_init14 < 8; ++oc_block_c_init14) {
      conv_global[(oc_block_c_init14 + 112)] = 0.000000e+00f;
    }
    for (int32_t oc_block_c_init15 = 0; oc_block_c_init15 < 8; ++oc_block_c_init15) {
      conv_global[(oc_block_c_init15 + 120)] = 0.000000e+00f;
    }
    for (int32_t ic_outer = 0; ic_outer < 16; ++ic_outer) {
      for (int32_t kh = 0; kh < 3; ++kh) {
        for (int32_t kw = 0; kw < 3; ++kw) {
          for (int32_t ic_inner = 0; ic_inner < 8; ++ic_inner) {
            for (int32_t oc_block_c = 0; oc_block_c < 8; ++oc_block_c) {
              conv_global[oc_block_c] = (conv_global[oc_block_c] + ((( float*)data_vec)[(((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c)]));
            }
            for (int32_t oc_block_c1 = 0; oc_block_c1 < 8; ++oc_block_c1) {
              conv_global[(oc_block_c1 + 8)] = (conv_global[(oc_block_c1 + 8)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 1)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c1)]));
            }
            for (int32_t oc_block_c2 = 0; oc_block_c2 < 8; ++oc_block_c2) {
              conv_global[(oc_block_c2 + 16)] = (conv_global[(oc_block_c2 + 16)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 2)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c2)]));
            }
            for (int32_t oc_block_c3 = 0; oc_block_c3 < 8; ++oc_block_c3) {
              conv_global[(oc_block_c3 + 24)] = (conv_global[(oc_block_c3 + 24)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 3)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c3)]));
            }
            for (int32_t oc_block_c4 = 0; oc_block_c4 < 8; ++oc_block_c4) {
              conv_global[(oc_block_c4 + 32)] = (conv_global[(oc_block_c4 + 32)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 4)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c4)]));
            }
            for (int32_t oc_block_c5 = 0; oc_block_c5 < 8; ++oc_block_c5) {
              conv_global[(oc_block_c5 + 40)] = (conv_global[(oc_block_c5 + 40)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 5)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c5)]));
            }
            for (int32_t oc_block_c6 = 0; oc_block_c6 < 8; ++oc_block_c6) {
              conv_global[(oc_block_c6 + 48)] = (conv_global[(oc_block_c6 + 48)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 6)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c6)]));
            }
            for (int32_t oc_block_c7 = 0; oc_block_c7 < 8; ++oc_block_c7) {
              conv_global[(oc_block_c7 + 56)] = (conv_global[(oc_block_c7 + 56)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 7)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c7)]));
            }
            for (int32_t oc_block_c8 = 0; oc_block_c8 < 8; ++oc_block_c8) {
              conv_global[(oc_block_c8 + 64)] = (conv_global[(oc_block_c8 + 64)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 8)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c8)]));
            }
            for (int32_t oc_block_c9 = 0; oc_block_c9 < 8; ++oc_block_c9) {
              conv_global[(oc_block_c9 + 72)] = (conv_global[(oc_block_c9 + 72)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 9)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c9)]));
            }
            for (int32_t oc_block_c10 = 0; oc_block_c10 < 8; ++oc_block_c10) {
              conv_global[(oc_block_c10 + 80)] = (conv_global[(oc_block_c10 + 80)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 10)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c10)]));
            }
            for (int32_t oc_block_c11 = 0; oc_block_c11 < 8; ++oc_block_c11) {
              conv_global[(oc_block_c11 + 88)] = (conv_global[(oc_block_c11 + 88)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 11)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c11)]));
            }
            for (int32_t oc_block_c12 = 0; oc_block_c12 < 8; ++oc_block_c12) {
              conv_global[(oc_block_c12 + 96)] = (conv_global[(oc_block_c12 + 96)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 12)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c12)]));
            }
            for (int32_t oc_block_c13 = 0; oc_block_c13 < 8; ++oc_block_c13) {
              conv_global[(oc_block_c13 + 104)] = (conv_global[(oc_block_c13 + 104)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 13)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c13)]));
            }
            for (int32_t oc_block_c14 = 0; oc_block_c14 < 8; ++oc_block_c14) {
              conv_global[(oc_block_c14 + 112)] = (conv_global[(oc_block_c14 + 112)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 14)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c14)]));
            }
            for (int32_t oc_block_c15 = 0; oc_block_c15 < 8; ++oc_block_c15) {
              conv_global[(oc_block_c15 + 120)] = (conv_global[(oc_block_c15 + 120)] + ((( float*)data_vec)[((((((((ic_outer * 18) + kh) + (ax1_outer_ax2_fused % 16)) * 8) + ic_inner) * 18) + kw) + 15)] * (( float*)kernel_vec)[(((((((((((ax1_outer_ax2_fused / 16) * 16) + ic_outer) * 3) + kh) * 3) + kw) * 8) + ic_inner) * 8) + oc_block_c15)]));
            }
          }
        }
      }
    }
    for (int32_t ax3_inner = 0; ax3_inner < 16; ++ax3_inner) {
      for (int32_t ax1_inner = 0; ax1_inner < 8; ++ax1_inner) {
        T_relu[(((((((ax1_outer_ax2_fused / 16) * 8) + ax1_inner) * 16) + (ax1_outer_ax2_fused % 16)) * 16) + ax3_inner)] = (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)])) > (0.000000e+00f) ? (((conv_global[((ax3_inner * 8) + ax1_inner)] * placeholder2[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)]) + placeholder3[(((ax1_outer_ax2_fused / 16) * 8) + ax1_inner)])) : (0.000000e+00f);
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, kernel_vec) != 0) {
    return -1364;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, data_vec) != 0) {
    return -1365;
  }
  return 0;
}

