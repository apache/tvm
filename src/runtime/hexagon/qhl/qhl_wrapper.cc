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
#if defined(__hexagon__)
#include <hexagon_types.h>
#include <stdio.h>
#include <tvm/runtime/logging.h>

#define restrict __restrict__
#define LOG2VLEN 7

// QHL functions with 1 input arg
#define TVM_QHL_WRAPPER_DECL_1IP(NAME) HVX_Vector tvm_vect_##NAME(HVX_Vector input);

// QHL functions with 2 input args
#define TVM_QHL_WRAPPER_DECL_2IP(NAME) HVX_Vector tvm_vect_##NAME(HVX_Vector ip1, HVX_Vector ip2);

#define TVM_QHL_WRAPPER_AHF_1IP(NAME) \
  HVX_Vector tvm_vect_##NAME(HVX_Vector input) { return wrapper_api<__fp16>(input, NAME, #NAME); }

#define TVM_QHL_WRAPPER_AHF_2IP(NAME)                          \
  HVX_Vector tvm_vect_##NAME(HVX_Vector ip1, HVX_Vector ip2) { \
    return wrapper_api<__fp16>(ip1, ip2, NAME, #NAME);         \
  }

extern "C" {
#include "hvx_internal.h"
#include "qhmath_hvx.h"
#include "qhmath_hvx_vector.h"
using qhlFptr = int (*)(__fp16*, __fp16*, uint32_t);
using qhlFptr2 = int (*)(__fp16*, __fp16*, __fp16*, uint32_t);
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_ceil_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_cos_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_exp_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_floor_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_sin_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_sigmoid_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_sqrt_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_tan_ahf)
TVM_QHL_WRAPPER_DECL_1IP(qhmath_hvx_tanh_ahf)

// QHL functions with 2 input args
TVM_QHL_WRAPPER_DECL_2IP(qhmath_hvx_pow_ahf)
}
template <typename T>
HVX_Vector wrapper_api(HVX_Vector input, qhlFptr qhl_api, const char* qhl_api_name) {
  HVX_Vector output;
  int32_t res = qhl_api(reinterpret_cast<T*>(&input), reinterpret_cast<T*>(&output), 64);
  if (res != 0) LOG(FATAL) << "Error. Failed execution of " << qhl_api_name << "  Error=" << res;
  return output;
}

template <typename T>
HVX_Vector wrapper_api(HVX_Vector ip1, HVX_Vector ip2, qhlFptr2 qhl_api, const char* qhl_api_name) {
  HVX_Vector output;
  int32_t res = qhl_api(reinterpret_cast<T*>(&ip1), reinterpret_cast<T*>(&ip2),
                        reinterpret_cast<T*>(&output), 64);
  if (res != 0) LOG(FATAL) << "Error. Failed execution of " << qhl_api_name << "Error=" << res;
  return output;
}

TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_ceil_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_cos_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_exp_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_floor_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_sin_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_sigmoid_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_sqrt_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_tan_ahf);
TVM_QHL_WRAPPER_AHF_1IP(qhmath_hvx_tanh_ahf);

TVM_QHL_WRAPPER_AHF_2IP(qhmath_hvx_pow_ahf);

#endif
