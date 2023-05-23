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

/*!
 * \file builtin_fp16.cc
 * \brief Functions for conversion between fp32 and fp16
 */
#include <builtin_fp16.h>
#include <tvm/runtime/c_runtime_api.h>

extern "C" {

// disable under msvc
#ifndef _MSC_VER

TVM_DLL TVM_WEAK uint16_t __gnu_f2h_ieee(float a) {
  return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(a);
}

TVM_DLL TVM_WEAK float __gnu_h2f_ieee(uint16_t a) {
  return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

TVM_DLL uint16_t __truncdfhf2(double a) {
  return __truncXfYf2__<double, uint64_t, 52, uint16_t, uint16_t, 10>(a);
}

#else

TVM_DLL uint16_t __gnu_f2h_ieee(float a) {
  return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(a);
}

TVM_DLL float __gnu_h2f_ieee(uint16_t a) {
  return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

TVM_DLL uint16_t __truncdfhf2(double a) {
  return __truncXfYf2__<double, uint64_t, 52, uint16_t, uint16_t, 10>(a);
}

#endif
}
