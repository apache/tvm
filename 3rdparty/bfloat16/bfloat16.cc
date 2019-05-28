/*
    Copyright (c) 2019 by Contributors
   \file tvm/src/codegen/custom_datatypes/mybfloat16.cc
   \brief Small bfloat16 library for use in unittests

  Code originally from TensorFlow; taken and simplified. Original license:

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include <tvm/runtime/c_runtime_api.h>
#include <cstddef>
#include <cstdint>

void FloatToBFloat16(const float* src, uint16_t* dst, size_t size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p += 2, q++, size--) {
    *q = p[0];
  }
#else
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
  }
#endif
}

void BFloat16ToFloat(const uint16_t* src, float* dst, size_t size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p++, q += 2, size--) {
    q[0] = *p;
    q[1] = 0;
  }
#else
  for (; size != 0; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
#endif
}

void BFloat16Add(const uint16_t* a, const uint16_t* b, uint16_t* dst,
                 size_t size) {
  float a_f, b_f;
  BFloat16ToFloat(a, &a_f, 1);
  BFloat16ToFloat(b, &b_f, 1);
  float out_f = a_f + b_f;
  FloatToBFloat16(&out_f, dst, 1);
}

extern "C" {
TVM_DLL TVM_DLL uint16_t FloatToBFloat16_wrapper(float in) {
  uint16_t out;
  FloatToBFloat16(&in, &out, 1);
  return out;
}

TVM_DLL float BFloat16ToFloat_wrapper(uint16_t in) {
  float out;
  BFloat16ToFloat(&in, &out, 1);
  return out;
}

TVM_DLL uint16_t BFloat16Add_wrapper(uint16_t a, uint16_t b) {
  uint16_t out;
  BFloat16Add(&a, &b, &out, 1);
  return out;
}
}
