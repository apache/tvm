/*!
 * Copyright (c) 2018 by Contributors
 * \file builtin_fp16.cc
 * \brief Functions for conversion between fp32 and fp16
*/
#include <builtin_fp16.h>
#include <tvm/runtime/c_runtime_api.h>

extern "C" {

// disable under msvc
#ifndef _MSC_VER

TVM_WEAK uint16_t __gnu_f2h_ieee(float a) {
  return __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(a);
}

TVM_WEAK float __gnu_h2f_ieee(uint16_t a) {
  return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

#endif
}
