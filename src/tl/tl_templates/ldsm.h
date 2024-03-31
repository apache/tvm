#pragma once

#include "common.h"

namespace tl {

TL_DEVICE void ptx_ldmatrix(void const* const smem_ptr, int32_t& value) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(value)
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_trans(void const* const smem_ptr, int32_t& value) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(value)
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_stmatrix(void const* const smem_ptr, const int32_t& value) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(smem_int_ptr),
               "r"(value));
}

TL_DEVICE void ptx_stmatrix_trans(void const* const smem_ptr, const int32_t& value) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(smem_int_ptr),
               "r"(value));
}

}  // namespace tl