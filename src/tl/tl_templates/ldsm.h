#pragma once

#include "common.h"

namespace tl {

TL_DEVICE void ptx_ldmatrix_x1(void const* const smem_ptr, void* const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t* value = reinterpret_cast<int32_t*>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(value[0])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x2(void const* const smem_ptr, void* const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t* value = reinterpret_cast<int32_t*>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(value[0]), "=r"(value[1])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x4(void const* const smem_ptr, void* const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t* value = reinterpret_cast<int32_t*>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x1_trans(void const* const smem_ptr, void* const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t* value = reinterpret_cast<int32_t*>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(value[0])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x2_trans(void const* const smem_ptr, void* const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t* value = reinterpret_cast<int32_t*>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(value[0]), "=r"(value[1])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x4_trans(void const* const smem_ptr, void* const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t* value = reinterpret_cast<int32_t*>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_stmatrix_x1(void const* const smem_ptr, const int32_t& value0) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(smem_int_ptr),
               "r"(value0));
}

TL_DEVICE void ptx_stmatrix_x2(void const* const smem_ptr, const int32_t& value0,
                               const int32_t& value1) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(smem_int_ptr),
               "r"(value0), "r"(value1));
}

TL_DEVICE void ptx_stmatrix_x4(void const* const smem_ptr, const int32_t& value0,
                               const int32_t& value1, const int32_t& value2,
                               const int32_t& value3) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_int_ptr),
      "r"(value0), "r"(value1), "r"(value2), "r"(value3));
}

TL_DEVICE void ptx_stmatrix_x1_trans(void const* const smem_ptr, const int32_t& value0) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(smem_int_ptr),
               "r"(value0));
}

TL_DEVICE void ptx_stmatrix_x2_trans(void const* const smem_ptr, const int32_t& value0,
                                     const int32_t& value1) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(smem_int_ptr),
      "r"(value0), "r"(value1));
}

TL_DEVICE void ptx_stmatrix_x4_trans(void const* const smem_ptr, const int32_t& value0,
                                     const int32_t& value1, const int32_t& value2,
                                     const int32_t& value3) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(
                   smem_int_ptr),
               "r"(value0), "r"(value1), "r"(value2), "r"(value3));
}

}  // namespace tl