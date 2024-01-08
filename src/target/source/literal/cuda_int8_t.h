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
 * \file cuda_int8_t.h
 * \brief Extra int8 intrisic for cuda codegen.
 */
#ifndef TVM_TARGET_SOURCE_LITERAL_CUDA_INT8_T_H_
#define TVM_TARGET_SOURCE_LITERAL_CUDA_INT8_T_H_

static constexpr const char* _cuda_int8_t_def = R"(

#if defined(__CUDACC_RTC__)
#define __SM_61_INTRINSICS_DECL__ __device__
#else /* !__CUDACC_RTC__ */
#define __SM_61_INTRINSICS_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */

#ifndef __CUDA_ARCH__
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ */

__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) __DEF_IF_HOST
__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) __DEF_IF_HOST

#undef __DEF_IF_HOST

#if !defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)
__SM_61_INTRINSICS_DECL__ int __dp4a(unsigned int srcA, int srcB, int c) {
    int ret;
    asm volatile ("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}

__SM_61_INTRINSICS_DECL__ int __dp4a(int srcA, unsigned int srcB, int c) {
    int ret;
    asm volatile ("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(srcA), "r"(srcB), "r"(c));
    return ret;
}
#endif /* !__CUDACC_RTC__ && defined(__CUDA_ARCH__) */

#undef __SM_61_INTRINSICS_DECL__

)";

#endif  // TVM_TARGET_SOURCE_LITERAL_CUDA_INT8_T_H_
