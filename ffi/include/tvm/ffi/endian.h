
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

/*
 * \file tvm/ffi/endian.h
 * \brief Endian detection and handling
 */
#ifndef TVM_FFI_ENDIAN_H_
#define TVM_FFI_ENDIAN_H_

#include <cstddef>
#include <cstdint>

#ifndef TVM_FFI_IO_USE_LITTLE_ENDIAN
#define TVM_FFI_IO_USE_LITTLE_ENDIAN 1
#endif

#ifdef TVM_FFI_CMAKE_LITTLE_ENDIAN
// If compiled with CMake, use CMake's endian detection logic
#define TVM_FFI_LITTLE_ENDIAN TVM_FFI_CMAKE_LITTLE_ENDIAN
#else
#if defined(__APPLE__) || defined(_WIN32)
#define TVM_FFI_LITTLE_ENDIAN 1
#elif defined(__GLIBC__) || defined(__GNU_LIBRARY__) || defined(__ANDROID__) || defined(__RISCV__)
#include <endian.h>
#define TVM_FFI_LITTLE_ENDIAN (__BYTE_ORDER == __LITTLE_ENDIAN)
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/endian.h>
#define TVM_FFI_LITTLE_ENDIAN (_BYTE_ORDER == _LITTLE_ENDIAN)
#elif defined(__QNX__)
#include <sys/param.h>
#define TVM_FFI_LITTLE_ENDIAN (BYTE_ORDER == LITTLE_ENDIAN)
#elif defined(__EMSCRIPTEN__) || defined(__hexagon__)
#define TVM_FFI_LITTLE_ENDIAN 1
#elif defined(__sun) || defined(sun)
#include <sys/isa_defs.h>
#if defined(_LITTLE_ENDIAN)
#define TVM_FFI_LITTLE_ENDIAN 1
#else
#define TVM_FFI_LITTLE_ENDIAN 0
#endif
#else
#error "Unable to determine endianness of your machine; use CMake to compile"
#endif
#endif

/*! \brief whether serialize using little endian */
#define TVM_FFI_IO_NO_ENDIAN_SWAP (TVM_FFI_LITTLE_ENDIAN == TVM_FFI_IO_USE_LITTLE_ENDIAN)

namespace tvm {
namespace ffi {
/*!
 * \brief A generic inplace byte swapping function.
 * \param data The data pointer.
 * \param elem_bytes The number of bytes of the data elements
 * \param num_elems Number of elements in the data.
 * \note Always try pass in constant elem_bytes to enable
 *       compiler optimization
 */
inline void ByteSwap(void* data, size_t elem_bytes, size_t num_elems) {
  for (size_t i = 0; i < num_elems; ++i) {
    uint8_t* bptr = reinterpret_cast<uint8_t*>(data) + elem_bytes * i;
    for (size_t j = 0; j < elem_bytes / 2; ++j) {
      uint8_t v = bptr[elem_bytes - 1 - j];
      bptr[elem_bytes - 1 - j] = bptr[j];
      bptr[j] = v;
    }
  }
}
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_ENDIAN_H_
