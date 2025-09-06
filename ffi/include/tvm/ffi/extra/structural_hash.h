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
 * \file tvm/ffi/extra/structural_hash.h
 * \brief Structural hash
 */
#ifndef TVM_FFI_EXTRA_STRUCTURAL_HASH_H_
#define TVM_FFI_EXTRA_STRUCTURAL_HASH_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/extra/base.h>

namespace tvm {
namespace ffi {

/*
 * \brief Structural hash
 */
class StructuralHash {
 public:
  /*!
   * \brief Hash an Any value.
   * \param value The Any value to hash.
   * \param map_free_vars Whether to map free variables.
   * \param skip_tensor_content Whether to skip comparingn darray data content,
   *                             useful for cases where we don't care about parameters content.
   * \return The hash value.
   */
  TVM_FFI_EXTRA_CXX_API static uint64_t Hash(const Any& value, bool map_free_vars = false,
                                             bool skip_tensor_content = false);
  /*!
   * \brief Hash an Any value.
   * \param value The Any value to hash.
   * \return The hash value.
   */
  TVM_FFI_INLINE uint64_t operator()(const Any& value) const { return Hash(value); }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_STRUCTURAL_HASH_H_
