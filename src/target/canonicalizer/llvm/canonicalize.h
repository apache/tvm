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
 * \file tvm/target/canonicalizer/llvm/canonicalize.h
 * \brief Target canonicalizer for LLVM-based CPU targets
 */

#ifndef TVM_TARGET_CANONICALIZER_LLVM_CANONICALIZE_H_
#define TVM_TARGET_CANONICALIZER_LLVM_CANONICALIZE_H_

#include <tvm/target/target.h>

namespace tvm {
namespace target {
namespace canonicalizer {
namespace llvm {

ffi::Map<ffi::String, ffi::Any> Canonicalize(ffi::Map<ffi::String, ffi::Any> target);

}  // namespace llvm
}  // namespace canonicalizer
}  // namespace target
}  // namespace tvm

#endif  // TVM_TARGET_CANONICALIZER_LLVM_CANONICALIZE_H_
