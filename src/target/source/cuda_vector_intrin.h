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
 * \file cuda_vector_intrin.h
 * \brief Code generation with vector intrinsics in CUDA.
 */
#ifndef TVM_TARGET_SOURCE_CUDA_VECTOR_INTRIN_H_
#define TVM_TARGET_SOURCE_CUDA_VECTOR_INTRIN_H_

#include <string>

namespace tvm {
namespace codegen {
namespace cuda {

std::string PrintHalf2BinaryOp(const std::string& op, const std::string lhs, const std::string rhs);

std::string PrintNVBFloat162BinaryOp(const std::string& op, const std::string lhs,
                                     const std::string rhs);

}  // namespace cuda
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CUDA_VECTOR_INTRIN_H_