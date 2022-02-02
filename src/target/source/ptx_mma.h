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
 * \file ptx_mma.h
 * \brief MMA code generation with inlined PTX code.
 */
#ifndef TVM_TARGET_SOURCE_PTX_MMA_H_
#define TVM_TARGET_SOURCE_PTX_MMA_H_

#include <tvm/runtime/logging.h>

#include <string>
#include <tuple>

namespace tvm {
namespace codegen {

std::string PrintMMAAssembly(const std::string& shape, const std::string& A_layout,
                             const std::string& B_layout, const std::string& A_dtype,
                             const std::string& B_dtype, const std::string& C_dtype,
                             const std::string& a_ref, const std::string& a_bias,
                             const std::string& b_ref, const std::string& b_bias,
                             const std::string& c_ref, const std::string& c_bias, bool saturate);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_PTX_MMA_H_
