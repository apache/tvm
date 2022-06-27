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
 * \file codegen_params.h
 */

#ifndef TVM_TARGET_LLVM_CODEGEN_PARAMS_H_
#define TVM_TARGET_LLVM_CODEGEN_PARAMS_H_

#include <tvm/runtime/ndarray.h>

namespace llvm {
class ConstantArray;
class LLVMContext;
}  // namespace llvm

namespace tvm {
namespace codegen {

/*!
 * \brief Convert an NDArray to an LLVM array of constants.
 *
 * The supplied NDArray is flattened, and each element is converted to the appropriate LLVM type.
 *
 * \param ctx LLVM context used to create the various primitive datatypes.
 * \param arr NDArray to convert.
 * \return LLVM array containing the array data.
 */
llvm::ConstantArray* NDArrayToLLVMArray(llvm::LLVMContext* ctx, tvm::runtime::NDArray arr);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_LLVM_CODEGEN_PARAMS_H_
