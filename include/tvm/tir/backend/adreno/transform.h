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
 * \file tvm/tir/backend/adreno/transform.h
 * \brief TIR specific Adreno GPU transformation passes.
 */
#ifndef TVM_TIR_BACKEND_ADRENO_TRANSFORM_H_
#define TVM_TIR_BACKEND_ADRENO_TRANSFORM_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>

#include <string>
#include <vector>

namespace tvm {
namespace tir {
namespace backend {
namespace adreno {
namespace transform {

using tvm::tir::transform::CreatePrimFuncPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
using tvm::transform::PassContextNode;
using tvm::transform::PassInfo;
using tvm::transform::PassInfoNode;
using tvm::transform::PassNode;
using tvm::transform::Sequential;

/*!
 * \brief Texture flattening pass.
 * \return The pass.
 */
TVM_DLL Pass TextureFlatten();

/*!
 * \brief Inject Texture Allocation intrinsic.
 * \return The pass.
 */
TVM_DLL Pass InjectTextureAlloc();

}  // namespace transform
}  // namespace adreno
}  // namespace backend
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_BACKEND_ADRENO_TRANSFORM_H_
