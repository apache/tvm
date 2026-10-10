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
#ifndef TVM_TIRX_SCRIPT_IR_BUILDER_UTILS_H_
#define TVM_TIRX_SCRIPT_IR_BUILDER_UTILS_H_

#include <tvm/ffi/cast.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/script/ir_builder/frame.h>
#include <tvm/tirx/script/ir_builder/ir.h>
#include <tvm/tirx/stmt.h>

#include "../../../script/ir_builder/utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

using ir::AddToParent;
using ir::AsStmt;

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is FunctionFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of FunctionFrame.
 */
inline FunctionFrame FindFunctionFrame(const ffi::String& method) {
  if (ffi::Optional<FunctionFrame> frame = IRBuilder::Current()->GetLastFrame<FunctionFrame>()) {
    return frame.value();
  } else if (ffi::Optional<FunctionFrame> frame =
                 IRBuilder::Current()->FindFrame<FunctionFrame>()) {
    TVM_FFI_THROW(ValueError)
        << method << " must be called at the top of a Function.  "
        << "While " << method << " did occur within the Function \"" << frame.value()->name
        << "\", other frames (e.g. block/if/else/let) had been introduced since the "
        << "Function's frame";
  } else {
    TVM_FFI_THROW(ValueError) << method << " must be called at the top of a Function, "
                              << "but " << method << " occurred outside of any T.function() frame";
  }
  throw;
}

/*!
 * \brief Convert TensorLoad to TensorRegion.
 * \param buffer_load The TensorLoad.
 * \return The converted TensorRegion.
 */
inline tvm::TensorRegion TensorRegionFromLoad(tvm::TensorLoad buffer_load) {
  return ffi::TypeTraits<tvm::TensorRegion>::ConvertFallbackValue(buffer_load);
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIRX_SCRIPT_IR_BUILDER_UTILS_H_
