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
#ifndef TVM_TIRX_SCRIPT_BUILDER_UTILS_H_
#define TVM_TIRX_SCRIPT_BUILDER_UTILS_H_

#include <tvm/ffi/cast.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/script/builder/frame.h>
#include <tvm/tirx/script/builder/ir.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

/*!
 * \brief Add tirx Stmt to the top frame in IRBuilder frame stack.
 * \param stmt The Stmt.
 */
inline void AddToParent(tvm::tirx::Stmt stmt) {
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    TVM_FFI_CHECK(!builder->result.defined(), ValueError) << "Builder.result has already been set";
    builder->result = stmt;
  } else if (const auto* tir_frame = builder->frames.back().as<TIRFrameNode>()) {
    ffi::GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported frame type: " << builder->frames.back();
  }
}

/*!
 * \brief Convert array of tirx Stmt to single Stmt.
 * \param stmt The array of Stmt.
 * \return The SeqStmt.
 */
inline tvm::tirx::Stmt AsStmt(const ffi::Array<tvm::tirx::Stmt>& stmt) {
  return tvm::tirx::SeqStmt::Flatten(stmt);
}

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is PrimFuncFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of PrimFuncFrame.
 */
inline PrimFuncFrame FindPrimFuncFrame(const ffi::String& method) {
  if (ffi::Optional<PrimFuncFrame> frame = IRBuilder::Current()->GetLastFrame<PrimFuncFrame>()) {
    return frame.value();
  } else if (ffi::Optional<PrimFuncFrame> frame =
                 IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    TVM_FFI_THROW(ValueError)
        << method << " must be called at the top of a PrimFunc.  "
        << "While " << method << " did occur within the PrimFunc \"" << frame.value()->name
        << "\", other frames (e.g. block/if/else/let) had been introduced since the "
        << "PrimFunc's frame";
  } else {
    TVM_FFI_THROW(ValueError) << method << " must be called at the top of a PrimFunc, "
                              << "but " << method << " occurred outside of any T.prim_func() frame";
  }
  throw;
}

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is SBlockFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of SBlockFrame.
 */
inline SBlockFrame FindSBlockFrame(const ffi::String& method) {
  if (ffi::Optional<SBlockFrame> frame = IRBuilder::Current()->FindFrame<SBlockFrame>()) {
    return frame.value();
  } else if (ffi::Optional<SBlockFrame> frame = IRBuilder::Current()->FindFrame<SBlockFrame>()) {
    TVM_FFI_THROW(ValueError)
        << method << " must be called at the top of a T.sblock().  "
        << "While " << method << " did occur within the block \"" << frame.value()->name
        << "\", other frames (e.g. if/else/let) had been introduced since the T.sblock(\""
        << frame.value()->name << "\") frame";
  } else {
    TVM_FFI_THROW(ValueError) << method << " must be called at the top of a T.sblock(), "
                              << "but " << method << " occurred outside of any T.sblock() frame";
  }
  throw;
}

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is IfFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of IfFrame.
 */
inline IfFrame FindIfFrame(const ffi::String& method) {
  if (ffi::Optional<IfFrame> frame = IRBuilder::Current()->GetLastFrame<IfFrame>()) {
    return frame.value();
  } else if (ffi::Optional<IfFrame> frame = IRBuilder::Current()->FindFrame<IfFrame>()) {
    TVM_FFI_THROW(ValueError) << method << " must be called at the top of a T.if_().  "
                              << "While " << method
                              << " did occur within the conditional based on ("
                              << frame.value()->condition
                              << "), other frames (e.g. if/else/let) had been introduced since the "
                              << "IfThenElse frame";
  } else {
    TVM_FFI_THROW(ValueError) << "IfThenElse frame not find. Please ensure '" << method
                              << "' is called under T.if_()";
  }
  throw;
}

/*!
 * \brief Convert BufferLoad to BufferRegion.
 * \param buffer_load The BufferLoad.
 * \return The converted BufferRegion.
 */
inline tvm::tirx::BufferRegion BufferRegionFromLoad(tvm::tirx::BufferLoad buffer_load) {
  ffi::Array<Range> ranges;
  for (const PrimExpr& index : buffer_load->indices) {
    ranges.push_back(Range::FromMinExtent(index, IntImm(index->dtype, 1)));
  }
  return tvm::tirx::BufferRegion(buffer_load->buffer, ranges);
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIRX_SCRIPT_BUILDER_UTILS_H_
