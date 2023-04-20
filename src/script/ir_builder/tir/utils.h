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
#ifndef TVM_SCRIPT_IR_BUILDER_TIR_UTILS_H_
#define TVM_SCRIPT_IR_BUILDER_TIR_UTILS_H_

#include <tvm/script/ir_builder/tir/frame.h>
#include <tvm/script/ir_builder/tir/ir.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

/*!
 * \brief Add tir Stmt to the top frame in IRBuilder frame stack.
 * \param stmt The Stmt.
 */
inline void AddToParent(tvm::tir::Stmt stmt) {
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    ICHECK(!builder->result.defined()) << "ValueError: Builder.result has already been set";
    builder->result = stmt;
  } else if (const auto* tir_frame = builder->frames.back().as<TIRFrameNode>()) {
    GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
  } else {
    LOG(FATAL) << "TypeError: Unsupported frame type: " << builder->frames.back();
  }
}

/*!
 * \brief Convert array of tir Stmt to single Stmt.
 * \param stmt The array of Stmt.
 * \return The SeqStmt.
 */
inline tvm::tir::Stmt AsStmt(const Array<tvm::tir::Stmt>& stmt) {
  return tvm::tir::SeqStmt::Flatten(stmt);
}

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is PrimFuncFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of PrimFuncFrame.
 */
inline PrimFuncFrame FindPrimFuncFrame(const String& method) {
  if (Optional<PrimFuncFrame> frame = IRBuilder::Current()->GetLastFrame<PrimFuncFrame>()) {
    return frame.value();
  } else if (Optional<PrimFuncFrame> frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>()) {
    LOG(FATAL) << "ValueError: " << method << " must be called at the top of a PrimFunc.  "
               << "While " << method << " did occur within the PrimFunc \"" << frame.value()->name
               << "\", other frames (e.g. block/if/else/let) had been introduced since the "
               << "PrimFunc's frame";
  } else {
    LOG(FATAL) << "ValueError: " << method << " must be called at the top of a PrimFunc, "
               << "but " << method << " occurred outside of any T.prim_func() frame";
  }
  throw;
}

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is BlockFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of BlockFrame.
 */
inline BlockFrame FindBlockFrame(const String& method) {
  if (Optional<BlockFrame> frame = IRBuilder::Current()->FindFrame<BlockFrame>()) {
    return frame.value();
  } else if (Optional<BlockFrame> frame = IRBuilder::Current()->FindFrame<BlockFrame>()) {
    LOG(FATAL) << "ValueError: " << method << " must be called at the top of a T.block().  "
               << "While " << method << " did occur within the block \"" << frame.value()->name
               << "\", other frames (e.g. if/else/let) had been introduced since the T.block(\""
               << frame.value()->name << "\") frame";
  } else {
    LOG(FATAL) << "ValueError: " << method << " must be called at the top of a T.block(), "
               << "but " << method << " occurred outside of any T.block() frame";
  }
  throw;
}

/*!
 * \brief Check whether the top frame in IRBuilder frame stack is IfFrame.
 * \param method The method name to be printed when throwing exception.
 * \return The top frame of IfFrame.
 */
inline IfFrame FindIfFrame(const String& method) {
  if (Optional<IfFrame> frame = IRBuilder::Current()->GetLastFrame<IfFrame>()) {
    return frame.value();
  } else if (Optional<IfFrame> frame = IRBuilder::Current()->FindFrame<IfFrame>()) {
    LOG(FATAL) << "ValueError: " << method << " must be called at the top of a T.if_().  "
               << "While " << method << " did occur within the conditional based on ("
               << frame.value()->condition
               << "), other frames (e.g. if/else/let) had been introduced since the "
               << "IfThenElse frame";
  } else {
    LOG(FATAL) << "ValueError: IfThenElse frame not find. Please ensure '" << method
               << "' is called under T.if_()";
  }
  throw;
}

/*!
 * \brief Convert BufferLoad to BufferRegion.
 * \param buffer_load The BufferLoad.
 * \return The converted BufferRegion.
 */
inline tvm::tir::BufferRegion BufferRegionFromLoad(tvm::tir::BufferLoad buffer_load) {
  Array<Range> ranges;
  for (const PrimExpr& index : buffer_load->indices) {
    ranges.push_back(Range::FromMinExtent(index, IntImm(index->dtype, 1)));
  }
  return tvm::tir::BufferRegion(buffer_load->buffer, ranges);
}

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_TIR_UTILS_H_
