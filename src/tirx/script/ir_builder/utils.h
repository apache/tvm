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

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

/*!
 * \brief Add tirx Stmt to the top frame in IRBuilder frame stack.
 * \param stmt The Stmt.
 * \param span The stored result location, which may be undefined.
 */
inline void AddToParent(tvm::tirx::Stmt stmt, Span span) {
  IRBuilder builder = IRBuilder::Current();
  // A deferred frame owns its location even when that location is undefined.
  // Preserve an existing body location when flattening returns the body itself.
  if (stmt.defined() && !stmt->span.defined()) stmt->span = std::move(span);
  if (builder->frames.empty()) {
    if (!builder->result.has_value()) {
      if (stmt.as<tvm::tirx::SeqStmtNode>()) {
        auto normalized = tvm::tirx::SeqStmt(stmt);
        if (!normalized->span.defined()) normalized->span = stmt->span;
        builder->result = std::move(normalized);
      } else {
        builder->result = std::move(stmt);
      }
      return;
    }
    TVM_FFI_CHECK(builder->result.as<tvm::tirx::StmtNode>(), ValueError)
        << "Builder.result has already been set";
    ffi::Array<tvm::tirx::Stmt> incoming = tvm::tirx::SeqStmt(stmt)->seq;
    if (incoming.empty()) return;
    if (builder->result.as<tvm::tirx::SeqStmtNode>()) {
      // Move the builder's ownership so unobserved results can grow in place.
      // Copy-on-write preserves sequences and arrays retained by callers.
      auto sequence = std::move(builder->result).value().as_or_throw<tvm::tirx::SeqStmt>();
      auto* node = sequence.CopyOnWrite();
      for (const auto& child : incoming) node->seq.push_back(child);
      builder->result = std::move(sequence);
    } else {
      builder->result = tvm::tirx::SeqStmt(
          {builder->result.value().as_or_throw<tvm::tirx::Stmt>(), tvm::tirx::SeqStmt(incoming)});
    }
  } else if (const auto* tir_frame = builder->frames.back().as<TIRFrameNode>()) {
    ffi::GetRef<TIRFrame>(tir_frame)->stmts.push_back(stmt);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported frame type: " << builder->frames.back();
  }
}

/*! \brief Add an eager statement under the current source-call context. */
inline void AddToParent(tvm::tirx::Stmt stmt) {
  // Some builder paths use an undefined statement as an omitted branch.
  if (stmt.defined()) IRBuilder::Current()->SetCurrentSourceSpan(stmt);
  AddToParent(std::move(stmt), Span());
}

/*!
 * \brief Convert array of tirx Stmt to single Stmt.
 * \param stmt The array of Stmt.
 * \return The SeqStmt.
 */
inline tvm::tirx::SeqStmt AsStmt(const ffi::Array<tvm::tirx::Stmt>& stmt) {
  return tvm::tirx::SeqStmt(stmt);
}

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
 * \brief Convert TensorLoad to TensorRegion.
 * \param buffer_load The TensorLoad.
 * \return The converted TensorRegion.
 */
inline tvm::TensorRegion TensorRegionFromLoad(tvm::TensorLoad buffer_load) {
  ffi::Array<Range> ranges;
  for (const PrimExpr& index : buffer_load->indices) {
    ranges.push_back(Range::FromMinExtent(index, IntImm(index.ty(), 1)));
  }
  return tvm::tirx::BufferRegion(buffer_load->source.as_or_throw<tvm::tirx::TensorVar>(), ranges);
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_TIRX_SCRIPT_IR_BUILDER_UTILS_H_
