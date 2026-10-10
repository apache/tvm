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
#ifndef TVM_SCRIPT_IR_BUILDER_UTILS_H_
#define TVM_SCRIPT_IR_BUILDER_UTILS_H_

#include <tvm/script/ir_builder/frame.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

inline IRModuleFrame FindModuleFrame(const ffi::String& method) {
  IRBuilder builder = IRBuilder::Current();
  if (ffi::Optional<IRModuleFrame> frame = builder->FindFrame<IRModuleFrame>()) {
    const ffi::Optional<IRModuleFrame>& last_module_frame = builder->GetLastFrame<IRModuleFrame>();
    if (last_module_frame.has_value() && last_module_frame.value() == frame.value()) {
      return frame.value();
    }
  } else {
    TVM_FFI_THROW(ValueError) << "IRModule frame not find. Please ensure '" << method
                              << "' is called under I.ir_module()";
  }
  TVM_FFI_THROW(ValueError) << "'" << method << "' must be called immediately under I.ir_module()";
  throw;
}

inline IRModuleFrame FindModuleFrame() {
  IRBuilder builder = IRBuilder::Current();
  if (ffi::Optional<IRModuleFrame> frame = builder->FindFrame<IRModuleFrame>()) {
    return frame.value();
  } else {
    TVM_FFI_THROW(ValueError) << "IRModule frame not find. Please ensure it"
                              << " is called under I.ir_module()";
  }
  throw;
}

/*!
 * \brief Add a core IR statement to the top frame in IRBuilder frame stack.
 * \param stmt The Stmt.
 * \param loc The stored result location, which may be undefined.
 */
inline void AddToParent(tvm::Stmt stmt, Location loc) {
  IRBuilder builder = IRBuilder::Current();
  // A deferred frame owns its location even when that location is undefined.
  // Preserve an existing body location when flattening returns the body itself.
  if (stmt.defined() && stmt->loc.as<UnknownLocNode>()) stmt->loc = std::move(loc);
  if (builder->frames.empty()) {
    if (!builder->result.has_value()) {
      if (stmt.as<tvm::SeqStmtNode>()) {
        auto normalized = tvm::SeqStmt(stmt);
        if (normalized->loc.as<UnknownLocNode>()) normalized->loc = stmt->loc;
        builder->result = std::move(normalized);
      } else {
        builder->result = std::move(stmt);
      }
      return;
    }
    TVM_FFI_CHECK(builder->result.as<tvm::StmtNode>(), ValueError)
        << "Builder.result has already been set";
    ffi::Array<tvm::Stmt> incoming = tvm::SeqStmt(stmt)->seq;
    if (incoming.empty()) return;
    if (builder->result.as<tvm::SeqStmtNode>()) {
      // Move the builder's ownership so unobserved results can grow in place.
      // Copy-on-write preserves sequences and arrays retained by callers.
      auto sequence = std::move(builder->result).value().as_or_throw<tvm::SeqStmt>();
      auto* node = sequence.CopyOnWrite();
      for (const auto& child : incoming) node->seq.push_back(child);
      builder->result = std::move(sequence);
    } else {
      builder->result =
          tvm::SeqStmt({builder->result.value().as_or_throw<tvm::Stmt>(), tvm::SeqStmt(incoming)});
    }
  } else if (const auto* stmt_frame = builder->frames.back().as<StmtFrameNode>()) {
    ffi::GetRef<StmtFrame>(stmt_frame)->stmts.push_back(stmt);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported frame type: " << builder->frames.back();
  }
}

/*! \brief Add an eager statement under the current source-call context. */
inline void AddToParent(tvm::Stmt stmt) {
  // Some builder paths use an undefined statement as an omitted branch.
  if (stmt.defined()) IRBuilder::Current()->SetCurrentLoc(stmt);
  AddToParent(std::move(stmt), Location());
}

/*!
 * \brief Convert array of core IR statements to single Stmt.
 * \param stmt The array of Stmt.
 * \return The SeqStmt.
 */
inline tvm::SeqStmt AsStmt(const ffi::Array<tvm::Stmt>& stmt) { return tvm::SeqStmt(stmt); }

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
                              << "If frame";
  } else {
    TVM_FFI_THROW(ValueError) << "If frame not find. Please ensure '" << method
                              << "' is called under T.if_()";
  }
  throw;
}

/*!
 * \brief Validate a user-requested loop / scope-id var dtype.
 * \note Only scalar int32 and uint32 are supported.
 */
inline void CheckExplicitIndexDtype(const PrimType& dtype) {
  TVM_FFI_ICHECK(dtype.IsScalar() && dtype.bits() == 32 &&
                 dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
      << "ValueError: dtype of a loop/scope-id var must be \"int32\" or \"uint32\", got " << dtype;
}

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_UTILS_H_
