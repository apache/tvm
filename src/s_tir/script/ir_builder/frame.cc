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
#include <tvm/s_tir/script/ir_builder/frame.h>
#include <tvm/tirx/function.h>

#include "./script_complete.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {
TVM_FFI_STATIC_INIT_BLOCK() {
  FunctionFrameNode::RegisterReflection();
  SBlockFrameNode::RegisterReflection();
  BlockInitFrameNode::RegisterReflection();
  tirx::FunctionFrameNode::RegisterAttrValidator(
      tvm::attr::kSTir, [](const tirx::FunctionFrameNode* frame, const ffi::Any& value) {
        TVM_FFI_CHECK(frame->IsInstance<FunctionFrameNode>(), ValueError)
            << "The s_tir attribute requires Ts.function";
        bool enabled = false;
        if (auto flag = value.as<bool>()) {
          enabled = *flag;
        } else if (auto flag = value.as<int64_t>()) {
          enabled = *flag != 0;
        } else if (auto flag = value.as<tvm::IntImm>()) {
          enabled = !tvm::prim::IsZero(*flag);
        }
        TVM_FFI_CHECK(enabled, ValueError) << "Ts.function cannot disable the s_tir attribute";
      });
}

tvm::tirx::Function FunctionFrameNode::FinalizeFunction(tvm::tirx::Function func) {
  TVM_FFI_CHECK(!is_declaration || root_alloc_buffers.empty(), ValueError)
      << "A function declaration cannot allocate buffers";
  if (!is_declaration) {
    func = WithAttr(std::move(func), tvm::attr::kSTir, true);
    func = tvm::s_tir::ScriptComplete(std::move(func), root_alloc_buffers);
  }
  return func;
}

void SBlockFrameNode::BindBufferRegion(tvm::tirx::TensorVar buffer, tvm::TensorRegion region) {
  match_buffers.push_back(tvm::s_tir::MatchBufferRegion(std::move(buffer), std::move(region)));
}

void SBlockFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();

  // Shared operations remain usable in S-TIR and raw builder contexts, but
  // a TIRx function cannot contain an S-TIR block, even with validation disabled.
  if (auto function = IRBuilder::Current()->FindFrame<tirx::FunctionFrame>()) {
    TVM_FFI_CHECK(function.value().as<FunctionFrameNode>() != nullptr, ValueError)
        << "S-TIR blocks require Ts.function; T.function only accepts TIRx";
  }

  ffi::Array<tvm::tirx::TensorVar> tir_alloc_buffers;
  for (const tvm::tirx::TensorVar& buffer : alloc_buffers) {
    tir_alloc_buffers.push_back(buffer);
  }
  ffi::Map<ffi::String, Any> attrs = annotations.value_or({});
  if (int detect_access = (!reads.has_value()) | (!writes.has_value() << 1)) {
    attrs.Set("tirx.script_parsing_detect_access", tvm::IntImm::Int64(detect_access));
  }
  tvm::s_tir::SBlock block(iter_vars, reads.value_or(ffi::Array<tvm::TensorRegion>()),
                           writes.value_or(ffi::Array<tvm::TensorRegion>()), name, AsStmt(stmts),
                           init, tir_alloc_buffers, match_buffers, attrs, source_span);
  if (no_realize) {
    TVM_FFI_CHECK(iter_values.empty(), ValueError)
        << "Block bindings are not allowed when `no_realize=True`";
    TVM_FFI_CHECK(!predicate.has_value(), ValueError)
        << "`Ts.where` is not allowed when `no_realize=True`";
    AddToParent(block, source_span);
  } else {
    AddToParent(tvm::s_tir::SBlockRealize(iter_values, predicate.value_or(IntImm::Bool(true)),
                                          block, source_span),
                source_span);
  }
}

void BlockInitFrameNode::EnterWithScope() {
  SBlockFrame frame = FindSBlockFrame("Ts.init");
  if (frame->init.has_value()) {
    TVM_FFI_THROW(ValueError) << "Duplicate block init declaration";
  }
  TIRFrameNode::EnterWithScope();
}

void BlockInitFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  SBlockFrame frame = FindSBlockFrame("Ts.init");
  frame->init = AsStmt(stmts);
}

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
