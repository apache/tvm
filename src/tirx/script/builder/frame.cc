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
#include <tvm/runtime/logging.h>
#include <tvm/script/ir_builder/ir/ir.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/script/builder/frame.h>
#include <tvm/tirx/stmt_functor.h>

#include "../../../tirx/ir/script/script_complete.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tirx {

namespace {

// In s_tir functions, buffer-typed parameters must not carry a layout (the
// s_tir IR doesn't track per-buffer layouts on params). When `T.Buffer(...)` is
// used as a parameter annotation, the parser evaluates the annotation outside
// the PrimFunc frame; if the annotation captures an outer-scope variable (e.g.
// `dtype` in a closure-based generator), the evaluation happens *before*
// `_current_s_tir()` becomes true, so the resulting Buffer is built with the
// default tile layout instead of None. Direct annotations using only literals
// are re-evaluated inside the frame and correctly get layout=None.
//
// This normalizer runs at PrimFunc construction time: it strips any defined
// layout from buffers in `buffer_map` / `root_alloc_buffers` and rewrites
// matching body references through the StmtExprMutator's built-in
// `buffer_remap_` machinery, so the body remains well-formed.
class STirBufferLayoutNormalizer : public tvm::tirx::StmtExprMutator {
 public:
  void Register(const tvm::tirx::Buffer& old_buf, const tvm::tirx::Buffer& new_buf) {
    this->buffer_remap_.Set(old_buf, new_buf);
  }
  bool Empty() const { return this->buffer_remap_.empty(); }
  tvm::tirx::Buffer Lookup(const tvm::tirx::Buffer& buf) const {
    auto it = this->buffer_remap_.find(buf);
    if (it != this->buffer_remap_.end()) {
      return (*it).second;
    }
    return buf;
  }
};

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  TIRFrameNode::RegisterReflection();
  PrimFuncFrameNode::RegisterReflection();
  SBlockFrameNode::RegisterReflection();
  ExecScopeFrameNode::RegisterReflection();
  BlockInitFrameNode::RegisterReflection();
  ForFrameNode::RegisterReflection();
  AssertFrameNode::RegisterReflection();
  LaunchThreadFrameNode::RegisterReflection();
  AttrFrameNode::RegisterReflection();
  WhileFrameNode::RegisterReflection();
  IfFrameNode::RegisterReflection();
  ThenFrameNode::RegisterReflection();
  ElseFrameNode::RegisterReflection();
  ComposeOpFrameNode::RegisterReflection();
  DeclBufferFrameNode::RegisterReflection();
  AllocBufferFrameNode::RegisterReflection();
  HintFrameNode::RegisterReflection();
}

void PrimFuncFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  // if the prim func is not private and there isn't already a global symbol,
  // add a global symbol
  auto insert_attr = [&](ffi::String key, ffi::Any value) {
    if (!attrs.defined()) {
      attrs = {{key, value}};
    } else if (!attrs.count(key)) {
      // copy over attributes (can't mutate the dict inside the optional in-place)
      ffi::Map<ffi::String, ffi::Any> new_attrs;
      for (auto kv : attrs) {
        new_attrs.Set(kv.first, kv.second);
      }
      new_attrs.Set(key, value);
      attrs = std::move(new_attrs);
    }
  };
  if (!is_private && name.has_value() && !attrs.count(tvm::attr::kGlobalSymbol)) {
    insert_attr(tvm::attr::kGlobalSymbol, name.value());
  }
  if (s_tir) {
    insert_attr(tvm::attr::kSTir, true);
  }
  if (persistent) {
    insert_attr(tvm::tirx::attr::kPersistentKernel, true);
  }
  // s_tir-mode normalization: drop stale default layouts (see comment on
  // STirBufferLayoutNormalizer above) and rewrite body references coherently.
  ffi::Map<tvm::tirx::Var, tvm::tirx::Buffer> effective_buffer_map = buffer_map;
  ffi::Array<tvm::tirx::Buffer> effective_root_alloc_buffers = root_alloc_buffers;
  tvm::tirx::Stmt body = AsStmt(stmts);
  if (s_tir) {
    STirBufferLayoutNormalizer normalizer;
    ffi::Map<tvm::tirx::Var, tvm::tirx::Buffer> new_buffer_map;
    for (const auto& kv : buffer_map) {
      tvm::tirx::Buffer buf = kv.second;
      if (buf->layout.has_value()) {
        tvm::tirx::Buffer new_buf = buf;
        new_buf.CopyOnWrite()->layout = std::nullopt;
        normalizer.Register(buf, new_buf);
        new_buffer_map.Set(kv.first, new_buf);
      } else {
        new_buffer_map.Set(kv.first, buf);
      }
    }
    if (!normalizer.Empty()) {
      body = normalizer(std::move(body));
      ffi::Array<tvm::tirx::Buffer> new_root_alloc_buffers;
      for (const tvm::tirx::Buffer& buf : root_alloc_buffers) {
        new_root_alloc_buffers.push_back(normalizer.Lookup(buf));
      }
      effective_buffer_map = std::move(new_buffer_map);
      effective_root_alloc_buffers = std::move(new_root_alloc_buffers);
    }
  }
  tvm::tirx::PrimFunc func(
      /*params=*/args,
      /*body=*/body,
      /*ret_type=*/ret_type.value_or(TupleType::Empty()),
      /*buffer_map=*/effective_buffer_map,
      /*attrs=*/attrs.defined() ? DictAttrs(attrs) : DictAttrs(),
      /*span=*/tvm::Span());
  func = tvm::tirx::ScriptComplete(func, effective_root_alloc_buffers, s_tir);
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    TVM_FFI_CHECK(!builder->result.defined(), ValueError) << "Builder.result has already been set";
    builder->result = func;
  } else if (ffi::Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    TVM_FFI_CHECK(name.has_value(), ValueError)
        << "The function name must be defined before exiting the "
           "function scope, if it's defined in a Module";
    const ir::IRModuleFrame& frame = opt_frame.value();
    const ffi::String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name)) {
      // Case. First time visiting the function.
      ir::DeclFunction(func_name, func);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    ir::DefFunction(func_name, func);
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot find where to insert PrimFunc";
  }
}

void SBlockFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();

  // Allow SBlock construction in raw IRBuilder context (no enclosing PrimFuncFrame)
  // so test fixtures can construct blocks/block-realizes directly.

  ffi::Array<tvm::tirx::Buffer> tir_alloc_buffers;
  for (const tvm::tirx::Buffer& buffer : alloc_buffers) {
    tir_alloc_buffers.push_back(buffer);
  }
  ffi::Map<ffi::String, Any> attrs = annotations.value_or({});
  if (int detect_access = (!reads.defined()) | (!writes.defined() << 1)) {
    attrs.Set("tirx.script_parsing_detect_access", tvm::IntImm(DataType::Int(64), detect_access));
  }
  tvm::tirx::SBlock block(iter_vars, reads.value_or(ffi::Array<tvm::tirx::BufferRegion>()),
                          writes.value_or(ffi::Array<tvm::tirx::BufferRegion>()), name,
                          AsStmt(stmts), init, tir_alloc_buffers, match_buffers, attrs,
                          tvm::Span());
  if (no_realize) {
    TVM_FFI_CHECK(iter_values.empty(), ValueError)
        << "Block bindings are not allowed when `no_realize=True`";
    TVM_FFI_CHECK(!predicate.defined(), ValueError)
        << "`T.where` is not allowed when `no_realize=True`";
    AddToParent(block);
  } else {
    AddToParent(tvm::tirx::SBlockRealize(iter_values, predicate.value_or(Bool(true)), block));
  }
}

void ExecScopeFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  TVM_FFI_ICHECK(exec_scope.defined())
      << "InternalError: ExecScopeFrame must have an execution scope";
  tvm::tirx::Stmt body = AsStmt(stmts);
  tvm::tirx::Stmt stmt = tvm::tirx::ExecScopeStmt(exec_scope.value(), body);
  ffi::Optional<PrimExpr> guard = std::nullopt;
  for (const PrimExpr& predicate : guards) {
    guard = guard.defined() ? PrimExpr(guard.value() && predicate) : predicate;
  }
  if (guard.defined()) {
    stmt = tvm::tirx::IfThenElse(guard.value(), stmt);
  }
  AddToParent(stmt);
}

void BlockInitFrameNode::EnterWithScope() {
  SBlockFrame frame = FindSBlockFrame("T.init");
  if (frame->init.defined()) {
    TVM_FFI_THROW(ValueError) << "Duplicate block init declaration";
  }
  TIRFrameNode::EnterWithScope();
}

void BlockInitFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  SBlockFrame frame = FindSBlockFrame("T.init");
  frame->init = AsStmt(stmts);
}

void ForFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(this->f_make_for_loop(vars, doms, steps, AsStmt(stmts)));
}

void AssertFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (stmts.empty()) {
    AddToParent(tvm::tirx::AssertStmt(condition, error_kind, message_parts));
  } else {
    ffi::Array<tvm::tirx::Stmt> seq;
    seq.push_back(tvm::tirx::AssertStmt(condition, error_kind, message_parts));
    for (const auto& stmt : stmts) {
      seq.push_back(stmt);
    }
    AddToParent(tvm::tirx::SeqStmt(seq));
  }
}

void LaunchThreadFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::AttrStmt(iter_var, attr_key, extent, AsStmt(stmts)));
}

void AttrFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::AttrStmt(node, attr_key, value, AsStmt(stmts)));
}

void WhileFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::While(condition, AsStmt(stmts)));
}

void IfFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (!stmts.empty()) {
    TVM_FFI_THROW(InternalError)
        << "stmt within IfThenElse frame should be either in ThenFrame or ElseFrame";
  }
  if (!then_stmts.defined()) {
    TVM_FFI_THROW(InternalError) << "IfThenElse frame should have at least one then branch";
  }
  AddToParent(tvm::tirx::IfThenElse(
      condition, AsStmt(then_stmts.value()),
      else_stmts.defined() ? AsStmt(else_stmts.value()) : tvm::tirx::Stmt(nullptr)));
}

void ThenFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.then_");
  if (frame->then_stmts.defined()) {
    TVM_FFI_THROW(ValueError) << "Duplicate then branch declaration, previous one is "
                              << frame->then_stmts.value();
  }
  TIRFrameNode::EnterWithScope();
}

void ThenFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  FindIfFrame("T.then_")->then_stmts = stmts;
}

void ElseFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.else_");
  if (!frame->then_stmts.defined()) {
    TVM_FFI_THROW(InternalError) << "The else branch should follow then branch";
  }
  if (frame->else_stmts.defined()) {
    TVM_FFI_THROW(ValueError) << "Duplicate else branch declaration, previous one is "
                              << frame->else_stmts.value();
  }
  TIRFrameNode::EnterWithScope();
}

void ElseFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  FindIfFrame("T.else_")->else_stmts = stmts;
}

void DeclBufferFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (allocated) {
    AddToParent(tvm::tirx::SeqStmt::Flatten(tvm::tirx::DeclBuffer(buffer), AsStmt(stmts)));
  } else {
    // data is undefined in `decl_buffer(...)`, lower to `alloc_buffer(...)`.
    AddToParent(tvm::tirx::SeqStmt::Flatten(tvm::tirx::AllocBuffer(buffer), AsStmt(stmts)));
  }
}

void ComposeOpFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  ffi::Array<ffi::ObjectRef> ops;
  for (const auto& stmt : stmts) {
    auto op_call = stmt.as<tvm::tirx::TilePrimitiveCallNode>();
    TVM_FFI_ICHECK(op_call) << "ValueError: Only TIRx op calls allowed in ComposeOp. Violated by "
                            << stmt;
    ops.push_back(ffi::GetRef<tvm::tirx::TilePrimitiveCall>(op_call));
  }
  auto compose_op_op = tvm::Op::Get("tirx.compose_op");
  AddToParent(tvm::tirx::TilePrimitiveCall(compose_op_op, ops, workspace, config, dispatch));
}

void AllocBufferFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::SeqStmt::Flatten(tvm::tirx::AllocBuffer(buffer), AsStmt(stmts)));
}

void HintFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  // Always store attrs as a structured Map in the node field
  ffi::Map<ffi::String, Any> full_attrs;
  if (!message.empty()) {
    full_attrs.Set("message", ffi::String(message));
  }
  for (const auto& [k, v] : attrs) {
    full_attrs.Set(k, v);
  }
  AddToParent(
      tvm::tirx::AttrStmt(full_attrs, "tirx_hint", IntImm(DataType::Int(32), 1), AsStmt(stmts)));
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
