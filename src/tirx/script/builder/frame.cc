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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/script/ir_builder/ir/ir.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/script/builder/frame.h>

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
// `_current_s_tir()` becomes true, so the resulting BufferVar is built with the
// default tile layout instead of None. Direct annotations using only literals
// are re-evaluated inside the frame and correctly get layout=None.
//
// This normalizer runs at PrimFunc construction time: it strips any defined
// layout from buffers in `buffer_map` / `root_alloc_buffers` and rewrites
// matching body references through the s_tir::StmtExprMutator's built-in
// variable remapping, so the body remains well-formed.
class STirBufferLayoutNormalizer : public tvm::tirx::StmtExprMutator {
 public:
  using tvm::tirx::StmtExprMutator::Mutate;
  using tvm::tirx::StmtExprMutator::Mutate_;
  void Register(const tvm::tirx::BufferVar& old_buf, const tvm::tirx::BufferVar& new_buf) {
    VarRemapSet(old_buf, new_buf);
  }
  bool Empty() const { return var_remap_.empty(); }
  tvm::tirx::BufferVar Lookup(const tvm::tirx::BufferVar& buf) {
    if (auto mapped = VarRemapGet(buf); mapped != nullptr) {
      return mapped.as_or_throw<tvm::tirx::BufferVar>();
    }
    return buf;
  }
};

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  TIRFrameNode::RegisterReflection();
  PrimFuncFrameNode::RegisterReflection();
  SBlockFrameNode::RegisterReflection();
  BlockInitFrameNode::RegisterReflection();
  ForFrameNode::RegisterReflection();
  AssertFrameNode::RegisterReflection();
  LaunchThreadFrameNode::RegisterReflection();
  AttrFrameNode::RegisterReflection();
  WhileFrameNode::RegisterReflection();
  IfFrameNode::RegisterReflection();
  ThenFrameNode::RegisterReflection();
  ElseFrameNode::RegisterReflection();
  DeclBufferFrameNode::RegisterReflection();
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
  // Default attributes belong to the completed function. A declaration must
  // leave body-level func_attr free to supply those values on the same frame.
  if (!is_declaration && !is_private && name.has_value() &&
      !attrs.count(tvm::attr::kGlobalSymbol)) {
    insert_attr(tvm::attr::kGlobalSymbol, name.value());
  }
  if (!is_declaration && s_tir) {
    insert_attr(tvm::attr::kSTir, true);
  }
  if (!is_declaration && persistent) {
    insert_attr(tvm::tirx::attr::kPersistentKernel, true);
  }
  // s_tir-mode normalization: drop stale default layouts (see comment on
  // STirBufferLayoutNormalizer above) and rewrite body references coherently.
  ffi::Array<tvm::tirx::BufferVar> effective_root_alloc_buffers = root_alloc_buffers;
  TVM_FFI_CHECK(!is_declaration || (stmts.empty() && root_alloc_buffers.empty()), ValueError)
      << "A function declaration cannot contain body statements";
  tvm::tirx::Stmt body = is_declaration ? tvm::tirx::Stmt() : AsStmt(stmts);
  auto normalizer = ffi::make_object<STirBufferLayoutNormalizer>();
  ffi::Array<tvm::tirx::Var> effective_args;
  ffi::Map<tvm::tirx::Var, tvm::Expr> param_replacements;
  for (const tvm::tirx::Var& arg : args) {
    ffi::Optional<tvm::tirx::BufferVar> opt_buffer = buffer_map.Get(arg);
    bool replaces_legacy_param = opt_buffer.has_value();
    if (!opt_buffer.has_value() && arg->ty.as<tvm::tirx::BufferTypeNode>()) {
      opt_buffer = tvm::tirx::BufferVar(arg);
    }
    if (!opt_buffer.has_value()) {
      effective_args.push_back(arg);
      continue;
    }
    tvm::tirx::BufferVar buffer = opt_buffer.value();
    if (s_tir && buffer->layout.has_value()) {
      ffi::ObjectPtr<tvm::tirx::BufferTypeNode> type = tvm::tirx::CopyBufferType(buffer);
      type->layout = std::nullopt;
      tvm::tirx::BufferVar new_buffer = tvm::tirx::RebuildBufferVar(buffer, std::move(type));
      normalizer->Register(buffer, new_buffer);
      buffer = new_buffer;
    }
    effective_args.push_back(buffer.var());
    if (replaces_legacy_param && !arg.same_as(buffer.var()) &&
        !arg->ty.as<tvm::tirx::BufferTypeNode>()) {
      tvm::Expr data = buffer.data();
      param_replacements.Set(arg, ffi::StructuralEqual()(arg->ty, data->ty)
                                      ? data
                                      : tvm::prim::reinterpret(arg->ty, std::move(data)));
    }
  }
  if (!normalizer->Empty()) {
    if (!is_declaration) {
      body = normalizer->Mutate(body, InplaceMode::kAllow).ValueOrUnchanged(body);
    }
    ffi::Array<tvm::tirx::BufferVar> new_root_alloc_buffers;
    for (const tvm::tirx::BufferVar& buffer : root_alloc_buffers) {
      new_root_alloc_buffers.push_back(normalizer->Lookup(buffer));
    }
    effective_root_alloc_buffers = std::move(new_root_alloc_buffers);
  }
  if (!is_declaration && !param_replacements.empty()) {
    auto f_substitute =
        [&param_replacements](
            const tvm::tirx::Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = param_replacements.Get(var)) {
        return ffi::Any(*std::move(repl));
      }
      return ffi::Unchanged();
    };
    body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(body), f_substitute)
               .as_or_throw<tvm::tirx::Stmt>();
  }
  tvm::tirx::PrimFunc func(
      /*params=*/effective_args,
      /*body=*/body,
      /*ret_type=*/ret_type.value_or(TupleType::Empty()),
      /*attrs=*/attrs.defined() ? DictAttrs(attrs) : DictAttrs(),
      /*span=*/source_span);
  if (!is_declaration) {
    func = tvm::tirx::ScriptComplete(func, effective_root_alloc_buffers, s_tir);
  }
  function = func;
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    TVM_FFI_CHECK(!builder->result.has_value(), ValueError)
        << "Builder.result has already been set";
    if (!is_declaration) builder->result = func;
  } else if (ffi::Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    TVM_FFI_CHECK(name.has_value(), ValueError)
        << "The function name must be defined before exiting the "
           "function scope, if it's defined in a Module";
    const ir::IRModuleFrame& frame = opt_frame.value();
    const ffi::String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name) ||
        !frame->functions.count(frame->global_var_map.at(func_name))) {
      // Case. First time visiting the function.
      global_var = ir::DeclFunction(func_name, func);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    if (!global_var.has_value()) {
      TVM_FFI_CHECK(!is_declaration, ValueError) << "function " << func_name << " already exists";
      global_var = frame->global_var_map.at(func_name);
    }
    if (!is_declaration) {
      ir::DefFunction(func_name, func);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot find where to insert PrimFunc";
  }
  is_declaration = false;
}

void SBlockFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();

  // Allow s_tir::SBlock construction in raw IRBuilder context (no enclosing PrimFuncFrame)
  // so test fixtures can construct blocks/block-realizes directly.

  ffi::Array<tvm::tirx::BufferVar> tir_alloc_buffers;
  for (const tvm::tirx::BufferVar& buffer : alloc_buffers) {
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
        << "`T.where` is not allowed when `no_realize=True`";
    AddToParent(block, source_span);
  } else {
    AddToParent(tvm::s_tir::SBlockRealize(iter_values, predicate.value_or(IntImm::Bool(true)),
                                          block, source_span),
                source_span);
  }
}

void BlockInitFrameNode::EnterWithScope() {
  SBlockFrame frame = FindSBlockFrame("T.init");
  if (frame->init.has_value()) {
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
  AddToParent(this->f_make_for_loop(vars, doms, steps, AsStmt(stmts), source_span), source_span);
}

void ForFrameNode::SetNames(
    ffi::Optional<ffi::Variant<ffi::String, ffi::Array<ffi::String>>> names) {
  if (!names.has_value()) return;
  if (IRBuilder::IsInScope()) {
    for (const IRBuilderFrame& frame : IRBuilder::Current()->frames) {
      TVM_FFI_CHECK(frame.get() != this, ValueError)
          << "Loop names must be configured before entering the frame";
    }
  }
  std::vector<ffi::String> targets;
  if (auto name = names.value().as<ffi::String>()) {
    for (size_t i = 0; i < vars.size(); ++i) {
      targets.push_back(vars.size() == 1 ? name.value() : name.value() + "_" + std::to_string(i));
    }
  } else {
    auto source_names = names.value().as<ffi::Array<ffi::String>>().value();
    bool expanded = false;
    for (const ffi::String& name : source_names) {
      if (!name.empty() && name.data()[0] == '*') {
        TVM_FFI_CHECK(!expanded && name.size() > 1, ValueError)
            << "A loop target may contain only one named starred target";
        expanded = true;
        int count = static_cast<int>(vars.size()) - static_cast<int>(source_names.size()) + 1;
        TVM_FFI_CHECK(count >= 0, ValueError)
            << "Loop target count differs from iteration dimensions";
        for (int i = 0; i < count; ++i) {
          targets.push_back(std::string(name).substr(1) + "_" + std::to_string(i));
        }
      } else {
        targets.push_back(name);
      }
    }
  }
  TVM_FFI_CHECK(targets.size() == vars.size(), ValueError)
      << "Loop target count differs from iteration dimensions";
  for (size_t i = 0; i < targets.size(); ++i) {
    TVM_FFI_CHECK(!targets[i].empty(), ValueError) << "Loop variable names must be nonempty";
  }
  for (size_t i = 0; i < targets.size(); ++i) {
    const_cast<tvm::VarNode*>(vars[i].get())->name = targets[i];
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def_method("script.ir_builder.tirx.ForFrameSetNames",
                                               &ForFrameNode::SetNames);
}

void AssertFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (stmts.empty()) {
    AddToParent(tvm::tirx::AssertStmt(condition, error_kind, message_parts, source_span),
                source_span);
  } else {
    ffi::Array<tvm::tirx::Stmt> seq;
    seq.push_back(tvm::tirx::AssertStmt(condition, error_kind, message_parts, source_span));
    for (const auto& stmt : stmts) {
      seq.push_back(stmt);
    }
    AddToParent(tvm::tirx::SeqStmt(seq, source_span), source_span);
  }
}

void LaunchThreadFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::AttrStmt(iter_var, attr_key, extent, AsStmt(stmts), source_span),
              source_span);
}

void AttrFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::AttrStmt(node, attr_key, value, AsStmt(stmts), source_span), source_span);
}

void WhileFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tirx::While(condition, AsStmt(stmts), source_span), source_span);
}

void IfFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (!stmts.empty()) {
    TVM_FFI_THROW(InternalError)
        << "stmt within IfThenElse frame should be either in ThenFrame or ElseFrame";
  }
  if (!then_stmts.has_value()) {
    TVM_FFI_THROW(InternalError) << "IfThenElse frame should have at least one then branch";
  }
  AddToParent(tvm::tirx::IfThenElse(
                  condition, AsStmt(then_stmts.value()),
                  else_stmts.has_value() ? AsStmt(else_stmts.value()) : tvm::tirx::Stmt(nullptr),
                  source_span),
              source_span);
}

void ThenFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.then_");
  if (frame->then_stmts.has_value()) {
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
  if (!frame->then_stmts.has_value()) {
    TVM_FFI_THROW(InternalError) << "The else branch should follow then branch";
  }
  if (frame->else_stmts.has_value()) {
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
    AddToParent(tvm::tirx::SeqStmt::Flatten(tvm::tirx::DeclBuffer(buffer, data, source_span),
                                            AsStmt(stmts)),
                source_span);
  } else {
    // data is undefined in `decl_buffer(...)`, lower to `alloc_buffer(...)`.
    AddToParent(
        tvm::tirx::SeqStmt::Flatten(tvm::tirx::AllocBuffer(buffer, {}, source_span), AsStmt(stmts)),
        source_span);
  }
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
      tvm::tirx::AttrStmt(full_attrs, "tirx_hint", IntImm::Int32(1), AsStmt(stmts), source_span),
      source_span);
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
