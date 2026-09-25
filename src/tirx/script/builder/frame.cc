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
#include <tvm/script/ir_builder/ir/ir.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/script/builder/frame.h>

#include <map>

#include "./utils.h"

namespace tvm {
namespace script {

namespace ir_builder {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  TIRFrameNode::RegisterReflection();
  PrimFuncFrameNode::RegisterReflection();
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

namespace {
std::map<ffi::String, PrimFuncFrameNode::AttrValidator>& AttrValidators() {
  static std::map<ffi::String, PrimFuncFrameNode::AttrValidator> validators;
  return validators;
}
}  // namespace

void PrimFuncFrameNode::RegisterAttrValidator(ffi::String key, AttrValidator validator) {
  TVM_FFI_ICHECK(AttrValidators().emplace(std::move(key), std::move(validator)).second)
      << "Duplicate function attribute validator";
}

void PrimFuncFrameNode::ValidateAttrs() const {
  for (const auto& [key, value] : attrs) {
    auto it = AttrValidators().find(key);
    if (it != AttrValidators().end()) it->second(this, value);
  }
}

void TIRFrameNode::BindBufferRegion(tvm::tirx::BufferVar buffer, tvm::TensorRegion region) {
  TVM_FFI_THROW(ValueError) << "match_buffer requires a frame that supports region aliases";
}

tvm::tirx::PrimFunc PrimFuncFrameNode::FinalizeFunction(tvm::tirx::PrimFunc func) { return func; }

void PrimFuncFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  ValidateAttrs();
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
  if (!is_declaration && persistent) {
    insert_attr(tvm::tirx::attr::kPersistentKernel, true);
  }
  TVM_FFI_CHECK(!is_declaration || stmts.empty(), ValueError)
      << "A function declaration cannot contain body statements";
  tvm::tirx::Stmt body = is_declaration ? tvm::tirx::Stmt() : AsStmt(stmts);
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
    effective_args.push_back(buffer.var());
    if (replaces_legacy_param && !arg.same_as(buffer.var()) &&
        !arg->ty.as<tvm::tirx::BufferTypeNode>()) {
      tvm::Expr data = buffer.data();
      param_replacements.Set(arg, ffi::StructuralEqual()(arg->ty, data->ty)
                                      ? data
                                      : tvm::prim::reinterpret(arg->ty, std::move(data)));
    }
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
  func = FinalizeFunction(std::move(func));
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
