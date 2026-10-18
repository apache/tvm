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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/script/ir_builder/frame.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

void AddToParent(tvm::Stmt stmt, Location loc) {
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
  } else if (const auto* stmt_frame = builder->frames.back().as<StmtFrameObj>()) {
    ffi::GetRef<StmtFrame>(stmt_frame)->stmts.push_back(stmt);
  } else {
    TVM_FFI_THROW(TypeError) << "Unsupported frame type: " << builder->frames.back();
  }
}

IRModuleFrame FindModuleFrame() {
  IRBuilder builder = IRBuilder::Current();
  if (ffi::Optional<IRModuleFrame> frame = builder->FindFrame<IRModuleFrame>()) {
    return frame.value();
  } else {
    TVM_FFI_THROW(ValueError) << "IRModule frame not find. Please ensure it"
                              << " is called under I.ir_module()";
  }
  throw;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRModuleFrameObj::RegisterReflection();
  ffi::reflection::GlobalDef().def("script.ir_builder.ir.IRModuleFrameAbort",
                                   [](const IRModuleFrame& frame) {
                                     if (!IRBuilder::IsInScope()) return;
                                     auto& frames = IRBuilder::Current()->frames;
                                     for (size_t i = frames.size(); i > 0; --i) {
                                       if (frames[i - 1].same_as(frame)) {
                                         // Drop this failed scope without callbacks or
                                         // finalization.
                                         while (frames.size() >= i) frames.pop_back();
                                         return;
                                       }
                                     }
                                   });
}

void IRModuleFrameObj::ExitWithScope() {
  IRBuilderFrameObj::ExitWithScope();
  ffi::Map<GlobalVar, BaseFunc> func_map;
  TVM_FFI_ICHECK_EQ(functions.size(), global_var_map.size())
      << "All functions must be defined in the IRModule. Got " << global_var_map.size()
      << "declared function(s), but only " << functions.size() << "defined function(s).";
  for (const auto& kv : functions) {
    const GlobalVar& gv = kv.first;
    const BaseFunc& func = kv.second;
    TVM_FFI_CHECK(func.defined(), ValueError) << "function " << gv->name_hint << " is not defined";
    func_map.Set(gv, func);
  }
  IRBuilder builder = IRBuilder::Current();
  TVM_FFI_CHECK(!builder->result.has_value(), ValueError) << "Builder.result has already been set";
  auto dict_attrs = DictAttrs(attrs);
  builder->result = tvm::IRModule(func_map, dict_attrs, global_infos);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  StmtFrameObj::RegisterReflection();
  ForFrameObj::RegisterReflection();
  AssertFrameObj::RegisterReflection();
  RegionFrameObj::RegisterReflection();
  WhileFrameObj::RegisterReflection();
  IfFrameObj::RegisterReflection();
  ThenFrameObj::RegisterReflection();
  ElseFrameObj::RegisterReflection();
}

void ForFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  AddToParent(this->f_make_for_loop(vars, doms, steps, AsStmt(stmts), loc), loc);
}

void ForFrameObj::SetNames(
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
  tvm::ffi::reflection::GlobalDef().def_method("script.ir_builder.ForFrameSetNames",
                                               &ForFrameObj::SetNames);
}

void AssertFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  if (stmts.empty()) {
    AddToParent(tvm::AssertStmt(condition, error_kind, message_parts, loc), loc);
  } else {
    ffi::Array<tvm::Stmt> seq;
    seq.push_back(tvm::AssertStmt(condition, error_kind, message_parts, loc));
    for (const auto& stmt : stmts) {
      seq.push_back(stmt);
    }
    AddToParent(tvm::SeqStmt(seq, loc), loc);
  }
}

void RegionFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  AddToParent(tvm::RegionStmt(op, args, body_params, attrs, AsStmt(stmts), {}, loc), loc);
}

void WhileFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  AddToParent(tvm::While(condition, AsStmt(stmts), loc), loc);
}

void IfFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  if (!stmts.empty()) {
    TVM_FFI_THROW(InternalError)
        << "stmt within If frame should be either in ThenFrame or ElseFrame";
  }
  if (!then_stmts.has_value()) {
    TVM_FFI_THROW(InternalError) << "If frame should have at least one then branch";
  }
  AddToParent(
      tvm::If(condition, AsStmt(then_stmts.value()),
              else_stmts.has_value() ? ffi::Optional<tvm::SeqStmt>(AsStmt(else_stmts.value()))
                                     : std::nullopt,
              loc),
      loc);
}

void ThenFrameObj::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.then_");
  if (frame->then_stmts.has_value()) {
    TVM_FFI_THROW(ValueError) << "Duplicate then branch declaration, previous one is "
                              << frame->then_stmts.value();
  }
  StmtFrameObj::EnterWithScope();
}

void ThenFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  FindIfFrame("T.then_")->then_stmts = stmts;
}

void ElseFrameObj::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.else_");
  if (!frame->then_stmts.has_value()) {
    TVM_FFI_THROW(InternalError) << "The else branch should follow then branch";
  }
  if (frame->else_stmts.has_value()) {
    TVM_FFI_THROW(ValueError) << "Duplicate else branch declaration, previous one is "
                              << frame->else_stmts.value();
  }
  StmtFrameObj::EnterWithScope();
}

void ElseFrameObj::ExitWithScope() {
  StmtFrameObj::ExitWithScope();
  FindIfFrame("T.else_")->else_stmts = stmts;
}

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
