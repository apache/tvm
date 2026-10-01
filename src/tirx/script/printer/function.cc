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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/function.h>
#include <tvm/ir/global_info.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../script/printer/dialect_prefix.h"
#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> PrimFuncDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object* destination) {
  const auto* func =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::PrimFuncNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  auto module =
      d->GetOrCreateExtraState<ffi::Optional<ffi::Dict<ffi::String, ffi::List<GlobalInfo>>>>(
          "ir.global_info_map");
  auto names = d->GetExtraConfig<ffi::Array<ffi::String>>("script.binding_names", {});
  if (!module.has_value() && !names.empty()) {
    if (auto symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      TVM_FFI_CHECK(symbol.value() == names.back(), TypeError)
          << "printer PrimFunc global_symbol must match its definition name";
    }
  }
  VarScope vars(d);

  bool legacy_s_tir = func->attrs->dict.count(tvm::attr::kSTir);

  ffi::String name = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value_or("main");
  FunctionDoc doc(ffi::UnsafeInit{});
  {
    ffi::Array<AssignDoc> args;
    ffi::Array<IdDoc> param_ids;
    for (const Var& var : func->params) param_ids.push_back(VarDoc(d, var));
    size_t param_index = 0;
    for (const Var& var : func->params) {
      IdDoc lhs = param_ids[param_index++];
      ExprDoc annotation = d->Translate(var->ty).value();
      d->RecordOrigin(annotation, var->ty);
      AssignDoc argument(lhs, std::nullopt, annotation);
      d->RecordOrigin(argument, var);
      args.push_back(argument);
    }

    auto signature_candidates = CopyImplicitDefs(d);
    // Hoist truly free buffer declarations out of conditional statement lists.
    // They may occur in diagnostic IR that intentionally disables well-formedness
    // checks; declaring them at first use can leave later branches unbound.
    ffi::Array<Var> free_vars;
    try {
      free_vars = tirx::UndefinedVars(func->body, func->params);
    } catch (const ffi::Error&) {
      // Keep diagnostic printing available even for malformed non-SSA bodies.
    }
    auto free_declarations = d->WithDocScope([&]() {
      for (const Var& var : free_vars) {
        if (var->ty.as<tirx::BufferTypeNode>()) {
          d->Translate(var).value();
        }
      }
    });
    std::unordered_map<const ffi::Object*, size_t> thread_counts;
    std::vector<tirx::IterVar> thread_vars;
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        func->body, [&](const tirx::AttrStmt& attr) -> ffi::Expected<ffi::WalkResult> {
          if (attr->attr_key == "thread_extent" || attr->attr_key == tirx::attr::virtual_thread) {
            if (auto iter = attr->node.as<tirx::IterVar>()) {
              if (thread_counts[iter.value()->var.get()]++ == 0)
                thread_vars.push_back(iter.value());
            }
          }
          return ffi::WalkResult::Advance();
        });
    ffi::Array<StmtDoc> thread_declarations;
    for (const auto& iter : thread_vars) {
      if (thread_counts[iter->var.get()] > 1) {
        IdDoc lhs = VarDoc(d, iter->var);
        thread_declarations.push_back(
            AssignDoc(lhs,
                      NamespaceDoc("tirx")
                          ->Attr("env_thread")
                          ->Call({LiteralDoc::Str(iter->thread_tag, std::nullopt)}),
                      std::nullopt));
      }
    }
    ExprDoc decorator = legacy_s_tir ? NamespaceDoc("s_tir")->Attr("prim_func")
                                     : NamespaceDoc("tirx")->Attr("prim_func");
    ffi::Array<ffi::String> decorator_keys;
    ffi::Array<ExprDoc> decorator_values;
    if (!func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      decorator_keys.push_back("private");
      decorator_values.push_back(LiteralDoc::Boolean(true, std::nullopt));
    }
    if (func->attrs->dict.count(tirx::attr::kPersistentKernel)) {
      decorator_keys.push_back("persistent");
      decorator_values.push_back(LiteralDoc::Boolean(true, std::nullopt));
    }
    if (!decorator_keys.empty()) decorator = decorator->Call({}, decorator_keys, decorator_values);
    ffi::Array<StmtDoc> body;
    if (auto root = func->body.as<s_tir::SBlockRealizeNode>();
        root && root->iter_values.empty() && tvm::prim::is_one(root->predicate) &&
        root->block->annotations.empty() && root->block->match_buffers.empty() &&
        root->block->reads.empty() && root->block->writes.empty() &&
        !root->block->init.has_value() && root->block->iter_vars.empty() &&
        (!root->block->alloc_buffers.empty() ||
         (!root->block->body.as<s_tir::SBlockRealizeNode>() &&
          tirx::ContainsNode<s_tir::SBlockRealizeNode>(root->block->body)))) {
      auto docs = d->WithDocScope([&]() {
        for (const tirx::BufferVar& buffer : root->block->alloc_buffers) {
          CallDoc rhs = d->Translate(buffer.var()->ty).value().as_or_throw<CallDoc>();
          TVM_FFI_CHECK(rhs->callee.as_or_throw<AttrAccessDoc>()->name == "Buffer", TypeError)
              << "Ts.sblock_alloc_buffer cannot reconstruct this nonrepresentable BufferType";
          const auto* buffer_type = buffer.var()->ty.as<tirx::BufferTypeNode>();
          TVM_FFI_CHECK(
              buffer_type->allocated_addr.empty() || (buffer_type->storage_scope != "global" &&
                                                      buffer_type->storage_scope != "shared" &&
                                                      buffer_type->storage_scope != "shared.dyn" &&
                                                      buffer_type->storage_scope != "local"),
              TypeError)
              << "Ts.sblock_alloc_buffer does not accept allocated_addr in "
              << buffer_type->storage_scope;
          rhs->callee = NamespaceDoc("s_tir")->Attr("sblock_alloc_buffer");
          IdDoc lhs = VarDoc(d, buffer);
          d->Emit(AssignDoc(lhs, rhs, std::nullopt), buffer);
        }
        d->Translate(root->block->body);
      });
      ffi::Array<StmtDoc> root_body = ToStmtDocArray(docs);
      ScopeDoc scope(std::nullopt,
                     NamespaceDoc("s_tir")->Attr("sblock")->Call(
                         {LiteralDoc::Str(root->block->name_hint, std::nullopt)}),
                     root_body);
      d->RecordOrigin(scope, root->block);
      ffi::Array<StmtDoc> elided = {CommentDoc("with Ts.sblock(\"root\"):")};
      elided.insert(elided.end(), root_body.begin(), root_body.end());
      if (SyntaxSugar(d) && !IsAnnotated(d, root->block) &&
          !IsAnnotated(d, ffi::GetRef<s_tir::SBlockRealize>(root)) &&
          !IsUnderlined(d, root->block) &&
          !IsUnderlined(d, ffi::GetRef<s_tir::SBlockRealize>(root))) {
        body = elided;
      } else {
        d->RecordOrigin(scope->rhs, ffi::GetRef<s_tir::SBlockRealize>(root));
        body = {scope};
      }
    } else {
      body = Body(func->body, d);
    }
    body.insert(body.begin(), thread_declarations.begin(), thread_declarations.end());
    ffi::Array<StmtDoc> free_statements = ToStmtDocArray(free_declarations);
    body.insert(body.begin(), free_statements.begin(), free_statements.end());
    std::vector<std::pair<ffi::String, ffi::Any>> attrs;
    for (const auto& [key, value] : func->attrs->dict) {
      if (key != tvm::attr::kGlobalSymbol && key != tvm::attr::kSTir &&
          key != tirx::attr::kPersistentKernel)
        attrs.emplace_back(key, value);
    }
    if (!attrs.empty()) {
      std::sort(attrs.begin(), attrs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      ffi::Array<ExprDoc> keys;
      ffi::Array<ExprDoc> values;
      for (const auto& [key, value] : attrs) {
        keys.push_back(LiteralDoc::Str(key, std::nullopt));
        values.push_back(AnyValue(d, value));
      }
      body.insert(
          body.begin(),
          ExprStmtDoc(NamespaceDoc("tirx")->Attr("func_attr")->Call({DictDoc(keys, values)})));
    }
    ffi::Optional<ExprDoc> ret_type = std::nullopt;
    if (!func->ret_type.IsMissing() && !IsVoidType(func->ret_type)) {
      ret_type = d->Translate(func->ret_type).value();
    }
    doc = FunctionDoc(IdDoc(name), args, {decorator}, ret_type, body);
    FinalizeFunctionDefinitions(d, signature_candidates, doc);
  }
  vars.Close();
  d->Emit(doc, ffi::GetRef<ffi::ObjectRef>(func));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterDialectPrefix("tirx.prefix", "T");
  ffi::reflection::TypeAttrDef<tirx::PrimFuncNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&PrimFuncDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
