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
#include <tvm/s_tir/stmt.h>

#include <utility>

#include "./utils.h"

namespace tvm {
namespace script {

namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tirx::PrimFunc>(
      "", [](tirx::PrimFunc func, AccessPath p, IRDocsifier d) -> Doc {
        With<TIRFrame> f(d, func);
        (*f)->AddDispatchToken(d, "tirx");
        IdDoc func_name = IdDoc(FindFunctionName(d, func).value_or("main"));
        d->SetCommonPrefix(func, [](const ffi::ObjectRef& obj) {
          return obj->IsInstance<tirx::VarNode>() || obj->IsInstance<tirx::BufferTypeNode>();
        });
        std::unordered_set<const VarNode*> runtime_params;
        for (const tirx::Var& param : func->params) {
          runtime_params.insert(param.get());
        }
        std::unordered_set<const VarNode*> type_vars;
        auto collect_type_vars = [&](const PrimExpr& expr) {
          for (const tirx::Var& var : tirx::UndefinedVars(expr)) {
            const auto* var_ty_node = var->ty.as<PrimTypeNode>();
            if (var_ty_node == nullptr) {
              continue;
            }
            PrimType var_ty(var_ty_node->dtype);
            if (!runtime_params.count(var.get()) && var_ty.IsScalar()) {
              type_vars.insert(var.get());
            }
          }
        };
        for (const tirx::Var& param : func->params) {
          if (!param->ty.as<tirx::BufferTypeNode>()) {
            continue;
          }
          tirx::BufferVar buffer(param);
          for (const PrimExpr& extent : buffer->shape) {
            collect_type_vars(extent);
          }
          for (const PrimExpr& stride : buffer->strides) {
            collect_type_vars(stride);
          }
          collect_type_vars(buffer->elem_offset);
          for (const PrimExpr& address : buffer->allocated_addr) {
            collect_type_vars(address);
          }
        }
        auto type_var_docs = DefineTypeVarDocs(type_vars, d);
        int n_args = func->params.size();
        // Step 1. Handle `func->params`
        ffi::Array<AssignDoc> args;
        args.reserve(n_args);
        std::unordered_map<const tirx::VarNode*, ExprDoc> scalar_param_docs;
        // Define scalar docs up front so a preceding Buffer parameter can render
        // a reference to a later scalar parameter. Reserve their names for the
        // whole script so later hoisted symbols cannot capture these annotations.
        bool has_dependent_annotations = false;
        for (const tirx::Var& param : func->params) {
          if (!param->ty.as<tirx::BufferTypeNode>()) {
            scalar_param_docs.emplace(param.get(), DefineVar(param, d->frames.front(), d));
          }
        }
        for (int i = 0; i < n_args; ++i) {
          tirx::Var var = func->params[i];
          AccessPath var_p = p->Attr("params")->ArrayItem(i);
          if (var->ty.as<tirx::BufferTypeNode>()) {
            tirx::BufferVar buffer(var);
            auto check_annotation_var =
                [&](const tirx::Var& annotation_var) -> ffi::Expected<ffi::WalkResult> {
              has_dependent_annotations =
                  has_dependent_annotations || runtime_params.count(annotation_var.get());
              return ffi::WalkResult::Advance();
            };
            if (buffer->layout.has_value() &&
                !ffi::StructuralEqual()(buffer->layout,
                                        tirx::TileLayoutNode::DefaultLayout(buffer->shape))) {
              ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(buffer->layout, check_annotation_var);
            }
            for (const PrimExpr& extent : buffer->shape) {
              ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(extent, check_annotation_var);
            }
            for (const PrimExpr& stride : buffer->strides) {
              ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(stride, check_annotation_var);
            }
            ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(buffer->elem_offset,
                                                            check_annotation_var);
            for (const PrimExpr& address : buffer->allocated_addr) {
              ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(address, check_annotation_var);
            }
            IdDoc lhs = DefineBuffer(buffer, *f, d);
            ExprDoc annotation = BufferAttn(buffer, var_p->Attr("ty"), *f, d);
            args.push_back(AssignDoc(lhs, std::nullopt, annotation));
            continue;
          }
          ExprDoc a = d->AsDoc<ExprDoc>(var->ty, var_p->Attr("ty"));
          args.push_back(AssignDoc(scalar_param_docs.at(var.get()), std::nullopt, a));
        }
        if (has_dependent_annotations) {
          d->ir_usage.insert("future_annotations");
        }
        ffi::Optional<ExprDoc> ret_type = std::nullopt;
        if (!func->ret_type.IsMissing()) {
          const auto* as_tuple = func->ret_type.as<TupleTypeNode>();
          if (!as_tuple || as_tuple->fields.size()) {
            ret_type = d->AsDoc<ExprDoc>(func->ret_type, p->Attr("ret_type"));
          }
        }
        // Step 2. Handle `func->attrs`
        if (!func->attrs->dict.empty()) {
          // for global symbol, don't display it if it matches the func name
          std::unordered_set<ffi::String> keys_to_remove;
          if (func->attrs->dict.count(tvm::attr::kGlobalSymbol) &&
              func->attrs->dict.at(tvm::attr::kGlobalSymbol).as_or_throw<ffi::String>() ==
                  func_name->name) {
            keys_to_remove.insert(tvm::attr::kGlobalSymbol);
          }
          // s_tir is shown in decorator, not in attr dict.
          if (func->attrs->dict.count(tvm::attr::kSTir)) {
            keys_to_remove.insert(tvm::attr::kSTir);
          }
          // for persistent, don't display it (shown in decorator)
          if (func->attrs->dict.count(tirx::attr::kPersistentKernel)) {
            keys_to_remove.insert(tirx::attr::kPersistentKernel);
          }
          ffi::Map<ffi::String, Any> new_attrs;
          for (auto kv : func->attrs->dict) {
            if (!keys_to_remove.count(kv.first)) {
              new_attrs.Set(kv.first, kv.second);
            }
          }
          if (!new_attrs.empty()) {
            (*f)->stmts.push_back(ExprStmtDoc(
                TIR(d, "func_attr")  //
                    ->Call({d->AsDoc<ExprDoc>(DictAttrs(new_attrs), p->Attr("attrs"))})));
          }
        }
        // Step 3. Handle `func->body`
        ffi::Optional<s_tir::SBlock> implicit_root_block = [&]() -> ffi::Optional<s_tir::SBlock> {
          const s_tir::SBlockRealizeNode* root_block_realize =
              func->body.as<s_tir::SBlockRealizeNode>();
          if (root_block_realize && !root_block_realize->iter_values.size() &&
              tvm::prim::is_one(root_block_realize->predicate)) {
            s_tir::SBlock root_block = root_block_realize->block;
            if (!root_block->annotations.size() && !root_block->match_buffers.size() &&
                !root_block->reads.size() && !root_block->writes.size() &&
                !root_block->init.has_value()) {
              const s_tir::SBlockRealizeNode* block_realize =
                  root_block->body.as<s_tir::SBlockRealizeNode>();
              if (root_block->alloc_buffers.size() ||
                  (block_realize && block_realize->block->iter_vars.size()) ||
                  (!block_realize &&
                   tirx::ContainsNode<s_tir::SBlockRealizeNode>(root_block->body))) {
                return root_block;
              }
            }
          }
          return std::nullopt;
        }();
        if (d->cfg->syntax_sugar && implicit_root_block) {
          s_tir::SBlock root_block = implicit_root_block.value();
          AccessPath root_block_p = p->Attr("body")->Attr("block");
          (*f)->stmts.push_back(CommentDoc("with Ts.sblock(\"root\"):"));
          // Handle root block `alloc_buffer`
          for (int i = 0, n = root_block->alloc_buffers.size(); i < n; ++i) {
            tirx::BufferVar buffer = root_block->alloc_buffers[i];
            AccessPath buffer_p = root_block_p->Attr("alloc_buffers")->ArrayItem(i);
            IdDoc lhs = DefineBuffer(buffer, *f, d);
            ExprDoc rhs = BufferDecl(buffer, "sblock_alloc_buffer", {}, buffer_p, *f, d,
                                     BufferVarDefinition::DataPointer);
            (*f)->stmts.push_back(AssignDoc(lhs, rhs, std::nullopt));
          }
          AsDocBody(root_block->body, root_block_p->Attr("body"), f->get(), d);
        } else {
          AsDocBody(func->body, p->Attr("body"), f->get(), d);
        }
        // Step 5. Determine if we need to display the private annotation in the decorator
        ExprDoc decorator =
            func->attrs->dict.count(tvm::attr::kSTir) ? STIR(d, "prim_func") : TIR(d, "prim_func");
        ffi::Array<ffi::String, void> kwargs_keys;
        ffi::Array<ExprDoc, void> kwargs_values;
        // mark private if there is no global symbol
        if (!func->attrs->dict.count(tvm::attr::kGlobalSymbol)) {
          kwargs_keys.push_back("private");
          kwargs_values.push_back(LiteralDoc::Boolean(true, ffi::Optional<AccessPath>()));
        }
        if (func->attrs->dict.count(tirx::attr::kPersistentKernel)) {
          kwargs_keys.push_back("persistent");
          kwargs_values.push_back(LiteralDoc::Boolean(true, ffi::Optional<AccessPath>()));
        }
        // Only emit ``@T.prim_func(...)`` when there is at least one keyword
        // argument; otherwise print bare ``@T.prim_func`` to match apache.
        if (!kwargs_keys.empty()) {
          ffi::Array<ExprDoc> pos_args;
          decorator = std::move(decorator->Call(pos_args, kwargs_keys, kwargs_values));
        }
        return WrapFunctionDocWithTypeVars(d,
                                           FunctionDoc(
                                               /*name=*/func_name,
                                               /*args=*/args,
                                               /*decorators=*/{decorator},
                                               /*return_type=*/ret_type,
                                               /*body=*/(*f)->stmts),
                                           type_var_docs);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() { TVMScriptPrinter::Register<tirx::PrimFuncNode>(ReprPrintTIR); }

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tvm::GlobalVar>(                       //
      "tirx", [](tvm::GlobalVar n, AccessPath n_p, IRDocsifier d) -> Doc {  //
        if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(n)) {
          return doc.value();
        } else {
          IdDoc ret(n->name_hint);
          ret->source_paths.push_back(n_p);
          return ret;
        }
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<tvm::IRModule>(                         //
      "tirx", [](tvm::IRModule mod, AccessPath n_p, IRDocsifier d) -> Doc {  //
        ffi::Optional<ExprDoc> doc = d->GetVarDoc(mod);
        TVM_FFI_ICHECK(doc) << "Unable to print IRModule before definition in TIR.";
        return doc.value();
      });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
