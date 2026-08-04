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
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

bool IsSimpleBuffer(const tirx::BufferVar& buf, bool s_tir) {
  if (!buf->strides.empty()) {
    return false;
  }
  for (const PrimExpr& shp_i : buf->shape) {
    if (!tirx::UndefinedVars(shp_i).empty()) {
      return false;
    }
  }
  for (const PrimExpr& stride_i : buf->strides) {
    if (!tirx::UndefinedVars(stride_i).empty()) {
      return false;
    }
  }
  if (!tirx::UndefinedVars(buf->elem_offset).empty()) {
    return false;
  } else if (buf->elem_offset->IsInstance<IntImmNode>()) {
    IntImm elem_offset = buf->elem_offset.as_or_throw<IntImm>();
    if (elem_offset->value != 0) {
      return false;
    }
  }
  if (s_tir) {
    if (buf->layout.has_value() &&
        !ffi::StructuralEqual()(buf->layout, tirx::TileLayoutNode::DefaultLayout(buf->shape))) {
      return false;
    }
  } else {
    if (!buf->layout.has_value() ||
        !ffi::StructuralEqual()(buf->layout, tirx::TileLayoutNode::DefaultLayout(buf->shape))) {
      return false;
    }
  }
  if (!buf->allocated_addr.empty()) {
    return false;
  }
  return buf.scope() == "global" && buf->data_alignment == runtime::kAllocAlignment &&
         buf->offset_factor == 1;
}

int CountVarOccurrence(const tirx::PrimFunc& f, const tirx::Var& v) {
  OccurrenceCounter counter(v.get());
  counter(f->body);
  for (const tirx::Var& v : f->params) {
    counter.VisitVar(v);
  }
  return counter.count;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tirx::PrimFunc>("", [](tirx::PrimFunc func, AccessPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> f(d, func);
      (*f)->AddDispatchToken(d, "tirx");
      IdDoc func_name = IdDoc(FindFunctionName(d, func).value_or("main"));
      d->SetCommonPrefix(func, [](const ffi::ObjectRef& obj) {
        return obj->IsInstance<tirx::VarNode>() || obj->IsInstance<tirx::BufferTypeNode>();
      });
      int n_args = func->params.size();
      // Step 1. Handle `func->params`
      ffi::Array<AssignDoc> args;
      args.reserve(n_args);
      std::unordered_map<const tirx::VarNode*, ExprDoc> scalar_param_docs;
      // Define scalar docs up front so a preceding Buffer parameter can render
      // a reference to a later scalar parameter.  `bound_signature_vars`
      // separately tracks source order: the first shape expression that sees
      // an unbound Var must be quoted because Buffer shapes are match scopes.
      std::unordered_set<const tirx::VarNode*> bound_signature_vars;
      for (const tirx::Var& param : func->params) {
        if (!param->ty.as<tirx::BufferTypeNode>()) {
          scalar_param_docs.emplace(param.get(), DefineVar(param, *f, d));
        }
      }
      for (int i = 0; i < n_args; ++i) {
        tirx::Var var = func->params[i];
        AccessPath var_p = p->Attr("params")->ArrayItem(i);
        if (var->ty.as<tirx::BufferTypeNode>()) {
          tirx::BufferVar buffer(var);
          std::unordered_set<const tirx::VarNode*> stringify_shape_vars;
          std::unordered_set<const tirx::VarNode*> shape_vars;
          for (const PrimExpr& shape : buffer->shape) {
            tirx::PostOrderVisit(shape, [&](const ffi::ObjectRef& obj) {
              if (const auto* shape_var = obj.as<tirx::VarNode>()) {
                shape_vars.insert(shape_var);
                if (!bound_signature_vars.count(shape_var)) {
                  stringify_shape_vars.insert(shape_var);
                }
              }
            });
          }
          IdDoc lhs = DefineBuffer(buffer, *f, d);
          ExprDoc annotation = BufferAttn(buffer, var_p->Attr("ty"), *f, d, stringify_shape_vars);
          args.push_back(AssignDoc(lhs, std::nullopt, annotation));
          for (const tirx::VarNode* shape_var : shape_vars) {
            bound_signature_vars.insert(shape_var);
          }
          continue;
        }
        ExprDoc a = d->AsDoc<ExprDoc>(var->ty, var_p->Attr("ty"));
        args.push_back(AssignDoc(scalar_param_docs.at(var.get()), std::nullopt, a));
        bound_signature_vars.insert(var.get());
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
          (*f)->stmts.push_back(
              ExprStmtDoc(TIR(d, "func_attr")  //
                              ->Call({d->AsDoc<ExprDoc>(DictAttrs(new_attrs), p->Attr("attrs"))})));
        }
      }
      // Step 3. Handle `func->body`
      ffi::Optional<tirx::SBlock> implicit_root_block = [&]() -> ffi::Optional<tirx::SBlock> {
        const tirx::SBlockRealizeNode* root_block_realize =
            func->body.as<tirx::SBlockRealizeNode>();
        if (root_block_realize && !root_block_realize->iter_values.size() &&
            tirx::is_one(root_block_realize->predicate)) {
          tirx::SBlock root_block = root_block_realize->block;
          if (!root_block->annotations.size() && !root_block->match_buffers.size() &&
              !root_block->reads.size() && !root_block->writes.size() &&
              !root_block->init.has_value()) {
            const tirx::SBlockRealizeNode* block_realize =
                root_block->body.as<tirx::SBlockRealizeNode>();
            if (root_block->alloc_buffers.size() ||
                (block_realize && block_realize->block->iter_vars.size()) ||
                (!block_realize && tirx::ContainsNode<tirx::SBlockRealizeNode>(root_block->body))) {
              return root_block;
            }
          }
        }
        return std::nullopt;
      }();
      if (d->cfg->syntax_sugar && implicit_root_block) {
        tirx::SBlock root_block = implicit_root_block.value();
        AccessPath root_block_p = p->Attr("body")->Attr("block");
        (*f)->stmts.push_back(CommentDoc("with T.sblock(\"root\"):"));
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
      ExprDoc decorator = TIR(d, "prim_func");
      ffi::Array<ffi::String, void> kwargs_keys;
      ffi::Array<ExprDoc, void> kwargs_values;
      // mark private if there is no global symbol
      if (!func->attrs->dict.count(tvm::attr::kGlobalSymbol)) {
        kwargs_keys.push_back("private");
        kwargs_values.push_back(LiteralDoc::Boolean(true, ffi::Optional<AccessPath>()));
      }
      if (func->attrs->dict.count(tvm::attr::kSTir)) {
        kwargs_keys.push_back("s_tir");
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
      return HeaderWrapper(d, FunctionDoc(
                                  /*name=*/func_name,
                                  /*args=*/args,
                                  /*decorators=*/{decorator},
                                  /*return_type=*/ret_type,
                                  /*body=*/(*f)->stmts));
    });

TVM_REGISTER_SCRIPT_AS_REPR(tirx::PrimFuncNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::GlobalVar>(                                            //
        "tirx", [](tvm::GlobalVar n, AccessPath n_p, IRDocsifier d) -> Doc {  //
          if (ffi::Optional<ExprDoc> doc = d->GetVarDoc(n)) {
            return doc.value();
          } else {
            IdDoc ret(n->name_hint);
            ret->source_paths.push_back(n_p);
            return ret;
          }
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::IRModule>(                                              //
        "tirx", [](tvm::IRModule mod, AccessPath n_p, IRDocsifier d) -> Doc {  //
          ffi::Optional<ExprDoc> doc = d->GetVarDoc(mod);
          TVM_FFI_ICHECK(doc) << "Unable to print IRModule before definition in TIR.";
          return doc.value();
        });

}  // namespace printer
}  // namespace script
}  // namespace tvm
