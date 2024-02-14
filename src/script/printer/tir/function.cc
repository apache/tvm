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

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

bool IsSimpleBuffer(const tir::Buffer& buf) {
  if (!buf->strides.empty()) {
    return false;
  }
  for (const PrimExpr& shp_i : buf->shape) {
    if (!tir::UndefinedVars(shp_i).empty()) {
      return false;
    }
  }
  for (const PrimExpr& stride_i : buf->strides) {
    if (!tir::UndefinedVars(stride_i).empty()) {
      return false;
    }
  }
  if (!tir::UndefinedVars(buf->elem_offset).empty()) {
    return false;
  } else if (buf->elem_offset->IsInstance<IntImmNode>()) {
    IntImm elem_offset = Downcast<IntImm>(buf->elem_offset);
    if (elem_offset->value != 0) {
      return false;
    }
  }
  return buf.scope() == "global" && buf->data_alignment == runtime::kAllocAlignment &&
         buf->offset_factor == 1 && buf->buffer_type == tir::BufferType::kDefault &&
         !buf->axis_separators.size();
}

int CountVarOccurrence(const tir::PrimFunc& f, const tir::Var& v) {
  OccurrenceCounter counter(v.get());
  counter(f->body);
  for (const tir::Var& v : f->params) {
    counter(v);
  }
  for (const auto& pair : f->buffer_map) {
    counter(pair.first);
    counter.VisitBuffer(pair.second.get());
  }
  return counter.count;
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tir::PrimFunc>("", [](tir::PrimFunc func, ObjectPath p, IRDocsifier d) -> Doc {
      With<TIRFrame> f(d, func);
      (*f)->AddDispatchToken(d, "tir");
      IdDoc func_name = IdDoc(FindFunctionName(d, func).value_or("main"));
      d->SetCommonPrefix(func, [](const ObjectRef& obj) {
        return obj->IsInstance<tir::VarNode>() || obj->IsInstance<tir::BufferNode>();
      });
      int n_args = func->params.size();
      std::unordered_map<const tir::VarNode*, int> buffer_data_counter;
      for (const auto& pair : func->buffer_map) {
        const tir::VarNode* data_var = pair.second->data.get();
        if (!buffer_data_counter.count(data_var)) {
          buffer_data_counter.insert({data_var, 0});
        }
        ++buffer_data_counter.at(data_var);
      }
      // Step 1. Handle `func->params`
      Array<AssignDoc> args;
      args.reserve(n_args);
      std::unordered_set<const tir::BufferNode*> buffer_inlined;
      for (int i = 0; i < n_args; ++i) {
        tir::Var var = func->params[i];
        ObjectPath var_p = p->Attr("params")->ArrayIndex(i);
        if (d->cfg->syntax_sugar && CountVarOccurrence(func, var) == 2 &&
            func->buffer_map.count(var)) {
          tir::Buffer buffer = func->buffer_map[var];
          if (IsSimpleBuffer(buffer) && buffer_data_counter.at(buffer->data.get()) == 1) {
            ObjectPath buffer_p = p->Attr("buffer_map")->MapValue(var);
            IdDoc lhs = DefineBuffer(buffer, *f, d);
            ExprDoc annotation = BufferAttn(buffer, buffer_p, *f, d);
            args.push_back(AssignDoc(lhs, NullOpt, annotation));
            buffer_inlined.insert(buffer.get());
            continue;
          }
        }
        ExprDoc a = d->AsDoc<ExprDoc>(var->type_annotation, var_p->Attr("type_annotation"));
        args.push_back(AssignDoc(DefineVar(var, *f, d), NullOpt, a));
      }
      // Step 2. Handle `func->attrs`
      if (func->attrs.defined() && !func->attrs->dict.empty()) {
        // for global symbol, don't display it if it matches the func name
        if (func->attrs->dict.count(tvm::attr::kGlobalSymbol) &&
            Downcast<String>(func->attrs->dict.at(tvm::attr::kGlobalSymbol)) == func_name->name) {
          Map<String, ObjectRef> new_attrs;
          for (auto kv : func->attrs->dict) {
            if (kv.first != tvm::attr::kGlobalSymbol) {
              new_attrs.Set(kv.first, kv.second);
            }
          }
          if (!new_attrs.empty()) {
            (*f)->stmts.push_back(ExprStmtDoc(
                TIR(d, "func_attr")  //
                    ->Call({d->AsDoc<ExprDoc>(DictAttrs(new_attrs), p->Attr("attrs"))})));
          }
        } else {
          (*f)->stmts.push_back(
              ExprStmtDoc(TIR(d, "func_attr")  //
                              ->Call({d->AsDoc<ExprDoc>(func->attrs, p->Attr("attrs"))})));
        }
      }
      // Step 3. Handle `func->buffer_map`
      for (int i = 0; i < n_args; ++i) {
        tir::Var param = func->params[i];
        if (func->buffer_map.count(param)) {
          tir::Buffer buffer = func->buffer_map[param];
          if (buffer_inlined.count(buffer.get())) {
            continue;
          }
          ExprDoc param_doc = args[i]->lhs;
          ObjectPath buffer_p = p->Attr("buffer_map")->MapValue(param);
          ExprDoc lhs = DefineBuffer(buffer, *f, d);
          ExprDoc rhs = BufferDecl(buffer, "match_buffer", {param_doc}, buffer_p, *f, d,
                                   BufferVarDefinition::MatchBuffer);
          (*f)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
      }
      // Step 4. Handle `func->body`
      Optional<tir::Block> implicit_root_block = [&]() -> Optional<tir::Block> {
        const tir::BlockRealizeNode* root_block_realize = func->body.as<tir::BlockRealizeNode>();
        if (root_block_realize && !root_block_realize->iter_values.size() &&
            tir::is_one(root_block_realize->predicate)) {
          tir::Block root_block = root_block_realize->block;
          if (!root_block->annotations.size() && !root_block->match_buffers.size() &&
              !root_block->reads.size() && !root_block->writes.size() &&
              !root_block->init.defined()) {
            const tir::BlockRealizeNode* block_realize =
                root_block->body.as<tir::BlockRealizeNode>();
            if (root_block->alloc_buffers.size() ||
                (block_realize && block_realize->block->iter_vars.size()) ||
                (!block_realize && tir::ContainsNode<tir::BlockRealizeNode>(root_block->body))) {
              return root_block;
            }
          }
        }
        return NullOpt;
      }();
      if (d->cfg->syntax_sugar && implicit_root_block) {
        tir::Block root_block = implicit_root_block.value();
        ObjectPath root_block_p = p->Attr("body")->Attr("block");
        (*f)->stmts.push_back(CommentDoc("with T.block(\"root\"):"));
        // Handle root block `alloc_buffer`
        for (int i = 0, n = root_block->alloc_buffers.size(); i < n; ++i) {
          tir::Buffer buffer = root_block->alloc_buffers[i];
          ObjectPath buffer_p = root_block_p->Attr("alloc_buffers")->ArrayIndex(i);
          IdDoc lhs = DefineBuffer(buffer, *f, d);
          ExprDoc rhs = BufferDecl(buffer, "alloc_buffer", {}, buffer_p, *f, d,
                                   BufferVarDefinition::DataPointer);
          (*f)->stmts.push_back(AssignDoc(lhs, rhs, NullOpt));
        }
        AsDocBody(root_block->body, root_block_p->Attr("body"), f->get(), d);
      } else {
        AsDocBody(func->body, p->Attr("body"), f->get(), d);
      }
      Optional<ExprDoc> ret_type = NullOpt;
      if (func->ret_type.defined()) {
        const auto* as_tuple = func->ret_type.as<TupleTypeNode>();
        if (!as_tuple || as_tuple->fields.size()) {
          ret_type = d->AsDoc<ExprDoc>(func->ret_type, p->Attr("ret_type"));
        }
      }
      // Step 5. Determine if we need to display the private annotation in the decorator
      ExprDoc decorator = TIR(d, "prim_func");
      // mark private if there is no global symbol
      if (!func->attrs.defined() || !func->attrs->dict.count(tvm::attr::kGlobalSymbol)) {
        Array<ExprDoc> pos_args;
        decorator = std::move(decorator->Call(pos_args, {"private"},
                                              {LiteralDoc::Boolean(true, Optional<ObjectPath>())}));
      }

      return HeaderWrapper(d, FunctionDoc(
                                  /*name=*/func_name,
                                  /*args=*/args,
                                  /*decorators=*/{decorator},
                                  /*return_type=*/ret_type,
                                  /*body=*/(*f)->stmts));
    });

TVM_SCRIPT_REPR(tir::PrimFuncNode, ReprPrintTIR);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::GlobalVar>(                                           //
        "tir", [](tvm::GlobalVar n, ObjectPath n_p, IRDocsifier d) -> Doc {  //
          if (Optional<ExprDoc> doc = d->GetVarDoc(n)) {
            return doc.value();
          } else {
            IdDoc ret(n->name_hint);
            ret->source_paths.push_back(n_p);
            return ret;
          }
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<tvm::IRModule>(                                             //
        "tir", [](tvm::IRModule mod, ObjectPath n_p, IRDocsifier d) -> Doc {  //
          Optional<ExprDoc> doc = d->GetVarDoc(mod);
          ICHECK(doc) << "Unable to print IRModule before definition in TIR.";
          return doc.value();
        });

}  // namespace printer
}  // namespace script
}  // namespace tvm
