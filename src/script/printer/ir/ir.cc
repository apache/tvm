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
#include <tvm/ir/type.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() { IRFrameNode::RegisterReflection(); }

struct SortableFunction {
  int priority;
  GlobalVar gv;
  BaseFunc func;

  explicit SortableFunction(const std::pair<GlobalVar, BaseFunc>& obj)
      : priority(0), gv(obj.first), func(obj.second) {
    if (gv->name_hint == "main") {
      priority = 1000;
    } else if (func.defined()) {
      if (func->GetTypeKey() == "tirx.PrimFunc") {
        priority = 1;
      } else if (func->GetTypeKey() == "relax.expr.ExternFunc") {
        priority = 2;
      } else if (func->GetTypeKey() == "relax.expr.Function") {
        priority = 3;
      } else {
        TVM_FFI_THROW(TypeError) << "TVMScript cannot print functions of type: "
                                 << func->GetTypeKey();
      }
    } else {
      // PrimFuncPass may leave undefined GlobalVar slots when transforming
      // this function (see tirx/ir/transform.cc); this transient state may
      // be encountered during the internal call Dump(mod) executed in
      // PrimFuncPass during debugging.
      priority = 999;
      LOG(INFO) << "Function " << gv->name_hint << " is undefined";
    }
  }

  bool operator<(const SortableFunction& other) const {
    if (this->priority != other.priority) {
      return this->priority < other.priority;
    }
    return this->gv->name_hint < other.gv->name_hint;
  }
};

ffi::Optional<ffi::String> GetDynamicDeclarationName(const StmtDoc& stmt) {
  const auto* assign = stmt.as<AssignDocNode>();
  if (assign == nullptr || !assign->rhs.has_value()) {
    return std::nullopt;
  }
  const auto* lhs = assign->lhs.as<IdDocNode>();
  const auto* call = assign->rhs.value().as<CallDocNode>();
  if (lhs == nullptr || call == nullptr) {
    return std::nullopt;
  }
  const auto* callee = call->callee.as<AttrAccessDocNode>();
  if (callee == nullptr || callee->name != "dynamic") {
    return std::nullopt;
  }
  return lhs->name;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<IRModule>(
      "", [](IRModule mod, AccessPath p, IRDocsifier d) -> Doc {
        std::vector<SortableFunction> functions;
        ffi::Array<StmtDoc> dynamic_decls;
        std::unordered_set<ffi::String> declared_dynamic_names;
        for (const auto& kv : mod->functions) {
          functions.push_back(SortableFunction(kv));
        }
        std::sort(functions.begin(), functions.end());
        With<IRFrame> f(d);
        (*f)->AddDispatchToken(d, "ir");
        IdDoc module_doc = d->Define(mod, f(), GetBindingName(d).value_or("Module"));
        (*f)->global_infos = &mod->global_infos;
        if (!mod->attrs->dict.empty()) {
          (*f)->stmts.push_back(
              ExprStmtDoc(IR(d, "module_attrs")  //
                              ->Call({d->AsDoc<ExprDoc>(mod->attrs, p->Attr("attrs"))})));
        }
        if (mod->global_infos.defined() && !mod->global_infos.empty()) {
          (*f)->stmts.push_back(ExprStmtDoc(
              IR(d, "module_global_infos")  //
                  ->Call({d->AsDoc<ExprDoc>(mod->global_infos, p->Attr("global_infos"))})));
        }
        // Declare GlobalVars first
        IdDoc module_alias =
            d->cfg->module_alias.empty() ? module_doc : IdDoc(d->cfg->module_alias);
        for (const auto& entry : functions) {
          const GlobalVar& gv = entry.gv;
          d->Define(gv, f(), [=]() {
            return d->AsDoc<ExprDoc>(mod, p->Attr("global_vars"))->Attr(gv->name_hint);
          });
        }
        // Print functions
        for (const auto& entry : functions) {
          const GlobalVar& gv = entry.gv;
          const BaseFunc& base_func = entry.func;
          d->cfg->binding_names.push_back(gv->name_hint);
          Doc doc = d->AsDoc(base_func, p->Attr("functions")->MapItem(gv));
          d->cfg->binding_names.pop_back();
          if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
            for (const StmtDoc& stmt : stmt_block->stmts) {
              if (ffi::Optional<ffi::String> name = GetDynamicDeclarationName(stmt)) {
                if (!declared_dynamic_names.count(name.value())) {
                  declared_dynamic_names.insert(name.value());
                  dynamic_decls.push_back(stmt);
                }
              }
            }
            (*f)->stmts.push_back(stmt_block->stmts.back());
            (*f)->stmts.back()->source_paths = std::move(doc->source_paths);
          } else if (auto stmt = doc.as<StmtDoc>()) {
            (*f)->stmts.push_back(stmt.value());
          } else if (auto func = doc.as<FunctionDoc>()) {
            (*f)->stmts.push_back(func.value());
          } else if (auto expr = doc.as<ExprDoc>()) {
            ExprDoc lhs = IdDoc(gv->name_hint);
            AssignDoc assignment(lhs, expr.value(), std::nullopt);
            (*f)->stmts.push_back(assignment);
          } else {
            TVM_FFI_THROW(TypeError)
                << "Expected IRModule to only contain functions, "
                << " but mod[" << gv->name_hint << "] with type  " << base_func->GetTypeKey()
                << " produced Doc type of " << doc->GetTypeKey();
          }
        }
        ClassDoc class_doc(module_doc, {IR(d, "ir_module")}, (*f)->stmts);
        if (dynamic_decls.empty()) {
          return HeaderWrapper(d, class_doc);
        }
        dynamic_decls.push_back(class_doc);
        return HeaderWrapper(d, StmtBlockDoc(dynamic_decls));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<DictAttrs>(
      "", [](DictAttrs attrs, AccessPath p, IRDocsifier d) -> Doc {
        return d->AsDoc(attrs->dict, p->Attr("dict"));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<GlobalVar>(
      "", [](GlobalVar gv, AccessPath p, IRDocsifier d) -> Doc {
        return IR(d, "GlobalVar")->Call({LiteralDoc::Str(gv->name_hint, p->Attr("name_hint"))});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<Op>("", [](Op op, AccessPath p, IRDocsifier d) -> Doc {
    return IR(d, "Op")->Call({LiteralDoc::Str(op->name, p->Attr("name"))});
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<FuncType>(
      "", [](FuncType func_type, AccessPath p, IRDocsifier d) -> Doc {
        return IR(d, "FuncType")
            ->Call({
                d->AsDoc<ExprDoc>(func_type->arg_types, p->Attr("arg_types")),
                d->AsDoc<ExprDoc>(func_type->ret_type, p->Attr("ret_type")),
            });
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<Range>(
      "ir", [](Range range, AccessPath p, IRDocsifier d) -> Doc {
        return IR(d, "Range")
            ->Call({
                d->AsDoc<ExprDoc>(range->min, p->Attr("min")),
                d->AsDoc<ExprDoc>(range->extent + range->min, p->Attr("extent")),
            });
      });
}

std::string ReprPrintIRModule(const ffi::ObjectRef& mod, const PrinterConfig& cfg) {
  return ReprPrintIR(mod, cfg);
}

TVM_REGISTER_SCRIPT_AS_REPR(GlobalVarNode, ReprPrintIR);
TVM_REGISTER_SCRIPT_AS_REPR(DictAttrsNode, ReprPrintIR);
TVM_REGISTER_SCRIPT_AS_REPR(FuncTypeNode, ReprPrintIR);
TVM_REGISTER_SCRIPT_AS_REPR(RangeNode, ReprPrintIR);
TVM_REGISTER_SCRIPT_AS_REPR(IRModuleNode, ReprPrintIRModule);

}  // namespace printer
}  // namespace script
}  // namespace tvm
