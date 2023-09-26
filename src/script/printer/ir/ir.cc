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
#include <tvm/ir/tensor_type.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_REGISTER_NODE_TYPE(IRFrameNode);

struct SortableFunction {
  int priority;
  GlobalVar gv;
  BaseFunc func;

  explicit SortableFunction(const std::pair<GlobalVar, BaseFunc>& obj)
      : priority(0), gv(obj.first), func(obj.second) {
    if (gv->name_hint == "main") {
      priority = 1000;
    } else if (obj.second->GetTypeKey() == "tir.PrimFunc") {
      priority = 1;
    } else if (obj.second->GetTypeKey() == "relax.expr.ExternFunc") {
      priority = 2;
    } else if (obj.second->GetTypeKey() == "relax.expr.Function") {
      priority = 3;
    } else {
      LOG(FATAL) << "TypeError: TVMScript cannot print functions of type: "
                 << obj.second->GetTypeKey();
    }
  }

  bool operator<(const SortableFunction& other) const {
    if (this->priority != other.priority) {
      return this->priority < other.priority;
    }
    return this->gv->name_hint < other.gv->name_hint;
  }
};

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IRModule>("", [](IRModule mod, ObjectPath p, IRDocsifier d) -> Doc {
      std::vector<SortableFunction> functions;
      for (const auto& kv : mod->functions) {
        functions.push_back(SortableFunction(kv));
      }
      std::sort(functions.begin(), functions.end());
      With<IRFrame> f(d);
      (*f)->AddDispatchToken(d, "ir");
      IdDoc module_doc = d->Define(mod, f(), GetBindingName(d).value_or("Module"));
      (*f)->global_infos = &mod->global_infos;
      if (mod->attrs.defined() && !mod->attrs->dict.empty()) {
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
      IdDoc module_alias = d->cfg->module_alias.empty() ? module_doc : IdDoc(d->cfg->module_alias);
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
        Doc doc = d->AsDoc(base_func, p->Attr("functions")->MapValue(gv));
        d->cfg->binding_names.pop_back();
        if (const auto* stmt_block = doc.as<StmtBlockDocNode>()) {
          (*f)->stmts.push_back(stmt_block->stmts.back());
          (*f)->stmts.back()->source_paths = std::move(doc->source_paths);
        } else if (auto stmt = doc.as<StmtDoc>()) {
          (*f)->stmts.push_back(stmt.value());
        } else if (auto func = doc.as<FunctionDoc>()) {
          (*f)->stmts.push_back(func.value());
        } else if (auto expr = doc.as<ExprDoc>()) {
          ExprDoc lhs = IdDoc(gv->name_hint);
          AssignDoc assignment(lhs, expr.value(), NullOpt);
          (*f)->stmts.push_back(assignment);
        } else {
          LOG(FATAL) << "TypeError: "
                     << "Expected IRModule to only contain functions, "
                     << " but mod[" << gv->name_hint << "] with type  " << base_func->GetTypeKey()
                     << " produced Doc type of " << doc->GetTypeKey();
        }
      }
      return HeaderWrapper(d, ClassDoc(module_doc, {IR(d, "ir_module")}, (*f)->stmts));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<DictAttrs>("", [](DictAttrs attrs, ObjectPath p, IRDocsifier d) -> Doc {
      return d->AsDoc(attrs->dict, p->Attr("dict"));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<GlobalVar>("", [](GlobalVar gv, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "GlobalVar")->Call({LiteralDoc::Str(gv->name_hint, p->Attr("name_hint"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<DummyGlobalInfo>("", [](GlobalInfo ginfo, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "dummy_global_info")->Call({});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<VDevice>("", [](VDevice vdev, ObjectPath p, IRDocsifier d) -> Doc {
      d->AddGlobalInfo("vdevice", vdev);
      Map<String, ObjectRef> config = vdev->target->Export();
      return IR(d, "vdevice")
          ->Call({d->AsDoc<ExprDoc>(config, p),
                  LiteralDoc::Int(vdev->vdevice_id, p->Attr("vdevice_id")),
                  LiteralDoc::Str(vdev->memory_scope, p->Attr("memory_scope"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Op>("", [](Op op, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "Op")->Call({LiteralDoc::Str(op->name, p->Attr("name"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TypeVar>("", [](TypeVar var, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "TypeVar")
          ->Call({LiteralDoc::Str(var->name_hint, p->Attr("name_hint")),  //
                  LiteralDoc::Str(TypeKind2String(var->kind), p->Attr("kind"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<GlobalTypeVar>(  //
        "", [](GlobalTypeVar var, ObjectPath p, IRDocsifier d) -> Doc {
          return IR(d, "GlobalTypeVar")
              ->Call({LiteralDoc::Str(var->name_hint, p->Attr("name_hint")),
                      LiteralDoc::Str(TypeKind2String(var->kind), p->Attr("kind"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<RelayRefType>("", [](RelayRefType ref, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "RelayRef")->Call({d->AsDoc<ExprDoc>(ref->value, p->Attr("value"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TensorType>("", [](TensorType type, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "TensorType")
          ->Call({d->AsDoc<ExprDoc>(type->shape, p->Attr("shape")),
                  LiteralDoc::DataType(type->dtype, p->Attr("dtype"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<FuncType>("", [](FuncType func_type, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "FuncType")
          ->Call({
              d->AsDoc<ExprDoc>(func_type->type_params, p->Attr("type_params")),
              d->AsDoc<ExprDoc>(func_type->arg_types, p->Attr("arg_types")),
              d->AsDoc<ExprDoc>(func_type->ret_type, p->Attr("ret_type")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IncompleteType>("", [](IncompleteType ty, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "IncompleteType")->Call({});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Range>("ir", [](Range range, ObjectPath p, IRDocsifier d) -> Doc {
      return IR(d, "Range")
          ->Call({
              d->AsDoc<ExprDoc>(range->min, p->Attr("min")),
              d->AsDoc<ExprDoc>(range->extent + range->min, p->Attr("extent")),
          });
    });

std::string ReprPrintIRModule(const ObjectRef& mod, const PrinterConfig& cfg) {
  if (const auto* f = runtime::Registry::Get("relay.ir.PrintRelayModule")) {
    if (Optional<String> s = (*f)(mod)) {
      return s.value();
    }
  }
  return ReprPrintIR(mod, cfg);
}

TVM_SCRIPT_REPR(TypeVarNode, ReprPrintIR);
TVM_SCRIPT_REPR(GlobalTypeVarNode, ReprPrintIR);
TVM_SCRIPT_REPR(GlobalVarNode, ReprPrintIR);
TVM_SCRIPT_REPR(DictAttrsNode, ReprPrintIR);
TVM_SCRIPT_REPR(RelayRefTypeNode, ReprPrintIR);
TVM_SCRIPT_REPR(FuncTypeNode, ReprPrintIR);
TVM_SCRIPT_REPR(IncompleteTypeNode, ReprPrintIR);
TVM_SCRIPT_REPR(RangeNode, ReprPrintIR);
TVM_SCRIPT_REPR(IRModuleNode, ReprPrintIRModule);

}  // namespace printer
}  // namespace script
}  // namespace tvm
