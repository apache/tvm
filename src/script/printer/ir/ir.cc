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

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IRModule>("", [](IRModule mod, ObjectPath p, IRDocsifier d) -> Doc {
      std::vector<std::pair<GlobalVar, BaseFunc>> functions{mod->functions.begin(),
                                                            mod->functions.end()};
      // print "main" first
      std::sort(functions.begin(), functions.end(), [](const auto& lhs, const auto& rhs) {
        String lhs_name = lhs.first->name_hint;
        String rhs_name = rhs.first->name_hint;
        if (lhs_name == "main") {
          lhs_name = "";
        }
        if (rhs_name == "main") {
          rhs_name = "";
        }
        return lhs_name < rhs_name;
      });
      ICHECK(!d->mod.defined());
      d->mod = mod;
      {
        With<IRFrame> f(d);
        (*f)->AddDispatchToken(d, "ir");
        for (const auto& kv : functions) {
          GlobalVar gv = kv.first;
          BaseFunc func = kv.second;
          (*f)->stmts.push_back(d->AsDoc<FunctionDoc>(func, p->Attr("functions")->MapValue(gv)));
        }
        return ClassDoc(IdDoc("Module"), {IR("ir_module")}, (*f)->stmts);
      }
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<DictAttrs>("", [](DictAttrs attrs, ObjectPath p, IRDocsifier d) -> Doc {
      return d->AsDoc(attrs->dict, p->Attr("dict"));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<GlobalVar>("", [](GlobalVar gv, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("GlobalVar")->Call({LiteralDoc::Str(gv->name_hint)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Op>("", [](Op op, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("Op")->Call({LiteralDoc::Str(op->name)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TypeVar>("", [](TypeVar type_var, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("TypeVar")->Call({LiteralDoc::Str(type_var->name_hint),  //
                                  LiteralDoc::Str(TypeKind2String(type_var->kind))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<GlobalTypeVar>(  //
        "", [](GlobalTypeVar type_var, ObjectPath p, IRDocsifier d) -> Doc {
          return IR("GlobalTypeVar")
              ->Call({LiteralDoc::Str(type_var->name_hint),  //
                      LiteralDoc::Str(TypeKind2String(type_var->kind))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<RelayRefType>("", [](RelayRefType ref, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("RelayRef")->Call({d->AsDoc<ExprDoc>(ref->value, p->Attr("value"))});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<TensorType>("", [](TensorType type, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("TensorType")
          ->Call({d->AsDoc<ExprDoc>(type->shape, p->Attr("shape")),
                  LiteralDoc::DataType(type->dtype)});
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<FuncType>("", [](FuncType func_type, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("FuncType")
          ->Call({
              d->AsDoc<ExprDoc>(func_type->type_params, p->Attr("type_params")),
              d->AsDoc<ExprDoc>(func_type->arg_types, p->Attr("arg_types")),
              d->AsDoc<ExprDoc>(func_type->ret_type, p->Attr("ret_type")),
          });
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<IncompleteType>("", [](IncompleteType ty, ObjectPath p, IRDocsifier d) -> Doc {
      return IR("IncompleteType")->Call({});
    });

void ReprPrintIRModule(const ObjectRef& mod, ReprPrinter* p) {
  if (const auto* f = runtime::Registry::Get("relay.ir.PrintRelayModule")) {
    if (Optional<String> s = (*f)(mod)) {
      p->stream << s.value();
      return;
    }
  }
  std::string res =
      DocToPythonScript(IRDocsifier()->AsDoc(Downcast<IRModule>(mod), ObjectPath::Root()));
  p->stream << res;
}

TVM_SCRIPT_REPR(TypeVarNode, ReprPrintIR);
TVM_SCRIPT_REPR(GlobalTypeVarNode, ReprPrintIR);
TVM_SCRIPT_REPR(GlobalVarNode, ReprPrintIR);
TVM_SCRIPT_REPR(DictAttrsNode, ReprPrintIR);
TVM_SCRIPT_REPR(RelayRefTypeNode, ReprPrintIR);
TVM_SCRIPT_REPR(FuncTypeNode, ReprPrintIR);
TVM_SCRIPT_REPR(IncompleteTypeNode, ReprPrintIR);
TVM_SCRIPT_REPR(IRModuleNode, ReprPrintIRModule);

}  // namespace printer
}  // namespace script
}  // namespace tvm
