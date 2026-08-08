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
#include <algorithm>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

static bool HasDefaultExternFuncType(const relax::ExternFunc& n) {
  const auto* ty = n->ty.as<relax::FuncTypeNode>();
  if (ty == nullptr || ty->params.has_value() || ty->purity ||
      !ty->ret->IsInstance<relax::AnyTypeNode>()) {
    return false;
  }
  return true;
}

bool AtTopLevelFunction(const IRDocsifier& d) {
  // fewer than 2 frames: not in a function at all
  if (d->frames.size() < 2) {
    return false;
  }
  // if the first frame is a RelaxFrame, then this is not inside a module.
  // 2 frames => we are at a function (more than 2 => nested function)
  if (d->frames[0]->IsInstance<RelaxFrameNode>()) {
    return d->frames.size() == 2;
  }
  // otherwise the first two frames pertain to an IR module,
  // so 3 frames => we are at a top-level function (more than 3 => nested function)
  return d->frames.size() == 3;
}

TVM_FFI_STATIC_INIT_BLOCK() { RelaxFrameNode::RegisterReflection(); }

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Function>("", [](relax::Function n, AccessPath n_p, IRDocsifier d) -> Doc {
      std::unordered_set<const VarNode*> func_vars;
      std::unordered_set<const VarNode*> type_vars;
      std::unordered_set<const VarNode*> prim_params;
      With<RelaxFrame> f(d);

      IdDoc func_name("");
      // if we are binding a local definition, then calling d->Define
      // will result in a repeated definition and an incorrect displayed name
      if (ffi::Optional<ffi::String> name = GetBindingName(d)) {
        func_name = IdDoc(name.value());
      } else {
        func_name = IdDoc(FindFunctionName(d, n).value_or("main"));
      }
      (*f)->AddDispatchToken(d, "relax");
      (*f)->is_func = true;
      (*f)->func_vars = &func_vars;
      (*f)->type_vars = &type_vars;
      (*f)->prim_params = &prim_params;
      for (const Var& param : n->params) {
        if (param->ty.as<PrimTypeNode>()) {
          prim_params.insert(param.get());
        }
      }
      // Step 1. Print params
      ffi::Array<AssignDoc> params;
      {
        AccessPath params_p = n_p->Attr("params");
        for (int i = 0, l = n->params.size(); i < l; ++i) {
          params.push_back(AssignDoc(
              /*lhs=*/DefineRelaxVar(n->params[i], *f, d),
              /*rhs=*/std::nullopt,
              TypeAsAnn(n->params[i], params_p->ArrayItem(i), d, std::nullopt)));
        }
      }
      // Step 2. Print the return type
      ffi::Optional<ExprDoc> ret_type = d->AsDoc<ExprDoc>(n->ret_ty, n_p->Attr("ret_ty"));
      // Step 3. Clean up func variables
      (*f)->func_vars = nullptr;
      (*f)->type_vars = nullptr;
      (*f)->prim_params = nullptr;
      // Step 4. Print attributes
      ffi::Map<ffi::String, Any> printable_attrs;
      for (const auto& [key, value] : n->attrs->dict) {
        // A matching global symbol is implicit for a top-level function.
        if (key == tvm::attr::kGlobalSymbol && AtTopLevelFunction(d) &&
            value.as_or_throw<ffi::String>() == func_name->name) {
          continue;
        }
        printable_attrs.Set(key, value);
      }
      if (!printable_attrs.empty()) {
        (*f)->stmts.push_back(ExprStmtDoc(
            Relax(d, "func_attr")  //
                ->Call({d->AsDoc<ExprDoc>(DictAttrs(printable_attrs), n_p->Attr("attrs"))})));
      }
      // Step 5. Prepare the decorator (include purity if it's impure)
      ExprDoc decorator = Relax(d, "function");
      ffi::Array<ExprDoc, void> pos_args = {};
      ffi::Array<ffi::String, void> dec_keys;
      ffi::Array<ExprDoc, void> dec_values;
      if (!n->is_pure) {
        dec_keys.push_back("pure");
        dec_values.push_back(LiteralDoc::Boolean(false, ffi::Optional<AccessPath>()));
      }
      // if the function is global or is not in a module and does not have a global symbol,
      // indicate that it's private
      if (AtTopLevelFunction(d) && !n->attrs->dict.count(tvm::attr::kGlobalSymbol)) {
        dec_keys.push_back("private");
        dec_values.push_back(LiteralDoc::Boolean(true, ffi::Optional<AccessPath>()));
      }
      if (dec_keys.size()) {
        decorator = decorator->Call(pos_args, dec_keys, dec_values);
      }

      // Step 6. Print body
      ffi::Array<StmtDoc> body = PrintSeqExpr(n->body, n_p->Attr("body"), d, /*use_ret=*/true);
      (*f)->stmts.insert((*f)->stmts.end(), body.begin(), body.end());
      auto type_var_docs = DefineTypeVarDocs(type_vars, ffi::GetRef<Frame>((*f).get()), d);
      return WrapFunctionDocWithTypeVars(
          d, FunctionDoc(func_name, params, {decorator}, ret_type, (*f)->stmts), type_var_docs);
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ExternFunc>(  //
        "", [](relax::ExternFunc n, AccessPath n_p, IRDocsifier d) -> Doc {
          ffi::Array<ExprDoc> args;
          args.push_back(LiteralDoc::Str(n->global_symbol, n_p->Attr("global_symbol")));
          if (!HasDefaultExternFuncType(n)) {
            args.push_back(d->AsDoc<ExprDoc>(n->ty, n_p->Attr("ty")));
          }
          return Relax(d, "ExternFunc")->Call(args);
        });

TVM_REGISTER_SCRIPT_AS_REPR(relax::FunctionNode, ReprPrintRelax);
TVM_REGISTER_SCRIPT_AS_REPR(relax::ExternFuncNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
