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
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

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

TVM_REGISTER_NODE_TYPE(RelaxFrameNode);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Function>("", [](relax::Function n, ObjectPath n_p, IRDocsifier d) -> Doc {
      std::unordered_set<const tir::VarNode*> func_vars;
      With<RelaxFrame> f(d);

      IdDoc func_name("");
      // if we are binding a local definition, then calling d->Define
      // will result in a repeated definition and an incorrect displayed name
      if (Optional<String> name = GetBindingName(d)) {
        func_name = std::move(IdDoc(name.value()));
      } else {
        func_name = std::move(IdDoc(FindFunctionName(d, n).value_or("main")));
      }
      (*f)->AddDispatchToken(d, "relax");
      (*f)->is_func = true;
      (*f)->func_vars = &func_vars;
      // Step 1. Print the return type
      Optional<ExprDoc> ret_type = NullOpt;
      if (const auto& func_sinfo = relax::MatchStructInfo<relax::FuncStructInfo>(n)) {
        ret_type = d->AsDoc<ExprDoc>(func_sinfo.value()->ret,  //
                                     n_p->Attr("struct_info_")->Attr("ret"));
      }
      // Step 2. Print params
      Array<AssignDoc> params;
      {
        ObjectPath params_p = n_p->Attr("params");
        for (int i = 0, l = n->params.size(); i < l; ++i) {
          params.push_back(AssignDoc(
              /*lhs=*/DefineVar(n->params[i], *f, d),
              /*rhs=*/NullOpt, StructInfoAsAnn(n->params[i], params_p->ArrayIndex(i), d, NullOpt)));
        }
      }
      // Step 3. Clean up func variables
      (*f)->func_vars = nullptr;
      // Step 4. Print attributes
      if (n->attrs.defined() && !n->attrs->dict.empty()) {
        // If the function is a global function and has a global symbol,
        // then don't print the global symbol (it will be implicit from not being private).
        // For a function without an IR module whose global symbol
        // doesn't match the function name, we should still print the global symbol attribute.
        if (AtTopLevelFunction(d) && n->attrs->dict.count(tvm::attr::kGlobalSymbol) &&
            Downcast<String>(n->attrs->dict.at(tvm::attr::kGlobalSymbol)) == func_name->name) {
          Map<String, ObjectRef> new_attrs;
          for (auto kv : n->attrs->dict) {
            if (kv.first != tvm::attr::kGlobalSymbol) {
              new_attrs.Set(kv.first, kv.second);
            }
          }
          if (!new_attrs.empty()) {
            (*f)->stmts.push_back(ExprStmtDoc(
                Relax(d, "func_attr")  //
                    ->Call({d->AsDoc<ExprDoc>(DictAttrs(new_attrs), n_p->Attr("attrs"))})));
          }
        } else {
          (*f)->stmts.push_back(
              ExprStmtDoc(Relax(d, "func_attr")  //
                              ->Call({d->AsDoc<ExprDoc>(n->attrs, n_p->Attr("attrs"))})));
        }
      }
      // Step 5. Prepare the decorator (include purity if it's impure)
      ExprDoc decorator = Relax(d, "function");
      Array<ExprDoc, void> pos_args = {};
      Array<String, void> dec_keys;
      Array<ExprDoc, void> dec_values;
      if (!n->is_pure) {
        dec_keys.push_back("pure");
        dec_values.push_back(LiteralDoc::Boolean(false, Optional<ObjectPath>()));
      }
      // if the function is global or is not in a module and does not have a global symbol,
      // indicate that it's private
      if (AtTopLevelFunction(d) &&
          (!n->attrs.defined() || !n->attrs->dict.count(tvm::attr::kGlobalSymbol))) {
        dec_keys.push_back("private");
        dec_values.push_back(LiteralDoc::Boolean(true, Optional<ObjectPath>()));
      }
      if (dec_keys.size()) {
        decorator = std::move(decorator->Call(pos_args, dec_keys, dec_values));
      }

      // Step 6. Print body
      Array<StmtDoc> body =
          PrintSeqExpr(Downcast<relax::SeqExpr>(n->body), n_p->Attr("body"), d, /*use_ret=*/true);
      (*f)->stmts.insert((*f)->stmts.end(), body.begin(), body.end());
      return HeaderWrapper(d, FunctionDoc(func_name, params, {decorator}, ret_type, (*f)->stmts));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ExternFunc>(  //
        "", [](relax::ExternFunc n, ObjectPath n_p, IRDocsifier d) -> Doc {
          // TODO(@junrushao): print more information out of extern function.
          return Relax(d, "ExternFunc")->Call({LiteralDoc::Str(n->global_symbol, n_p)});
        });

TVM_SCRIPT_REPR(relax::FunctionNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::ExternFuncNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
