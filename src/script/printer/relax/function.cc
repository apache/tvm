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

TVM_REGISTER_NODE_TYPE(RelaxFrameNode);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Function>("", [](relax::Function n, ObjectPath n_p, IRDocsifier d) -> Doc {
      std::unordered_set<const tir::VarNode*> func_vars;
      With<RelaxFrame> f(d);
      IdDoc func_name = d->Define(n, f(), FindFunctionName(d, n).value_or("main"));
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
        (*f)->stmts.push_back(
            ExprStmtDoc(Relax(d, "func_attr")  //
                            ->Call({d->AsDoc<ExprDoc>(n->attrs, n_p->Attr("attrs"))})));
      }
      // Step 5. Print body
      Array<StmtDoc> body =
          PrintSeqExpr(Downcast<relax::SeqExpr>(n->body), n_p->Attr("body"), d, /*use_ret=*/true);
      (*f)->stmts.insert((*f)->stmts.end(), body.begin(), body.end());
      return HeaderWrapper(
          d, FunctionDoc(func_name, params, {Relax(d, "function")}, ret_type, (*f)->stmts));
    });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ExternFunc>(  //
        "", [](relax::ExternFunc n, ObjectPath n_p, IRDocsifier d) -> Doc {
          // TODO(@junrushao): print more information out of extern function.
          return ExprStmtDoc(LiteralDoc::Str(n->global_symbol, n_p));
        });

TVM_SCRIPT_REPR(relax::FunctionNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::ExternFuncNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
