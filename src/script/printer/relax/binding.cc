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

IfDoc PrintIfExpr(const relax::If& n, const ObjectPath& n_p, const IRDocsifier& d,  //
                  const Optional<ExprDoc>& var, const Optional<ExprDoc>& ann) {
  using relax::SeqExpr;
  ExprDoc cond = d->AsDoc<ExprDoc>(n->cond, n_p->Attr("cond"));
  std::vector<Array<StmtDoc>> branches;
  // normalizer adds a new binding, need to figure out if the seqexpr.body was bound
  auto is_return = [](const SeqExpr& seq_expr, const IRDocsifier& dd) {
    relax::BindingBlock last_block = seq_expr->blocks[seq_expr->blocks.size() - 1];
    relax::Binding last_binding = last_block->bindings[last_block->bindings.size() - 1];
    if (auto* var_binding = last_binding.as<relax::VarBindingNode>()) {
      auto last_var_binding = GetRef<relax::VarBinding>(var_binding);
      if (last_var_binding->var.same_as(seq_expr->body) &&
          dd->relax_return_exprs.find(last_var_binding->value) != dd->relax_return_exprs.end()) {
        return true;
      }
    }
    return false;
  };

  auto true_seq_expr = Downcast<SeqExpr>(n->true_branch);
  auto false_seq_expr = Downcast<SeqExpr>(n->false_branch);
  bool ret_true_branch = is_return(true_seq_expr, d);
  bool ret_false_branch = is_return(false_seq_expr, d);
  branches.push_back(PrintSeqExpr(true_seq_expr, n_p->Attr("true_branch"), d, ret_true_branch));
  branches.push_back(PrintSeqExpr(false_seq_expr, n_p->Attr("false_branch"), d, ret_false_branch));

  if (var.defined()) {
    for (Array<StmtDoc>& stmts : branches) {
      if (!stmts.back()->IsInstance<ReturnDocNode>()) {
        ExprDoc ret = Downcast<ExprStmtDoc>(stmts.back())->expr;
        stmts.Set(stmts.size() - 1, AssignDoc(var.value(), ret, ann));
      }
    }
  }
  return IfDoc(cond, branches[0], branches[1]);
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::MatchCast>(
        "", [](relax::MatchCast n, ObjectPath n_p, IRDocsifier d) -> Doc {
          using relax::StructInfo;
          using relax::MatchStructInfo;
          Optional<ExprDoc> ann = StructInfoAsAnn(n->var, n_p->Attr("var"), d, n->value);
          ExprDoc rhs = Relax(d, "match_cast")
                            ->Call({d->AsDoc<ExprDoc>(n->value, n_p->Attr("value")),
                                    d->AsDoc<ExprDoc>(n->struct_info, n_p->Attr("struct_info_"))});
          ExprDoc lhs = DefineVar(n->var, d->frames.back(), d);
          return AssignDoc(lhs, rhs, ann);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::VarBinding>(  //
        "", [](relax::VarBinding n, ObjectPath n_p, IRDocsifier d) -> Doc {
          if (const auto if_ = n->value.as<relax::IfNode>()) {
            Optional<ExprDoc> ann = StructInfoAsAnn(n->var, n_p->Attr("var"), d, n->value);
            ExprDoc lhs = DefineVar(n->var, d->frames.back(), d);
            return PrintIfExpr(GetRef<relax::If>(if_), n_p->Attr("value"), d, lhs, ann);
          } else if (n->value->IsInstance<tvm::BaseFuncNode>()) {
            IdDoc lhs = DefineVar(n->var, d->frames.back(), d);
            d->cfg->binding_names.push_back(lhs->name);
            Doc ret = d->AsDoc(n->value, n_p->Attr("value"));
            d->cfg->binding_names.pop_back();
            return ret;
          } else {
            ExprDoc rhs = d->AsDoc<ExprDoc>(n->value, n_p->Attr("value"));
            Optional<ExprDoc> ann = StructInfoAsAnn(n->var, n_p->Attr("var"), d, n->value);
            ExprDoc lhs = DefineVar(n->var, d->frames.back(), d);
            return AssignDoc(lhs, rhs, ann);
          }
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::If>("", [](relax::If n, ObjectPath n_p, IRDocsifier d) -> Doc {
      return PrintIfExpr(n, n_p, d, NullOpt, NullOpt);
    });

TVM_SCRIPT_REPR(relax::MatchCastNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::VarBindingNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::IfNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
