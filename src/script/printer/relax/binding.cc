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
  // todo(yongwww): looks the relax_return_exprs are the values, and normalizer adds a new binding
  // need to figure out a way to get if the seqexpr.body was bound to one of relax_return_exprs, too
  // complicated!
  for (auto ret_expr : d->relax_return_exprs) {
    LOG(INFO) << "yongwww 33 ret_expr: " << ret_expr;
  }
  auto true_seq_expr = Downcast<SeqExpr>(n->true_branch);
  auto false_seq_expr = Downcast<SeqExpr>(n->false_branch);
  if (const auto* var_node = true_seq_expr->body.as<relax::VarNode>()) {
    auto t_var = GetRef<relax::Var>(var_node);
    LOG(INFO) << "yongwww true_seq_expr->body: " << t_var << " -- val: " << d->LookupBinding(t_var);
  }

  for (auto ele : d->binding_table_) {
    LOG(INFO) << "ele k: " << ele.first << " - value: " << ele.second;
  }

  if (const auto* var_node = false_seq_expr->body.as<relax::VarNode>()) {
    auto t_var = GetRef<relax::Var>(var_node);
    LOG(INFO) << "yongwww false_seq_expr->body: " << t_var
              << " -- val: " << d->LookupBinding(t_var);
  }
  bool ret_true_branch = false;
  bool ret_false_branch = false;
  relax::BindingBlock last_block_true = true_seq_expr->blocks[true_seq_expr->blocks.size() - 1];
  relax::Binding last_binding_true =
      last_block_true->bindings[last_block_true->bindings.size() - 1];
  if (auto* var_binding = last_binding_true.as<relax::VarBindingNode>()) {
    auto last_var_binding_true = GetRef<relax::VarBinding>(var_binding);
    if (last_var_binding_true->var.same_as(true_seq_expr->body) &&
        d->relax_return_exprs.find(last_var_binding_true->value) != d->relax_return_exprs.end()) {
      ret_true_branch = true;
      LOG(INFO) << "yongwww  ret_true_branch true";
    }
  }

  relax::BindingBlock last_block_false = false_seq_expr->blocks[false_seq_expr->blocks.size() - 1];
  relax::Binding last_binding_false =
      last_block_false->bindings[last_block_false->bindings.size() - 1];
  if (auto* var_binding = last_binding_false.as<relax::VarBindingNode>()) {
    auto last_var_binding_false = GetRef<relax::VarBinding>(var_binding);
    if (last_var_binding_false->var.same_as(false_seq_expr->body) &&
        d->relax_return_exprs.find(last_var_binding_false->value) != d->relax_return_exprs.end()) {
      ret_false_branch = true;
      LOG(INFO) << "yongwww  ret_false_branch true";
    }
  }

  if (d->relax_return_exprs.find(true_seq_expr->body) != d->relax_return_exprs.end()) {
    branches.push_back(PrintSeqExpr(true_seq_expr, n_p->Attr("true_branch"), d, ret_true_branch));
  } else {
    branches.push_back(PrintSeqExpr(true_seq_expr, n_p->Attr("true_branch"), d, ret_true_branch));
  }

  if (d->relax_return_exprs.find(false_seq_expr->body) != d->relax_return_exprs.end()) {
    branches.push_back(
        PrintSeqExpr(false_seq_expr, n_p->Attr("false_branch"), d, ret_false_branch));
  } else {
    branches.push_back(
        PrintSeqExpr(false_seq_expr, n_p->Attr("false_branch"), d, ret_false_branch));
  }

  if (var.defined()) {
    for (Array<StmtDoc>& stmts : branches) {
      if (!stmts.back()->IsInstance<ReturnDocNode>()) {
        ExprDoc ret = Downcast<ExprStmtDoc>(stmts.back())->expr;
        stmts.Set(stmts.size() - 1, AssignDoc(var.value(), ret, ann));
      }
      LOG(INFO) << "yongwww stmts.back() key: " << stmts.back()->GetTypeKey();
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
          d->binding_table_[n->var->vid] = n->value;
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
