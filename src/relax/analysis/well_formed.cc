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

/*!
 * \file relax/analysis/well_formed.cc
 * \brief Check if the IRModule is well-formed.
 *
 * This pass is supposed to be applied to normalized Relax AST.
 * If it's malformed, messages will be logged as Warning.
 * This pass will check:
 *    1. Each Expr should have `struct_info_` field already populated, when
 *      `check_struct_info` is true.
 *    2. GlobalVars are defined before use.
 *    3. When a Function has a corresponding GlobalVar and a `global_symbol`
 *       attribute, the name of the GlobalVar must equal the value of the
 *       `global_symbol` attribute value.
 *    4. Any variable cannot used as different function parameters in the same IRModule
 *    5. Vars are defined before use.
 *    6. Vars are defined exactly once.
 *    7. Symbolic Vars are defined before use.
 *    8. DataflowVars cannot be defined inside BindingBlock.
 *    9. Vars defined in IfNode, except the return Var, are invisible
 *       out of the If body.(May change for new AST designs)
 *    10. SeqExpr only serves as function body, or in the true and
 *       false branches in IfNode.
 *    11. The IR is in ANF:
 *       (a) Expressions cannot contain nested complex expressions.
 *           Here are the expressions that may be nested inside other expressions:
 *           Var, DataflowVar, GlobalVar, Constant, ShapeExpr,
 *           Op, Tuple (we call these "leaf" expressions).
 *       (b) The right-hand side of a binding may contain a non-leaf expression
 *           (where all expressions nested in it are leaf expressions),
 *           other than SeqExprs (see rule 6)
 *       (c) Exceptions: The body of a Function node and the true branch
 *           and false branch of If nodes *must* be SeqExprs.
 *       (d) Places where non-leaf expressions cannot appear:
 *           * The tuple_value field of TupleGetItem nodes
 *           * The cond field of If nodes
 *           * The op or args fields of Call nodes
 *           * Inside the fields of Tuple nodes
 *    12. Expr always has checked_type_ (with the exception of Op).
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/relax/utils.h>
#include <tvm/tir/expr_functor.h>

#include <unordered_set>

namespace tvm {
namespace relax {

// TODO(relax-team): Consider further refactor using
// Scope Frame to store manage the var context.
//
/*! \brief Helper to implement well formed check.*/
class WellFormedChecker : public relax::ExprVisitor,
                          public relax::StructInfoVisitor,
                          public tir::ExprVisitor {
 public:
  static bool Check(IRModule mod, bool check_struct_info) {
    WellFormedChecker well_formed_checker = WellFormedChecker(mod, check_struct_info);

    for (const auto& it : mod->functions) {
      // visit relax.Function
      if (auto* n = it.second.as<FunctionNode>()) {
        Function func = GetRef<Function>(n);
        well_formed_checker.CheckGlobalVarAndGsymbolConsistency(it.first, func);
        well_formed_checker.VisitExpr(func);
      }
    }
    return well_formed_checker.well_formed_;
  }

 private:
  explicit WellFormedChecker(IRModule mod, bool check_struct_info)
      : mod_(std::move(mod)), check_struct_info_(check_struct_info) {}

  using relax::ExprVisitor::VisitExpr_;
  using tir::ExprVisitor::VisitExpr;
  using tir::ExprVisitor::VisitExpr_;

  // Possible mode of visitor
  enum class VisitMode {
    /*!
     * \brief Check all vars are well-defined
     */
    kDefault,
    /*!
     * \brief Match define the vars on first occurance.
     * Do not check the well-defined property of composite expr.
     */
    kMatchVarDef
  };

  void Malformed(Diagnostic diag) {
    well_formed_ = false;
    LOG(WARNING) << "This IR is not well formed: " << diag->message;
  }

  void CheckGlobalVarAndGsymbolConsistency(GlobalVar var, Function func) {
    // check name in global var and gsymbol
    Optional<String> gsymbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    if (gsymbol.defined() && gsymbol != var->name_hint) {
      Malformed(Diagnostic::Error(func->span)
                << "Name in GlobalVar is not equal to name in gsymbol: " << var->name_hint
                << " != " << gsymbol.value());
    }
  }

  void VisitExpr(const Expr& expr) final {
    if (!expr.as<OpNode>() && !expr->checked_type_.defined()) {
      Malformed(Diagnostic::Error(expr) << "The checked_type_ of Expr " << expr << " is nullptr.");
    }
    relax::ExprVisitor::VisitExpr(expr);
  }

  void VisitExpr_(const GlobalVarNode* op) final {
    GlobalVar var = GetRef<GlobalVar>(op);
    if (!(mod_->ContainGlobalVar(var->name_hint) &&
          mod_->GetGlobalVar(var->name_hint).same_as(var))) {
      Malformed(Diagnostic::Error(var) << "GlobalVar " << op->name_hint << " is not defined.");
    }

    if (op->checked_type_.defined()) {
      if ((!op->checked_type_->IsInstance<FuncTypeNode>()) &&
          (!op->checked_type_->IsInstance<PackedFuncTypeNode>())) {
        Malformed(Diagnostic::Error(var) << "The checked_type_ of GlobalVar " << op->name_hint
                                         << " must be either FuncType or PackedFuncType.");
      }
    }

    CheckStructInfo(op);
  }

  void VisitExpr_(const TupleNode* op) final {
    for (size_t i = 0; i < op->fields.size(); i++) {
      Expr expr = op->fields[i];
      if (IsLeafOrTuple(expr)) {
        this->VisitExpr(expr);
      } else {
        Malformed(Diagnostic::Error(expr)
                  << "Tuple is not in ANF form, field " << i << " gets " << expr->GetTypeKey());
      }
    }

    CheckStructInfo(op);
  }

  void VisitExpr_(const TupleGetItemNode* op) final {
    if (IsLeafOrTuple(op->tuple)) {
      this->VisitExpr(op->tuple);
    } else {
      Malformed(Diagnostic::Error(op)
                << "The tuple value in a TupleGetItem node must be a leaf expression.");
    }
    CheckStructInfo(op);
  }

  void VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    if (var_set_.count(var) == 0 && recur_vars_.count(var) == 0) {
      Malformed(Diagnostic::Error(var) << "Var " << op->name_hint() << " is not defined.");
    }
    CheckStructInfo(op);
  }

  void VisitExpr_(const DataflowVarNode* op) final {
    DataflowVar var = GetRef<DataflowVar>(op);
    if (!is_dataflow_) {
      Malformed(Diagnostic::Error(var)
                << "DataflowVar " << op->name_hint() << " is used outside DataflowBlock.");
    }
    if (dataflow_var_set_.count(var) == 0) {
      Malformed(Diagnostic::Error(var) << "DataflowVar " << op->name_hint() << " is not defined.");
    }
    CheckStructInfo(op);
  }

  void VisitExpr_(const FunctionNode* op) final {
    // save the var_set_ for local function
    auto prev_var_set = var_set_;
    auto prev_dataflow_var_set = dataflow_var_set_;
    auto prev_symbolic_var_set = symbolic_var_set_;
    bool old_dataflow_state = is_dataflow_;
    // symbolic var is not captured across function boundaries
    symbolic_var_set_.clear();
    is_dataflow_ = false;

    // first populate defs in params
    WithMode(VisitMode::kMatchVarDef, [&]() {
      ICHECK(mode_ == VisitMode::kMatchVarDef);
      for (Var param : op->params) {
        relax::StructInfoVisitor::VisitStructInfo(GetStructInfo(param));
      }
    });

    // check all expr are well defined.
    for (Var param : op->params) {
      this->VisitVarDef(param);

      if (param_var_func_map_.count(param) == 1) {
        // TODO(relax-team): Complete this error info after we integrate printer
        Malformed(Diagnostic::Error(param->span)
                  << "Relax variable " << param->name_hint()
                  << " is repeatedly used as parameters in function.");
      }
      param_var_func_map_.insert({param, GetRef<Function>(op)});
    }
    // check function ret_struct_info
    if (op->ret_struct_info.defined()) {
      this->VisitStructInfo(op->ret_struct_info);
    } else {
      Malformed(Diagnostic::Error(op) << "Function must have defined ret_struct_info");
    }

    if (auto seq = op->body.as<SeqExprNode>()) {
      this->VisitSeqExpr(seq);
    } else {
      Malformed(Diagnostic::Error(op) << "Function bodies must be sequence expressions");
    }

    is_dataflow_ = old_dataflow_state;
    dataflow_var_set_ = prev_dataflow_var_set;
    var_set_ = prev_var_set;
    symbolic_var_set_ = prev_symbolic_var_set;
  }

  void VisitExpr_(const CallNode* op) final {
    if (IsLeafOrTuple(op->op)) {
      this->VisitExpr(op->op);
    } else {
      Malformed(Diagnostic::Error(op) << "The called expression must be a leaf expression");
    }
    for (size_t i = 0; i < op->args.size(); i++) {
      Expr arg = op->args[i];
      if (IsLeafOrTuple(arg)) {
        this->VisitExpr(arg);
      } else {
        Malformed(Diagnostic::Error(arg->span)
                  << "Call is not in ANF form, arg " << i << " gets " << arg->GetTypeKey());
      }
    }

    for (const StructInfo& sinfo_arg : op->sinfo_args) {
      this->VisitStructInfo(sinfo_arg);
    }

    CheckStructInfo(op);
  }

  void VisitExpr_(const IfNode* op) final {
    if (IsLeafOrTuple(op->cond)) {
      this->VisitExpr(op->cond);
    } else {
      Malformed(Diagnostic::Error(op) << "The condition for an if node must be a leaf expression.");
    }
    auto true_seq = op->true_branch.as<SeqExprNode>();
    auto false_seq = op->false_branch.as<SeqExprNode>();
    if (true_seq && false_seq) {
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> previous_var_set = var_set_;
      std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> previous_symbolic_var_set =
          symbolic_var_set_;
      this->VisitSeqExpr(true_seq);
      var_set_ = previous_var_set;
      symbolic_var_set_ = previous_symbolic_var_set;
      this->VisitSeqExpr(false_seq);
      var_set_ = previous_var_set;
      symbolic_var_set_ = previous_symbolic_var_set;
    } else {
      Malformed(Diagnostic::Error(op) << "If node branches must be seq exprs");
    }
    CheckStructInfo(op);
  }

  void VisitExpr_(const ShapeExprNode* op) final {
    for (PrimExpr expr : op->values) {
      // check if the symbolic vars in the expr are defined, e.g, 2 * m
      tir::ExprVisitor::VisitExpr(expr);
      if (!expr.dtype().is_int()) {
        Malformed(Diagnostic::Error(expr)
                  << "Shape expressions must be of integer type, but got " << expr.dtype());
      }
    }
    CheckStructInfo(op);
  }

  void VisitExpr_(const SeqExprNode* op) final {
    Malformed(Diagnostic::Error(op) << "SeqExpr only serves as the function body in FunctionNode, "
                                       "or the true/false branch body in IfNode.");
  }

  void VisitSeqExpr(const SeqExprNode* op) {
    // a special call only if SeqExpr is the function body
    // in FunctionNode or the true/false branch body in IfNode
    for (BindingBlock block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    if (!IsLeafOrTuple(op->body)) {
      Malformed(Diagnostic::Error(op) << "SeqExpr bodies must be leaf expressions.");
    }
    this->VisitExpr(op->body);
    CheckStructInfo(op);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    bool is_lambda = false;
    if (binding->value->IsInstance<FunctionNode>()) {
      is_lambda = true;
      recur_vars_.insert(binding->var);
    }
    if (binding->value->IsInstance<tir::PrimFuncNode>()) {
      Malformed(Diagnostic::Error(binding->value) << "Inline PrimFunc is disallowed in Relax IR.");
    } else {
      this->VisitExpr(binding->value);
    }
    this->VisitVarDef(binding->var);
    if (is_lambda) {
      recur_vars_.erase(binding->var);
    }
  }

  void VisitBinding_(const MatchCastNode* binding) final {
    this->VisitExpr(binding->value);
    // define the vars
    WithMode(VisitMode::kMatchVarDef, [&]() { this->VisitStructInfo(binding->struct_info); });

    this->VisitStructInfo(binding->struct_info);
    this->VisitVarDef(binding->var);
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) final {
    bool old_is_dataflow_ = is_dataflow_;
    is_dataflow_ = true;
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    is_dataflow_ = old_is_dataflow_;
    dataflow_var_set_.clear();
  }

  void VisitVarDef_(const DataflowVarNode* var) final {
    if (!is_dataflow_) {
      Malformed(Diagnostic::Error(var)
                << "DataflowVar " << var->name_hint() << " is defined outside DataflowBlock.");
    }
    DataflowVar lv = GetRef<DataflowVar>(var);
    if (dataflow_var_set_.count(lv) == 1) {
      Malformed(Diagnostic::Error(var)
                << "DataflowVar " << lv->name_hint() << " is defined more than once.");
    }
    // register DataflowVar
    dataflow_var_set_.insert(lv);
    CheckStructInfo(var);
  }

  void VisitVarDef_(const VarNode* var) final {
    Var gv = GetRef<Var>(var);
    if (var_set_.count(gv) == 1) {
      Malformed(Diagnostic::Error(var)
                << "Var " << gv->name_hint() << " is defined more than once.");
    }
    // register Var
    var_set_.insert(gv);
    CheckStructInfo(var);
  }

  void VisitVarDef(const Var& var) final {
    if (const DataflowVarNode* lv_node = var.as<DataflowVarNode>()) {
      VisitVarDef_(lv_node);
    } else if (const VarNode* gv_node = var.as<VarNode>()) {
      VisitVarDef_(gv_node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << var->GetTypeKey();
    }
  }

  void VisitExpr_(const tir::VarNode* op) final {
    tir::Var var = GetRef<tir::Var>(op);
    // default mode, check defined.
    if (symbolic_var_set_.count(var) == 0) {
      this->Malformed(Diagnostic::Error(var)
                      << "Symbolic Var " << var->name_hint << " is not defined.");
    }
  }

  void VisitStructInfoExprField(const Expr& expr) final {
    if (mode_ == VisitMode::kMatchVarDef) {
      // populate symbolic var in first occurrence
      if (auto* op = expr.as<relax::VarNode>()) {
        auto var = GetRef<relax::Var>(op);
        if (var_set_.count(var) == 0) {
          var_set_.insert(var);
        }
      }
      if (auto* shape = expr.as<relax::ShapeExprNode>()) {
        for (auto val : shape->values) {
          this->VisitStructInfoExprField(val);
        }
      }
    } else {
      relax::ExprVisitor::VisitExpr(expr);
    }
  }

  void VisitStructInfoExprField(const PrimExpr& expr) final {
    if (mode_ == VisitMode::kMatchVarDef) {
      // populate symbolic var in first occurrence
      if (auto* op = expr.as<tir::VarNode>()) {
        auto var = GetRef<tir::Var>(op);
        if (symbolic_var_set_.count(var) == 0) {
          symbolic_var_set_.insert(var);
        }
      }
    } else {
      tir::ExprVisitor::VisitExpr(expr);
    }
  }

  void CheckStructInfo(const ExprNode* op) {
    if (!check_struct_info_) {
      return;
    }

    auto* sinfo = op->struct_info_.as<StructInfoNode>();
    if (sinfo != nullptr) {
      this->VisitStructInfo(GetRef<StructInfo>(sinfo));
    } else {
      Malformed(Diagnostic::Error(op) << "Expr must have struct_info populated. "
                                      << " Expr.type_key=" << op->GetTypeKey());
    }
  }

  // Run callback with mode.
  template <typename FType>
  void WithMode(VisitMode mode, FType callback) {
    std::swap(mode_, mode);
    callback();
    std::swap(mode_, mode);
  }

  IRModule mod_;
  const bool check_struct_info_;
  bool well_formed_ = true;
  bool is_dataflow_;
  // Current visit mode.
  VisitMode mode_ = VisitMode::kDefault;
  // set of context variables.
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> recur_vars_;
  std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> dataflow_var_set_;
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> symbolic_var_set_;
  std::unordered_map<Var, Function, ObjectPtrHash, ObjectPtrEqual> param_var_func_map_;
};

bool WellFormed(IRModule m, bool check_struct_info) {
  return WellFormedChecker::Check(std::move(m), check_struct_info);
}

TVM_REGISTER_GLOBAL(("relax.analysis.well_formed"))
    .set_body_typed([](IRModule m, bool check_struct_info) {
      return WellFormed(m, check_struct_info);
    });

}  // namespace relax
}  // namespace tvm
