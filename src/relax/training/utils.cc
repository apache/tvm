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
 * \file src/relax/transform/append_loss.cc
 * \brief A tool to append the loss function to the backbone function in an IRModule.
 */

#include "utils.h"

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_set>

#include "../transform/utils.h"

namespace tvm {
namespace relax {

/*! \brief Append the loss function to the backbone function in an IRModule.*/
class AppendLossMutator : private ExprMutator {
 public:
  static IRModule Transform(IRModule mod, String func_name, Function loss_function,
                            int num_backbone_outputs, Optional<String> new_func_name) {
    auto* old_func = mod->Lookup(func_name).as<FunctionNode>();
    CHECK(old_func) << func_name << "is not a Relax Function";

    // functions should be copied to satisfy the well-formed check
    Function new_func = CopyWithNewVars(GetRef<Function>(old_func));
    Function new_loss_func = CopyWithNewVars(loss_function);

    AppendLossMutator mutator(mod, new_loss_func, num_backbone_outputs);
    auto new_func_transformed =
        WithAttr(Downcast<Function>(mutator.VisitExpr(new_func)), tvm::attr::kGlobalSymbol,
                 new_func_name.value_or(func_name + "_loss"));

    auto new_module = GetRef<IRModule>(mod.CopyOnWrite());
    auto new_var = GlobalVar(new_func_name.value_or(func_name + "_loss"));
    new_module->Add(new_var, new_func_transformed);
    return new_module;
  }

 private:
  AppendLossMutator(const IRModule& module, const Function& loss_function, int num_backbone_outputs)
      : ExprMutator(module),
        loss_function_(loss_function),
        num_backbone_outputs_(num_backbone_outputs) {}

  Expr VisitExpr_(const FunctionNode* func) final {
    CHECK(func->body->IsInstance<SeqExprNode>() && loss_function_->body->IsInstance<SeqExprNode>())
        << "The bodies of the backbone and the loss function must be SeqExpr.";

    // Well-formed checks and setting up class members
    loss_body_ = Downcast<SeqExpr>(loss_function_->body);
    CheckLossBody();
    BackboneReturnToArr(func->body.as<SeqExprNode>()->body);
    CheckAndRemapBackboneReturn();
    CheckAndRemapLossParams(loss_function_->params);

    Array<Var> new_params = func->params;
    new_params.insert(new_params.end(), loss_function_->params.begin() + num_backbone_outputs_,
                      loss_function_->params.end());
    Expr new_body = this->VisitExpr(func->body);

    return Function(new_params, new_body, NullOpt, func->is_pure, func->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) final {
    CHECK(seq_expr->blocks.size() == 1 && seq_expr->blocks[0]->IsInstance<DataflowBlockNode>())
        << "Backbone should have only one DataflowBlock";

    auto new_blocks = Array<BindingBlock>({this->VisitBindingBlock(seq_expr->blocks[0])});
    auto ret = Array<Expr>({loss_body_->body});
    ret.insert(ret.end(), backbone_return_arr_.begin() + num_backbone_outputs_,
               backbone_return_arr_.end());
    return SeqExpr(new_blocks, ret.size() == 1 ? ret[0] : Tuple(ret));
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    builder_->BeginDataflowBlock();
    // Emit original bindings.
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // Emit bindings in the loss function.
    for (const Binding& binding : loss_body_->blocks[0]->bindings) {
      this->VisitBinding(binding);
    }

    return builder_->EndBlock();
  }

  /*!
   * \brief Using VisitExpr to remap the defined variable. This is different from the standard
   * behaviour of VisitVarDef.
   */
  Var VisitVarDef(const Var& var) final { return Downcast<Var>(this->VisitExpr(var)); }

  /*! \brief Checks the loss function have only one DataflowBlock, and returns a scalar Var. */
  void CheckLossBody() {
    CHECK(loss_body_->blocks.size() == 1 && loss_body_->blocks[0]->IsInstance<DataflowBlockNode>())
        << "The loss function should have only one DataflowBlock";
    auto var_node = loss_body_->body.as<VarNode>();
    CHECK(var_node && IsScalarTensor(GetRef<Var>(var_node)))
        << "The loss function must return a scalar(0-dim Tensor) Var";
  }

  /*!
   * \brief Convert the return value of the backbone to Array<Var>. The backbone should return one
   * or a tuple of Vars.
   */
  void BackboneReturnToArr(const Expr& backbone_return) {
    if (auto* var = backbone_return.as<VarNode>()) {
      backbone_return_arr_.push_back(GetRef<Var>(var));
    } else if (auto* tuple = backbone_return.as<TupleNode>()) {
      for (auto i : tuple->fields) {
        auto var = i.as<VarNode>();
        CHECK(var) << "The return value of the backbone should be either a Var or a Tuple of Vars";
        backbone_return_arr_.push_back(GetRef<Var>(var));
      }
    } else {
      LOG(FATAL) << "The return value of the backbone should be either a Var or a Tuple of Vars";
    }
  }

  /*!
   * \brief Check the number of elements in loss_func_params is no less than num_backbone_outputs,
   * and the elements in backbone_return_arr_ and loss_func_params have matched struct_info. Also
   * sets up var_remap_ from loss parameter Vars to backbone returned Vars.
   */
  void CheckAndRemapLossParams(const Array<Var>& loss_func_params) {
    static StructuralEqual checker;
    CHECK(static_cast<int>(loss_func_params.size()) >= num_backbone_outputs_)
        << "The number of parameters of the loss function is " << loss_func_params.size()
        << ", which is less than the given num_backbone_outputs " << num_backbone_outputs_;
    for (int i = 0; i < num_backbone_outputs_; ++i) {
      Var loss_param = loss_func_params[i];
      Var backbone_ret = backbone_return_arr_[i];
      auto loss_param_sinfo = GetStructInfo(loss_param);
      auto backbone_ret_sinfo = GetStructInfo(backbone_ret);

      CHECK(checker(backbone_ret_sinfo, loss_param_sinfo))
          << "The struct info of the " << i
          << "-th return value of backbone function is: " << backbone_ret_sinfo
          << " while the corresponding struct info of parameter of loss function is "
          << loss_param_sinfo << ", which is different.";

      this->var_remap_[loss_param->vid] = backbone_ret;
    }
  }

  /*!
   * \brief Check the number of elements in backbone_return_arr_ is no less than
   * num_backbone_outputs. Then remap Vars in backbone return values that satisfy these conditions
   * from Var to DataflowVar:
   *
   * 1. Is used in prediction_outputs of the backbone function,
   * 2. Is not used in other_outputs of the backbone function.
   *
   * Because such Vars are no longer the outputs of the new function.
   */
  void CheckAndRemapBackboneReturn() {
    CHECK(static_cast<int>(backbone_return_arr_.size()) >= num_backbone_outputs_)
        << "The number of return values of the backbone function is " << backbone_return_arr_.size()
        << ", which is less than the given num_backbone_outputs " << num_backbone_outputs_;
    std::unordered_set<Var, ObjectPtrHash> other_outputs_var(
        backbone_return_arr_.begin() + num_backbone_outputs_, backbone_return_arr_.end());
    for (int i = 0; i < num_backbone_outputs_; ++i) {
      auto var = backbone_return_arr_[i];
      if (other_outputs_var.count(var) == 0) {
        auto new_var = DataflowVar(var->vid, GetStructInfo(var), var->span);
        this->var_remap_[var->vid] = new_var;
        backbone_return_arr_.Set(i, new_var);
      }
    }
  }

  /*! \brief The loss function. */
  Function loss_function_;
  /*! \brief The number of prediction_outputs of the backbone function. */
  int num_backbone_outputs_;
  /*! \brief The body of the loss function */
  SeqExpr loss_body_;
  /*! \brief The unpacked return values of the backbone. All return values should be Vars. */
  Array<Var> backbone_return_arr_;
};

namespace transform {

Pass AppendLoss(String func_name, Function loss_function, int num_backbone_outputs,
                Optional<String> new_func_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    return relax::AppendLossMutator::Transform(mod, func_name, loss_function, num_backbone_outputs,
                                               new_func_name);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"AppendLoss",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.training.AppendLoss").set_body_typed(AppendLoss);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
