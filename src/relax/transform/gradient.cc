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
 * \file src/relax/transform/gradient.cc
 * \brief Reverse-mode automatic differentiation.
 *
 * Now only supports differentiating one function in the IRModule with one dataflow block
 * with respect to the only return value of the function, which needs to be scalar.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>

#include <unordered_set>

#include "../op/tensor/binary.h"
#include "../op/tensor/create.h"
#include "utils.h"

namespace tvm {
namespace relax {

using AdjointMsg = NestedMsg<Expr>;

// A tool class for GradientMutator
// Visit the forward bindings and generate the backward bindings
class BackwardBindingGenerator : private ExprVisitor {
 public:
  /*!
   * \brief Generate the backward bindings for the corresponding GradientMutator
   *
   * \param builder The BlockBuilder of GradientMutator, used to generate bindings
   * \param forward_block The forward DataflowBlock
   * \param require_grads The Var list to differentiate w.r.t.
   * \param target_var The target Var to differentiate
   * \param orig_return_value The original return value of the function. The new return value is a
   * 2-tuple, containing the original return value, and a tuple of the adjoints of parameters.
   * \return The return expr of new adjoint function.
   */
  static Expr Generate(const BlockBuilder& builder, const DataflowBlock& forward_block,
                       const Array<Var>& require_grads, const Var& target_var,
                       const Expr& orig_return_value) {
    BackwardBindingGenerator generator(builder);

    // Initialize the adjoint of target_var as ones op. We have already check the target.
    auto* target_sinfo = GetStructInfoAs<TensorStructInfoNode>(target_var);
    const Expr& target_adjoint = ones(target_sinfo->shape.value(), target_sinfo->dtype);
    UpdateStructInfo(target_adjoint, GetRef<StructInfo>(target_sinfo));
    generator.adjoint_msg_map_.Set(target_var, AdjointMsg(target_adjoint));

    // We do reverse-mode ad, so visit bindings backwards
    for (auto it = forward_block->bindings.rbegin(); it != forward_block->bindings.rend(); ++it) {
      generator.VisitBinding(*it);
    }

    return generator.Epilogue(require_grads, orig_return_value);
  }

 private:
  explicit BackwardBindingGenerator(const BlockBuilder& builder) : builder_(builder) {}

  void VisitBinding(const Binding& binding) final {
    // TODO(chaofan, yixin): support other types of bindings
    CHECK(binding->IsInstance<VarBindingNode>()) << "now only support VarBindingNode";
    auto* var_binding = binding.as<VarBindingNode>();

    auto it = adjoint_msg_map_.find(var_binding->var);
    if (it == adjoint_msg_map_.end()) {
      // This var is not used in the following bindings
      return;
    }

    // Meet the definition of binding->var
    // Create the adjoint var and bind the adjoint value to it
    EmitAdjoint(var_binding->var, (*it).second, true);

    Expr value = var_binding->value;
    // TODO(chaofan, yixin): support other types of binding values
    CHECK(value->IsInstance<CallNode>() || value->IsInstance<TupleNode>() ||
          value->IsInstance<TupleGetItemNode>() || value->IsInstance<VarNode>() ||
          value->IsInstance<ConstantNode>())
        << "now does not support the type of binding value: " << value;

    ExprVisitor::VisitBinding_(var_binding);
  }

  // Handle the adjoint expr of the inputs of binding
  // For call node, we would call the registered gradient functions
  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) final {
    static const OpAttrMap<FPrimalGradient>& gradient_op_map =
        Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

    Var adjoint_var = adjoint_var_map_[binding->var];
    const Op& call_op = Downcast<Op>(call->op);
    const Array<Expr>& partials =
        gradient_op_map[call_op](binding->var, GetRef<Call>(call), adjoint_var, builder_);
    ICHECK(partials.size() == call->args.size()) << "partials number != inputs number";

    for (size_t i = 0; i < partials.size(); ++i) {
      Expr partial = partials[i];
      if (IsCallNoGrad(partial)) {  // no grad: don't update
        continue;
      }
      if (!partial->struct_info_.defined()) {
        UpdateStructInfo(partial, GetStructInfo(call->args[i]));
      }
      UpdateAdjoint(call->args[i], partial);
    }
  }

  // For Tuple nodes, we would iterate over the input tuple and update adjoint exprs for each input
  // e.g.
  // a = (b, c)
  // b_adjoint += a_adjoint_var[0], c_adjoint += a_adjoint_var[1]
  // a = ((b, c), d)
  // b_adjoint += a_adjoint_var[0][0], c_adjoint += a_adjoint_var[0][1],
  // d_adjoint += a_adjoint_var[1]
  //
  // Here we use adjoint_var to simplify calculation
  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple) final {
    UpdateAdjoint(GetRef<Tuple>(tuple), adjoint_var_map_[binding->var]);
  }

  // For TupleGetItem nodes, we do a partial update
  // e.g.
  // b = a[0]
  // a_adjoint[0] += b_adjoint_var
  // If a_adjoint does not exist, we would create a zeros tuple as a_adjoint first, and then add
  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* tuple_get_item) final {
    ICHECK(tuple_get_item->tuple->IsInstance<VarNode>())
        << "The tuple field of a TupleGetItem is not bound to a Var";
    auto* tuple_sinfo = GetStructInfoAs<TupleStructInfoNode>(tuple_get_item->tuple);
    ICHECK(tuple_sinfo) << "The tuple field of a TupleGetItem must has a TupleStructInfo";

    const Var& tuple_var = Downcast<Var>(tuple_get_item->tuple);
    if (adjoint_msg_map_.count(tuple_var) == 0) {
      const AdjointMsg& init = InitZerosAdjointNested(GetRef<StructInfo>(tuple_sinfo));
      adjoint_msg_map_.Set(tuple_var, init);
    }

    adjoint_msg_map_.Set(tuple_var,
                         AddInAdjointMsg(adjoint_msg_map_[tuple_var], tuple_get_item->index,
                                         ExprToAdjointMsg(adjoint_var_map_[binding->var])));
  }

  // For assign nodes, we add the adjoint of output to the adjoint of input
  void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* var) final {
    UpdateAdjoint(GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) final {
    UpdateAdjoint(GetRef<Var>(var), adjoint_var_map_[binding->var]);
  }

  // For constant nodes, we do not have to handle it because it does not contribute to the adjoint
  void VisitBinding_(const VarBindingNode* binding, const ConstantNode* var) final { return; }

  // Add partial (Expr type) to the adjoint of expr
  void UpdateAdjoint(const Expr& expr, const Expr& partial) {
    DecomposeNestedMsg(expr, ExprToAdjointMsg(partial), [&](Expr leaf, AdjointMsg msg) {
      if (leaf->IsInstance<VarNode>()) {
        const Var& v = Downcast<Var>(leaf);
        if (adjoint_msg_map_.count(v) == 0) {
          adjoint_msg_map_.Set(v, msg);
        } else {
          adjoint_msg_map_.Set(v, TupleAwareAdd(adjoint_msg_map_[v], msg));
        }
      } else if (leaf->IsInstance<ConstantNode>()) {
        // nothing to do
      } else if (leaf->IsInstance<ShapeExprNode>()) {
        // must be no grad
        ICHECK(IsCallNoGrad(partial));
      } else {
        LOG(FATAL) << "UpdateAdjoint: leaf type not supported. Currently Var and Constant leaves "
                      "are supported.";
      }
    });
  }

  // Transform the adjoint expressed as NestedMsg<Expr> into adjoint Expr, and then emit it
  // If the adjoint is assigned to a DataflowVar (the adjoint corresponds to a non-output binding),
  // it would be stored in adjoint_var_map_ for future lookup
  Var EmitAdjoint(const Var& source_var, const AdjointMsg& adjoint, bool is_dataflow_var) {
    Var adjoint_var;
    if (is_dataflow_var) {
      adjoint_var = builder_->Emit(AdjointMsgToExpr(adjoint), source_var->name_hint() + "_adjoint");
      adjoint_var_map_.Set(source_var, adjoint_var);
    } else {
      adjoint_var =
          builder_->EmitOutput(AdjointMsgToExpr(adjoint), source_var->name_hint() + "_adjoint");
    }
    return adjoint_var;
  }

  // Handle the return value of the AD function.
  // Returns the new return value, which would be like:
  // Tuple(original_return_value,
  //       Tuple(adjoint_of_require_grads_1, adjoint_of_require_grads_2, ...))
  Expr Epilogue(const Array<Var>& require_grads, const Expr& orig_return_value) {
    // create adjoint variables for inputs, and then bind adjoints
    Array<Expr> out_adjoints;

    for (Var var : require_grads) {
      // If the var don't have adjoint msg, it do not contribute to the target
      // so its adjoint is zeros
      AdjointMsg adjoint =
          adjoint_msg_map_.Get(var).value_or(InitZerosAdjointNested(GetStructInfo(var)));
      Var adjoint_var = EmitAdjoint(var, adjoint, false);
      out_adjoints.push_back(adjoint_var);
    }

    return Tuple({orig_return_value, Tuple(out_adjoints)});
  }

  static bool IsCallZeros(const Expr& expr) {
    return expr->IsInstance<CallNode>() && Downcast<Call>(expr)->op == Op::Get("relax.zeros");
  }

  static bool IsCallNoGrad(const Expr& expr) {
    return expr->IsInstance<CallNode>() &&
           Downcast<Call>(expr)->op == Op::Get("relax.grad.no_grad");
  }

  static Expr AdjointMsgToExpr(AdjointMsg msg) {
    return NestedMsgToExpr<Expr>(msg, [](Optional<Expr> leaf_expr) {
      if (!leaf_expr.defined()) {
        LOG(FATAL) << "Null should not exist in AdjointMsg.";
      }
      return leaf_expr.value();
    });
  }

  static AdjointMsg ExprToAdjointMsg(Expr expr) {
    return MapToNestedMsgBySInfo<Expr>(expr, [](Expr leaf) {
      ICHECK(GetStructInfoAs<TensorStructInfoNode>(leaf))
          << "The leaf of adjoint: " << leaf << " should have StructInfo and be a Tensor.";
      return AdjointMsg(leaf);
    });
  }

  // Create a zeros AdjointMsg with specified struct info
  // When sinfo is TupleStructInfo, we would create a nested zeros Tuple
  static AdjointMsg InitZerosAdjointNested(const StructInfo& sinfo) {
    return MapToNestedMsg<Expr>(sinfo, [](StructInfo sinfo) {
      auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>();
      ICHECK(tensor_sinfo) << "The leaf of adjoint should be a Tensor.";
      ICHECK(tensor_sinfo->shape.defined()) << "Error: missing shape when building zeros tuple.";
      const Expr& init = zeros(tensor_sinfo->shape.value(), tensor_sinfo->dtype);
      UpdateStructInfo(init, sinfo);
      return init;
    });
  }

  // Return base + increment. A tuple-aware addition.
  static AdjointMsg TupleAwareAdd(const AdjointMsg& base, const AdjointMsg& increment) {
    return CombineNestedMsg(base, increment, [&](Expr lhs, Expr rhs) {
      // a small optimization: a+0=a, 0+a=a.
      if (IsCallZeros(lhs)) {
        return rhs;
      } else if (IsCallZeros(rhs)) {
        return lhs;
      }
      auto* sinfo = GetStructInfoAs<TensorStructInfoNode>(lhs);
      ICHECK(sinfo) << "The leaf of adjoint should have StructInfo and be a Tensor.";
      ICHECK(GetStructInfoAs<TensorStructInfoNode>(rhs))
          << "The leaf of adjoint should have StructInfo and be a Tensor.";
      Expr res = add(lhs, rhs);
      UpdateStructInfo(res, GetRef<StructInfo>(sinfo));
      return res;
    });
  }

  // Perform an addition in a specified position of tuple.
  // e.g. tuple=(a, b, c), index=1, increment=d, then return (a, b+d, c)
  static AdjointMsg AddInAdjointMsg(const AdjointMsg& adjoint, int index,
                                    const AdjointMsg& increment) {
    ICHECK(adjoint.IsNested()) << "The adjoint should be nested.";
    Array<AdjointMsg> arr = adjoint.NestedArray();
    ICHECK(index >= 0 && index < static_cast<int>(arr.size()));
    arr.Set(index, TupleAwareAdd(arr[index], increment));
    return AdjointMsg(arr);
  }

  // The block builder of the corresponding GradientMutator, to emit bindings
  BlockBuilder builder_;
  // Forward Var to its adjoint Var
  Map<Var, Var> adjoint_var_map_;
  // Forward Var to its adjoint NestedMsg<Expr>
  // We use NestedMsg<Expr> to save the adjoint information (equivalent to adjoint Expr)
  // When emitting, adjoint information will be transformed into adjoint Expr
  Map<Var, AdjointMsg> adjoint_msg_map_;
};

class GradientMutator : private ExprMutator {
 public:
  static IRModule Transform(IRModule mod, String func_name, Optional<Array<Var>> require_grads,
                            int target_index) {
    auto* old_func_ptr = mod->Lookup(func_name).as<FunctionNode>();
    CHECK(old_func_ptr) << func_name << "is not a Relax Function";
    auto old_func = GetRef<Function>(old_func_ptr);

    // when require_grads is not specified, it would be set to all params of the function
    auto require_grads_value = require_grads.value_or(old_func->params);

    CheckRequireGrads(require_grads_value, old_func->params, func_name);

    Function new_func = CopyWithNewVars(old_func);
    // map the parameter list into new params
    for (size_t i = 0; i < require_grads_value.size(); ++i) {
      int idx =
          std::find(old_func->params.begin(), old_func->params.end(), require_grads_value[i]) -
          old_func->params.begin();
      require_grads_value.Set(i, new_func->params[idx]);
    }

    GradientMutator mutator(mod, require_grads_value, target_index);
    Function new_func_transformed = Downcast<Function>(mutator.VisitExpr(new_func));

    IRModule new_module = GetRef<IRModule>(mod.CopyOnWrite());
    new_module->Add(GlobalVar(func_name + "_adjoint"), new_func_transformed);
    return new_module;
  }

 private:
  GradientMutator(const IRModule& module, const Array<Var>& require_grads, int target_index)
      : ExprMutator(module), require_grads_(require_grads), target_index_(target_index) {}

  Expr VisitExpr_(const FunctionNode* func) final {
    CHECK(func->body->IsInstance<SeqExprNode>()) << "The body of the function must be SeqExpr.";

    Expr new_body = this->VisitExpr(func->body);

    return Function(func->params, new_body, NullOpt, func->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) final {
    // TODO(chaofan, yixin): multiple blocks AD
    CHECK(seq_expr->blocks.size() == 1) << "now only support one dataflow block";
    // TODO(chaofan, yixin): AD in non-dataflow block.
    CHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>())
        << "now only support one dataflow block";

    // the return value should be a VarNode, and a scalar
    orig_return_expr_ = seq_expr->body;
    CheckAndSetTarget(seq_expr->body, target_index_);

    BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[0]);
    return SeqExpr({new_block}, this->return_expr_);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    builder_->BeginDataflowBlock();
    // accept bindings in the original block
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    // generate backward bindings and the return value
    return_expr_ = BackwardBindingGenerator::Generate(this->builder_, GetRef<DataflowBlock>(block),
                                                      this->require_grads_, this->target_var_,
                                                      orig_return_expr_);

    return builder_->EndBlock();
  }

  static bool IsFloatTensorSInfo(const StructInfo& sinfo) {
    auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>();
    return tensor_sinfo && tensor_sinfo->dtype.is_float();
  }

  // When the return value is a Var, it is the target;
  // when the return value is a Tuple, the target is the target_index-th field of the return value
  // Check that the target should be a Var of scalar tensor struct_info
  void CheckAndSetTarget(const Expr& e, int target_index) {
    if (auto* var = e.as<VarNode>()) {
      CHECK_EQ(target_index, 0) << "When the function has only one return value, target_index can "
                                   "only be 0. But the target_index specified is "
                                << target_index;
      target_var_ = GetRef<Var>(var);
    } else if (auto* tuple = e.as<TupleNode>()) {
      CHECK(target_index >= 0 && target_index < static_cast<int>(tuple->fields.size()))
          << "target_index should be in the range of the number of return values of the function. "
             "But the specified target_index is "
          << target_index << ", while the number of return values is " << tuple->fields.size();
      auto* var = tuple->fields[target_index].as<VarNode>();
      CHECK(var) << "Target must be a Var, but the specified target is "
                 << tuple->fields[target_index];
      target_var_ = GetRef<Var>(var);
    } else {
      LOG(FATAL) << "The return value of the function must be Var or Tuple. However, the return "
                    "value of the given function is "
                 << e;
    }
    auto target_sinfo = GetStructInfo(target_var_);
    CHECK(IsScalarTensor(target_sinfo) && IsFloatTensorSInfo(target_sinfo))
        << "The differentiation target must be a float scalar (0-dim Tensor), but the StructInfo "
           "of the given target "
        << target_var_ << " is " << GetStructInfo(target_var_);
  }

  // Check every Var in require_grads:
  // 1. there should be no duplicate var
  // 2. every var should be a parameter of the function
  // 3. the type of the input var should be Tensor of floating point dtype, or Tuple of that
  static void CheckRequireGrads(const Array<Var>& require_grads, const Array<Var>& func_params,
                                const String& func_name) {
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set;
    for (const auto& var : require_grads) {
      CHECK(std::find(func_params.begin(), func_params.end(), var) != func_params.end())
          << "There is no Var named " << var->name_hint() << " in the parameters of the function "
          << func_name;
      CHECK_EQ(var_set.count(var), 0) << "Var " << var->name_hint() << " appears more than once";
      var_set.emplace(var);

      CHECK(IsNestedTensorConditioned(GetStructInfo(var), IsFloatTensorSInfo))
          << "Only Tensors of floating point dtype or Tuples of float "
             "Tensors can require gradients, but the StructInfo of Var "
          << var->name_hint() << " is " << GetStructInfo(var);
    }
  }

  // differentiation sources
  Array<Var> require_grads_;
  // the differentiation target
  int target_index_;
  Var target_var_;
  // the return value of the original function and the differentiated function
  Expr orig_return_expr_;
  Expr return_expr_;
};

namespace transform {

Pass Gradient(String func_name, Optional<Array<Var>> require_grads, int target_index) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext pc) {
    return relax::GradientMutator::Transform(mod, func_name, require_grads, target_index);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"Gradient",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.Gradient").set_body_typed(Gradient);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
