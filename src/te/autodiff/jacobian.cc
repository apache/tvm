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
 * \file jacobian.cc
 * \brief Calculate Jacobian of two tensors dY/dX.
 *        X must be direct input tensor of Y.
 *        The result Jacobian shape will be (Y.shape, X.shape)
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/autodiff.h>
#include <tvm/tir/stmt_functor.h>

#include <memory>

#include "ad_utils.h"

namespace tvm {
namespace te {

#define NOT_IMPLEMENTED                                                                   \
  {                                                                                       \
    LOG(FATAL) << "Derivative of this expr is not implemented: " << GetRef<PrimExpr>(op); \
    throw;                                                                                \
  }

/*! \brief Differentiate an expression wrt a variable or a tensor element */
class JacobianMutator : public ExprMutator {
 public:
  /*!
   * \brief Differentiate wrt `input(indices)`.
   * \param input The input tensor.
   * \param indices The indices of the element with respect to which to differentiate.
   */
  explicit JacobianMutator(Tensor input, Array<PrimExpr> indices)
      : input_(input), indices_(indices) {}
  /*!
   * \brief Differentiate wrt the input variable.
   * \param input The input variable.
   */
  explicit JacobianMutator(Var input) : input_var_(input) {}

  PrimExpr Mutate(PrimExpr e) {
    if (e.dtype().is_int() || e.dtype().is_uint()) {
      LOG(WARNING) << "For now we assume that the derivative of any integer expression is always 0."
                   << " e = " << e;
      return make_zero(e.dtype());
    } else {
      return ExprMutator::VisitExpr(e);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    if (input_var_.get() && input_var_.get() == op && op->dtype.is_float()) {
      return FloatImm(op->dtype, 1.0);
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const LetNode* op) NOT_IMPLEMENTED;

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto tensor = Downcast<te::Tensor>(op->producer);
    if (input_.get() && tensor == input_) {
      // Tensor(indices)
      ICHECK_EQ(indices_.size(), op->indices.size());
      PrimExpr condition = const_true();
      for (size_t i = 0; i < input_.ndim(); ++i) {
        condition = And(condition, EQ(indices_[i], op->indices[i]));
      }
      return Cast(op->dtype, condition);
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const CallNode* op) {
    PrimExpr expr = GetRef<PrimExpr>(op);
    if (op->op.same_as(op_exp_)) {
      return Mul(Mutate(op->args[0]), expr);
    } else if (op->op.same_as(op_log_)) {
      return Div(Mutate(op->args[0]), op->args[0]);
    } else if (op->op.same_as(op_sigmoid_)) {
      return Mul(Mutate(op->args[0]), Mul(expr, Sub(FloatImm(expr.dtype(), 1.0), expr)));
    } else if (op->op.same_as(op_sqrt_)) {
      return Div(Mutate(op->args[0]), Mul(expr, FloatImm(expr.dtype(), 2.0)));
    } else if (op->op.same_as(op_tanh_)) {
      return Mul(Mutate(op->args[0]), Sub(FloatImm(expr.dtype(), 1.0), Mul(expr, expr)));
    } else if (op->op.same_as(op_pow_)) {
      auto x = op->args[0], y = op->args[1];
      return expr * (Mutate(y) * log(x) + Mutate(x) * y / x);
    } else if (op->op.same_as(op_fabs_)) {
      auto type = op->args[0].dtype();
      return Mul(Mutate(op->args[0]), Select(GE(op->args[0], make_zero(type)), FloatImm(type, 1.0),
                                             FloatImm(type, -1.0)));
    } else if (op->op.same_as(op_if_then_else_)) {
      Array<PrimExpr> new_args = {op->args[0], Mutate(op->args[1]), Mutate(op->args[2])};
      return Call(op->dtype, op->op, new_args);
    } else if (piecewise_const.count(op->op)) {
      return FloatImm(expr.dtype(), 0.0);
    } else {
      LOG(FATAL) << "Derivative of this intrinsic is not implemented: " << op->op;
    }
  }

  PrimExpr VisitExpr_(const AddNode* op) { return Add(Mutate(op->a), Mutate(op->b)); }

  PrimExpr VisitExpr_(const SubNode* op) { return Sub(Mutate(op->a), Mutate(op->b)); }

  PrimExpr VisitExpr_(const MulNode* op) {
    return Add(Mul(Mutate(op->a), op->b), Mul(op->a, Mutate(op->b)));
  }

  PrimExpr VisitExpr_(const DivNode* op) {
    return Div(Sub(Mul(Mutate(op->a), op->b), Mul(op->a, Mutate(op->b))), Mul(op->b, op->b));
  }

  PrimExpr VisitExpr_(const ModNode* op) NOT_IMPLEMENTED;

  PrimExpr VisitExpr_(const FloorDivNode* op) {
    return FloorDiv(Sub(Mul(Mutate(op->a), op->b), Mul(op->a, Mutate(op->b))), Mul(op->b, op->b));
  }

  PrimExpr VisitExpr_(const FloorModNode* op) NOT_IMPLEMENTED;

  PrimExpr VisitExpr_(const MinNode* op) {
    return Select(LE(op->a, op->b), Mutate(op->a), Mutate(op->b));
  }

  PrimExpr VisitExpr_(const MaxNode* op) {
    return Select(GE(op->a, op->b), Mutate(op->a), Mutate(op->b));
  }

  PrimExpr VisitExpr_(const EQNode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const NENode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const LTNode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const LENode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const GTNode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const GENode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const AndNode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const OrNode* op) NOT_IMPLEMENTED;

  PrimExpr VisitExpr_(const ReduceNode* op) {
    // This case is relatively difficult because a reduction expression
    // may use an arbitrary combiner.
    // The resulting reduction expression will return a tuple containing
    // both derivatives and the original results (in exactly this order).
    // The order matters when original init value is different from its derivative init value,
    // and they depend on each other during gradient calculation,
    // we must calculate derivatives first (using origin's init value),
    // switching the order (original results first, then derivatives)
    // makes the origin value be replaced before using,
    // produces incorrect results.

    // Example of a ReduceNode,
    // reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0f]),
    //   source=[A(k)], axis=[iter_var(k, range(min=0, ext=5))], where=(bool)1, value_index=0)

    // We have to clone the reduction axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    PrimExpr expr_with_new_axes = te::CloneReduction(GetRef<PrimExpr>(op));
    const ReduceNode* new_op = expr_with_new_axes.as<ReduceNode>();

    ICHECK(new_op->init.empty())
        << "Derivative of Reduction with initialization is not implemented";

    // New lhs and rhs variables of the new combiner consist of
    // variables representing derivatives (which are later derived from new_op->source)
    // followed by the original variables.
    Array<Var> new_lhs;
    for (const auto& var : new_op->combiner->lhs) {
      new_lhs.push_back(var.copy_with_suffix(".jac"));
    }
    for (const auto& var : new_op->combiner->lhs) {
      new_lhs.push_back(var);
    }

    Array<Var> new_rhs;
    for (const auto& var : new_op->combiner->rhs) {
      new_rhs.push_back(var.copy_with_suffix(".jac"));
    }
    for (const auto& var : new_op->combiner->rhs) {
      new_rhs.push_back(var);
    }

    // The new combiner result also consists of the resulting derivatives
    // followed by the original results.
    Array<PrimExpr> new_result;
    for (const auto& res : new_op->combiner->result) {
      // Each resulting derivative is computed as a sum of derivatives
      // wrt lhs and rhs multiplied by the derivatives of lhs and rhs
      PrimExpr new_res = make_zero(res.dtype());
      for (size_t i = 0; i < new_op->combiner->lhs.size(); ++i) {
        PrimExpr res_di = Derivative(res, new_op->combiner->lhs[i]);
        // new_lhs[i] is the derivative of lhs[i] (wrt our input tensor)
        new_res = Add(new_res, Mul(new_lhs[i], res_di));
      }
      for (size_t i = 0; i < new_op->combiner->rhs.size(); ++i) {
        PrimExpr res_di = Derivative(res, new_op->combiner->rhs[i]);
        // new_rhs[i] is the derivative of rhs[i] (wrt our input tensor)
        new_res = Add(new_res, Mul(new_rhs[i], res_di));
      }
      new_result.push_back(new_res);
    }
    // add original results
    for (const auto& res : new_op->combiner->result) {
      new_result.push_back(res);
    }

    // The identity is transformed in a similar way
    Array<PrimExpr> new_identity;
    for (const auto& id : new_op->combiner->identity_element) {
      new_identity.push_back(Mutate(id));
    }
    for (const auto& id : new_op->combiner->identity_element) {
      new_identity.push_back(id);
    }

    // Same as source
    Array<PrimExpr> new_source;
    for (const auto& src : new_op->source) {
      new_source.push_back(Mutate(src));
    }
    for (const auto& src : new_op->source) {
      new_source.push_back(src);
    }

    CommReducer new_combiner = CommReducer(new_lhs, new_rhs, new_result, new_identity);
    // Also simplify the resulting combiner
    // (mostly to get rid of unused components, e.g., the original expressions)
    return analyzer_.Simplify(Reduce(new_combiner, new_source, new_op->axis, new_op->condition,
                                     new_op->value_index, new_op->init));
  }

  PrimExpr VisitExpr_(const CastNode* op) {
    if (op->dtype.is_float()) {
      return Cast(op->dtype, Mutate(op->value));
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const NotNode* op) NOT_IMPLEMENTED;

  PrimExpr VisitExpr_(const SelectNode* op) {
    return Select(op->condition, Mutate(op->true_value), Mutate(op->false_value));
  }

  PrimExpr VisitExpr_(const RampNode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const BroadcastNode* op) NOT_IMPLEMENTED;
  PrimExpr VisitExpr_(const ShuffleNode* op) NOT_IMPLEMENTED;

  PrimExpr VisitExpr_(const IntImmNode* op) { return IntImm(op->dtype, 0); }

  PrimExpr VisitExpr_(const FloatImmNode* op) { return FloatImm(op->dtype, 0); }

  PrimExpr VisitExpr_(const StringImmNode* op) NOT_IMPLEMENTED;

 private:
  Tensor input_;
  Array<PrimExpr> indices_;
  Var input_var_;
  arith::Analyzer analyzer_;

  const Op& op_exp_ = Op::Get("tir.exp");
  const Op& op_log_ = Op::Get("tir.log");
  const Op& op_sigmoid_ = Op::Get("tir.sigmoid");
  const Op& op_sqrt_ = Op::Get("tir.sqrt");
  const Op& op_tanh_ = Op::Get("tir.tanh");
  const Op& op_pow_ = Op::Get("tir.pow");
  const Op& op_fabs_ = Op::Get("tir.fabs");
  const Op& op_if_then_else_ = Op::Get("tir.if_then_else");
  std::unordered_set<RelayExpr, ObjectPtrHash, ObjectPtrEqual> piecewise_const = {
      Op::Get("tir.floor"), Op::Get("tir.ceil"), Op::Get("tir.trunc"), Op::Get("tir.round")};
};

PrimExpr Derivative(const PrimExpr& expr, const Var& var) {
  return JacobianMutator(var).Mutate(expr);
}

PrimExpr Jacobian(const PrimExpr& expr, const Tensor& input, const Array<PrimExpr>& indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Tensor Jacobian(const Tensor& output, const Tensor& input) {
  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  ICHECK(op) << "Derivative of this operation is not implemented: " << output->op;
  bool is_input_tensor = false;
  for (const Tensor& child : op->InputTensors()) {
    if (input == child) {
      is_input_tensor = true;
      break;
    }
  }
  ICHECK(is_input_tensor) << "Jacobian is called on a pair of tensors such that the output "
                          << "does not directly depend on the input.";

  // We have to clone the iteration axes because otherwise the original expression
  // cannot be used together with the derivative (it will lead to errors during lowering)
  auto [new_axis, vmap] = te::CloneIterVars(op->axis);

  Array<PrimExpr> input_indices;
  size_t i = 0;
  for (PrimExpr ext : input->shape) {
    IterVar new_v =
        IterVar(Range(0, ext), Var("jac_i" + std::to_string(i++)), IterVarType::kDataPar);
    // Append jacobian iter to new_axis
    new_axis.push_back(new_v);
    // Differentiate wrt input[input_indices]
    input_indices.push_back(new_v);
  }
  arith::Analyzer analzyer;
  // Compute Jacobian
  PrimExpr new_body =
      Jacobian(Substitute(op->body[output->value_index], vmap), input, input_indices);
  new_body = analzyer.Simplify(new_body);

  int value_index = 0;
  Array<PrimExpr> new_bodies;

  // If this is a reduction then it may return a tuple and we have
  // to repeat the body several times
  if (const ReduceNode* red = new_body.as<ReduceNode>()) {
    value_index = red->value_index;
    for (size_t idx = 0; idx < red->source.size(); ++idx) {
      new_bodies.push_back(
          Reduce(red->combiner, red->source, red->axis, red->condition, idx, red->init));
    }
  } else {
    new_bodies.push_back(new_body);
  }

  auto new_op = ComputeOp(op->name + ".jacobian", op->tag, op->attrs, new_axis, new_bodies);

  // Jacobian shape = output.shape + input.shape
  Array<PrimExpr> new_shape = output->shape;
  for (const auto& e : input->shape) {
    new_shape.push_back(e);
  }

  Tensor ret = Tensor(new_shape, output->dtype, new_op, value_index);
  ret = RemoveJacobianAndLiftNonzeroCond(ret);
  return ret;
}

}  // namespace te
}  // namespace tvm
