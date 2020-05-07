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
#include <tvm/te/autodiff.h>
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include <memory>
#include "ad_util.h"

namespace tvm {
namespace te {

#define NOT_IMPLEMENTED \
  { LOG(FATAL) << "Derivative of this expr is not implemented: " << GetRef<PrimExpr>(op); throw; }

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

  PrimExpr VisitExpr_(const LoadNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LetNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const CallNode* op) {
    PrimExpr expr = GetRef<PrimExpr>(op);
    if (op->call_type == CallNode::CallType::Halide) {
      if (input_.get() && op->func.same_as(input_->op) &&
          op->value_index == input_->value_index) {
        // Tensor(indices)
        CHECK_EQ(indices_.size(), op->args.size());
        PrimExpr condition = const_true();
        for (size_t i = 0; i < input_.ndim(); ++i) {
          condition = AndNode::make(condition, EQNode::make(indices_[i], op->args[i]));
        }
        return CastNode::make(op->dtype, condition);
      } else {
        return make_zero(op->dtype);
      }
    } else if (op->call_type == CallNode::CallType::PureIntrinsic) {
      static std::unordered_set<std::string> piecewise_const = {"floor", "ceil", "trunc", "round"};
      if (op->name == "exp") {
        return MulNode::make(Mutate(op->args[0]), expr);
      } else if (op->name == "log") {
        return DivNode::make(Mutate(op->args[0]), op->args[0]);
      } else if (op->name == "sigmoid") {
        return MulNode::make(Mutate(op->args[0]),
                             MulNode::make(expr, SubNode::make(FloatImm(expr.dtype(), 1.0), expr)));
      } else if (op->name == "sqrt") {
        return DivNode::make(Mutate(op->args[0]),
                             MulNode::make(expr, FloatImm(expr.dtype(), 2.0)));
      } else if (op->name == "tanh") {
        return MulNode::make(Mutate(op->args[0]),
                             SubNode::make(FloatImm(expr.dtype(), 1.0), MulNode::make(expr, expr)));
      } else if (op->name == "pow") {
        auto x = op->args[0], y = op->args[1];
        return expr * (Mutate(y)*log(x) + Mutate(x)*y/x);
      } else if (op->name == "fabs") {
        auto type = op->args[0].dtype();
        return MulNode::make(Mutate(op->args[0]),
                             SelectNode::make(GENode::make(op->args[0], make_zero(type)),
                                              FloatImm(type, 1.0), FloatImm(type, -1.0)));
      } else if (op->name == intrinsic::tvm_if_then_else) {
        Array<PrimExpr> new_args = {op->args[0],
                                    Mutate(op->args[1]),
                                    Mutate(op->args[2])};
        return CallNode::make(op->dtype, op->name, new_args,
                              op->call_type, op->func, op->value_index);
      } else if (piecewise_const.count(op->name)) {
        return FloatImm(expr.dtype(), 0.0);
      } else {
        throw dmlc::Error("Derivative of this intrinsic is not implemented: " + op->name);
      }
    }
    NOT_IMPLEMENTED
  }

  PrimExpr VisitExpr_(const AddNode* op) {
    return AddNode::make(Mutate(op->a), Mutate(op->b));
  }

  PrimExpr VisitExpr_(const SubNode* op) {
    return SubNode::make(Mutate(op->a), Mutate(op->b));
  }

  PrimExpr VisitExpr_(const MulNode* op) {
    return AddNode::make(
        MulNode::make(Mutate(op->a), op->b),
        MulNode::make(op->a, Mutate(op->b)));
  }

  PrimExpr VisitExpr_(const DivNode* op) {
    return DivNode::make(
        SubNode::make(
            MulNode::make(Mutate(op->a), op->b),
            MulNode::make(op->a, Mutate(op->b))),
        MulNode::make(op->b, op->b));
  }

  PrimExpr VisitExpr_(const ModNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const FloorDivNode* op) {
    return FloorDivNode::make(
        SubNode::make(
            MulNode::make(Mutate(op->a), op->b),
            MulNode::make(op->a, Mutate(op->b))),
        MulNode::make(op->b, op->b));
  }

  PrimExpr VisitExpr_(const FloorModNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const MinNode* op) {
    return SelectNode::make(LENode::make(op->a, op->b),
        Mutate(op->a), Mutate(op->b));
  }

  PrimExpr VisitExpr_(const MaxNode* op) {
    return SelectNode::make(GENode::make(op->a, op->b),
        Mutate(op->a), Mutate(op->b));
  }

  PrimExpr VisitExpr_(const EQNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const NENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LTNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const GTNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const GENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const AndNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const OrNode* op) NOT_IMPLEMENTED

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
        new_res = AddNode::make(new_res, MulNode::make(new_lhs[i], res_di));
      }
      for (size_t i = 0; i < new_op->combiner->rhs.size(); ++i) {
        PrimExpr res_di = Derivative(res, new_op->combiner->rhs[i]);
        // new_rhs[i] is the derivative of rhs[i] (wrt our input tensor)
        new_res = AddNode::make(new_res, MulNode::make(new_rhs[i], res_di));
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

    CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
    // Also simplify the resulting combiner
    // (mostly to get rid of unused components, e.g., the original expressions)
    return analyzer_.Simplify(
        ReduceNode::make(new_combiner, new_source, new_op->axis,
                         new_op->condition, new_op->value_index));
  }

  PrimExpr VisitExpr_(const CastNode* op) {
    if (op->dtype.is_float()) {
      return CastNode::make(op->dtype, Mutate(op->value));
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const NotNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const SelectNode* op) {
    return SelectNode::make(op->condition,
        Mutate(op->true_value), Mutate(op->false_value));
  }

  PrimExpr VisitExpr_(const RampNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const BroadcastNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const ShuffleNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const IntImmNode* op) {
    return IntImm(op->dtype, 0);
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) {
    return FloatImm(op->dtype, 0);
  }

  PrimExpr VisitExpr_(const StringImmNode* op) NOT_IMPLEMENTED

 private:
  Tensor input_;
  Array<PrimExpr> indices_;
  Var input_var_;
  arith::Analyzer analyzer_;
};

PrimExpr Derivative(const PrimExpr& expr, const Var& var) {
  return JacobianMutator(var).Mutate(expr);
}

PrimExpr Jacobian(const PrimExpr& expr, const Tensor& input, const Array<PrimExpr>& indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Tensor Jacobian(const Tensor& output, const Tensor& input) {
  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op) << "Derivative of this operation is not implemented: " << output->op;
  bool is_input_tensor = false;
  for (const Tensor& child : op->InputTensors()) {
    if (input == child) {
      is_input_tensor = true;
      break;
    }
  }
  CHECK(is_input_tensor) << "Jacobian is called on a pair of tensors such that the output "
                         << "does not directly depend on the input.";

  // We have to clone the iteration axes because otherwise the original expression
  // cannot be used together with the derivative (it will lead to errors during lowering)
  Array<IterVar> new_axis;
  Map<Var, PrimExpr> vmap;
  std::tie(new_axis, vmap) = te::CloneIterVars(op->axis);

  Array<PrimExpr> input_indices;
  size_t i = 0;
  for (PrimExpr ext : input->shape) {
    IterVar new_v = IterVarNode::make(Range(0, ext), Var("jac_i" + std::to_string(i++)),
        IterVarType::kDataPar);
    // Append jacobian iter to new_axis
    new_axis.push_back(new_v);
    // Differentiate wrt input[input_indices]
    input_indices.push_back(new_v);
  }
  arith::Analyzer analzyer;
  // Compute Jacobian
  PrimExpr new_body = Jacobian(
      Substitute(op->body[output->value_index], vmap), input, input_indices);
  new_body = analzyer.Simplify(new_body);

  int value_index = 0;
  Array<PrimExpr> new_bodies;

  // If this is a reduction then it may return a tuple and we have
  // to repeat the body several times
  if (const ReduceNode* red = new_body.as<ReduceNode>()) {
    value_index = red->value_index;
    for (size_t idx = 0; idx < red->source.size(); ++idx) {
      new_bodies.push_back(
            ReduceNode::make(red->combiner, red->source, red->axis, red->condition, idx));
    }
  } else {
    new_bodies.push_back(new_body);
  }

  auto new_op = ComputeOpNode::make(
      op->name + ".jacobian", op->tag, op->attrs, new_axis, new_bodies);

  // Jacobian shape = output.shape + input.shape
  Array<PrimExpr> new_shape = output->shape;
  for (const auto& e : input->shape) {
    new_shape.push_back(e);
  }

  return TensorNode::make(new_shape, output->dtype, new_op, value_index);
}

}  // namespace te
}  // namespace tvm
