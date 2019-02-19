/*!
 *  Copyright (c) 2018 by Contributors
 * \file autodiff.cc
 * \brief Automatic differentiation of tensor expressions
 */
#include <tvm/autodiff.h>

#include <tvm/api_registry.h>
#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/tags.h>
#include <topi/transform.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include <vector>

#include "../op/op_util.h"
#include "zero_elimination.h"

namespace tvm {
namespace ir {


DifferentiationResult DifferentiationResultNode::make(Array<Tensor> result,
                                                      Map<Tensor, Tensor> adjoints,
                                                      Map<Tensor, Map<Tensor, Tensor>> summands) {
  auto n = make_node<DifferentiationResultNode>();
  n->result = std::move(result);
  n->adjoints = adjoints;
  n->adjoint_summands = summands;
  return DifferentiationResult(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<DifferentiationResultNode>([](const DifferentiationResultNode* r, IRPrinter* p) {
    p->stream << "DifferentiationResult(result=" << r->result
              << ", adjoints=" << r->adjoints
              << ", adjoint_summands=" << r->adjoint_summands << ')';
  });

TVM_REGISTER_NODE_TYPE(DifferentiationResultNode);


#define NOT_IMPLEMENTED \
  { LOG(FATAL) << "Derivative of this expr is not implemented: " << e; throw; }

/*! \brief Differentiate an expression wrt a variable or a tensor element */
class JacobianMutator : public IRMutator {
 public:
  /*!
   * \brief Differentiate wrt `input(indices)`.
   * \param input The input tensor.
   * \param indices The indices of the element with respect to which to differentiate.
   */
  explicit JacobianMutator(Tensor input, Array<Expr> indices)
    : input_(input), indices_(indices) {}
  /*!
   * \brief Differentiate wrt the input variable.
   * \param input The input variable.
   */
  explicit JacobianMutator(VarExpr input)
    : input_var_(input) {}

  virtual Expr Mutate(Expr e) {
    if (e.type().is_int() || e.type().is_uint()) {
      // Assume that the derivative of any integer expression is always 0
      return make_zero(e.type());
    } else {
      return IRMutator::Mutate(e);
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) {
    if (input_var_.operator->() && input_var_.get() == op && op->type.is_float()) {
      return FloatImm::make(op->type, 1.0);
    } else {
      return make_zero(op->type);
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const Let* op, const Expr& e) NOT_IMPLEMENTED

  Expr Mutate_(const Call* op, const Expr& e) {
    if (op->call_type == Call::CallType::Halide) {
      if (input_.operator->() && op->func.same_as(input_->op) &&
          op->value_index == input_->value_index) {
        Expr condition = const_true();
        for (size_t i = 0; i < input_.ndim(); ++i) {
          condition = And::make(condition, EQ::make(indices_[i], op->args[i]));
        }
        return Cast::make(op->type, condition);
      } else {
        return make_zero(op->type);
      }
    } else if (op->call_type == Call::CallType::PureIntrinsic) {
      static std::unordered_set<std::string> piecewise_const = {"floor", "ceil", "trunc", "round"};
      if (op->name == "exp") {
        return Mul::make(Mutate(op->args[0]), e);
      } else if (op->name == "log") {
        return Div::make(Mutate(op->args[0]), op->args[0]);
      } else if (op->name == "sigmoid") {
        return Mul::make(Mutate(op->args[0]),
                         Mul::make(e, Sub::make(FloatImm::make(e.type(), 1.0), e)));
      } else if (op->name == "sqrt") {
        return Div::make(Mutate(op->args[0]), Mul::make(e, FloatImm::make(e.type(), 2.0)));
      } else if (op->name == "tanh") {
        return Mul::make(Mutate(op->args[0]),
                         Sub::make(FloatImm::make(e.type(), 1.0), Mul::make(e, e)));
      } else if (op->name == "pow") {
        auto x = op->args[0], y = op->args[1];
        return e * (Mutate(y)*log(x) + Mutate(x)*y/x);
      } else if (op->name == "fabs") {
        auto type = op->args[0].type();
        return Mul::make(Mutate(op->args[0]),
                         Select::make(GE::make(op->args[0], make_zero(type)),
                                      FloatImm::make(type, 1.0), FloatImm::make(type, -1.0)));
      } else if (op->name == intrinsic::tvm_if_then_else) {
        Array<Expr> new_args = {op->args[0], Mutate(op->args[1]), Mutate(op->args[2])};
        return Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
      } else if (piecewise_const.count(op->name)) {
        return FloatImm::make(e.type(), 0.0);
      } else {
        throw dmlc::Error("Derivative of this intrinsic is not implemented: " + op->name);
      }
    }
    NOT_IMPLEMENTED
  }

  Expr Mutate_(const Add* op, const Expr& e)  {
    return op->make(Mutate(op->a), Mutate(op->b));
  }

  Expr Mutate_(const Sub* op, const Expr& e)  {
    return op->make(Mutate(op->a), Mutate(op->b));
  }

  Expr Mutate_(const Mul* op, const Expr& e) {
    return Add::make(Mul::make(Mutate(op->a), op->b), Mul::make(op->a, Mutate(op->b)));
  }

  Expr Mutate_(const Div* op, const Expr& e) {
    return Div::make(
        Sub::make(Mul::make(Mutate(op->a), op->b), Mul::make(op->a, Mutate(op->b))),
        Mul::make(op->b, op->b));
  }

  Expr Mutate_(const Mod* op, const Expr& e) NOT_IMPLEMENTED

  Expr Mutate_(const Min* op, const Expr& e) {
    return Select::make(LE::make(op->a, op->b), Mutate(op->a), Mutate(op->b));
  }

  Expr Mutate_(const Max* op, const Expr& e) {
    return Select::make(GE::make(op->a, op->b), Mutate(op->a), Mutate(op->b));
  }

  Expr Mutate_(const EQ* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const NE* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const LT* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const LE* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const GT* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const GE* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const And* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const Or* op, const Expr& e) NOT_IMPLEMENTED

  Expr Mutate_(const Reduce*, const Expr& e) {
    // This case is relatively difficult because a reduction expression
    // may use an arbitrary combiner.
    // The resulting reduction expression will return a tuple containing
    // both derivatives and the original results (in exactly this order).

    // We have to clone the reduction axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    Expr expr_with_new_axes = CloneReduction(e);
    const Reduce* op = expr_with_new_axes.as<Reduce>();

    // New lhs and rhs variables of the new combiner consist of variables
    // representing derivatives followed by the original variables.
    Array<Var> new_lhs;
    for (const auto& var : op->combiner->lhs) {
      new_lhs.push_back(var.copy_with_suffix(".der"));
    }
    for (const auto& var : op->combiner->lhs) {
      new_lhs.push_back(var);
    }

    Array<Var> new_rhs;
    for (const auto& var : op->combiner->rhs) {
      new_rhs.push_back(var.copy_with_suffix(".der"));
    }
    for (const auto& var : op->combiner->rhs) {
      new_rhs.push_back(var);
    }

    // The new combiner result also consists of the resulting derivatives
    // followed by the original results.
    Array<Expr> new_result;
    for (const auto& res : op->combiner->result) {
      // Each resulting derivative is computed as a sum of derivatives
      // wrt lhs and rhs multiplied by the derivatives of lhs and rhs
      Expr new_res = make_zero(res.type());
      for (size_t i = 0; i < op->combiner->lhs.size(); ++i) {
        Expr res_di = Derivative(res, op->combiner->lhs[i]);
        // new_lhs[i] is the derivative of lhs[i] (wrt our input tensor)
        new_res = Add::make(new_res, Mul::make(new_lhs[i], res_di));
      }
      for (size_t i = 0; i < op->combiner->rhs.size(); ++i) {
        Expr res_di = Derivative(res, op->combiner->rhs[i]);
        new_res = Add::make(new_res, Mul::make(new_rhs[i], res_di));
      }
      new_result.push_back(new_res);
    }
    for (const auto& res : op->combiner->result) {
      new_result.push_back(res);
    }

    // The identity is transformed in a similar way
    Array<Expr> new_identity;
    for (const auto& id : op->combiner->identity_element) {
      new_identity.push_back(Mutate(id));
    }
    for (const auto& id : op->combiner->identity_element) {
      new_identity.push_back(id);
    }

    Array<Expr> new_source;
    for (const auto& src : op->source) {
      new_source.push_back(Mutate(src));
    }
    for (const auto& src : op->source) {
      new_source.push_back(src);
    }

    CommReducer new_combiner = CommReducerNode::make(new_lhs, new_rhs, new_result, new_identity);
    // Also simplify the resulting combiner (mostly to get rid of unused components)
    return Simplify(
        Reduce::make(new_combiner, new_source, op->axis, op->condition, op->value_index));
  }

  Expr Mutate_(const Cast* op, const Expr& e) {
    if (op->type.is_float()) {
      return Cast::make(op->type, Mutate(op->value));
    } else {
      return make_zero(op->type);
    }
  }

  Expr Mutate_(const Not* op, const Expr& e) NOT_IMPLEMENTED

  Expr Mutate_(const Select* op, const Expr& e) {
    return Select::make(op->condition, Mutate(op->true_value), Mutate(op->false_value));
  }

  Expr Mutate_(const Ramp* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const Broadcast* op, const Expr& e) NOT_IMPLEMENTED

  Expr Mutate_(const IntImm* op, const Expr& e) { return op->make(op->type, 0); }
  Expr Mutate_(const UIntImm* op, const Expr& e) { return op->make(op->type, 0); }
  Expr Mutate_(const FloatImm* op, const Expr& e) { return op->make(op->type, 0); }

  Expr Mutate_(const StringImm* op, const Expr& e) NOT_IMPLEMENTED
  Expr Mutate_(const Shuffle* op, const Expr& e) NOT_IMPLEMENTED

 private:
  Tensor input_;
  Array<Expr> indices_;
  VarExpr input_var_;
};

Expr Jacobian(const Expr& expr, const Tensor& input, const Array<Expr>& indices) {
  return JacobianMutator(input, indices).Mutate(expr);
}

Expr Derivative(const Expr& expr, const VarExpr& var) {
  return JacobianMutator(var).Mutate(expr);
}

Tensor Jacobian(const Tensor& output, const Tensor& input, bool optimize) {
  if (const ComputeOpNode* op = output->op.as<ComputeOpNode>()) {
    bool is_input_tensor = false;
    for (const Tensor& child : op->InputTensors()) {
      if (input == child) {
        is_input_tensor = true;
        break;
      }
    }
    CHECK(is_input_tensor) << "Jacobian is called on a pair of tensors such that the output "
      << "does not depend on the input. This is probably a mistake.";

    // We have to clone the iteration axes because otherwise the original expression
    // cannot be used together with the derivative (it will lead to errors during lowering)
    Array<IterVar> new_axis;
    std::unordered_map<const Variable*, Expr> vmap;
    for (IterVar iv : op->axis) {
      IterVar new_v =
        IterVarNode::make(iv->dom, iv->var.copy_with_suffix(""),
            iv->iter_type, iv->thread_tag);
      new_axis.push_back(new_v);
      vmap[iv->var.operator->()] = new_v;
    }

    // Generate new itervars for the input
    Array<Expr> input_itervars;
    size_t i = 0;
    for (Expr ext : input->shape) {
      IterVar new_v =
        IterVarNode::make(Range(0, ext), Var("jac_i" + std::to_string(i)),
            IterVarType::kDataPar);
      // Append them to new_axis
      new_axis.push_back(new_v);
      // We also need a separate array of these itervars
      input_itervars.push_back(new_v);
      ++i;
    }

    // The differentiation itself happens here
    Expr new_body =
      Jacobian(Substitute(op->body[output->value_index], vmap), input, input_itervars);
    new_body = Simplify(new_body);

    int value_index = 0;
    Array<Expr> new_bodies;

    // If this is a reduction then it may return a tuple and we have
    // to repeat the body several times
    if (const Reduce* red = new_body.as<Reduce>()) {
      value_index = red->value_index;
      for (size_t i = 0; i < red->source.size(); ++i) {
        new_bodies.push_back(
            Reduce::make(red->combiner, red->source, red->axis, red->condition, i));
      }
    } else {
      new_bodies.push_back(new_body);
    }

    auto new_op =
      ComputeOpNode::make(op->name + ".jacobian", op->tag, op->attrs, new_axis, new_bodies);

    // new_shape = output.shape + input.shape
    Array<Expr> new_shape = output->shape;
    for (const auto& e : input->shape) {
      new_shape.push_back(e);
    }

    Tensor tensor = TensorNode::make(new_shape, output->dtype, new_op, value_index);

    if (optimize) {
      tensor = OptimizeAndLiftNonzeronessConditions(tensor);
    }

    return tensor;
  } else {
    LOG(FATAL) << "Derivative of this op is not implemented: " << output->op;
    throw;
  }
}

Tensor DiffBuildingBlock(const Tensor& output, const Tensor& input, const Tensor& head) {
  Tensor jac_output_input = Jacobian(output, input);
  Tensor result = topi::tensordot(head, jac_output_input, output->shape.size(),
                                  output->op->name + "." + input->op->name + ".grad");
  // TODO(sgrechanik-h): Here we inline only jac_output_input because otherwise there will be
  // performance problems. A better solution would be to inline smartly.
  result = InlineTensors(result, {jac_output_input});
  result = OptimizeAndLiftNonzeronessConditions(result);
  result = InlineTailCall(result);
  return result;
}

DifferentiationResult Differentiate(const Tensor& output,
                                    const Array<Tensor>& inputs,
                                    const Tensor& head_or_null,
                                    const FDiffBuildingBlock& fdiff,
                                    const Map<Tensor, Array<Tensor>>& override_deps) {
  Tensor head = head_or_null;

  // If the head is a null pointer, create an identity tensor
  if (!head.get()) {
    Array<Expr> shape = output->shape;
    for (auto e : output->shape) {
      shape.push_back(e);
    }
    auto func =
      [&output](const Array<tvm::Var>& input_indices) {
        Expr res = const_true();
        for (size_t i = 0; i < output->shape.size(); ++i) {
          res = res && Expr(input_indices[i]) == Expr(input_indices[output->shape.size() + i]);
        }
        return Cast::make(output->dtype, res);
      };
    head = tvm::compute(shape, func, "identity");
  }

  // This map maps a tensor to the list of tensors immediately depending on it (using it in their
  // bodies)
  std::unordered_map<Tensor, std::vector<Tensor>> reverse_dependencies;

  // Map doesn't work correctly for Tensors, so convert it to std::unordered_map
  std::unordered_map<Tensor, Array<Tensor>> override_deps_map;
  for (auto pair : override_deps) {
    override_deps_map.insert(pair);
  }

  // Collect reverse dependencies
  std::vector<Tensor> stack({output});
  while (!stack.empty()) {
    Tensor tensor = stack.back();
    stack.pop_back();

    auto it = override_deps_map.find(tensor);
    Array<Tensor> deps = it != override_deps_map.end() ? it->second : tensor->op->InputTensors();

    for (const Tensor& child : deps) {
      if (!reverse_dependencies.count(child)) {
        stack.push_back(child);
      }
      reverse_dependencies[child].push_back(tensor);
    }
  }

  // Individual summands of the adjoints
  std::unordered_map<Tensor, Map<Tensor, Tensor>> summands;

  // This map maps tensors to the corresponding adjoints (dLoss/dTensor)
  std::unordered_map<Tensor, Tensor> adjoints;
  // head is the adjoint of output by definition
  adjoints[output] = head;

  // This is a recursive function that does all the work. It computes the adjoint for a given
  // tensor, adds it to the map, and returns it
  std::function<Tensor(const Tensor&)> compute_adjoint;
  compute_adjoint =
    [&compute_adjoint, &adjoints, &summands, &reverse_dependencies, &fdiff, &head, &output]
    (const Tensor& tensor) {
      if (!adjoints.count(tensor)) {
        // Here the adjoint hasn't been computed yet
        Tensor res_adjoint;
        std::vector<Tensor> deps = reverse_dependencies[tensor];
        if (deps.empty()) {
          // No reverse dependencies means that the output does not depend on this tensor,
          // return a zero tensor of the appropriate shape
          Array<tvm::Expr> result_shape(head->shape.begin(),
                                        head->shape.end() + (-output->shape.size()));
          for (auto e : tensor->shape) {
            result_shape.push_back(e);
          }
          res_adjoint = topi::full(result_shape, output->dtype, make_zero(output->dtype));
        } else {
          // The new adjoint is computed as a sum of the reverse dependencies' adjoints multiplied
          // by the corresponding "local" jacobians (dDep/dTensor). The computation of the jacobian
          // and the multiplication is done in the function fdiff (DiffBuildingBlock by default).
          for (const Tensor& dep : deps) {
            Tensor part = fdiff(dep, tensor, compute_adjoint(dep));
            res_adjoint = res_adjoint.get() ? topi::add(res_adjoint, part) : part;

            // Add this part to summands
            auto& summands_of_adjoint = summands[tensor];
            if (summands_of_adjoint.get()) {
              summands_of_adjoint.Set(dep, part);
            } else {
              summands_of_adjoint = Map<Tensor, Tensor>({{dep, part}});
            }
          }
        }

        adjoints[tensor] = res_adjoint;
        return res_adjoint;
      } else {
        return adjoints[tensor];
      }
    };

  // Adjoints corresponding to inputs
  Array<Tensor> result;

  // If inputs is empty, compute adjoints for all tensors, on which output depends
  if (inputs.empty()) {
    for (const auto& dep : reverse_dependencies) {
      compute_adjoint(dep.first);
    }
  }

  // Compute an adjoint for each input
  for (const Tensor& input : inputs) {
    result.push_back(compute_adjoint(input));
  }

  return DifferentiationResultNode::make(result, adjoints, summands);
}


TVM_REGISTER_API("autodiff.Jacobian")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() > 2) {
      *ret = Jacobian(args[0], args[1], args[2].operator bool());
    } else {
      *ret = Jacobian(args[0], args[1]);
    }
  });

TVM_REGISTER_API("autodiff.Derivative")
.set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Derivative(args[0], args[1]);
  });

TVM_REGISTER_API("autodiff.DiffBuildingBlock")
.set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = DiffBuildingBlock(args[0], args[1], args[2]);
  });

TVM_REGISTER_API("autodiff.Differentiate")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() <= 1) {
      *ret = Differentiate(args[0]);
    } else if (args.size() == 2) {
      *ret = Differentiate(args[0], args[1]);
    } else if (args.size() == 3) {
      *ret = Differentiate(args[0], args[1], args[2]);
    } else if (args.size() >= 4) {
      auto pfunc = args[3].operator PackedFunc();
      auto fdiff =
        [pfunc](const Tensor& o, const Tensor& i, const Tensor& h) {
          return pfunc(o, i, h);
        };
      if (args.size() >= 5) {
        *ret = Differentiate(args[0], args[1], args[2], fdiff, args[4]);
      } else {
        *ret = Differentiate(args[0], args[1], args[2], fdiff);
      }
    }
  });

}  // namespace ir
}  // namespace tvm
