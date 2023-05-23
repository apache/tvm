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
 * \file constant_folding.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include "../op/memory/on_device.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {
namespace transform {

namespace {
/*!
 * \brief Returns whether \p expr is a literal \p Constant, optionally wrapped by an "on_device"
 * annotation CallNode (which serves only to associate an \p VirtualDevice to the constant and has
 * no operational effect).
 */
bool IsSimpleConstant(const Expr& expr) {
  return AsIgnoringOnDevice<ConstantNode>(expr) != nullptr;
}

/*!
 * \brief Returns whether \p expr \p IsSimpleConstant directly or is a tuple of
 * \p IsComplexConstant expressions.
 */
bool IsComplexConstant(const Expr& expr) {
  if (IsSimpleConstant(expr)) {
    return true;
  } else if (const auto* tuple_node = AsIgnoringOnDevice<TupleNode>(expr)) {
    return std::all_of(tuple_node->fields.begin(), tuple_node->fields.end(), IsComplexConstant);
  } else {
    return false;
  }
}

// TODO(tvm-team) consider combine dead-code with constant folder.
// or make a more powerful partial evaluator.
class ConstantFolder : public MixedModeMutator {
 public:
  explicit ConstantFolder(IRModule module, bool fold_qnn)
      : module_(std::move(module)),
        fold_qnn_(fold_qnn),
        device_copy_op_(Op::Get("device_copy")),
        shape_of_op_(Op::Get("shape_of")),
        vm_shape_of_op_(Op::Get("vm.shape_of")),
        cast_op_(Op::Get("cast")),
        ndarray_size_op_(Op::Get("ndarray_size")) {}

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const LetNode* let_node) final {
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr new_value = Mutate(op->value);
      if (IsSimpleConstant(new_value)) {
        // Inline new value (along with any on_device annotation wrapping it) at all occurrences of
        // the variable.
        //
        // We need to retain any "on_device" annotation so that downstream 'device aware'
        // passes can still retrieve the virtual device for the constant in its new position(s). Eg:
        //   def @f(..., result_virtual_device=D) {
        //     let %x = on_device(... something we eval to a constant..., virtual_device=E)
        //     @f(..., %x, ...)
        //   }
        // Here the default virtual device is D, whereas the argument %x to @f is on E (and @f
        // expects that). No on_device annotation is required in the call according to the
        // convention used by the device-aware visitors.
        //
        // However once we've inlined the constant we need to insert an on_device, again to
        // respect the convention used by the device-aware visitors.
        //   def @f(..., result_virtual_device=D) {
        //     @f(..., on_device(...the constant..., virtual_device=E), ...)
        //   }
        VLOG(1) << "Replacing let-binding for " << op->var->name_hint()
                << " with constant:" << std::endl
                << PrettyPrint(new_value);
        memo_[op->var] = new_value;
      } else {
        this->Mutate(op->var);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      Expr expr = GetRef<Expr>(op);
      // Rely on the Memoizer to cache pre-visit values
      Expr new_value = this->Mutate(op->value);
      if (IsSimpleConstant(new_value)) {
        // The let-bound value has been inlined, drop the let-binding itself.
        this->memo_[expr] = Mutate(op->body);
      } else {
        Var new_var = Downcast<Var>(this->Mutate(op->var));
        Expr new_body = this->Mutate(op->body);
        if (new_var.same_as(op->var) && new_value.same_as(op->value) &&
            new_body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(new_var, new_value, new_body, op->span);
        }
      }
    };
    ExpandANormalForm(let_node, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let_node)];
  }

  Expr VisitExpr_(const FunctionNode* function_node) final {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      ICHECK_EQ(inside_primitive_, false);
      inside_primitive_ = true;
      auto ret = ExprMutator::VisitExpr_(function_node);
      inside_primitive_ = false;
      return ret;
    } else {
      return ExprMutator::VisitExpr_(function_node);
    }
  }

  Expr Rewrite_(const CallNode* pre_call_node, const Expr& post) final {
    Call pre_call = GetRef<Call>(pre_call_node);
    if (inside_primitive_) {
      return std::move(pre_call);
    }

    Call post_call = Downcast<Call>(post);

    if (post_call->args.empty()) {
      // We don't constant fold function with zero arguments.
      // This is a heuristic that is useful.
      // For example it is harmful to fold ones(shape=(4, 5)).
      return std::move(pre_call);
    }

    const auto* op_node = post_call->op.as<OpNode>();
    if (op_node == nullptr) {
      // Only evaluate primitives.
      return std::move(post_call);
    }
    Op op = GetRef<Op>(op_node);
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
    if (op_stateful.get(op, false)) {
      // skip stateful ops.
      return std::move(post_call);
    }
    // Try to evaluate shape_of and ndarray_size ops
    // Use the original call rather than new_call here since it still has valid checked_type
    // fields. These operators don't care about the value of their argument anyway.
    if (Optional<Expr> opt_result = EvaluateShapeOf(pre_call)) {
      return opt_result.value();
    }
    // Use the original call rather than new_call here since it still has valid checked_type
    // fields. This operator doesn't care about the value of its argument anyway.
    if (Optional<Expr> opt_result = EvaluateNdarraySize(pre_call)) {
      return opt_result.value();
    }
    static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
    static auto qnn_canonicalize = Op::GetAttrMap<FTVMLegalize>("FTVMQnnCanonicalize");
    bool is_no_qnn_canonicalized = !qnn_canonicalize.count(op);
    bool is_no_computational = fnoncomputational.count(op) && fnoncomputational[op];
    if (is_no_computational && (is_no_qnn_canonicalized || !fold_qnn_)) {
      return std::move(post_call);
    }
    if (op == device_copy_op_ || op == shape_of_op_ || op == vm_shape_of_op_ ||
        op == ndarray_size_op_) {
      // We should think about potentially constant evaluation over these ops too.
      return std::move(post_call);
    }
    if (!std::all_of(post_call->args.begin(), post_call->args.end(), IsComplexConstant)) {
      // At least one non-constant argument.
      return std::move(post_call);
    }
    // During evaluation we have obviously lost all on_device annotations. However any
    // on_device wrapping this call will be left in place.
    return ConstEvaluate(post_call);
  }

  Expr VisitExpr_(const IfNode* if_node) final {
    If new_if = Downcast<If>(ExprMutator::VisitExpr_(if_node));
    if (const auto* const_node = AsIgnoringOnDevice<ConstantNode>(new_if->cond)) {
      if (reinterpret_cast<uint8_t*>(const_node->data->data)[0]) {
        return new_if->true_branch;
      } else {
        return new_if->false_branch;
      }
    }
    return std::move(new_if);
  }

  Expr Rewrite_(const TupleGetItemNode* tuple_get_item_node,
                const Expr& post_tuple_get_item) final {
    const auto* post_tuple_get_item_node = post_tuple_get_item.as<TupleGetItemNode>();
    if (const auto* tuple_node = AsIgnoringOnDevice<TupleNode>(post_tuple_get_item_node->tuple)) {
      Expr result = tuple_node->fields[tuple_get_item_node->index];
      OnDeviceProps props = GetOnDeviceProps(post_tuple_get_item_node->tuple);
      if (props.body.defined()) {
        // (on_device((x, y, z), virtual_device=D).1 ==> on_device(y, virtual_device=D)
        return MaybeOnDeviceWithProps(result, props);
      } else {
        return result;
      }
    }
    return post_tuple_get_item;
  }

  // Convert value to expression.
  Expr ObjectToExpr(const ObjectRef& value) {
    if (value->IsInstance<runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<runtime::NDArray>(value);
      return Constant(nd_array);
    } else if (auto opt = value.as<runtime::ADT>()) {
      runtime::ADT adt = opt.value();
      Array<Expr> fields;
      for (size_t i = 0; i < adt.size(); ++i) {
        fields.push_back(ObjectToExpr(adt[i]));
      }
      return Tuple(fields);
    } else {
      LOG(FATAL) << "Cannot handle " << value->GetTypeKey();
    }
  }

  // Constant evaluate an expression.
  Expr ConstEvaluate(const Expr& expr) {
    VLOG_CONTEXT << "ConstEvaluate";
    VLOG(1) << "Evaluating :" << std::endl << PrettyPrint(expr);

    // We'll invoke the interpreter using the generic CPU device and target. Technically there's
    // no guarantee the results will be bitwise equal what we'd get on the true device, however to
    // support cross-compilation we don't want to assume the true device is available.

    // Use a fresh build context in case we are already in a build context.
    // needed for both execution and creation(due to JIT)
    With<transform::PassContext> fresh_build_ctx(transform::PassContext::Create());

    Map<String, ObjectRef> dict = (module_->attrs.defined())
                                      ? Map<String, ObjectRef>(module_->attrs.CopyOnWrite()->dict)
                                      : Map<String, ObjectRef>();

    // always use graph executor with no link-params
    dict.Set(tvm::attr::kExecutor,
             relay::Executor::Create("graph", {{"link-params", Bool(false)}}));
    Expr result = ObjectToExpr(Eval(expr, module_->type_definitions, module_->Imports(),
                                    eval_cpu_dev_, eval_cpu_target_, dict));
    VLOG(1) << "Evaluated to constant:" << std::endl << PrettyPrint(result);
    return result;
  }

  /*!
   * \brief Returns constant shape result of \p call if it of form \p shape_of(e) and \p e has
   * a non-dynamic tensor shape. Returns null otherwise.
   */
  Optional<Expr> EvaluateShapeOf(const Call& call) {
    if (call->op != shape_of_op_ && call->op != vm_shape_of_op_) {
      return {};
    }

    VLOG(1) << "Evaluating for shape_of:" << std::endl << PrettyPrint(call);
    ICHECK_EQ(call->args.size(), 1);
    const auto* param = call->attrs.as<ShapeOfAttrs>();
    ICHECK(param != nullptr);
    Expr input = call->args[0];

    tvm::Array<IndexExpr> ishape;
    if (Optional<tvm::Array<IndexExpr>> opt_shape = GetConstantShape(input)) {
      ishape = opt_shape.value();
    } else {
      return {};
    }

    // Get the constant shape
    runtime::NDArray value;
    DLDataType cdtype = DataType::Int(32);
    if (ishape.empty()) {
      value = runtime::NDArray::Empty({}, cdtype, eval_cpu_dev_);
    } else {
      ICHECK_NE(ishape.size(), 0);
      std::vector<int64_t> cshape = {static_cast<int64_t>(ishape.size())};
      value = runtime::NDArray::Empty(cshape, cdtype, eval_cpu_dev_);
      auto* dims = static_cast<int32_t*>(value->data);
      using ::tvm::tir::IntImmNode;
      for (size_t i = 0; i < ishape.size(); ++i) {
        if (const auto* dim = ishape[i].as<IntImmNode>()) {
          dims[i] = dim->value;
        } else {
          return {};
        }
      }
    }

    Constant shape = Downcast<Constant>(ObjectToExpr(value));

    if (shape->data.Shape().empty() && GetScalarFromConstant<int32_t>(shape) == 0) {
      auto ndarray = runtime::NDArray::Empty({}, cdtype, eval_cpu_dev_);
      shape = Constant(ndarray);
    }

    return CastValue(shape, param->dtype);
  }

  /*!
   * \brief Returns the constant NDArray size of result of \p call if it is of the form
   * \p ndarray_size(e) and \p e has non-dynamic tensor type. Returns null otherwise.
   */
  Optional<Expr> EvaluateNdarraySize(const Call& call) {
    if (call->op != ndarray_size_op_) {
      return {};
    }
    VLOG(1) << "Evaluating for ndarray_size:" << std::endl << PrettyPrint(call);
    ICHECK_EQ(call->args.size(), 1);
    Expr input = call->args[0];
    const auto* param = call->attrs.as<NdarraySizeAttrs>();
    ICHECK(param != nullptr);

    tvm::Array<IndexExpr> ishape;
    if (Optional<tvm::Array<IndexExpr>> opt_shape = GetConstantShape(input)) {
      ishape = opt_shape.value();
    } else {
      return {};
    }

    // Get the constant size
    runtime::NDArray value;
    DLDataType cdtype = DataType::Int(32);
    value = runtime::NDArray::Empty({}, cdtype, eval_cpu_dev_);
    auto* data = static_cast<int32_t*>(value->data);
    if (ishape.empty()) {
      *data = 0;
    } else {
      *data = 1;
      using ::tvm::tir::IntImmNode;
      for (size_t i = 0; i < ishape.size(); ++i) {
        if (const auto* dim = ishape[i].as<IntImmNode>()) {
          *data *= dim->value;
        } else {
          return {};
        }
      }
    }

    Constant size = Downcast<Constant>(ObjectToExpr(value));
    return CastValue(size, param->dtype);
  }

  Expr CastValue(const Expr& value, DataType dtype) {
    // Cast the constant into correct dtype
    auto cast_attrs = make_object<CastAttrs>();
    cast_attrs->dtype = dtype;
    Expr ret = Call(cast_op_, {value}, Attrs(cast_attrs), {});
    return ConstEvaluate(ret);
  }

  Optional<tvm::Array<IndexExpr>> GetConstantShape(const Expr& input) {
    if (const auto* const_node = AsIgnoringOnDevice<ConstantNode>(input)) {
      // TODO(mbs): This is not necessary since we only ever ask for the shapes for
      // pre-rewritten expressions which will always have a checked_type.
      return const_node->tensor_type()->shape;
    } else if (input->checked_type_.defined()) {
      return input->checked_type().as<TensorTypeNode>()->shape;
    } else {
      return {};
    }
  }

  // Module
  IRModule module_;

  // Whether to fold constants for QNN operations.
  bool fold_qnn_;

  // The kDLCPU device assumed to be available to the compiler. Used only when evaluating
  // sub-expressions.
  Device eval_cpu_dev_{kDLCPU, /*device_id=*/0};
  // The target for the above device assumed to be available to the compiler. Used only when
  // evaluating sub-expressions.
  Target eval_cpu_target_{"llvm"};

  // Cache the following ops for equivalence checking in this pass.
  const Op& device_copy_op_;
  const Op& shape_of_op_;
  const Op& vm_shape_of_op_;
  const Op& cast_op_;
  const Op& ndarray_size_op_;

  // True if currently within a "primitive" Relay Function.
  bool inside_primitive_ = false;
};

}  // namespace

TVM_REGISTER_GLOBAL("relay.analysis.check_constant").set_body_typed(IsComplexConstant);

Expr FoldConstantExpr(const Expr& expr, const IRModule& mod, bool fold_qnn) {
  VLOG_CONTEXT << "FoldConstantExpr";
  VLOG(1) << "folding:" << std::endl << PrettyPrint(expr);
  Expr result = ConstantFolder(mod, fold_qnn).VisitExpr(expr);
  VLOG(1) << "folded to:" << std::endl << PrettyPrint(result);
  return result;
}

Expr FoldConstantExpr(const Expr& expr, bool fold_qnn) {
  auto mod = IRModule::FromExpr(expr);
  return FoldConstantExpr(expr, mod, fold_qnn);
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstantExpr")
    .set_body_typed([](const Expr& expr, const IRModule& mod, bool fold_qnn) {
      return FoldConstantExpr(expr, mod, fold_qnn);
    });

Pass FoldConstant(bool fold_qnn) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext /* pc */) {
        return Downcast<Function>(FoldConstantExpr(f, m, fold_qnn));
      };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
