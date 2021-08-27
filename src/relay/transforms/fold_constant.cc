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
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

using FInterpreter = runtime::TypedPackedFunc<ObjectRef(Expr)>;

class ConstantChecker : private ExprVisitor {
 public:
  // Check whether an expression is constant. The results are memoized.
  bool Check(const Expr& expr) {
    // The `ConstantNode` case is common enough that we check directly for the
    // case here, to avoid the time overhead of dispatching through the vtable
    // and the space overhead of memoizing always-true results.
    if (expr.as<ConstantNode>()) {
      return true;
    }
    const auto it = memo_.find(expr);
    if (it != memo_.end()) return it->second;
    VisitExpr(expr);
    return memo_[expr];  // return memoized result or the default value false
  }

 private:
  std::unordered_map<Expr, bool, ObjectPtrHash, ObjectPtrEqual> memo_;

  void VisitExpr_(const TupleNode* n) final {
    bool result = true;
    for (const auto& field : n->fields) {
      if (!Check(field)) {
        result = false;
        break;
      }
    }
    memo_[GetRef<Tuple>(n)] = result;
  }
};

bool ConstantCheck(const Expr& e) { return ConstantChecker().Check(e); }

TVM_REGISTER_GLOBAL("relay.analysis.check_constant").set_body_typed(ConstantCheck);

// TODO(tvm-team) consider combine dead-code with constant folder.
// or make a more powerful partial evaluator.
class ConstantFolder : public MixedModeMutator {
 public:
  explicit ConstantFolder(IRModule module)
      : module_(module),
        device_copy_op_(Op::Get("device_copy")),
        shape_of_op_(Op::Get("shape_of")),
        vm_shape_of_op_(Op::Get("vm.shape_of")),
        cast_op_(Op::Get("cast")),
        ndarray_size_op_(Op::Get("ndarray_size")) {}

  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->Mutate(op->value);
      if (value.as<ConstantNode>()) {
        this->memo_[op->var] = value;
      } else {
        this->Mutate(op->var);
      }
    };
    auto post_visit = [this](const LetNode* op) {
      Expr expr = GetRef<Expr>(op);
      // Rely on the Memoizer to cache pre-visit values
      Expr value = this->Mutate(op->value);
      if (value.as<ConstantNode>()) {
        this->memo_[expr] = this->Mutate(op->body);
      } else {
        Var var = Downcast<Var>(this->Mutate(op->var));
        Expr body = this->Mutate(op->body);
        if (var.same_as(op->var) && value.same_as(op->value) && body.same_as(op->body)) {
          this->memo_[expr] = expr;
        } else {
          this->memo_[expr] = Let(var, value, body);
        }
      }
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  bool inside_primitive = false;
  Expr VisitExpr_(const FunctionNode* op) final {
    if (op->HasNonzeroAttr(attr::kPrimitive)) {
      ICHECK_EQ(inside_primitive, false);
      inside_primitive = true;
      auto ret = ExprMutator::VisitExpr_(op);
      inside_primitive = false;
      return ret;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  Expr VisitExpr_(const IfNode* op) final {
    auto new_cond = ExprMutator::VisitExpr(op->cond);
    if (auto const_cond = new_cond.as<ConstantNode>()) {
      if (reinterpret_cast<uint8_t*>(const_cond->data->data)[0]) {
        return ExprMutator::VisitExpr(op->true_branch);
      } else {
        return ExprMutator::VisitExpr(op->false_branch);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (inside_primitive) {
      return GetRef<Expr>(call);
    }
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");

    auto origin_args = call->args;
    call = post.as<CallNode>();
    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return post;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return post;
    // skip stateful ops.
    if (op_stateful.get(GetRef<Op>(op), false)) return post;
    // Try to evaluate shape_of op
    if (call->op == shape_of_op_ || call->op == vm_shape_of_op_) {
      return EvaluateShapeOf(post, origin_args, call->attrs);
    }

    if (call->op == ndarray_size_op_) {
      return EvaluateNdarraySize(post, origin_args, call->attrs);
    }

    // We should think about potentially constant evaluation over these ops too.
    static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
    if (const auto* call_node = call->op.as<OpNode>()) {
      Op op = GetRef<Op>(call_node);
      if ((fnoncomputational.count(op) && fnoncomputational[op]) || (call->op == device_copy_op_)) {
        return GetRef<Call>(call);
      }
    }

    bool all_const_args = true;
    for (Expr arg : call->args) {
      if (!checker_.Check(arg)) {
        all_const_args = false;
      }
    }
    if (all_const_args) {
      return ConstEvaluate(post);
    } else {
      return post;
    }
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) final {
    op = post.as<TupleGetItemNode>();
    if (const auto* tuple = op->tuple.as<TupleNode>()) {
      return tuple->fields[op->index];
    } else {
      return post;
    }
  }

 private:
  // Internal constant checker
  ConstantChecker checker_;
  // Module
  IRModule module_;

  // Cache the following ops for equivalence checking in this pass.
  const Op& device_copy_op_;
  const Op& shape_of_op_;
  const Op& vm_shape_of_op_;
  const Op& cast_op_;
  const Op& ndarray_size_op_;

  // Convert value to expression.
  Expr ObjectToExpr(const ObjectRef& value) {
    if (value->IsInstance<runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<runtime::NDArray>(value);
      return Constant(nd_array);
    } else if (const auto* val = value.as<runtime::ADTObj>()) {
      runtime::ADT adt = GetRef<runtime::ADT>(val);
      Array<Expr> fields;
      for (size_t i = 0; i < adt.size(); ++i) {
        fields.push_back(ObjectToExpr(adt[i]));
      }
      return Tuple(fields);
    } else {
      LOG(FATAL) << "Cannot handle " << value->GetTypeKey();
      return Expr();
    }
  }
  // Constant evaluate an expression.
  Expr ConstEvaluate(Expr expr) {
    Device dev;
    dev.device_type = kDLCPU;
    dev.device_id = 0;
    // Target("llvm") created here
    Target target = Target("llvm");

    // use a fresh build context in case we are already in a build context.
    // needed for both execution and creation(due to JIT)
    With<transform::PassContext> fresh_build_ctx(transform::PassContext::Create());

    return ObjectToExpr(Eval(expr, module_->type_definitions, module_->Imports(), dev, target));
  }

  // Evaluate a call to the shape_of operator for tensors with constant
  // shapes.
  Expr EvaluateShapeOf(Expr expr, Array<Expr> args, Attrs attrs) {
    Expr input = args[0];
    const auto* param = attrs.as<ShapeOfAttrs>();
    ICHECK(param != nullptr);

    tvm::Array<IndexExpr> ishape;
    if (auto opt = GetConstantShape(input)) {
      ishape = opt.value();
    } else {
      return expr;
    }

    // Get the constant shape
    Device dev;
    dev.device_type = kDLCPU;
    dev.device_id = 0;
    runtime::NDArray value;
    DLDataType cdtype = DataType::Int(32);
    if (ishape.size() == 0) {
      value = runtime::NDArray::Empty({}, cdtype, dev);
    } else {
      ICHECK_NE(ishape.size(), 0);
      std::vector<int64_t> cshape = {static_cast<int64_t>(ishape.size())};
      value = runtime::NDArray::Empty(cshape, cdtype, dev);
      int32_t* dims = static_cast<int32_t*>(value->data);
      using ::tvm::tir::IntImmNode;
      for (size_t i = 0; i < ishape.size(); ++i) {
        if (const IntImmNode* dim = ishape[i].as<IntImmNode>()) {
          dims[i] = dim->value;
        } else {
          return expr;
        }
      }
    }

    Constant shape = Downcast<Constant>(ObjectToExpr(value));

    if (shape->data.Shape().size() == 0 && GetScalarFromConstant<int32_t>(shape) == 0) {
      auto ndarray = runtime::NDArray::Empty({}, cdtype, dev);
      shape = Constant(ndarray);
    }

    return CastValue(shape, param->dtype);
  }

  // Evaluate a call to the ndarray_size operator for tensors with constant
  // shapes.
  Expr EvaluateNdarraySize(Expr expr, Array<Expr> args, Attrs attrs) {
    Expr input = args[0];
    const auto* param = attrs.as<NdarraySizeAttrs>();
    ICHECK(param != nullptr);

    tvm::Array<IndexExpr> ishape;
    if (auto opt = GetConstantShape(input)) {
      ishape = opt.value();
    } else {
      return expr;
    }

    // Get the constant size
    Device dev;
    dev.device_type = kDLCPU;
    dev.device_id = 0;
    runtime::NDArray value;
    DLDataType cdtype = DataType::Int(32);
    value = runtime::NDArray::Empty({}, cdtype, dev);
    int32_t* data = static_cast<int32_t*>(value->data);
    if (ishape.size() == 0) {
      *data = 0;
    } else {
      *data = 1;
      using ::tvm::tir::IntImmNode;
      for (size_t i = 0; i < ishape.size(); ++i) {
        if (const IntImmNode* dim = ishape[i].as<IntImmNode>()) {
          *data *= dim->value;
        } else {
          return expr;
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
    tvm::Array<IndexExpr> ishape;
    if (const ConstantNode* op = input.as<ConstantNode>()) {
      ishape = op->tensor_type()->shape;
    } else if (input->checked_type_.defined()) {
      ishape = input->checked_type().as<TensorTypeNode>()->shape;
    } else {
      return Optional<tvm::Array<IndexExpr>>(nullptr);
    }

    return Optional<tvm::Array<IndexExpr>>(ishape);
  }
};

Expr FoldConstant(const Expr& expr, const IRModule& mod) {
  return ConstantFolder(mod).Mutate(expr);
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstantExpr").set_body_typed(FoldConstant);

namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldConstant(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
