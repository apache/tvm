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
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/container.h>
#include "pattern_util.h"

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
    if (it != memo_.end())
      return it->second;
    VisitExpr(expr);
    return memo_[expr];  // return memoized result or the default value false
  }

 private:
  std::unordered_map<Expr, bool, ObjectHash, ObjectEqual> memo_;

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

bool ConstantCheck(const Expr& e) {
  return ConstantChecker().Check(e);
}

TVM_REGISTER_GLOBAL("relay.analysis.check_constant")
.set_body_typed(ConstantCheck);

// TODO(tvm-team) consider combine dead-code with constant folder.
// or make a more powerful partial evaluator.
class ConstantFolder : public ExprMutator {
 public:
  explicit ConstantFolder(FInterpreter executor, IRModule module)
      : executor_(executor),
        module_(module),
        shape_of_op_(Op::Get("shape_of")),
        invoke_tvm_op_(Op::Get("memory.invoke_tvm_op")),
        shape_func_op_(Op::Get("memory.shape_func")),
        alloc_tensor_op_(Op::Get("memory.alloc_tensor")),
        alloc_storage_op_(Op::Get("memory.alloc_storage")),
        cast_op_(Op::Get("cast")) {}

  Expr VisitExpr_(const LetNode* op) final {
    Expr value = this->Mutate(op->value);
    if (value.as<ConstantNode>()) {
      memo_[op->var] = value;
      return this->Mutate(op->body);
    } else {
      Var var = Downcast<Var>(this->Mutate(op->var));
      Expr body = this->Mutate(op->body);
      if (var.same_as(op->var) &&
          value.same_as(op->value) &&
          body.same_as(op->body)) {
        return GetRef<Expr>(op);
      } else {
        return Let(var, value, body);
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");

    std::unordered_set<std::string> skip_list{"zeros_like", "ones_like", "full_like", "full"};

    auto origin_args = call->args;
    Expr res = ExprMutator::VisitExpr_(call);
    call = res.as<CallNode>();
    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return res;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return res;
    if (skip_list.count(op->name)) {
        return res;
    }
    // skip stateful ops.
    if (op_stateful.get(GetRef<Op>(op), false)) return res;
    // Try to evaluate shape_of op
    if (call->op == shape_of_op_) {
      return EvaluateShapeOf(res, origin_args, call->attrs);
    }

    // We should think about potentially constant evaluation over these ops too.
    if (call->op == invoke_tvm_op_ ||
        call->op == shape_func_op_ ||
        call->op == alloc_tensor_op_ ||
        call->op == alloc_storage_op_) {
      return GetRef<Call>(call);
    }

    bool all_const_args = true;
    for (Expr arg : call->args) {
      if (!checker_.Check(arg)) {
        all_const_args = false;
      }
    }
    if (all_const_args) {
      return ConstEvaluate(res);
    } else {
      return res;
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    Expr res = ExprMutator::VisitExpr_(op);
    op = res.as<TupleGetItemNode>();
    if (const auto* tuple = op->tuple.as<TupleNode>()) {
      return tuple->fields[op->index];
    } else {
      return res;
    }
  }

 private:
  // Internal interepreter.
  FInterpreter executor_;
  // Internal constant checker
  ConstantChecker checker_;
  // Module
  IRModule module_;

  // Cache the following ops for equivalence checking in this pass.
  const Op& shape_of_op_;
  const Op& invoke_tvm_op_;
  const Op& shape_func_op_;
  const Op& alloc_tensor_op_;
  const Op& alloc_storage_op_;
  const Op& cast_op_;

  // Convert value to expression.
  Expr ObjectToExpr(const ObjectRef& value) {
    if (value->IsInstance<runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<runtime::NDArray>(value);
      for (auto dim : nd_array.Shape()) {
        CHECK_GT(dim, 0)
          << "invalid dimension after constant eval";
      }
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
  // Constant evaluate a expression.
  Expr ConstEvaluate(Expr expr) {
    std::vector<transform::Pass> passes = {transform::FuseOps(0),
                                           transform::ToANormalForm(),
                                           transform::InferType()};
    Function func;
    if (expr.as<FunctionNode>()) {
      func = Downcast<Function>(expr);
    } else {
      // TODO(@jroesch): fix this
      func = Function(FreeVars(expr), expr, Type(), FreeTypeVars(expr, module_), {});
    }
    auto mod = IRModule(
      {},
      module_->type_definitions,
      module_->Imports());
    auto global = GlobalVar("main");
    mod->Add(global, func);
    auto seq = transform::Sequential(passes);
    mod = seq(mod);
    auto entry_func = Downcast<Function>(mod->Lookup("main"));
    expr = expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
    return ObjectToExpr(executor_(expr));
  }

  // Evaluate a call to the shape_of operator for tensors with constant
  // shapes.
  Expr EvaluateShapeOf(Expr expr, Array<Expr> args, Attrs attrs) {
    Expr input = args[0];
    const auto* param = attrs.as<ShapeOfAttrs>();
    CHECK(param != nullptr);

    tvm::Array<IndexExpr> ishape;
    if (const ConstantNode* op = input.as<ConstantNode>()) {
      ishape = op->tensor_type()->shape;
    } else if (input->checked_type_.defined()) {
      ishape = input->checked_type().as<TensorTypeNode>()->shape;
    } else {
      return expr;
    }

    // Get the constant shape
    DLContext ctx;
    ctx.device_type = kDLCPU;
    ctx.device_id = 0;
    runtime::NDArray value;
    DLDataType cdtype = DataType::Int(32);
    if (ishape.size() == 0) {
      value = runtime::NDArray::Empty({}, cdtype, ctx);
    } else {
      CHECK_NE(ishape.size(), 0);
      std::vector<int64_t> cshape = { static_cast<int64_t>(ishape.size()) };
      value = runtime::NDArray::Empty(cshape, cdtype, ctx);
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
      auto ndarray = runtime::NDArray::Empty({}, cdtype, ctx);
      shape = Constant(ndarray);
    }

    // Cast the constant into correct dtype
    auto cast_attrs = make_object<CastAttrs>();
    cast_attrs->dtype = param->dtype;
    Expr ret = Call(cast_op_, { shape }, Attrs(cast_attrs), {});
    return ConstEvaluate(ret);
  }
};


Expr FoldConstant(const Expr& expr, const IRModule& mod) {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  Target target = Target::Create("llvm");
  // use a fresh build context
  // in case we are already in a build context.
  With<BuildConfig> fresh_build_ctx(BuildConfig::Create());

  return ConstantFolder(CreateInterpreter(mod, ctx, target), mod).Mutate(expr);
}

namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f, m));
  };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
