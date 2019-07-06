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
 * Copyright (c) 2018 by Contributors
 * \file constant_folding.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

using FInterpreter = runtime::TypedPackedFunc<Value(Expr)>;


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
  std::unordered_map<Expr, bool, NodeHash, NodeEqual> memo_;

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


// TODO(tvm-team) consider combine dead-code with constant folder.
// or make a more powerful partial evaluator.
class ConstantFolder : public ExprMutator {
 public:
  explicit ConstantFolder(FInterpreter executor)
      : executor_(executor) {
  }

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
        return LetNode::make(var, value, body);
      }
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");
    auto origin_args = call->args;
    Expr res = ExprMutator::VisitExpr_(call);
    call = res.as<CallNode>();
    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return res;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return res;
    // skip stateful ops.
    if (op_stateful.get(GetRef<Op>(op), false)) return res;
    // Try to evaluate shape_of op
    if (call->op.same_as(Op::Get("shape_of"))) {
      return EvaluateShapeOf(res, origin_args, call->attrs);
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

  // Convert value to expression.
  Expr ValueToExpr(Value value) {
    if (const auto* val = value.as<TensorValueNode>()) {
      return ConstantNode::make(val->data);
    } else if (const auto* val = value.as<TupleValueNode>()) {
      Array<Expr> fields;
      for (Value field : val->fields) {
        fields.push_back(ValueToExpr(field));
      }
      return TupleNode::make(fields);
    } else {
      LOG(FATAL) << "Cannot handle " << value->type_key();
      return Expr();
    }
  }
  // Constant evaluate a expression.
  Expr ConstEvaluate(Expr expr) {
    std::vector<transform::Pass> passes = {transform::FuseOps(0),
                                           transform::InferType()};
    auto mod = ModuleNode::FromExpr(expr);
    auto seq = transform::Sequential(passes);
    mod = seq(mod);
    auto entry_func = mod->Lookup("main");
    expr = expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
    return ValueToExpr(executor_(expr));
  }
  // Evaluate shape_of op
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
    auto val = runtime::NDArray::Empty(
        {(int64_t)ishape.size()}, Type2TVMType(Int(32)), ctx);
    int32_t* dims = static_cast<int32_t*>(val->data);
    using ::tvm::ir::IntImm;
    for (size_t i = 0; i < ishape.size(); ++i) {
      if (const IntImm* dim = ishape[i].as<IntImm>()) {
        dims[i] = dim->value;
      } else {
        return expr;
      }
    }
    Expr shape = ValueToExpr(TensorValueNode::make(val));
    // Cast the constant into correct dtype
    auto cast_attrs = make_node<CastAttrs>();
    cast_attrs->dtype = param->dtype;
    static const Op& cast_op = Op::Get("cast");
    Expr ret = CallNode::make(cast_op, {shape}, Attrs(cast_attrs), {});
    return ConstEvaluate(ret);
  }
};


Expr FoldConstant(const Expr& expr) {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  Target target = Target::Create("llvm");
  // use a fresh build context
  // in case we are already in a build context.
  With<BuildConfig> fresh_build_ctx(BuildConfig::Create());

  return ConstantFolder(CreateInterpreter(
      Module(nullptr), ctx, target)).Mutate(expr);
}

namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f));
  };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_API("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
