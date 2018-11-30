/*!
 * Copyright (c) 2018 by Contributors
 * \file constant_folding.cc
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/interpreter.h>

namespace tvm {
namespace relay {

using FInterpreter = runtime::TypedPackedFunc<Value(Expr)>;


class ConstantChecker : private ExprVisitor {
 public:
  // Check whether an expression is constant. The results are memorized.
  bool Check(const Expr& expr) {
    if (expr.as<ConstantNode>()) {
      return true;
    }
    const auto it = memo_.find(expr);
    if (it != memo_.end())
      return it->second;
    VisitExpr(expr);
    return memo_[expr];  // return memorized result or the default value false
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
    expr = InferType(expr, Module(nullptr));
    expr = FuseOps(expr, 0);
    expr = InferType(expr, Module(nullptr));
    return ValueToExpr(executor_(expr));
  }
};


Expr FoldConstant(const Expr& expr) {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  Target target = Target::create("llvm");
  // use a fresh build context
  // in case we are already in a build context.
  BuildConfigContext fresh_build_ctx(build_config());

  return ConstantFolder(CreateInterpreter(
      Module(nullptr), ctx, target)).Mutate(expr);
}

TVM_REGISTER_API("relay._ir_pass.FoldConstant")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = FoldConstant(args[0]);
});

}  // namespace relay
}  // namespace tvm
