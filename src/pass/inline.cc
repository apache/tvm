/*!
 *  Copyright (c) 2016 by Contributors
 * \file inline.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {
namespace {

// inliner to inline a function
// the result may not be SSA,
// ConvertSSA need to be applied after this pass
class IRInline : public IRMutator {
 public:
  IRInline(FunctionRef f, Array<Var> args, Expr body)
      : f_(f), args_(args), body_(body) {}

  Expr Mutate(Expr expr) final {
    expr = IRMutator::Mutate(expr);
    const Call* call = expr.as<Call>();
    if (call != nullptr && call->func == f_) {
      CHECK_EQ(call->value_index, 0);
      return InlineCall(call);
    } else {
      return expr;
    }
  }

  Stmt Mutate(Stmt stmt) final {
    return IRMutator::Mutate(stmt);
  }

 private:
  FunctionRef f_;
  Array<Var> args_;
  Expr body_;

  Expr InlineCall(const Call* op) {
    Expr expr = body_;

    CHECK_EQ(args_.size(), op->args.size())
        << op->args.size() << " vs " << args_.size();
    for (size_t i = 0; i < args_.size(); ++i) {
      expr = Let::make(args_[i], op->args[i], expr);
    }
    return expr;
  }
};

}  // namespace

Stmt Inline(FunctionRef f,
            Array<Var> args,
            Expr body,
            Stmt stmt) {
  CHECK_EQ(f->num_outputs(), 1)
      << "can only inline output single value operation";
  return ConvertSSA(IRInline(f, args, body).Mutate(stmt));
}
}  // namespace ir
}  // namespace tvm
