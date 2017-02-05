/*!
 *  Copyright (c) 2016 by Contributors
 * \file inline.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {

// inliner to inline a function
// the result may not be SSA,
// ConvertSSA need to be applied after this pass
class IRInline : public IRMutator {
 public:
  IRInline(FunctionRef f, Array<Var> args, Expr body)
      : f_(f), args_(args), body_(body) {}

  Expr Mutate_(const Call* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Call>();

    if (op->func == f_) {
      CHECK_EQ(op->value_index, 0);
      Expr expr = body_;
      CHECK_EQ(args_.size(), op->args.size());

      bool has_side_effect = false;
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (HasSideEffect(op->args[i])) has_side_effect = true;
      }

      if (has_side_effect) {
        for (size_t i = 0; i < args_.size(); ++i) {
          expr = Let::make(args_[i], op->args[i], expr);
        }
      } else {
        Map<Var, Expr> vmap;
        for (size_t i = 0; i < args_.size(); ++i) {
          vmap.Set(args_[i], op->args[i]);
        }
        expr = Substitute(
            Evaluate::make(expr), vmap).as<Evaluate>()->value;
      }
      return expr;
    } else {
      return e;
    }
  }

 private:
  FunctionRef f_;
  Array<Var> args_;
  Expr body_;
};

Stmt Inline(Stmt stmt,
            FunctionRef f,
            Array<Var> args,
            Expr body) {
  CHECK_EQ(f->num_outputs(), 1)
      << "can only inline output single value operation";
  return ConvertSSA(IRInline(f, args, body).Mutate(stmt));
}
}  // namespace ir
}  // namespace tvm
