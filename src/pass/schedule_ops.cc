/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {
namespace {

Stmt MakeCompute(const ComputeOpNode* op, const Array<Split>& splits) {
  Tensor output;
  std::vector<Expr> args(op->dim_var.size());
  for (size_t i = 0; i < args.size(); ++i) {
    args[i] = op->dim_var[i];
  }
  Array<Expr> values{op->body};
  Stmt stmt = Provide::make(output, values, args);
  // add splits from ousside most to outsidemost to innermost
  return stmt;
}


Stmt MakePipeline(const Schedule& sch, Stmt body) {
  return body;
}

// inject the operator's realization on the stmt.
class InjectRealize : public IRMutator {
 public:
  explicit InjectRealize(Schedule sch)
      : sch_(sch) {}

  Stmt Mutate(Stmt stmt) final {
    stmt = IRMutator::Mutate(stmt);
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr &&
        op->type_key == "Split" &&
        op->node == sch_->attach_parent) {
      return AttrStmt::make(
          op->node, op->type_key, op->value,
          MakePipeline(sch_, op->body));
    } else {
      return stmt;
    }
  }

 private:
  // the operations to be carried
  Schedule sch_;
};

}  // namespace
}  // namespace ir
}  // namespace tvm
