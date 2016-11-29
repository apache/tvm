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

// inject the operator's realization on the stmt.
class InjectRealize : public IRMutator {
 public:
  explicit InjectRealize(Schedule sch)
      : sch_(sch) {}

  Stmt Mutate(Stmt stmt) final {
    stmt = IRMutator::Mutate(stmt);
    const For* op = stmt.as<For>();
    return stmt;
  }

 private:
  // the operations to be carried
  Schedule sch_;
};

}  // namespace
}  // namespace ir
}  // namespace tvm
