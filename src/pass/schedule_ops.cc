/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule_ops.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include "./scope.h"

namespace tvm {
namespace ir {
namespace {

/*!
 * \brief make nest loops given list of stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body The inner-most body of the loop
 */
Stmt MakeLoop(std::vector<Stmt>&& nest, Stmt body) {
  while (!nest.empty()) {
    Stmt s = std::move(nest.back()); nest.pop_back();
    if (s.as<For>()) {
      auto n = std::make_shared<For>(*s.as<For>());
      n->body = body;
      body = Stmt(n);
    } else if (s.as<LetStmt>()) {
      auto n = std::make_shared<LetStmt>(*s.as<LetStmt>());
      n->body = body;
      body = Stmt(n);
    } else if (s.as<AttrStmt>()) {
      auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
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
