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

void MakeLoop(const DimSplitNode* op,
              const Split& s,
              Scope<AttrKey, Expr>* pscope,
              std::vector<Stmt>* nest) {
  auto& scope = *pscope;
  Expr out_min = scope[{op->var, "min"}];
  Expr out_ext = scope[{op->var, "extent"}];
  Expr stride = op->factor;
  Var offset(s->var->name_hint + ".offset", Int(32));
  // for loop with stride
  // TODO(tqchen) split the loop to deal with tails
  nest->emplace_back(
      For::make(
          offset, out_min, out_ext,
          ForType::Parallel, DeviceAPI::None, Stmt()));
  Expr in_min = offset + out_min;
  Expr in_ext = min(stride, out_ext - offset);
  // declare min and extent of the corresponding variable
  nest->emplace_back(AttrStmt::make(op->var, "min", in_min, Stmt()));
  nest->emplace_back(AttrStmt::make(op->var, "extent", in_ext, Stmt()));
  // declare this is  the loop
  nest->emplace_back(AttrStmt::make(s, "split", 0, Stmt()));
  // setup the scope.
  pscope->Push({op->var, "min"}, in_min);
  pscope->Push({op->var, "extent"}, in_ext);
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
    const AttrStmt* op = stmt.as<AttrStmt>();
    if (op != nullptr) {
      attr_scope_.Push({op->node, op->type_key}, op->value);
      stmt = IRMutator::Mutate(stmt);
      attr_scope_.Pop({op->node, op->type_key});
    } else {
      stmt = IRMutator::Mutate(stmt);
    }

    if (op != nullptr &&
        op->type_key == "split" &&
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
  Scope<AttrKey, Expr> attr_scope_;
};

}  // namespace
}  // namespace ir
}  // namespace tvm
