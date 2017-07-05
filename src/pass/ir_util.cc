/*!
 *  Copyright (c) 2017 by Contributors
 * \file ir_util.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "./ir_util.h"

namespace tvm {
namespace ir {

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (s.as<For>()) {
      auto n = std::make_shared<For>(*s.as<For>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<LetStmt>()) {
      auto n = std::make_shared<LetStmt>(*s.as<LetStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<AttrStmt>()) {
      auto n = std::make_shared<AttrStmt>(*s.as<AttrStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<IfThenElse>()) {
      auto n = std::make_shared<IfThenElse>(*s.as<IfThenElse>());
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (s.as<AssertStmt>()) {
      auto n = std::make_shared<AssertStmt>(*s.as<AssertStmt>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (s.as<Allocate>()) {
      auto n = std::make_shared<Allocate>(*s.as<Allocate>());
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else {
      LOG(FATAL) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt> >& nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

Stmt MergeSeq(const std::vector<Stmt>& seq) {
  if (seq.size() == 0) return Evaluate::make(0);
  Stmt body = seq[0];
  for (size_t i = 1; i < seq.size(); ++i) {
    body = Block::make(body, seq[i]);
  }
  return body;
}

}  // namespace ir
}  // namespace tvm
