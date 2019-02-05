/*!
 *  Copyright (c) 2017 by Contributors
 * \file ir_util.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "ir_util.h"

namespace tvm {
namespace ir {

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (const auto* for_ = s.as<For>()) {
      auto n = make_node<For>(*for_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* let = s.as<LetStmt>()) {
      auto n = make_node<LetStmt>(*let);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* attr = s.as<AttrStmt>()) {
      auto n = make_node<AttrStmt>(*attr);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* ite = s.as<IfThenElse>()) {
      auto n = make_node<IfThenElse>(*ite);
      CHECK(is_no_op(n->then_case));
      CHECK(!n->else_case.defined());
      n->then_case = body;
      body = Stmt(n);
    } else if (const auto* block = s.as<Block>()) {
      auto n = make_node<Block>(*block);
      CHECK(is_no_op(n->rest));
      n->rest = body;
      body = Stmt(n);
    } else if (const auto* assert_ = s.as<AssertStmt>()) {
      auto n = make_node<AssertStmt>(*assert_);
      CHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* alloc = s.as<Allocate>()) {
      auto n = make_node<Allocate>(*alloc);
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
