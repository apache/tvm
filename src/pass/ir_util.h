/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_util.h
 * \brief Helper functions to construct and compose IR nodes.
 */
#ifndef TVM_PASS_IR_UTIL_H_
#define TVM_PASS_IR_UTIL_H_

#include <tvm/ir.h>
#include <vector>

namespace tvm {
namespace ir {

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
inline Stmt MergeNest(std::vector<Stmt> nest, Stmt body) {
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
      body = Block::make(s, body);
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

/*!
 * \brief combine the nest stmt, whose body is not defined.
 * \param nest A list of For and LetStmt, whose body is not defined.
 * \param body body
 * \return The combined Stmt
 */
inline Stmt MergeNest(std::vector<std::vector<Stmt> > nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_IR_UTIL_H_
