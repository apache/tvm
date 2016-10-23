/*!
 *  Copyright (c) 2016 by Contributors
 * \file stmt.h
 * \brief The statement creation functions.
 *  The underlying container are defined in stmt_node.h
 */
#ifndef TVM_STMT_H_
#define TVM_STMT_H_

#include <type_traits>
#include "./base.h"
#include "./domain.h"

namespace tvm {

/*!
 * \brief a expression type, holds a ref to root of an AST
 */
class Stmt : public NodeRef {
 public:
  /*! \brief default constructor */
  Stmt() {}
  /*!
   * \brief constructor from node pointer
   * \param nptr Another node shared pointer
   */
  explicit Stmt(std::shared_ptr<Node>&& nptr) : NodeRef(std::move(nptr)) {
    CHECK(node_.get() != nullptr);
  }
};

/*!
 * \brief construct Store Stmt.
 * \param buffer The variable representing the buffer.
 * \param offset The offset in the buffer
 * \param src The source expression.
 */
Stmt Store(Var buffer, Expr offset, Expr src);

/*!
 * \brief construct ForRange Stmt
 * \param loop_var The loop variable
 * \param range The loop range
 * \param body The loop body
 */
Stmt ForRange(Var loop_var, Range range, Stmt body);

/*!
 * \brief construct a IfThenElse
 * \param cond The condition.
 * \param then_body The body to go to in then condition.
 * \param else_body The body to go to in else condition.
 */
Stmt IfThenElse(Expr cond, Stmt then_body, Stmt else_body);

}  // namespace tvm
#endif  // TVM_STMT_H_
