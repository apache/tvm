/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir_mutator.h
 * \brief Defines general IRMutation pass
 */
#ifndef TVM_IR_MUTATOR_H_
#define TVM_IR_MUTATOR_H_

#include <tvm/ir_node.h>
#include "./expr.h"

namespace tvm {
namespace ir {
/*!
 * \brief a base class for mutator to iterative mutate the IR
 *
 *  This IRMutator is implemented via IRFunctor instead of Visitor Pattern.
 *  This enables easy extensions of possible new IRNode.
 *  It also makes changing return types easier.
 *
 * \note If you want to return a different type other than Expr and Stmt,
 *       Simply following the same pattern as IRMutator and create a seperate class.
 * \sa IRFunctor
 */
class IRMutator {
 public:
  /*!
   * \brief mutate expression
   * \return the mutated expr
   */
  virtual Expr mutate(Expr expr) {
    static const FMutateExpr& f = vtable_expr();
    return f(expr, expr, this);
  }
  /*!
   * \brief mutate expression
   * \return the mutated stmt
   */
  virtual Stmt mutate(Stmt stmt) {
    static const FMutateStmt& f = vtable_stmt();
    return f(stmt, stmt, this);
  }
  /*! \brief destructor */
  virtual ~IRMutator() {}
  /*! \brief functor type of expr mutation */
  using FMutateExpr = IRFunctor<Expr(const IRNodeRef&, const Expr&, IRMutator*)>;
  /*! \brief functor type of stmt mutation */
  using FMutateStmt = IRFunctor<Stmt(const IRNodeRef&, const Stmt&, IRMutator*)>;
  /*! \return internal vtable of expr */
  static FMutateExpr& vtable_expr();  // NOLINT(*)
  /*! \return internal stmt of expr */
  static FMutateStmt& vtable_stmt();  // NOLINT(*)
};

/*!
 * \brief templatized base class of subclass of IRMutator
 *
 *  Use "curiously recurring template pattern" to implement mutate for you.
 *  Child class need to declare IRMutatorBase<T>::vtable_expr and IRMutatorBase<T>::vtable_stmt
 *
 * \note This only implement direct subclass from IRMutator, similar code
 *    can be created to implement deeper subclassing when needed.
 */
class IRMutatorExample : public IRMutator {
 public:
  Expr mutate(Expr expr) final {
    static const FMutateExpr& f = IRMutatorExample::vtable_expr();
    return (f.can_dispatch(expr) ?
            f(expr, expr, this) : IRMutator::mutate(expr));
  }
  Stmt mutate(Stmt stmt) final {
    static const FMutateStmt& f = IRMutatorExample::vtable_stmt();
    return (f.can_dispatch(stmt) ?
            f(stmt, stmt, this) : IRMutator::mutate(stmt));
  }
  // to be implemented by child class
  static FMutateExpr& vtable_expr();  // NOLINT(*)
  static FMutateStmt& vtable_stmt();  // NOLINT(*)
};

}  // namespace ir
}  // namespace tvm
#endif  // TVM_IR_MUTATOR_H_
