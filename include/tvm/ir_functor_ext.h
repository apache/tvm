/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/ir_functor_ext.h
 * \brief More powerful Visitor that allows define function signatures.
 */
#ifndef TVM_IR_FUNCTOR_EXT_H_
#define TVM_IR_FUNCTOR_EXT_H_

#include "tvm/node/ir_functor.h"
#include "ir.h"
#include <utility>

namespace tvm {
namespace ir {

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 *  This helps you to avoid to book-keep return value of Visitor via state,
 *  which can cause bugs easily when state is incorrectly maintained.
 *
 * \code
 *  // A functor that set variable to b. and calculate results.
 *  class MyExprFunctor
 *    : public ir::ExprFunctor<int(const Expr&, int)> {
 *   public:
 *    int VisitExpr_(const Variable* op, int b) final {
 *     return b;
 *    }
 *    int VisitExpr_(const IntImm* op, int b) final {
 *      return op->value;
 *    }
 *    int VisitExpr_(const Add* op, int b) final {
 *     return Visit(op->a, b) + Visit(op->b, b);
 *    }
 *  };
 *  MyExprFunctor f;
 *  Var x("x");
 *  CHECK_EQ(f(x + 1, 2), 3);
 * \endcode
 *
 * \note Why do we need this more powerful Functor:
 *
 *  We often need to implement a transformer tasks.
 *  Say we want to take Expr and transform it to some analysis result,
 *  This easily be done incorrectly using plain Visitor. See IRVisitor's
 *  document for possible error cases.
 *
 * \tparam FType function signiture
 *  This type if only defined for FType with function signiture R(const Expr&, Args...)
 */
template<typename FType>
class ExprFunctor;
/*!
 * \brief Same as ExprFunctor except it is applied on statements
 * \tparam FType The function signature.
 */
template<typename FType>
class StmtFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT {                                      \
    return VisitExprDefault_(op, std::forward<Args>(args)...);      \
  }
#define STMT_FUNCTOR_DEFAULT {                                      \
    return VisitStmtDefault_(op, std::forward<Args>(args)...);      \
}

#define IR_EXPR_FUNCTOR_DISPATCH(OP)                                    \
  vtable.template set_dispatch<OP>(                                     \
      [](const NodeRef& n, TSelf* self, Args... args) {                 \
        return self->VisitExpr_(static_cast<const OP*>(n.node_.get()),  \
                                std::forward<Args>(args)...);           \
      });                                                               \

#define IR_STMT_FUNCTOR_DISPATCH(OP)                                    \
  vtable.template set_dispatch<OP>(                                     \
      [](const NodeRef& n, TSelf* self, Args... args) {                 \
        return self->VisitStmt_(static_cast<const OP*>(n.node_.get()),  \
                                std::forward<Args>(args)...);           \
      });                                                               \

template<typename R, typename ...Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = IRFunctor<R(const NodeRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~ExprFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Expr& n, Args... args) {
    return VisitExpr(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const Expr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitExpr_(const Variable* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Load* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Let* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Call* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Add* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Sub* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Mul* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Div* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Mod* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloorDiv* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloorMod* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Min* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Max* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const EQ* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const NE* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LT* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LE* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GT* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GE* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const And* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Or* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Reduce* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Cast* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Not* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Select* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Ramp* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Broadcast* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const Shuffle* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IntImm* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const UIntImm* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloatImm* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const StringImm* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Node* op, Args ...) {
    LOG(FATAL) << "Do not have a default for " << op->type_key();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    IR_EXPR_FUNCTOR_DISPATCH(Variable);
    IR_EXPR_FUNCTOR_DISPATCH(Load);
    IR_EXPR_FUNCTOR_DISPATCH(Let);
    IR_EXPR_FUNCTOR_DISPATCH(Call);
    IR_EXPR_FUNCTOR_DISPATCH(Add);
    IR_EXPR_FUNCTOR_DISPATCH(Sub);
    IR_EXPR_FUNCTOR_DISPATCH(Mul);
    IR_EXPR_FUNCTOR_DISPATCH(Div);
    IR_EXPR_FUNCTOR_DISPATCH(Mod);
    IR_EXPR_FUNCTOR_DISPATCH(FloorDiv);
    IR_EXPR_FUNCTOR_DISPATCH(FloorMod);
    IR_EXPR_FUNCTOR_DISPATCH(Min);
    IR_EXPR_FUNCTOR_DISPATCH(Max);
    IR_EXPR_FUNCTOR_DISPATCH(EQ);
    IR_EXPR_FUNCTOR_DISPATCH(NE);
    IR_EXPR_FUNCTOR_DISPATCH(LT);
    IR_EXPR_FUNCTOR_DISPATCH(LE);
    IR_EXPR_FUNCTOR_DISPATCH(GT);
    IR_EXPR_FUNCTOR_DISPATCH(GE);
    IR_EXPR_FUNCTOR_DISPATCH(And);
    IR_EXPR_FUNCTOR_DISPATCH(Or);
    IR_EXPR_FUNCTOR_DISPATCH(Reduce);
    IR_EXPR_FUNCTOR_DISPATCH(Cast);
    IR_EXPR_FUNCTOR_DISPATCH(Not);
    IR_EXPR_FUNCTOR_DISPATCH(Select);
    IR_EXPR_FUNCTOR_DISPATCH(Ramp);
    IR_EXPR_FUNCTOR_DISPATCH(Shuffle);
    IR_EXPR_FUNCTOR_DISPATCH(Broadcast);
    IR_EXPR_FUNCTOR_DISPATCH(IntImm);
    IR_EXPR_FUNCTOR_DISPATCH(UIntImm);
    IR_EXPR_FUNCTOR_DISPATCH(FloatImm);
    IR_EXPR_FUNCTOR_DISPATCH(StringImm);
    return vtable;
  }
};

template<typename R, typename ...Args>
class StmtFunctor<R(const Stmt& n, Args... args)> {
 private:
  using TSelf = StmtFunctor<R(const Stmt& n, Args... args)>;
  using FType = IRFunctor<R(const NodeRef& n, TSelf* self, Args... args)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~StmtFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The stmt node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const Stmt& n, Args... args) {
    return VisitStmt(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The stmt node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitStmt(const Stmt& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitStmt_(const LetStmt* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AttrStmt* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const IfThenElse* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const For* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Allocate* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Store* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Free* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AssertStmt* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ProducerConsumer* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Provide* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Realize* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Prefetch* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Block* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Evaluate* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmtDefault_(const Node* op, Args ...) {
    LOG(FATAL) << "Do not have a default for " << op->type_key();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    IR_STMT_FUNCTOR_DISPATCH(LetStmt);
    IR_STMT_FUNCTOR_DISPATCH(AttrStmt);
    IR_STMT_FUNCTOR_DISPATCH(IfThenElse);
    IR_STMT_FUNCTOR_DISPATCH(For);
    IR_STMT_FUNCTOR_DISPATCH(Allocate);
    IR_STMT_FUNCTOR_DISPATCH(Store);
    IR_STMT_FUNCTOR_DISPATCH(Free);
    IR_STMT_FUNCTOR_DISPATCH(AssertStmt);
    IR_STMT_FUNCTOR_DISPATCH(ProducerConsumer);
    IR_STMT_FUNCTOR_DISPATCH(Provide);
    IR_STMT_FUNCTOR_DISPATCH(Realize);
    IR_STMT_FUNCTOR_DISPATCH(Prefetch);
    IR_STMT_FUNCTOR_DISPATCH(Block);
    IR_STMT_FUNCTOR_DISPATCH(Evaluate);
    return vtable;
  }
};

#undef IR_STMT_FUNCTOR_DISPATCH
#undef IR_EXPR_FUNCTOR_DISPATCH
#undef EXPR_FUNCTOR_DEFAULT
#undef STMT_FUNCTOR_DEFAULT

}  // namespace ir
}  // namespace tvm
#endif  // TVM_IR_FUNCTOR_EXT_H_
