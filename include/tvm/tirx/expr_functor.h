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
 * \file tvm/tirx/expr_functor.h
 *
 * \brief Functors for tirx expressions.
 */
#ifndef TVM_TIR_EXPR_FUNCTOR_H_
#define TVM_TIR_EXPR_FUNCTOR_H_

#include <tvm/ir/node_functor.h>
#include <tvm/ir/prim/expr.h>

#include <utility>

namespace tvm {
namespace tirx {

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
 *    : public tirx::ExprFunctor<int(const Expr&, int)> {
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
 *  TVM_FFI_ICHECK_EQ(f(x + 1, 2), 3);
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
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT                                   \
  {                                                            \
    return VisitExprDefault_(op, std::forward<Args>(args)...); \
  }

#define IR_EXPR_FUNCTOR_DISPATCH(OP)                                                        \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...);  \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = NodeFunctor<R(const ffi::ObjectRef& n, TSelf* self, Args...)>;

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
  R operator()(const Expr& n, Args... args) { return VisitExpr(n, std::forward<Args>(args)...); }
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
  virtual R VisitExpr_(const VarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TensorLoadNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpaqueExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleGetItemNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LetNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::AddNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::SubNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::MulNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::DivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::ModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::FloorDivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::FloorModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::MinNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::MaxNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::EQNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::NENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LTNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::LENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::GTNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::GENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::AndNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::OrNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::CastNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::NotNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::SelectNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::RampNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::BroadcastNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::ShuffleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IntImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloatImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const prim::StringImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    IR_EXPR_FUNCTOR_DISPATCH(VarNode);
    IR_EXPR_FUNCTOR_DISPATCH(TensorLoadNode);
    IR_EXPR_FUNCTOR_DISPATCH(OpaqueExprNode);
    IR_EXPR_FUNCTOR_DISPATCH(TupleNode);
    IR_EXPR_FUNCTOR_DISPATCH(TupleGetItemNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::LetNode);
    IR_EXPR_FUNCTOR_DISPATCH(CallNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::AddNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::SubNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::MulNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::DivNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::ModNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::FloorDivNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::FloorModNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::MinNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::MaxNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::EQNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::NENode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::LTNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::LENode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::GTNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::GENode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::AndNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::OrNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::CastNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::NotNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::SelectNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::RampNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::ShuffleNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::BroadcastNode);
    IR_EXPR_FUNCTOR_DISPATCH(IntImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(FloatImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(prim::StringImmNode);
    vtable.Finalize();
    return vtable;
  }
};

#undef IR_EXPR_FUNCTOR_DISPATCH
#undef EXPR_FUNCTOR_DEFAULT

/*!
 * \brief ExprVisitor
 */
class TVM_DLL ExprVisitor : public ExprFunctor<void(const Expr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const TensorLoadNode* op) override;
  void VisitExpr_(const OpaqueExprNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;
  void VisitExpr_(const prim::LetNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const prim::AddNode* op) override;
  void VisitExpr_(const prim::SubNode* op) override;
  void VisitExpr_(const prim::MulNode* op) override;
  void VisitExpr_(const prim::DivNode* op) override;
  void VisitExpr_(const prim::ModNode* op) override;
  void VisitExpr_(const prim::FloorDivNode* op) override;
  void VisitExpr_(const prim::FloorModNode* op) override;
  void VisitExpr_(const prim::MinNode* op) override;
  void VisitExpr_(const prim::MaxNode* op) override;
  void VisitExpr_(const prim::EQNode* op) override;
  void VisitExpr_(const prim::NENode* op) override;
  void VisitExpr_(const prim::LTNode* op) override;
  void VisitExpr_(const prim::LENode* op) override;
  void VisitExpr_(const prim::GTNode* op) override;
  void VisitExpr_(const prim::GENode* op) override;
  void VisitExpr_(const prim::AndNode* op) override;
  void VisitExpr_(const prim::OrNode* op) override;
  void VisitExpr_(const prim::CastNode* op) override;
  void VisitExpr_(const prim::NotNode* op) override;
  void VisitExpr_(const prim::SelectNode* op) override;
  void VisitExpr_(const prim::RampNode* op) override;
  void VisitExpr_(const prim::BroadcastNode* op) override;
  void VisitExpr_(const prim::ShuffleNode* op) override;
  void VisitExpr_(const IntImmNode* op) override;
  void VisitExpr_(const FloatImmNode* op) override;
  void VisitExpr_(const prim::StringImmNode* op) override;
};

/*!
 * \brief ExprMutator that mutates expressions.
 */
class TVM_DLL ExprMutator : protected ExprFunctor<Expr(const Expr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  /*! \brief Visit a primitive expression and verify that it remains primitive. */
  PrimExpr VisitPrimExpr(const PrimExpr& expr) { return VisitExpr(expr).as_or_throw<PrimExpr>(); }
  // list of functions to override.
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const TensorLoadNode* op) override;
  Expr VisitExpr_(const OpaqueExprNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;
  Expr VisitExpr_(const prim::LetNode* op) override;
  Expr VisitExpr_(const CallNode* op) override;
  Expr VisitExpr_(const prim::AddNode* op) override;
  Expr VisitExpr_(const prim::SubNode* op) override;
  Expr VisitExpr_(const prim::MulNode* op) override;
  Expr VisitExpr_(const prim::DivNode* op) override;
  Expr VisitExpr_(const prim::ModNode* op) override;
  Expr VisitExpr_(const prim::FloorDivNode* op) override;
  Expr VisitExpr_(const prim::FloorModNode* op) override;
  Expr VisitExpr_(const prim::MinNode* op) override;
  Expr VisitExpr_(const prim::MaxNode* op) override;
  Expr VisitExpr_(const prim::EQNode* op) override;
  Expr VisitExpr_(const prim::NENode* op) override;
  Expr VisitExpr_(const prim::LTNode* op) override;
  Expr VisitExpr_(const prim::LENode* op) override;
  Expr VisitExpr_(const prim::GTNode* op) override;
  Expr VisitExpr_(const prim::GENode* op) override;
  Expr VisitExpr_(const prim::AndNode* op) override;
  Expr VisitExpr_(const prim::OrNode* op) override;
  Expr VisitExpr_(const prim::CastNode* op) override;
  Expr VisitExpr_(const prim::NotNode* op) override;
  Expr VisitExpr_(const prim::SelectNode* op) override;
  Expr VisitExpr_(const prim::RampNode* op) override;
  Expr VisitExpr_(const prim::BroadcastNode* op) override;
  Expr VisitExpr_(const prim::ShuffleNode* op) override;
  Expr VisitExpr_(const IntImmNode* op) override;
  Expr VisitExpr_(const FloatImmNode* op) override;
  Expr VisitExpr_(const prim::StringImmNode* op) override;
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_EXPR_FUNCTOR_H_
