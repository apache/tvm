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
 * \file tvm/tir/expr_functor.h
 *
 * \brief Functors for tir expressions.
 */
#ifndef TVM_TIR_EXPR_FUNCTOR_H_
#define TVM_TIR_EXPR_FUNCTOR_H_

#include <tvm/node/functor.h>
#include <tvm/tir/expr.h>

#include <utility>

namespace tvm {
namespace tir {

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
 *    : public tir::ExprFunctor<int(const Expr&, int)> {
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
 *  ICHECK_EQ(f(x + 1, 2), 3);
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
#define EXPR_FUNCTOR_DEFAULT \
  { return VisitExprDefault_(op, std::forward<Args>(args)...); }

#define IR_EXPR_FUNCTOR_DISPATCH(OP)                                                       \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const PrimExpr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const PrimExpr& n, Args...)>;
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

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
  R operator()(const PrimExpr& n, Args... args) {
    return VisitExpr(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitExpr(const PrimExpr& n, Args... args) {
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitExpr_(const VarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SizeVarNode* op, Args... args) {
    return VisitExpr_(static_cast<const VarNode*>(op), std::forward<Args>(args)...);
  }
  virtual R VisitExpr_(const BufferLoadNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ProducerLoadNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LetNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const AddNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SubNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const MulNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloorDivNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloorModNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const MinNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const MaxNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const EQNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const NENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LTNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GTNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GENode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const AndNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OrNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ReduceNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CastNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const NotNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SelectNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const RampNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const BroadcastNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ShuffleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IntImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FloatImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const StringImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const AnyNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    IR_EXPR_FUNCTOR_DISPATCH(VarNode);
    IR_EXPR_FUNCTOR_DISPATCH(SizeVarNode);
    IR_EXPR_FUNCTOR_DISPATCH(BufferLoadNode);
    IR_EXPR_FUNCTOR_DISPATCH(ProducerLoadNode);
    IR_EXPR_FUNCTOR_DISPATCH(LetNode);
    IR_EXPR_FUNCTOR_DISPATCH(CallNode);
    IR_EXPR_FUNCTOR_DISPATCH(AddNode);
    IR_EXPR_FUNCTOR_DISPATCH(SubNode);
    IR_EXPR_FUNCTOR_DISPATCH(MulNode);
    IR_EXPR_FUNCTOR_DISPATCH(DivNode);
    IR_EXPR_FUNCTOR_DISPATCH(ModNode);
    IR_EXPR_FUNCTOR_DISPATCH(FloorDivNode);
    IR_EXPR_FUNCTOR_DISPATCH(FloorModNode);
    IR_EXPR_FUNCTOR_DISPATCH(MinNode);
    IR_EXPR_FUNCTOR_DISPATCH(MaxNode);
    IR_EXPR_FUNCTOR_DISPATCH(EQNode);
    IR_EXPR_FUNCTOR_DISPATCH(NENode);
    IR_EXPR_FUNCTOR_DISPATCH(LTNode);
    IR_EXPR_FUNCTOR_DISPATCH(LENode);
    IR_EXPR_FUNCTOR_DISPATCH(GTNode);
    IR_EXPR_FUNCTOR_DISPATCH(GENode);
    IR_EXPR_FUNCTOR_DISPATCH(AndNode);
    IR_EXPR_FUNCTOR_DISPATCH(OrNode);
    IR_EXPR_FUNCTOR_DISPATCH(ReduceNode);
    IR_EXPR_FUNCTOR_DISPATCH(CastNode);
    IR_EXPR_FUNCTOR_DISPATCH(NotNode);
    IR_EXPR_FUNCTOR_DISPATCH(SelectNode);
    IR_EXPR_FUNCTOR_DISPATCH(RampNode);
    IR_EXPR_FUNCTOR_DISPATCH(ShuffleNode);
    IR_EXPR_FUNCTOR_DISPATCH(BroadcastNode);
    IR_EXPR_FUNCTOR_DISPATCH(IntImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(FloatImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(StringImmNode);
    IR_EXPR_FUNCTOR_DISPATCH(AnyNode);
    return vtable;
  }
};

#undef IR_EXPR_FUNCTOR_DISPATCH
#undef EXPR_FUNCTOR_DEFAULT

/*!
 * \brief ExprVisitor
 */
class TVM_DLL ExprVisitor : public ExprFunctor<void(const PrimExpr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const SizeVarNode* op) override;
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitExpr_(const ProducerLoadNode* op) override;
  void VisitExpr_(const LetNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const AddNode* op) override;
  void VisitExpr_(const SubNode* op) override;
  void VisitExpr_(const MulNode* op) override;
  void VisitExpr_(const DivNode* op) override;
  void VisitExpr_(const ModNode* op) override;
  void VisitExpr_(const FloorDivNode* op) override;
  void VisitExpr_(const FloorModNode* op) override;
  void VisitExpr_(const MinNode* op) override;
  void VisitExpr_(const MaxNode* op) override;
  void VisitExpr_(const EQNode* op) override;
  void VisitExpr_(const NENode* op) override;
  void VisitExpr_(const LTNode* op) override;
  void VisitExpr_(const LENode* op) override;
  void VisitExpr_(const GTNode* op) override;
  void VisitExpr_(const GENode* op) override;
  void VisitExpr_(const AndNode* op) override;
  void VisitExpr_(const OrNode* op) override;
  void VisitExpr_(const ReduceNode* op) override;
  void VisitExpr_(const CastNode* op) override;
  void VisitExpr_(const NotNode* op) override;
  void VisitExpr_(const SelectNode* op) override;
  void VisitExpr_(const RampNode* op) override;
  void VisitExpr_(const BroadcastNode* op) override;
  void VisitExpr_(const ShuffleNode* op) override;
  void VisitExpr_(const IntImmNode* op) override;
  void VisitExpr_(const FloatImmNode* op) override;
  void VisitExpr_(const StringImmNode* op) override;
  void VisitExpr_(const AnyNode* op) override;
};

/*!
 * \brief ExprMutator that mutates expressions.
 */
class TVM_DLL ExprMutator : protected ExprFunctor<PrimExpr(const PrimExpr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  PrimExpr VisitExpr_(const VarNode* op) override;
  PrimExpr VisitExpr_(const SizeVarNode* op) override;
  PrimExpr VisitExpr_(const BufferLoadNode* op) override;
  PrimExpr VisitExpr_(const ProducerLoadNode* op) override;
  PrimExpr VisitExpr_(const LetNode* op) override;
  PrimExpr VisitExpr_(const CallNode* op) override;
  PrimExpr VisitExpr_(const AddNode* op) override;
  PrimExpr VisitExpr_(const SubNode* op) override;
  PrimExpr VisitExpr_(const MulNode* op) override;
  PrimExpr VisitExpr_(const DivNode* op) override;
  PrimExpr VisitExpr_(const ModNode* op) override;
  PrimExpr VisitExpr_(const FloorDivNode* op) override;
  PrimExpr VisitExpr_(const FloorModNode* op) override;
  PrimExpr VisitExpr_(const MinNode* op) override;
  PrimExpr VisitExpr_(const MaxNode* op) override;
  PrimExpr VisitExpr_(const EQNode* op) override;
  PrimExpr VisitExpr_(const NENode* op) override;
  PrimExpr VisitExpr_(const LTNode* op) override;
  PrimExpr VisitExpr_(const LENode* op) override;
  PrimExpr VisitExpr_(const GTNode* op) override;
  PrimExpr VisitExpr_(const GENode* op) override;
  PrimExpr VisitExpr_(const AndNode* op) override;
  PrimExpr VisitExpr_(const OrNode* op) override;
  PrimExpr VisitExpr_(const ReduceNode* op) override;
  PrimExpr VisitExpr_(const CastNode* op) override;
  PrimExpr VisitExpr_(const NotNode* op) override;
  PrimExpr VisitExpr_(const SelectNode* op) override;
  PrimExpr VisitExpr_(const RampNode* op) override;
  PrimExpr VisitExpr_(const BroadcastNode* op) override;
  PrimExpr VisitExpr_(const ShuffleNode* op) override;
  PrimExpr VisitExpr_(const IntImmNode* op) override;
  PrimExpr VisitExpr_(const FloatImmNode* op) override;
  PrimExpr VisitExpr_(const StringImmNode* op) override;
  PrimExpr VisitExpr_(const AnyNode* op) override;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_EXPR_FUNCTOR_H_
