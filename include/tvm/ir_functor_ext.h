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

#include <tvm/node/functor.h>
#include <tvm/ir.h>

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
      [](const ObjectRef& n, TSelf* self, Args... args) {               \
        return self->VisitExpr_(static_cast<const OP*>(n.get()),        \
                                std::forward<Args>(args)...);           \
      });                                                               \

#define IR_STMT_FUNCTOR_DISPATCH(OP)                                    \
  vtable.template set_dispatch<OP>(                                     \
      [](const ObjectRef& n, TSelf* self, Args... args) {               \
        return self->VisitStmt_(static_cast<const OP*>(n.get()),        \
                                std::forward<Args>(args)...);           \
      });                                                               \

template<typename R, typename ...Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
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
  virtual R VisitExprDefault_(const Object* op, Args ...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
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
  using FType = NodeFunctor<R(const ObjectRef& n, TSelf* self, Args... args)>;

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
  virtual R VisitStmt_(const SeqStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const Evaluate* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmtDefault_(const Object* op, Args ...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
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
    IR_STMT_FUNCTOR_DISPATCH(SeqStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(Evaluate);
    return vtable;
  }
};

#undef IR_STMT_FUNCTOR_DISPATCH
#undef IR_EXPR_FUNCTOR_DISPATCH
#undef EXPR_FUNCTOR_DEFAULT
#undef STMT_FUNCTOR_DEFAULT

/*!
 * \brief ExprVisitor
 */
class TVM_DLL ExprVisitor :
      public ExprFunctor<void(const Expr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  void VisitExpr_(const Variable* op) override;
  void VisitExpr_(const Load* op) override;
  void VisitExpr_(const Let* op) override;
  void VisitExpr_(const Call* op) override;
  void VisitExpr_(const Add* op) override;
  void VisitExpr_(const Sub* op) override;
  void VisitExpr_(const Mul* op) override;
  void VisitExpr_(const Div* op) override;
  void VisitExpr_(const Mod* op) override;
  void VisitExpr_(const FloorDiv* op) override;
  void VisitExpr_(const FloorMod* op) override;
  void VisitExpr_(const Min* op) override;
  void VisitExpr_(const Max* op) override;
  void VisitExpr_(const EQ* op) override;
  void VisitExpr_(const NE* op) override;
  void VisitExpr_(const LT* op) override;
  void VisitExpr_(const LE* op) override;
  void VisitExpr_(const GT* op) override;
  void VisitExpr_(const GE* op) override;
  void VisitExpr_(const And* op) override;
  void VisitExpr_(const Or* op) override;
  void VisitExpr_(const Reduce* op) override;
  void VisitExpr_(const Cast* op) override;
  void VisitExpr_(const Not* op) override;
  void VisitExpr_(const Select* op) override;
  void VisitExpr_(const Ramp* op) override;
  void VisitExpr_(const Broadcast* op) override;
  void VisitExpr_(const Shuffle* op) override;
  void VisitExpr_(const IntImm* op) override;
  void VisitExpr_(const UIntImm* op) override;
  void VisitExpr_(const FloatImm* op) override;
  void VisitExpr_(const StringImm* op) override;
};

/*!
 * \brief ExprMutator that mutates expressions.
 */
class TVM_DLL ExprMutator :
      protected ExprFunctor<Expr(const Expr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  Expr VisitExpr_(const Variable* op) override;
  Expr VisitExpr_(const Load* op) override;
  Expr VisitExpr_(const Let* op) override;
  Expr VisitExpr_(const Call* op) override;
  Expr VisitExpr_(const Add* op) override;
  Expr VisitExpr_(const Sub* op) override;
  Expr VisitExpr_(const Mul* op) override;
  Expr VisitExpr_(const Div* op) override;
  Expr VisitExpr_(const Mod* op) override;
  Expr VisitExpr_(const FloorDiv* op) override;
  Expr VisitExpr_(const FloorMod* op) override;
  Expr VisitExpr_(const Min* op) override;
  Expr VisitExpr_(const Max* op) override;
  Expr VisitExpr_(const EQ* op) override;
  Expr VisitExpr_(const NE* op) override;
  Expr VisitExpr_(const LT* op) override;
  Expr VisitExpr_(const LE* op) override;
  Expr VisitExpr_(const GT* op) override;
  Expr VisitExpr_(const GE* op) override;
  Expr VisitExpr_(const And* op) override;
  Expr VisitExpr_(const Or* op) override;
  Expr VisitExpr_(const Reduce* op) override;
  Expr VisitExpr_(const Cast* op) override;
  Expr VisitExpr_(const Not* op) override;
  Expr VisitExpr_(const Select* op) override;
  Expr VisitExpr_(const Ramp* op) override;
  Expr VisitExpr_(const Broadcast* op) override;
  Expr VisitExpr_(const Shuffle* op) override;
  Expr VisitExpr_(const IntImm* op) override;
  Expr VisitExpr_(const UIntImm* op) override;
  Expr VisitExpr_(const FloatImm* op) override;
  Expr VisitExpr_(const StringImm* op) override;
};

/*!
 * \brief StmtVisitor.
 */
class TVM_DLL StmtVisitor :
      protected StmtFunctor<void(const Stmt&)> {
 public:
  using StmtFunctor::operator();

 protected:
  using StmtFunctor::VisitStmt;
  /*!
   * \brief Visitor to Exprs, can be overriden
   *        to do recursive changes to Exprs.
   * \note A common pattern is to call ExprVisitor here,
   *       or have a class sub-class both StmtVisitor and ExprVisitor
   *       and redirect Visit to ExprMutator::VisitExpr(Expr)
   */
  virtual void VisitExpr(const Expr& e) {}
  // statement visitor
  void VisitStmt_(const AttrStmt* op) override;
  void VisitStmt_(const IfThenElse* op) override;
  void VisitStmt_(const LetStmt* op) override;
  void VisitStmt_(const For* op) override;
  void VisitStmt_(const Allocate* op) override;
  void VisitStmt_(const Store* op) override;
  void VisitStmt_(const Free* op) override;
  void VisitStmt_(const AssertStmt* op) override;
  void VisitStmt_(const ProducerConsumer* op) override;
  void VisitStmt_(const Provide* op) override;
  void VisitStmt_(const Realize* op) override;
  void VisitStmt_(const Prefetch* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const Evaluate* op) override;
};

/*!
 * \brief StmtMutator that mutates the statements.
 */
class TVM_DLL StmtMutator :
      protected StmtFunctor<Stmt(const Stmt&)> {
 public:
  /*!
   * \brief Mutate stmt.
   * \param stmt The input statement to be mutated.
   * \return The result of the call
   * \note It is important that stmt is passed by value.
   *       so copy on write can be triggered correctly.
   *       do mutator(std::move(stmt)) or when copy elison is triggered.
   */
  Stmt operator()(Stmt stmt) {
    allow_copy_on_write_ = true;
    return VisitStmt(stmt);
  }

 protected:
  // We perform copy on write optimizations on the StmtMutator
  // so that an unique copy of parent can be mutated inplace
  // when some of its children changed.
  // We only do such optimization for Stmt nests(instead of Exprs) for now
  // as Stmt's parent state is more likely remain unchanged when one of
  // its child block changes.
  /*!
   * \brief Internal state to indicate whether copy on write is enabled.
   *  COW is enabled iff all the parents of the node are unique.
   */
  bool allow_copy_on_write_{false};
  /*!
   * \brief Perform copy on write on node.
   *
   *  If CopyOnWrite is allowed, directly return
   *  a strong reference to the node container.
   *  Otherwise, return a copy of the node.
   *
   * \return The result object pointer.
   */
  template<typename TNode>
  ObjectPtr<TNode> CopyOnWrite(const TNode* node) {
    if (allow_copy_on_write_) {
      // return the old node.
      return runtime::GetObjectPtr<TNode>(const_cast<TNode*>(node));
    } else {
      // Make a new copy of the node.
      // need to rely on the default copy constructor
      return runtime::make_object<TNode>(*node);
    }
  }
  /*!
   * \brief Internal mutator that everyone calls.
   * \note To override mutate's behavior, override VisitExpr instead.
   * \param stmt The input stmt.
   * \return The mutated results.
   */
  Stmt VisitStmt(const Stmt& stmt) override {
    if (allow_copy_on_write_ && !stmt.unique()) {
      allow_copy_on_write_ = false;
      Stmt ret = StmtFunctor::VisitStmt(stmt);
      allow_copy_on_write_ = true;
      return ret;
    } else {
      return StmtFunctor::VisitStmt(stmt);
    }
  }
  /*!
   * \brief Visitor to Exprs, can be overriden
   *        to do recursive changes to Exprs.
   * \note A common pattern is to call ExprMutator here,
   *       or have a class sub-class both StmtMutator and ExprMutator
   *       and redirect Mutate to ExprMutator::Mutate(Expr)
   */
  virtual Expr VisitExpr(const Expr& e) {
    return e;
  }
  // statement visitor
  Stmt VisitStmt_(const AttrStmt* op) override;
  Stmt VisitStmt_(const IfThenElse* op) override;
  Stmt VisitStmt_(const LetStmt* op) override;
  Stmt VisitStmt_(const For* op) override;
  Stmt VisitStmt_(const Allocate* op) override;
  Stmt VisitStmt_(const Store* op) override;
  Stmt VisitStmt_(const Free* op) override;
  Stmt VisitStmt_(const AssertStmt* op) override;
  Stmt VisitStmt_(const ProducerConsumer* op) override;
  Stmt VisitStmt_(const Provide* op) override;
  Stmt VisitStmt_(const Realize* op) override;
  Stmt VisitStmt_(const Prefetch* op) override;
  Stmt VisitStmt_(const SeqStmtNode* op) override;
  Stmt VisitStmt_(const Evaluate* op) override;
  /*!
   * \brief Alternative advance method for SeqStmtNode.
   *
   *  This function can be called when a child class override
   *  VisitStmt_(const SeqStmtNode*) to introduce
   *  the special behavior to visit
   *
   * \param op The sequence.
   * \param flatten_before_visit Whether to flatten the sequence before visit.
   * \param fmutate The mutate function, can be nullptr, which defaults to Visit.
   * \return The mutated result.
   */
  Stmt VisitSeqStmt_(const SeqStmtNode* op,
                     bool flatten_before_visit,
                     std::function<Stmt(const Stmt&)> fmutate = nullptr);
  // internal helper.
  class Internal;
};

/*!
 * \brief Visitor that recursively visit stmts and exprs on them.
 */
class StmtExprVisitor :
      public StmtVisitor,
      public ExprVisitor {
 public:
  using StmtVisitor::operator();
  using ExprVisitor::operator();

 protected:
  using StmtVisitor::VisitStmt;
  using ExprVisitor::VisitExpr;

  void VisitExpr(const Expr& e) override {
    return ExprVisitor::VisitExpr(e);
  }
};

/*!
 * \brief Mutator that recursively mutates stmts and exprs on them.
 */
class StmtExprMutator :
      public StmtMutator,
      public ExprMutator {
 public:
  using StmtMutator::operator();
  using ExprMutator::operator();

 protected:
  using StmtMutator::VisitExpr;
  using ExprMutator::VisitExpr;

  Expr VisitExpr(const Expr& e) override {
    return ExprMutator::VisitExpr(e);
  }
};

/*!
 * \brief recursively visit the ir in post DFS order node, and transform it
 *
 * \param node The ir to be transformed.
 * \param preorder The function called in before recursive mutation
 *          If preorder returns None, then the transform will proceed to recursive call.
 *          If preorder returns a not None Stmt/Expr, the transformer will simply return it and
 *          won't do further recursion.
 * \param postorder The function called after recursive mutation.
 *          The recursive mutation result is passed to postorder for further mutation.
 * \param only_enable List of StringImm.
 *          If it is empty, all IRNode will call preorder/postorder
 *          If it is not empty, preorder/postorder will only be called
 *          when the IRNode's type key is in the list.
 */
TVM_DLL Stmt IRTransform(Stmt node,
                         const runtime::PackedFunc& preorder,
                         const runtime::PackedFunc& postorder,
                         const Array<Expr>& only_enable = {});

/*!
 * \brief recursively visit the ir in post DFS order node, apply fvisit
 * Each node is guaranteed to be visited only once.
 * \param node The ir to be visited.
 * \param fvisit The visitor function to be applied.
 */
TVM_DLL void PostOrderVisit(const ObjectRef& node, std::function<void(const ObjectRef&)> fvisit);


}  // namespace ir
}  // namespace tvm
#endif  // TVM_IR_FUNCTOR_EXT_H_
