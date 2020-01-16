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
  virtual R VisitExpr_(const LoadNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
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
  virtual R VisitExprDefault_(const Object* op, Args ...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    IR_EXPR_FUNCTOR_DISPATCH(VarNode);
    IR_EXPR_FUNCTOR_DISPATCH(SizeVarNode);
    IR_EXPR_FUNCTOR_DISPATCH(LoadNode);
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
  virtual R VisitStmt_(const LetStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AttrStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const IfThenElseNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ForNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AllocateNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const StoreNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const FreeNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AssertStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ProducerConsumerNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ProvideNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const RealizeNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const PrefetchNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const SeqStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const EvaluateNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmtDefault_(const Object* op, Args ...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    return R();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    IR_STMT_FUNCTOR_DISPATCH(LetStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(AttrStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(IfThenElseNode);
    IR_STMT_FUNCTOR_DISPATCH(ForNode);
    IR_STMT_FUNCTOR_DISPATCH(AllocateNode);
    IR_STMT_FUNCTOR_DISPATCH(StoreNode);
    IR_STMT_FUNCTOR_DISPATCH(FreeNode);
    IR_STMT_FUNCTOR_DISPATCH(AssertStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(ProducerConsumerNode);
    IR_STMT_FUNCTOR_DISPATCH(ProvideNode);
    IR_STMT_FUNCTOR_DISPATCH(RealizeNode);
    IR_STMT_FUNCTOR_DISPATCH(PrefetchNode);
    IR_STMT_FUNCTOR_DISPATCH(SeqStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(EvaluateNode);
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
      public ExprFunctor<void(const PrimExpr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const SizeVarNode* op) override;
  void VisitExpr_(const LoadNode* op) override;
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
};

/*!
 * \brief ExprMutator that mutates expressions.
 */
class TVM_DLL ExprMutator :
      protected ExprFunctor<PrimExpr(const PrimExpr&)> {
 public:
  using ExprFunctor::operator();

 protected:
  using ExprFunctor::VisitExpr;
  // list of functions to override.
  PrimExpr VisitExpr_(const VarNode* op) override;
  PrimExpr VisitExpr_(const SizeVarNode* op) override;
  PrimExpr VisitExpr_(const LoadNode* op) override;
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
  virtual void VisitExpr(const PrimExpr& e) {}
  // statement visitor
  void VisitStmt_(const AttrStmtNode* op) override;
  void VisitStmt_(const IfThenElseNode* op) override;
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const StoreNode* op) override;
  void VisitStmt_(const FreeNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const ProducerConsumerNode* op) override;
  void VisitStmt_(const ProvideNode* op) override;
  void VisitStmt_(const RealizeNode* op) override;
  void VisitStmt_(const PrefetchNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
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
  virtual PrimExpr VisitExpr(const PrimExpr& e) {
    return e;
  }
  // statement visitor
  Stmt VisitStmt_(const AttrStmtNode* op) override;
  Stmt VisitStmt_(const IfThenElseNode* op) override;
  Stmt VisitStmt_(const LetStmtNode* op) override;
  Stmt VisitStmt_(const ForNode* op) override;
  Stmt VisitStmt_(const AllocateNode* op) override;
  Stmt VisitStmt_(const StoreNode* op) override;
  Stmt VisitStmt_(const FreeNode* op) override;
  Stmt VisitStmt_(const AssertStmtNode* op) override;
  Stmt VisitStmt_(const ProducerConsumerNode* op) override;
  Stmt VisitStmt_(const ProvideNode* op) override;
  Stmt VisitStmt_(const RealizeNode* op) override;
  Stmt VisitStmt_(const PrefetchNode* op) override;
  Stmt VisitStmt_(const SeqStmtNode* op) override;
  Stmt VisitStmt_(const EvaluateNode* op) override;
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

  void VisitExpr(const PrimExpr& e) override {
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

  PrimExpr VisitExpr(const PrimExpr& e) override {
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
                         const Array<PrimExpr>& only_enable = {});

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
