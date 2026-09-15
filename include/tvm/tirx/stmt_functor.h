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
 * \file tvm/tirx/stmt_functor.h
 *
 * \brief Functors for tirx stmts
 *        utility functions to call common functors.
 */
#ifndef TVM_TIRX_STMT_FUNCTOR_H_
#define TVM_TIRX_STMT_FUNCTOR_H_

#include <tvm/ir/expr_functor.h>
#include <tvm/ir/object_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile_primitive.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace tirx {
/*!
 * \brief Same as ExprFunctor except it is applied on statements
 * \tparam FType The function signature.
 * \sa ExprFunctor
 */
template <typename FType>
class StmtFunctor;

#define STMT_FUNCTOR_DEFAULT                                   \
  {                                                            \
    return VisitStmtDefault_(op, std::forward<Args>(args)...); \
  }

#define IR_STMT_FUNCTOR_DISPATCH(OP)                                                       \
  vtable.template SetDispatch<OP>([](const ffi::ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitStmt_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class StmtFunctor<R(const Stmt& n, Args... args)> {
 private:
  using TSelf = StmtFunctor<R(const Stmt& n, Args... args)>;
  using FType = ObjectFunctor<R(const ffi::ObjectRef& n, TSelf* self, Args... args)>;

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
  R operator()(const Stmt& n, Args... args) { return VisitStmt(n, std::forward<Args>(args)...); }
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
  virtual R VisitStmt_(const BindNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AttrStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const IfThenElseNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ForNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const WhileNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ReturnNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const BreakNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ContinueNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AllocBufferNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const DeclBufferNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const BufferStoreNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const AssertStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const SeqStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const EvaluateNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const SBlockNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const SBlockRealizeNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const ScopeIdDefStmtNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmt_(const tirx::TilePrimitiveCallNode* op, Args... args) STMT_FUNCTOR_DEFAULT;
  virtual R VisitStmtDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    IR_STMT_FUNCTOR_DISPATCH(BindNode);
    IR_STMT_FUNCTOR_DISPATCH(AttrStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(IfThenElseNode);
    IR_STMT_FUNCTOR_DISPATCH(ForNode);
    IR_STMT_FUNCTOR_DISPATCH(WhileNode);
    IR_STMT_FUNCTOR_DISPATCH(ReturnNode);
    IR_STMT_FUNCTOR_DISPATCH(BreakNode);
    IR_STMT_FUNCTOR_DISPATCH(ContinueNode);
    IR_STMT_FUNCTOR_DISPATCH(AllocBufferNode);
    IR_STMT_FUNCTOR_DISPATCH(DeclBufferNode);
    IR_STMT_FUNCTOR_DISPATCH(AssertStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(SeqStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(EvaluateNode);
    IR_STMT_FUNCTOR_DISPATCH(BufferStoreNode);
    IR_STMT_FUNCTOR_DISPATCH(SBlockNode);
    IR_STMT_FUNCTOR_DISPATCH(SBlockRealizeNode);
    IR_STMT_FUNCTOR_DISPATCH(ScopeIdDefStmtNode);
    IR_STMT_FUNCTOR_DISPATCH(tirx::TilePrimitiveCallNode);
    vtable.Finalize();
    return vtable;
  }
};

#undef IR_STMT_FUNCTOR_DISPATCH
#undef STMT_FUNCTOR_DEFAULT

/*!
 * \brief Native visitor for TIRx statements and their expression operands.
 *
 * Inherits core expression dispatch and preserves TIRx traversal order and
 * buffer definition/use boundaries. Allocate visitors with ffi::make_object;
 * hooks return the first interrupt or throw on failure.
 *
 * Native hooks match exact types. Unregistered OpaqueExprNode subclasses use
 * structural traversal; leaf types register non-descending structural hooks.
 */
class TVM_DLL StmtExprVisitor : public tvm::ExprVisitor {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitor, tvm::ExprVisitor)

  using tvm::ExprVisitor::Visit;
  using tvm::ExprVisitor::Visit_;

  virtual ffi::Optional<VisitInterrupt> Visit_(const BindNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const AttrStmtNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const IfThenElseNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ForNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const WhileNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ReturnNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const BreakNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ContinueNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const AllocBufferNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const DeclBufferNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const BufferStoreNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SeqStmtNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const EvaluateNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const ScopeIdDefStmtNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TilePrimitiveCallNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const BufferRegionNode* op);

  // Preserve TIRx operand traversal where it differs from the shared defaults.
  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TupleNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::RampNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleNode* op) override;

 protected:
  // Visit definition metadata as uses, separately from the buffer Var definition.
  ffi::Optional<VisitInterrupt> VisitBufferMetadata(const BufferVar& buffer);

  explicit StmtExprVisitor(const VTable* vtable) : tvm::ExprVisitor(vtable) {}
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief StmtMutator that mutates the statements.
 */
class TVM_DLL StmtMutator : protected StmtFunctor<Stmt(const Stmt&)> {
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
  /*! \brief Map from old buffer to new buffer, populated by VisitBufferDef. */
  ffi::Map<BufferVar, BufferVar> buffer_remap_;
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
  template <typename TNode>
  ffi::ObjectPtr<TNode> CopyOnWrite(const TNode* node) {
    static_assert(std::is_base_of<StmtNode, TNode>::value,
                  "StmtMutator:: CopyOnWrite requires us to track uniqueness of all parent "
                  "nodes during the recursion. Because the child classes do not necessarily "
                  "check the Array, Expr and other structures during the visit, it is only safe to "
                  "call this function with StmtNodes for now. "
                  "Please create a new node directly in other cases.");
    if (allow_copy_on_write_) {
      // return the old node.
      return ffi::GetObjectPtr<TNode>(const_cast<TNode*>(node));
    } else {
      // Make a new copy of the node.
      // need to rely on the default copy constructor
      return ffi::make_object<TNode>(*node);
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
  virtual Expr VisitExpr(const Expr& e) { return e; }
  /*! \brief Mutate a primitive expression and verify that it remains primitive. */
  PrimExpr VisitPrimExpr(const PrimExpr& e) { return VisitExpr(e).as_or_throw<PrimExpr>(); }
  /*!
   * \brief Visit buffer at definition site. Visits shape/strides/elem_offset via VisitExpr.
   *  If any field changes, creates a new buffer and records it in buffer_remap_.
   * \param buffer The buffer being defined.
   * \param alloc_data If true, the buffer's data pointer is a new allocation (AllocBuffer);
   *              if false, data references an existing variable (DeclBuffer).
   * \return The (possibly new) buffer.
   */
  virtual BufferVar VisitBufferDef(const BufferVar& buffer, bool alloc_data);
  /*!
   * \brief Visit buffer at use site (BufferStore, BufferLoad, SBlock reads/writes).
   *  By default, returns the remapped buffer from buffer_remap_ if exists, otherwise
   *  returns the original buffer. BufferVar fields are visited at their definition site.
   * \return The (possibly remapped) buffer.
   */
  virtual BufferVar VisitBufferUse(const BufferVar& buffer);
  // statement visitor
  Stmt VisitStmt_(const BindNode* op) override;
  Stmt VisitStmt_(const AttrStmtNode* op) override;
  Stmt VisitStmt_(const IfThenElseNode* op) override;
  Stmt VisitStmt_(const ForNode* op) override;
  Stmt VisitStmt_(const WhileNode* op) override;
  Stmt VisitStmt_(const ReturnNode* op) override;
  Stmt VisitStmt_(const BreakNode* op) override;
  Stmt VisitStmt_(const ContinueNode* op) override;
  Stmt VisitStmt_(const AllocBufferNode* op) override;
  Stmt VisitStmt_(const DeclBufferNode* op) override;
  Stmt VisitStmt_(const BufferStoreNode* op) override;
  Stmt VisitStmt_(const AssertStmtNode* op) override;
  Stmt VisitStmt_(const SeqStmtNode* op) override;
  Stmt VisitStmt_(const EvaluateNode* op) override;
  Stmt VisitStmt_(const SBlockNode* op) override;
  Stmt VisitStmt_(const SBlockRealizeNode* op) override;
  Stmt VisitStmt_(const ScopeIdDefStmtNode* op) override;
  Stmt VisitStmt_(const tirx::TilePrimitiveCallNode* op) override;
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
  Stmt VisitSeqStmt_(const SeqStmtNode* op, bool flatten_before_visit,
                     std::function<Stmt(const Stmt&)> fmutate = nullptr);

  // internal helper.
  class Internal;
};

/*!
 * \brief Mutator that recursively mutates stmts and exprs on them.
 */
class TVM_DLL StmtExprMutator : public ExprMutator, public StmtMutator {
 public:
  using StmtMutator::operator();
  using ExprMutator::operator();

 protected:
  using ExprMutator::VisitExpr;
  using ExprMutator::VisitExpr_;
  using ExprMutator::VisitPrimExpr;
  using StmtMutator::VisitStmt;

  Expr VisitExpr(const Expr& e) override { return ExprMutator::VisitExpr(e); }
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const TensorLoadNode* op) override;
  Expr VisitExpr_(const BufferRegionNode* op) override;
};

/*!
 * \brief Substitute the var specified by vmap and legalize data types after substitution.
 * \param stmt The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 *
 * Substitution may change the data type of the expression.
 *
 * \return The result.
 */
TVM_DLL Stmt SubstituteWithDataTypeLegalization(
    Stmt stmt, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap);

/*!
 * \brief Substitute the var specified by vmap and legalize data types after substitution.
 * \param expr The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 *
 * Substitution may change the data type of the expression.
 *
 * \return The result.
 */
TVM_DLL PrimExpr SubstituteWithDataTypeLegalization(
    PrimExpr expr, std::function<ffi::Optional<PrimExpr>(const Var&)> vmap);

/*!
 * \brief Check if the statement contains the specified node type.
 *
 * This utility potentially walks the entire statement, and should
 * therefore not be used if it could otherwise be merged with another
 * pass.
 *
 * \param stmt The statement to be searched
 * \return Whether stmt contains Node
 */
template <typename Node, typename = std::enable_if_t<std::is_base_of_v<StmtNode, Node>>>
bool ContainsNode(const Stmt& stmt) {
  struct Visitor : StmtExprVisitor {
    // Early bail-out, if we already found the node. Skip expression operands.
    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) final {
      if (contains_node || value.as<ExprNode>()) {
        return std::nullopt;
      }
      return StmtExprVisitor::Visit(value);
    }

    ffi::Optional<VisitInterrupt> Visit_(const Node* block) override {
      contains_node = true;
      return std::nullopt;
    }

    bool contains_node{false};
  };

  auto visitor = ffi::make_object<Visitor>();
  visitor->Visit(stmt);
  return visitor->contains_node;
}

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIR_STMT_FUNCTOR_H_
