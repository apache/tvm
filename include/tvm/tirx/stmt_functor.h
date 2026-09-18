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
 * \brief Type-dispatched statement functor with a caller-selected signature.
 *
 * Override Dispatch_ for a node type or DispatchDefault_ for default behavior.
 * This functor does not traverse children automatically. Dispatch may use a
 * registered ancestor. Derived extensions can initialize a fresh inherited
 * table with InitVTable and register additional hooks with SetDispatch.
 * \tparam FType The function signature.
 * \sa ExprFunctor
 */
template <typename FType>
class StmtFunctor;

template <typename R, typename... Args>
class StmtFunctor<R(const Stmt&, Args...)> {
 private:
  using TSelf = StmtFunctor<R(const Stmt&, Args...)>;

 public:
  /*! \brief The result type of this functor. */
  using result_type = R;
  /*! \brief Construct a functor with the TIRx statement hooks. */
  StmtFunctor() : StmtFunctor(GlobalVTable()) {}
  /*! \brief Destroy through the statement functor base. */
  virtual ~StmtFunctor() = default;
  /*! \brief Dispatch a statement, forwarding additional arguments to its hook. */
  TVM_FFI_INLINE R operator()(const Stmt& node, Args... args) {
    return Dispatch(node, std::forward<Args>(args)...);
  }
  /*! \brief Dispatch to a node hook, including registered ancestor hooks. */
  TVM_FFI_INLINE virtual R Dispatch(const Stmt& node, Args... args) {
    TVM_FFI_ICHECK(node.defined()) << "Cannot dispatch a null statement";
    return (*vtable_)(node, this, std::forward<Args>(args)...);
  }

  virtual R Dispatch_(const BindNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const AttrStmtNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const IfThenElseNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ForNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const WhileNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ReturnNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const BreakNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ContinueNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const AllocBufferNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const DeclBufferNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const BufferStoreNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const AssertStmtNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const SeqStmtNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const EvaluateNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const ScopeIdDefStmtNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const tirx::TilePrimitiveCallNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*! \brief Default behavior for statement hooks not overridden by a subclass. */
  virtual R DispatchDefault_(const ffi::Object* node, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << node->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

 protected:
  /*! \brief Dispatch table shared by this signature and its subclasses. */
  using VTable = ObjectFunctor<R(const ffi::ObjectRef&, TSelf*, Args...)>;
  /*! \brief Construct with a finalized table that outlives the functor. */
  explicit StmtFunctor(const VTable* vtable) : vtable_(vtable) {}
  /*! \brief Register statement hooks in a fresh mutable table. */
  static void InitVTable(VTable* vtable) {
    vtable->template SetDispatch<StmtNode>(
        [](const ffi::ObjectRef& node, TSelf* self, Args... args) -> R {
          return self->DispatchDefault_(node.get(), std::forward<Args>(args)...);
        });
    SetDispatch<TSelf, BindNode>(vtable);
    SetDispatch<TSelf, AttrStmtNode>(vtable);
    SetDispatch<TSelf, IfThenElseNode>(vtable);
    SetDispatch<TSelf, ForNode>(vtable);
    SetDispatch<TSelf, WhileNode>(vtable);
    SetDispatch<TSelf, ReturnNode>(vtable);
    SetDispatch<TSelf, BreakNode>(vtable);
    SetDispatch<TSelf, ContinueNode>(vtable);
    SetDispatch<TSelf, AllocBufferNode>(vtable);
    SetDispatch<TSelf, DeclBufferNode>(vtable);
    SetDispatch<TSelf, BufferStoreNode>(vtable);
    SetDispatch<TSelf, AssertStmtNode>(vtable);
    SetDispatch<TSelf, SeqStmtNode>(vtable);
    SetDispatch<TSelf, EvaluateNode>(vtable);
    SetDispatch<TSelf, ScopeIdDefStmtNode>(vtable);
    SetDispatch<TSelf, tirx::TilePrimitiveCallNode>(vtable);
  }
  /*! \brief Register an additional node hook implemented by Self. */
  template <typename Self, typename Node>
  static void SetDispatch(VTable* vtable) {
    vtable->template SetDispatch<Node>(
        [](const ffi::ObjectRef& node, TSelf* self, Args... args) -> R {
          return static_cast<Self*>(self)->Dispatch_(static_cast<const Node*>(node.get()),
                                                     std::forward<Args>(args)...);
        });
  }

 private:
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
  const VTable* vtable_;
};

/*!
 * \brief Native visitor for TIRx statements and their expression operands.
 *
 * Inherits core expression dispatch and preserves TIRx traversal order and
 * buffer definition/use boundaries. Allocate visitors with ffi::make_object;
 * hooks return the first interrupt or throw on failure.
 * To preserve thrown ffi::Error subclasses, keep child traversal native:
 * Visit(array) crosses structural callbacks that may erase the C++ subtype.
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
  virtual ffi::Optional<VisitInterrupt> Visit_(const ScopeIdDefStmtNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const TilePrimitiveCallNode* op);

  // Preserve TIRx operand traversal where it differs from the shared defaults.
  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TupleNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const CallNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::RampNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastNode* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleNode* op) override;

  /*! \brief Visit definition metadata as uses, separately from the buffer Var definition. */
  ffi::Optional<VisitInterrupt> VisitBufferMetadata(const BufferVar& buffer);

 protected:
  explicit StmtExprVisitor(const VTable* vtable) : tvm::ExprVisitor(vtable) {}
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Mutator that recursively mutates stmts and exprs on them.
 *
 * Base hooks preserve stored types and derived fields. They rewrite structural
 * children without re-inferring types or repeating constructor validation,
 * including on in-place writes. Passes that change dtypes or index lanes must
 * provide the corresponding TensorLoad, BufferStore, or TensorRegion inference.
 */
class TVM_DLL StmtExprMutator : public tvm::ExprMutator {
 public:
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprMutator, tvm::ExprMutator)
  using tvm::ExprMutator::Mutate;
  using tvm::ExprMutator::Mutate_;

  /*! \brief Mutate a borrowed statement through the virtual generic entry.
   * \param stmt The borrowed statement.
   * \param inplace_mode Inherited permission along the complete ownership chain.
   * \return Unchanged or an owning statement replacement.
   * \note Entry overrides must preserve the statement category. Result storage
   *       is transferred without a runtime category check, as for Expr/PrimExpr.
   */
  TVM_FFI_INLINE UnchangedOr<Stmt> Mutate(const Stmt& stmt,
                                          InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<Stmt>(
        ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(stmt), inplace_mode)));
  }

  virtual UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const WhileNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ReturnNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const BreakNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ContinueNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const AssertStmtNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const SeqStmtNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const EvaluateNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const ScopeIdDefStmtNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const TilePrimitiveCallNode* op, InplaceMode inplace_mode);

 protected:
  explicit StmtExprMutator(const VTable* vtable) : tvm::ExprMutator(vtable) {}
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Substitute the var specified by vmap and legalize data types after substitution.
 * \param stmt The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 *
 * This statement overload legalizes only core TIRX nodes.  Dtype-changing
 * substitutions must be applied after lowering dialect blocks: structural
 * traversal of extension statements does not legalize their iterator domains or
 * bindings.  Type-preserving mappings continue to traverse schedulable blocks
 * structurally before lowering.
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
      if (value.as<Node>()) {
        contains_node = true;
        return std::nullopt;
      }
      return StmtExprVisitor::Visit(value);
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
