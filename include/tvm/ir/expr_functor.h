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
 * \file tvm/ir/expr_functor.h
 * \brief Native visiting and mutation of core IR expressions.
 */
#ifndef TVM_IR_EXPR_FUNCTOR_H_
#define TVM_IR_EXPR_FUNCTOR_H_
#include <tvm/ir/expr.h>
#include <tvm/ir/object_functor.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/vector_expr.h>

namespace tvm {

/*!
 * \brief Type-dispatched expression functor with a caller-selected signature.
 * \tparam FType A function signature of the form R(const Expr&, Args...).
 *
 * Override Dispatch_ for a node type, or DispatchDefault_ for the default
 * behavior. Unlike structural visitors and mutators, this functor does not
 * traverse children automatically. Dispatch may use a registered ancestor.
 * A derived class can add node types with a fresh inherited table.
 * Use tvm::ExprFunctor explicitly when dialect functors are also in scope.
 */
template <typename FType>
class ExprFunctor;

template <typename R, typename... Args>
class ExprFunctor<R(const Expr&, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr&, Args...)>;

 public:
  /*! \brief The result type of this functor. */
  using result_type = R;
  /*! \brief Construct a functor with the core expression hooks. */
  ExprFunctor() : ExprFunctor(GlobalVTable()) {}
  /*!
   * \brief Dispatch an expression through Dispatch.
   * \param node The borrowed expression.
   * \param args Additional arguments forwarded to the selected hook.
   * \return The hook result.
   */
  TVM_FFI_INLINE R operator()(const Expr& node, Args... args) {
    return Dispatch(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch to a node hook, including registered ancestor hooks.
   * \param node The non-null borrowed expression.
   * \param args Additional arguments forwarded to the selected hook.
   * \return The hook result.
   */
  TVM_FFI_INLINE virtual R Dispatch(const Expr& node, Args... args) {
    TVM_FFI_ICHECK(node.defined()) << "Cannot dispatch a null expression";
    return (*vtable_)(node, this, std::forward<Args>(args)...);
  }

  virtual R Dispatch_(const OpaqueExprNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TupleNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TupleGetItemNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const TensorLoadNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const VarNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const GlobalVarNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const CallNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const IntImmNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const FloatImmNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const OpNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::StringImmNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::CastNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::AddNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::SubNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::MulNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::DivNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::ModNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::FloorDivNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::FloorModNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::MinNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::MaxNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::EQNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::NENode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LTNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LENode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::GTNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::GENode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::AndNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::OrNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::NotNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::SelectNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::LetNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::RampNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::BroadcastNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  virtual R Dispatch_(const prim::ShuffleNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }

  /*!
   * \brief Default node behavior, overridden by subclasses that handle arbitrary core nodes.
   * \param node The borrowed node.
   * \param args Additional arguments supplied to the functor.
   * \return The hook result.
   */
  virtual R DispatchDefault_(const ffi::Object* node, Args... args) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << node->GetTypeKey();
    TVM_FFI_UNREACHABLE();
  }

 protected:
  /*! \brief Dispatch table shared by this signature and its subclasses. */
  using VTable = ObjectFunctor<R(const ffi::ObjectRef&, TSelf*, Args...)>;
  /*!
   * \brief Construct a functor with an extended finalized table.
   * \param vtable The table, which must outlive the functor.
   */
  explicit ExprFunctor(const VTable* vtable) : vtable_(vtable) {}
  /*!
   * \brief Register core expression hooks in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable) {
    SetDispatch<TSelf, OpaqueExprNode>(vtable);
    SetDispatch<TSelf, TupleNode>(vtable);
    SetDispatch<TSelf, TupleGetItemNode>(vtable);
    SetDispatch<TSelf, TensorLoadNode>(vtable);
    SetDispatch<TSelf, VarNode>(vtable);
    SetDispatch<TSelf, GlobalVarNode>(vtable);
    SetDispatch<TSelf, CallNode>(vtable);
    SetDispatch<TSelf, IntImmNode>(vtable);
    SetDispatch<TSelf, FloatImmNode>(vtable);
    SetDispatch<TSelf, OpNode>(vtable);
    SetDispatch<TSelf, prim::StringImmNode>(vtable);
    SetDispatch<TSelf, prim::CastNode>(vtable);
    SetDispatch<TSelf, prim::AddNode>(vtable);
    SetDispatch<TSelf, prim::SubNode>(vtable);
    SetDispatch<TSelf, prim::MulNode>(vtable);
    SetDispatch<TSelf, prim::DivNode>(vtable);
    SetDispatch<TSelf, prim::ModNode>(vtable);
    SetDispatch<TSelf, prim::FloorDivNode>(vtable);
    SetDispatch<TSelf, prim::FloorModNode>(vtable);
    SetDispatch<TSelf, prim::MinNode>(vtable);
    SetDispatch<TSelf, prim::MaxNode>(vtable);
    SetDispatch<TSelf, prim::EQNode>(vtable);
    SetDispatch<TSelf, prim::NENode>(vtable);
    SetDispatch<TSelf, prim::LTNode>(vtable);
    SetDispatch<TSelf, prim::LENode>(vtable);
    SetDispatch<TSelf, prim::GTNode>(vtable);
    SetDispatch<TSelf, prim::GENode>(vtable);
    SetDispatch<TSelf, prim::AndNode>(vtable);
    SetDispatch<TSelf, prim::OrNode>(vtable);
    SetDispatch<TSelf, prim::NotNode>(vtable);
    SetDispatch<TSelf, prim::SelectNode>(vtable);
    SetDispatch<TSelf, prim::LetNode>(vtable);
    SetDispatch<TSelf, prim::RampNode>(vtable);
    SetDispatch<TSelf, prim::BroadcastNode>(vtable);
    SetDispatch<TSelf, prim::ShuffleNode>(vtable);
  }
  /*!
   * \brief Register a hook for an additional expression type.
   * \tparam Self The subclass implementing the hook.
   * \tparam Node The node type handled by the hook.
   * \param vtable The mutable table to receive the registration.
   */
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
 * \brief Structural Expr visitor with native overrides for core expressions.
 *
 * Prefer StructuralVisit or StructuralWalk for common cases; use ExprVisitor
 * for extensive per-kind customization or optimization. Default hooks follow
 * the registered structural child order and skip primitive metadata.
 * Native hooks match exact node types; derived node types use structural fallback
 * unless separately registered in a fresh inherited table with SetDispatch.
 * Existing hooks only require overriding. For an extra MyExprNode derived from
 * ExprNode, initialize an inherited table and register the new virtual hook:
 * \code
 * class MyExprVisitor : public ExprVisitor {
 *  public:
 *   TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(MyExprVisitor, ExprVisitor)
 *   using ExprVisitor::Visit_;
 *   virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const MyExprNode* node);
 *
 *  protected:
 *   static void InitVTable(VTable* vtable) {
 *     ExprVisitor::InitVTable(vtable);
 *     SetDispatch<MyExprVisitor, MyExprNode>(vtable);
 *   }
 * };
 * \endcode
 * Use tvm::ExprVisitor explicitly when dialect visitors are also in scope.
 */
class TVM_DLL ExprVisitor : public ObjectVisitor {
 public:
  /*! \brief Construct a visitor with the core expression hooks. */
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(ExprVisitor, ObjectVisitor)

  using ObjectVisitor::VisitExpected;

  // Override existing hooks directly. Extra types need a fresh inherited table.
  // Hooks borrow the node and return None, an owning interrupt, or an Error.
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const OpaqueExprNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const TupleNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const TupleGetItemNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const TensorLoadNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const VarNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const GlobalVarNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const CallNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const IntImmNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const FloatImmNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const OpNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::StringImmNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::CastNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::AddNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::SubNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::MulNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::DivNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::ModNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::FloorDivNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::FloorModNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::MinNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::MaxNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::EQNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::NENode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::LTNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::LENode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::GTNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::GENode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::AndNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::OrNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::NotNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::SelectNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::LetNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::RampNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::BroadcastNode* node);
  virtual Expected<ffi::Optional<VisitInterrupt>> Visit_(const prim::ShuffleNode* node);

 protected:
  /*!
   * \brief Construct a visitor with an extended native dispatch table.
   * \param vtable The finalized table, which must outlive this visitor.
   */
  explicit ExprVisitor(const VTable* vtable) : ObjectVisitor(vtable) {}
  /*!
   * \brief Register core expression hooks in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable);
};

/*!
 * \brief Structural Expr mutator with native overrides for core expressions
 * and structural fallback for other objects.
 *
 * Prefer StructuralMap for common cases; use ExprMutator for extensive per-kind
 * customization or optimization.
 * The default Var hook leaves PrimType variables unchanged before remap lookup.
 *
 * Native hooks match exact node types; derived node types need their own registrations.
 * Existing hooks only require overriding. For an extra MyExprNode derived from
 * ExprNode, initialize an inherited table and register the new virtual hook:
 * \code
 * class MyExprMutator : public ExprMutator {
 *  public:
 *   TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(MyExprMutator, ExprMutator)
 *   using ExprMutator::Mutate_;
 *   virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const MyExprNode* node, bool allow_inplace);
 *
 *  protected:
 *   static void InitVTable(VTable* vtable) {
 *     ExprMutator::InitVTable(vtable);
 *     SetDispatch<MyExprMutator, MyExprNode>(vtable);
 *   }
 * };
 * \endcode
 * Use tvm::ExprMutator explicitly when dialect mutators are also in scope.
 */
class TVM_DLL ExprMutator : public ObjectMutator {
 public:
  /*! \brief Construct a mutator with the core expression hooks. */
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(ExprMutator, ObjectMutator)

  using ObjectMutator::MaybeInplaceMutateIfUniqueExpected;
  using ObjectMutator::MutateExpected;

  /*!
   * \brief Mutate a borrowed expression without allowing in-place changes.
   * \param value The borrowed expression to mutate.
   * \return An Expr replacement, Unchanged, or an Error if mutation fails.
   * \note This entry trusts the Expr replacement contract of native hooks and extensions.
   */
  TVM_FFI_INLINE Expected<UnchangedOr<Expr>> MutateExpected(const Expr& value) noexcept {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Expr>>(
        ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ObjectMutator::MutateExpected(value)));
  }
  /*!
   * \brief Forward inherited permission and check the expression's uniqueness.
   * \param value The borrowed expression to mutate.
   * \param allow_inplace Whether the path to this expression is already uniquely owned.
   * \return An Expr replacement, Unchanged, or an Error if mutation fails.
   * \note This entry trusts the Expr replacement contract of native hooks and extensions.
   */
  TVM_FFI_INLINE Expected<UnchangedOr<Expr>> MaybeInplaceMutateIfUniqueExpected(
      const Expr& value, bool allow_inplace = true) noexcept {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<UnchangedOr<Expr>>(
        ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
            ObjectMutator::MaybeInplaceMutateIfUniqueExpected(value, allow_inplace)));
  }

  // A downstream class overrides any existing hook without rebuilding the table.
  // Extra node types use a fresh inherited table and SetDispatch<Self, ExtraNode>.
  // Hooks borrow the node and return an owning Expr replacement in Any, Unchanged, or Error.
  // Narrower field types are checked separately. Forward allow_inplace on every child edge.
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const OpaqueExprNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const TupleNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const TupleGetItemNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const TensorLoadNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const VarNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const GlobalVarNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const CallNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const IntImmNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const FloatImmNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const OpNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::StringImmNode* node,
                                                  bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::CastNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::AddNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::SubNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::MulNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::DivNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::ModNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::FloorDivNode* node,
                                                  bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::FloorModNode* node,
                                                  bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::MinNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::MaxNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::EQNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::NENode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::LTNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::LENode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::GTNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::GENode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::AndNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::OrNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::NotNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::SelectNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::LetNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::RampNode* node, bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::BroadcastNode* node,
                                                  bool allow_inplace);
  virtual Expected<UnchangedOr<ffi::Any>> Mutate_(const prim::ShuffleNode* node,
                                                  bool allow_inplace);

 protected:
  /*!
   * \brief Construct a mutator with an extended native dispatch table.
   * \param vtable The finalized table, which must outlive this mutator.
   */
  explicit ExprMutator(const VTable* vtable) : ObjectMutator(vtable) {}
  /*!
   * \brief Register core expression hooks in a fresh mutable table.
   * \param vtable The table to initialize before adding derived registrations.
   */
  static void InitVTable(VTable* vtable);
};

}  // namespace tvm
#endif  // TVM_IR_EXPR_FUNCTOR_H_
