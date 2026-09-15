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

/*!
 * \brief Expression functor specialized for a result and additional arguments.
 * \tparam R The hook result type.
 * \tparam Args The additional hook argument types.
 */
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

  /*!
   * \brief Dispatch OpaqueExprNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const OpaqueExprNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch TupleNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const TupleNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch TupleGetItemNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const TupleGetItemNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch TensorLoadNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const TensorLoadNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch VarNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const VarNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch GlobalVarNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const GlobalVarNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch CallNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const CallNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch IntImmNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const IntImmNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch FloatImmNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const FloatImmNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch OpNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const OpNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::StringImmNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::StringImmNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::CastNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::CastNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::AddNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::AddNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::SubNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::SubNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::MulNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::MulNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::DivNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::DivNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::ModNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::ModNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::FloorDivNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::FloorDivNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::FloorModNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::FloorModNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::MinNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::MinNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::MaxNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::MaxNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::EQNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::EQNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::NENode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::NENode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::LTNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::LTNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::LENode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::LENode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::GTNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::GTNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::GENode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::GENode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::AndNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::AndNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::OrNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::OrNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::NotNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::NotNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::SelectNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::SelectNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::LetNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::LetNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::RampNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::RampNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::BroadcastNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
  virtual R Dispatch_(const prim::BroadcastNode* node, Args... args) {
    return DispatchDefault_(node, std::forward<Args>(args)...);
  }
  /*!
   * \brief Dispatch prim::ShuffleNode to the default hook.
   * \param node The borrowed expression node.
   * \param args Additional arguments forwarded to the default hook.
   * \return The default hook result.
   */
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
 *   virtual ffi::Optional<VisitInterrupt> Visit_(const MyExprNode* node);
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

  using ObjectVisitor::Visit;

  // Override existing hooks directly. Extra types need a fresh inherited table.
  // Hooks borrow the node and return None or an owning interrupt, and throw on failure.
  /*!
   * \brief Visit OpaqueExprNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const OpaqueExprNode* node);
  /*!
   * \brief Visit TupleNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const TupleNode* node);
  /*!
   * \brief Visit TupleGetItemNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const TupleGetItemNode* node);
  /*!
   * \brief Visit TensorLoadNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode* node);
  /*!
   * \brief Visit VarNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const VarNode* node);
  /*!
   * \brief Visit GlobalVarNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const GlobalVarNode* node);
  /*!
   * \brief Visit CallNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const CallNode* node);
  /*!
   * \brief Visit IntImmNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const IntImmNode* node);
  /*!
   * \brief Visit FloatImmNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const FloatImmNode* node);
  /*!
   * \brief Visit OpNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const OpNode* node);
  /*!
   * \brief Visit prim::StringImmNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::StringImmNode* node);
  /*!
   * \brief Visit prim::CastNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::CastNode* node);
  /*!
   * \brief Visit prim::AddNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AddNode* node);
  /*!
   * \brief Visit prim::SubNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SubNode* node);
  /*!
   * \brief Visit prim::MulNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MulNode* node);
  /*!
   * \brief Visit prim::DivNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::DivNode* node);
  /*!
   * \brief Visit prim::ModNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ModNode* node);
  /*!
   * \brief Visit prim::FloorDivNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorDivNode* node);
  /*!
   * \brief Visit prim::FloorModNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::FloorModNode* node);
  /*!
   * \brief Visit prim::MinNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MinNode* node);
  /*!
   * \brief Visit prim::MaxNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::MaxNode* node);
  /*!
   * \brief Visit prim::EQNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::EQNode* node);
  /*!
   * \brief Visit prim::NENode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NENode* node);
  /*!
   * \brief Visit prim::LTNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LTNode* node);
  /*!
   * \brief Visit prim::LENode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LENode* node);
  /*!
   * \brief Visit prim::GTNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GTNode* node);
  /*!
   * \brief Visit prim::GENode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::GENode* node);
  /*!
   * \brief Visit prim::AndNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::AndNode* node);
  /*!
   * \brief Visit prim::OrNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::OrNode* node);
  /*!
   * \brief Visit prim::NotNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::NotNode* node);
  /*!
   * \brief Visit prim::SelectNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::SelectNode* node);
  /*!
   * \brief Visit prim::LetNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::LetNode* node);
  /*!
   * \brief Visit prim::RampNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::RampNode* node);
  /*!
   * \brief Visit prim::BroadcastNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::BroadcastNode* node);
  /*!
   * \brief Visit prim::ShuffleNode using its structural fields.
   * \param node The borrowed expression node.
   * \return None on completion or an owning interrupt that halts traversal.
   */
  virtual ffi::Optional<VisitInterrupt> Visit_(const prim::ShuffleNode* node);

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
 *   virtual UnchangedOr<Expr> Mutate_(const MyExprNode* node, InplaceMode
 * inplace_mode);
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

  using ObjectMutator::Mutate;

  /*!
   * \brief Mutate primitive expression, preserving its expression category.
   * \param expr The borrowed primitive expression.
   * \param inplace_mode Inherited mutation permission.
   * \return Unchanged or an owning primitive expression replacement.
   * \note Calls the virtual AnyView entry. Overrides must preserve PrimExpr;
   *       the result storage is transferred without a runtime type check.
   */
  TVM_FFI_INLINE UnchangedOr<PrimExpr> Mutate(const PrimExpr& expr,
                                              InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<PrimExpr>(
        ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(expr), inplace_mode)));
  }

  /*!
   * \brief Mutate an expression, preserving its expression category.
   * \param expr The borrowed expression.
   * \param inplace_mode Inherited mutation permission.
   * \return Unchanged or an owning expression replacement.
   * \note Calls the virtual AnyView entry. Overrides must preserve Expr;
   *       the result storage is transferred without a runtime type check.
   */
  TVM_FFI_INLINE UnchangedOr<Expr> Mutate(const Expr& expr,
                                          InplaceMode inplace_mode = InplaceMode::kDisallow) {
    return ffi::details::UnchangedOrUnsafe::MoveFromTVMFFIAny<Expr>(
        ffi::details::UnchangedOrUnsafe::MoveToTVMFFIAny(Mutate(ffi::AnyView(expr), inplace_mode)));
  }

  // A downstream class overrides any existing hook without rebuilding the table.
  // Extra node types use a fresh inherited table and SetDispatch<Self, ExtraNode>.
  // Hooks borrow the node and return Unchanged or an owning replacement in its expression
  // category, throwing on failure. Narrower field types are checked separately. Forward
  // inplace_mode on every child edge.
  /*!
   * \brief Mutate OpaqueExprNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const OpaqueExprNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate TupleNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const TupleNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate TupleGetItemNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const TupleGetItemNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate TensorLoadNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate VarNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const VarNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate GlobalVarNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const GlobalVarNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate CallNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const CallNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate IntImmNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const IntImmNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate FloatImmNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const FloatImmNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate OpNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning expression replacement.
   */
  virtual UnchangedOr<Expr> Mutate_(const OpNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::StringImmNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::StringImmNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::CastNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::AddNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::AddNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::SubNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::SubNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::MulNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::MulNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::DivNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::DivNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::ModNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::ModNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::FloorDivNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::FloorModNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::MinNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::MinNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::MaxNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::EQNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::EQNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::NENode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::NENode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::LTNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LTNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::LENode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LENode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::GTNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::GTNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::GENode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::GENode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::AndNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::AndNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::OrNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::OrNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::NotNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::NotNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::SelectNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::LetNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::RampNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::RampNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::BroadcastNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* node, InplaceMode inplace_mode);
  /*!
   * \brief Mutate prim::ShuffleNode, preserving its expression category.
   * \param node The borrowed expression node.
   * \param inplace_mode Inherited permission to mutate unique nodes in place.
   * \return Unchanged or an owning primitive expression replacement.
   */
  virtual UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* node, InplaceMode inplace_mode);

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
