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
 * \file tvm/relax/expr_functor.h
 * \brief A more powerful visitor which enables defining arbitrary function
 * signatures with type based dispatch on first argument.
 */
#ifndef TVM_RELAX_EXPR_FUNCTOR_H_
#define TVM_RELAX_EXPR_FUNCTOR_H_

#include <tvm/arith/iter_affine_map.h>
#include <tvm/ir/node_functor.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>
#include <tvm/tirx/function.h>

#include <unordered_map>
#include <utility>
namespace tvm {
namespace relax {

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 * \sa tvm/ir_functor.h
 *
 * \tparam FType function signature
 *  This type is only defined for FType with function signature R(const Expr&,
 * Args...)
 */
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT                                   \
  {                                                            \
    return VisitExprDefault_(op, std::forward<Args>(args)...); \
  }

#define RELAX_EXPR_FUNCTOR_DISPATCH(OP)                                                     \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self, Args... args) { \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...);  \
  });

#define RELAX_PRIM_EXPR_NODE_DISPATCH_LIST(V) \
  V(::tvm::IntImmNode)                        \
  V(::tvm::FloatImmNode)                      \
  V(::tvm::tirx::VarNode)                     \
  V(::tvm::tirx::SizeVarNode)                 \
  V(::tvm::tirx::StringImmNode)               \
  V(::tvm::tirx::CastNode)                    \
  V(::tvm::tirx::AddNode)                     \
  V(::tvm::tirx::SubNode)                     \
  V(::tvm::tirx::MulNode)                     \
  V(::tvm::tirx::DivNode)                     \
  V(::tvm::tirx::ModNode)                     \
  V(::tvm::tirx::FloorDivNode)                \
  V(::tvm::tirx::FloorModNode)                \
  V(::tvm::tirx::MinNode)                     \
  V(::tvm::tirx::MaxNode)                     \
  V(::tvm::tirx::EQNode)                      \
  V(::tvm::tirx::NENode)                      \
  V(::tvm::tirx::LTNode)                      \
  V(::tvm::tirx::LENode)                      \
  V(::tvm::tirx::GTNode)                      \
  V(::tvm::tirx::GENode)                      \
  V(::tvm::tirx::AndNode)                     \
  V(::tvm::tirx::OrNode)                      \
  V(::tvm::tirx::NotNode)                     \
  V(::tvm::tirx::SelectNode)                  \
  V(::tvm::tirx::BufferLoadNode)              \
  V(::tvm::tirx::ProducerLoadNode)            \
  V(::tvm::tirx::RampNode)                    \
  V(::tvm::tirx::BroadcastNode)               \
  V(::tvm::tirx::LetNode)                     \
  V(::tvm::tirx::CallNode)                    \
  V(::tvm::tirx::ShuffleNode)                 \
  V(::tvm::tirx::ReduceNode)                  \
  V(::tvm::arith::IterSplitExprNode)          \
  V(::tvm::arith::IterSumExprNode)

#define PY_EXPR_VISITOR_DEFAULT(N, PY_FUNC, DEFAULT_FUNC) \
  {                                                       \
    if (PY_FUNC != nullptr)                               \
      PY_FUNC(N);                                         \
    else                                                  \
      DEFAULT_FUNC;                                       \
  }

#define PY_EXPR_MUTATOR_DEFAULT(N, PY_FUNC, DEFAULT_FUNC, RET_TYPE) \
  {                                                                 \
    if (PY_FUNC != nullptr) {                                       \
      RET_TYPE ret = PY_FUNC(N).cast<RET_TYPE>();                   \
      return ret;                                                   \
    } else {                                                        \
      return DEFAULT_FUNC;                                          \
    }                                                               \
  }

#define PY_EXPR_VISITOR_DISPATCH(OP, PY_FUNC)                                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    if (self->PY_FUNC != nullptr)                                             \
      self->PY_FUNC(n);                                                       \
    else                                                                      \
      self->VisitExpr_(static_cast<const OP*>(n.get()));                      \
  });

#define PY_EXPR_VISITOR_DISPATCH_PRIM_EXPR(OP) PY_EXPR_VISITOR_DISPATCH(OP, f_visit_prim_expr_)

#define PY_EXPR_MUTATOR_DISPATCH(OP, PY_FUNC)                                 \
  vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    if (self->PY_FUNC != nullptr) {                                           \
      Expr expr = self->PY_FUNC(n).cast<Expr>();                              \
      return expr;                                                            \
    } else {                                                                  \
      return self->VisitExpr_(static_cast<const OP*>(n.get()));               \
    }                                                                         \
  });

#define PY_EXPR_MUTATOR_DISPATCH_PRIM_EXPR(OP) PY_EXPR_MUTATOR_DISPATCH(OP, f_visit_prim_expr_)

#define PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(OP)                               \
  post_order_vtable.template set_dispatch<OP>([](const ffi::ObjectRef& n, TSelf* self) { \
    return self->VisitExprPostOrder_(static_cast<const OP*>(n.get()));                   \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ffi::ObjectRef& n, TSelf* self, Args...)>;

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
    TVM_FFI_ICHECK(n.defined())
        << "Found null pointer node while traversing AST. The previous pass may "
           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  // NOTE: cross dialect calls are invoked through global var
  // We do not expect inline PrimFunc to appear in relax IR.
  virtual R VisitExpr_(const ConstantNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const VarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DataflowVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ShapeExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ExternFuncNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GlobalVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const SeqExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IfNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleGetItemNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const PrimExprNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const StringImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DataTypeImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const ffi::Object* op, Args...) {
    TVM_FFI_THROW(InternalError) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAX_EXPR_FUNCTOR_DISPATCH(ConstantNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(TupleNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(VarNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(DataflowVarNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(ShapeExprNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(ExternFuncNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(GlobalVarNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(FunctionNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(CallNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(SeqExprNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(IfNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(OpNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(TupleGetItemNode);
    RELAX_PRIM_EXPR_NODE_DISPATCH_LIST(RELAX_EXPR_FUNCTOR_DISPATCH);
    RELAX_EXPR_FUNCTOR_DISPATCH(StringImmNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(DataTypeImmNode);
    vtable.Finalize();
    return vtable;
  }
};

/*!
 * \brief A simple visitor wrapper around ExprFunctor.
 *  Recursively visit the content.
 */
class ExprVisitor : public ExprFunctor<void(const Expr&)> {
 public:
  /*!
   * \brief Generic dispatcher for Expr.
   * \param expr The expr to be visited.
   */
  void VisitExpr(const Expr& expr) override;
  // specific leaf level visitor functions
  void VisitExpr_(const ConstantNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const DataflowVarNode* op) override;
  void VisitExpr_(const ShapeExprNode* op) override;
  void VisitExpr_(const ExternFuncNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const SeqExprNode* op) override;
  void VisitExpr_(const IfNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;
  void VisitExpr_(const PrimExprNode* op) override;
  void VisitExpr_(const StringImmNode* op) override;
  void VisitExpr_(const DataTypeImmNode* op) override;

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  virtual void VisitBinding(const Binding& binding);
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingNode* binding);
  virtual void VisitBinding_(const MatchCastNode* binding);
  // second level dispatching based on binding value type.
  // these dispatching functions get called from first-level dispatch on VarBinding
  virtual void VisitBinding_(const VarBindingNode* binding, const ConstantNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const TupleNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const VarNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const ShapeExprNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const ExternFuncNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const GlobalVarNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const CallNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const SeqExprNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const IfNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const OpNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const PrimExprNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const StringImmNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const DataTypeImmNode* val);
  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   */
  virtual void VisitBindingBlock(const BindingBlock& block);
  // specific leaf level visitor functions
  virtual void VisitBindingBlock_(const BindingBlockNode* block);
  virtual void VisitBindingBlock_(const DataflowBlockNode* block);

  /*!
   * \brief Generic dispatcher for visiting the var definition site.
   * \param var The var to be visited.
   * \note VisitExpr_(const VarNode*) will only visit the usage site of an Var
   */
  virtual void VisitVarDef(const Var& var);

  /*!
   * \brief Visit ty may recursively contain Expr/PrimExpr.
   *
   * By default, this function recurse into type such as
   * TensorType and ShapeType and call VisitExpr/VisitPrimExpr
   * accordingly. It does not recurse into FunctionType as it does
   * not contain Expr defined in the current scope.
   *
   * Pass writers can overload this function to change to other behaviors.
   * For example, if we are not interested in Expr in Type, we can
   * override this function by a no-op.
   *
   * \param ty Input type field.
   */
  virtual void VisitExprDepTypeField(const Type& ty);

  // specific leaf level visitor functions
  virtual void VisitVarDef_(const VarNode* var);
  virtual void VisitVarDef_(const DataflowVarNode* var);

  virtual void VisitSpan(const Span& span);
  virtual void VisitPrimExpr(const PrimExpr& expr);

 private:
  using TSelf = ExprVisitor;
  using VisitBindingVTable = tvm::NodeFunctor<void(const ffi::ObjectRef& n, ExprVisitor* self,
                                                   const VarBindingNode* binding)>;
  // initialize the vtable.
  static VisitBindingVTable InitVisitBindingVTable();
  /*!
   * \brief Private internal type field visitor.
   *
   *  Support default visiting of type field and recursive into
   *  their Expr fields.
   *
   *  We use component instead of sub-classing so there can be other
   *  joint inheritance between ExprVisitor and TypeVisitor.
   */
  class DefaultTypeFieldVisitor : public TypeVisitor {
   public:
    explicit DefaultTypeFieldVisitor(ExprVisitor* parent);

    // Override defaults in type visitor.
    void VisitTypeExprField(const Expr& expr) final;
    void VisitTypeExprField(const PrimExpr& expr) final;
    void VisitType_(const FuncTypeNode* op) final;

   private:
    ExprVisitor* parent_;
  };
  // This visitor is not visible to child classes and only
  // used to supported default visiting behavior.
  DefaultTypeFieldVisitor default_tyfield_visitor_{this};
};

void PostOrderVisit(const Expr& node, std::function<void(const Expr&)> fvisit);

/*!
 * \brief A mutator works in unnormalized form.
 *
 * ExprMutatorBase expects input AST to be in the unnormalized form, i.e., ty
 * of expressions can be nullptr, and the expressions may nest(and as a result the AST is not in
 * ANF).
 */

class ExprMutatorBase : public ExprFunctor<Expr(const Expr&)> {
 public:
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const DataflowVarNode* op) override;
  Expr VisitExpr_(const ShapeExprNode* op) override;
  Expr VisitExpr_(const ExternFuncNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* op) override;
  Expr VisitExpr_(const SeqExprNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;
  Expr VisitExpr_(const PrimExprNode* op) override;
  Expr VisitExpr_(const StringImmNode* op) override;
  Expr VisitExpr_(const DataTypeImmNode* op) override;

  /*!
   * \brief Mutate BindingBlock.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  virtual BindingBlock VisitBindingBlock(const BindingBlock& block);

  /*!
   * \brief Used to visit the PrimExpr inside of expressions.
   *
   * Can be overloaded to transform the shape expressions.
   */
  virtual PrimExpr VisitPrimExpr(const PrimExpr& expr);

  /*!
   * \brief Visit ty that may recursively contain Expr/PrimExpr.
   *
   * By default, this function recurse into type such as
   * TensorType and ShapeType and call VisitExpr/VisitPrimExpr
   * accordingly. It does not recurse into FunctionType as it does
   * not contain Expr defined in the current scope.
   *
   * Pass writers can overload this function to change to other behaviors.
   * For example, if in Expr in Type won't change, we can
   * override this function by an identity function.
   *
   * \param ty Input type field.
   * \return The updated type.
   */
  virtual Type VisitExprDepTypeField(const Type& ty);

 protected:
  /*!
   * \brief Check whether VisitExprDepTypeField change ty.
   * \return Whether type changed.
   * \note This function is used by mutator implementations to check if
   *       previous Expr update will trigger a change in ty.
   *       If change is detected, the implementation can generate a fresh
   *       node without ty, and trigger normalizer to re-derive.
   */
  bool VisitAndCheckTypeFieldUnchanged(const ffi::ObjectRef& ty) {
    if (const TypeNode* ty_node = ty.as<TypeNode>()) {
      return this->VisitExprDepTypeField(ffi::GetRef<Type>(ty_node)).same_as(ty);
    } else {
      return true;
    }
  }

 private:
  /*!
   * \brief Private internal type field visitor to support
   *  Default visiting of type field and recursive into their Expr fields.
   *
   *  We use component instead of sub-classing so there can be other
   *  joint inheritance between ExprMutator and TypeMutator.
   */
  class DefaultTypeFieldMutator : public TypeMutator {
   public:
    explicit DefaultTypeFieldMutator(ExprMutatorBase* parent);

    // Override defaults in type visitor.
    Expr VisitTypeExprField(const Expr& expr) final;
    PrimExpr VisitTypeExprField(const PrimExpr& expr) final;
    Type VisitType_(const FuncTypeNode* op) final;

   private:
    ExprMutatorBase* parent_;
  };
  // This visitor is not visible to child classes and only
  // used to supported default visiting behavior.
  DefaultTypeFieldMutator default_tyfield_mutator_{this};
};

/*!
 * \brief A mutator works in normal form.
 *
 * ExprMutator expects input AST to be in the normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all ty of expressions are
 * available.
 */
class ExprMutator : public ExprMutatorBase {
 public:
  using ExprMutatorBase::VisitExpr_;

  ExprMutator(ffi::Optional<IRModule> mod = std::nullopt) { builder_ = BlockBuilder::Create(mod); }
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const DataflowVarNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const SeqExprNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;

  /*!
   * \brief Generic dispatcher for bindings.
   * \param binding The binding to be visited.
   */
  virtual void VisitBinding(const Binding& binding);
  // specific leaf level visitor functions
  virtual void VisitBinding_(const VarBindingNode* binding);
  virtual void VisitBinding_(const MatchCastNode* binding);
  // second level dispatching based on binding value type.
  // these dispatching functions get called from first-level dispatch on VarBinding
  virtual void VisitBinding_(const VarBindingNode* binding, const ConstantNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const TupleNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const VarNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const DataflowVarNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const ShapeExprNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const ExternFuncNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const GlobalVarNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const FunctionNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const CallNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const SeqExprNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const IfNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const OpNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const PrimExprNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const StringImmNode* val);
  virtual void VisitBinding_(const VarBindingNode* binding, const DataTypeImmNode* val);
  /*!
   * \brief Generic dispatcher for binding blocks.
   * \param block The binding block to be visited.
   * \return The binding block after transformation.
   */
  virtual BindingBlock VisitBindingBlock(const BindingBlock& block) override;  // NOLINT(*)
  // specific leaf level visitor functions
  virtual BindingBlock VisitBindingBlock_(const BindingBlockNode* block);
  virtual BindingBlock VisitBindingBlock_(const DataflowBlockNode* block);

  /*!
   * \brief Generic dispatcher for rewriting the var definition site.
   * \param var The var to be visited.
   * \return The var after post-order rewritten.
   * \note VisitExpr_(const VarNode*) will only visit the usage site of an Var
   */
  virtual Var VisitVarDef(const Var& var);
  // specific leaf level visitor functions
  virtual Var VisitVarDef_(const VarNode* var);
  virtual Var VisitVarDef_(const DataflowVarNode* var);

 protected:
  /*!
   * \brief Try to remit binding and bind it to a new_value
   *
   * This function is called after VisitExpr(binding->value) in
   * VisitBinding_(const VarBinding*).
   * It will try to reuse the current binding when the new value's shape/type
   * matches the original binding and no changes in var is needed.
   *
   * Otherwise, a new binding will be emitted to replace the var specified in
   * the current binding.
   */
  void ReEmitBinding(const VarBindingNode* binding, Expr new_value);

  /*!
   * \brief Rewrite the expr with a new scope, used in a Function's body.
   *
   * Visit an expression that may neither access variables from the
   * current scope, nor may export definitions into the current scope.
   *
   * \param body_expr The body to be visited.
   * \param params Optional parameters that are visible within the scope.
   * \return The expr after visiting.
   *
   * \note The body_expr must be an SeqExpr in the normal form.
   */
  Expr VisitWithNewScope(const Expr& body_expr,
                         ffi::Optional<ffi::Array<Var>> params = std::nullopt);

  /*!
   * \brief Rewrite the expr with a new scope, used in the branches of If.
   *
   * Visit an expression that may access variables from the current
   * scope, but may not export definitions into the current scope.
   *
   * \param body_expr The body to be visited.
   *
   * \return The expr after visiting.
   *
   * \sa VisitWithNewScope
   *
   * \note The body_expr must be an SeqExpr in the normal form.
   */
  Expr VisitWithInnerScope(const Expr& body_expr);

  /*!
   * \brief Look up the value bound to a variable.
   * \param var The var to be looked up.
   * \return The value bound to the input \p var.
   * \note For function parameters, this function returns std::nullopt.
   */
  ffi::Optional<Expr> LookupBinding(const Var& var);

  /*!
   * \brief Post-order rewrite a node and normalize.
   * \tparam T The node type to be rewritten.
   * \param op The node to be rewritten.
   * \return The node after post rewritten.
   */
  template <typename T>
  Expr VisitExprPostOrder_(const T* op) {
    return builder_->Normalize(ExprMutator::VisitExpr_(op));
  }

  /*!
   * \brief Create a new var with specified type if the original var's shape or type does not
   * match with the specified ones.
   * \param var The var to be updated.
   * \param ty The type to be updated.
   * \return The var filled with type information.
   */
  Var WithType(Var var, Type ty);

  /*! \brief Internal block builder to emit bindings during rewriting. */
  BlockBuilder builder_;

  /*! \brief Remap a var to a new var in use-site. */
  std::unordered_map<Id, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> var_remap_;

 private:
  using TSelf = ExprMutator;
  using VisitBindingVTable = tvm::NodeFunctor<void(const ffi::ObjectRef& n, ExprMutator* self,
                                                   const VarBindingNode* binding)>;
  // initialize the vtable.
  static VisitBindingVTable InitVisitBindingVTable();
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_EXPR_FUNCTOR_H_
