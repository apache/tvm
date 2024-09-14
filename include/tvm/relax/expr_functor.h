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

#include <tvm/node/functor.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/relay/op.h>
#include <tvm/tir/function.h>

#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace tvm {
namespace relax {

/*!
 * \brief A dynamical functor that dispatches on in the first Expr argument.
 *  You can use this as a more powerful Visitor, since it allows you to
 *  define function signatures of Visit Function.
 *
 * \sa tvm/ir_functor.h
 *
 * \tparam FType function signiture
 *  This type is only defined for FType with function signature R(const Expr&,
 * Args...)
 */
template <typename FType>
class ExprFunctor;

// functions to be overriden.
#define EXPR_FUNCTOR_DEFAULT \
  { return VisitExprDefault_(op, std::forward<Args>(args)...); }

#define RELAX_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

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
      RET_TYPE ret = PY_FUNC(N);                                    \
      return ret;                                                   \
    } else {                                                        \
      return DEFAULT_FUNC;                                          \
    }                                                               \
  }

#define PY_EXPR_VISITOR_DISPATCH(OP, PY_FUNC)                            \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) { \
    if (self->PY_FUNC != nullptr)                                        \
      self->PY_FUNC(n);                                                  \
    else                                                                 \
      self->VisitExpr_(static_cast<const OP*>(n.get()));                 \
  });

#define PY_EXPR_MUTATOR_DISPATCH(OP, PY_FUNC)                            \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) { \
    if (self->PY_FUNC != nullptr) {                                      \
      Expr expr = self->PY_FUNC(n);                                      \
      return expr;                                                       \
    } else {                                                             \
      return self->VisitExpr_(static_cast<const OP*>(n.get()));          \
    }                                                                    \
  });

#define PY_EXPR_MUTATOR_VISIT_EXPR_POST_ORDER_DISPATCH(OP)                          \
  post_order_vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self) { \
    return self->VisitExprPostOrder_(static_cast<const OP*>(n.get()));              \
  });

template <typename R, typename... Args>
class ExprFunctor<R(const Expr& n, Args...)> {
 private:
  using TSelf = ExprFunctor<R(const Expr& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

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
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
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
  virtual R VisitExpr_(const PrimValueNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const StringImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const DataTypeImmNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExprDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
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
    RELAX_EXPR_FUNCTOR_DISPATCH(PrimValueNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(StringImmNode);
    RELAX_EXPR_FUNCTOR_DISPATCH(DataTypeImmNode);
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
  void VisitExpr_(const PrimValueNode* op) override;
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
  virtual void VisitBinding_(const VarBindingNode* binding, const PrimValueNode* val);
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
   * \brief Visit struct_info may recursively contain Expr/PrimExpr.
   *
   * By default, this function recurse into struct info such as
   * TensorStructInfo and ShapeStructInfo and call VisitExpr/VisitPrimExpr
   * accordingly. It does not recurse into FunctionStructInfo as it does
   * not contain Expr defined in the current scope.
   *
   * Pass writers can overload this function to change to other behaviors.
   * For example, if we are not interested in Expr in StructInfo, we can
   * override this function by a no-op.
   *
   * \param struct_info Input struct info field.
   */
  virtual void VisitExprDepStructInfoField(const StructInfo& struct_info);

  // specific leaf level visitor functions
  virtual void VisitVarDef_(const VarNode* var);
  virtual void VisitVarDef_(const DataflowVarNode* var);

  virtual void VisitSpan(const Span& span);
  virtual void VisitPrimExpr(const PrimExpr& expr);

 private:
  using TSelf = ExprVisitor;
  using VisitBindingVTable =
      tvm::NodeFunctor<void(const ObjectRef& n, ExprVisitor* self, const VarBindingNode* binding)>;
  // initialize the vtable.
  static VisitBindingVTable InitVisitBindingVTable();
  /*!
   * \brief Private internal struct info field visitor.
   *
   *  Support default visiting of struct info field and recursive into
   *  their Expr fields.
   *
   *  We use component instead of sub-classing so there can be other
   *  joint inheritance between ExprVisitor and StructInfoVisitor.
   */
  class DefaultStructInfoFieldVisitor : public StructInfoVisitor {
   public:
    explicit DefaultStructInfoFieldVisitor(ExprVisitor* parent);

    // Override defaults in struct info visitor.
    void VisitStructInfoExprField(const Expr& expr) final;
    void VisitStructInfoExprField(const PrimExpr& expr) final;
    void VisitStructInfo_(const FuncStructInfoNode* op) final;

   private:
    ExprVisitor* parent_;
  };
  // This visitor is not visible to child classes and only
  // used to supported default visiting behavior.
  DefaultStructInfoFieldVisitor default_struct_info_field_visitor_{this};
};

void PostOrderVisit(const Expr& node, std::function<void(const Expr&)> fvisit);

/*!
 * \brief A mutator works in unnormalized form.
 *
 * ExprMutatorBase expects input AST to be in the unnormalized form, i.e., checked_type_ and shape_
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
  Expr VisitExpr_(const PrimValueNode* op) override;
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
   * \brief Visit struct_info that may recursively contain Expr/PrimExpr.
   *
   * By default, this function recurse into struct info such as
   * TensorStructInfo and ShapeStructInfo and call VisitExpr/VisitPrimExpr
   * accordingly. It does not recurse into FunctionStructInfo as it does
   * not contain Expr defined in the current scope.
   *
   * Pass writers can overload this function to change to other behaviors.
   * For example, if in Expr in StructInfo won't change, we can
   * override this function by an identity function.
   *
   * \param struct_info Input struct info field.
   * \return The updated struct info.
   */
  virtual StructInfo VisitExprDepStructInfoField(const StructInfo& struct_info);

 protected:
  /*!
   * \brief Check whether VisitExprDepStructInfoField change struct_info.
   * \return Whether struct info changed.
   * \note This function is used by mutator implementations to check if
   *       previous Expr update will trigger a change in struct_info.
   *       If change is detected, the implementation can generate a fresh
   *       node without struct_info, and trigger normalizer to re-derive.
   */
  bool VisitAndCheckStructInfoFieldUnchanged(const ObjectRef& struct_info) {
    if (const StructInfoNode* sinfo = struct_info.as<StructInfoNode>()) {
      return this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo)).same_as(struct_info);
    } else {
      return true;
    }
  }

 private:
  /*!
   * \brief Private internal struct info field visitor to support
   *  Default visiting of struct info field and recursive into their Expr fields.
   *
   *  We use component instead of sub-classing so there can be other
   *  joint inheritance between ExprMutator and StructInfoMutator.
   */
  class DefaultStructInfoFieldMutator : public StructInfoMutator {
   public:
    explicit DefaultStructInfoFieldMutator(ExprMutatorBase* parent);

    // Override defaults in struct info visitor.
    Expr VisitStructInfoExprField(const Expr& expr) final;
    PrimExpr VisitStructInfoExprField(const PrimExpr& expr) final;
    StructInfo VisitStructInfo_(const FuncStructInfoNode* op) final;

   private:
    ExprMutatorBase* parent_;
  };
  // This visitor is not visible to child classes and only
  // used to supported default visiting behavior.
  DefaultStructInfoFieldMutator default_struct_info_field_mutator_{this};
};

/*!
 * \brief A mutator works in normal form.
 *
 * ExprMutator expects input AST to be in the normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all checked_type_ and shape_ of expressions are
 * available.
 */
class ExprMutator : public ExprMutatorBase {
 public:
  using ExprMutatorBase::VisitExpr_;

  ExprMutator(Optional<IRModule> mod = NullOpt) { builder_ = BlockBuilder::Create(mod); }
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
  virtual void VisitBinding_(const VarBindingNode* binding, const PrimValueNode* val);
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
  Expr VisitWithNewScope(const Expr& body_expr, Optional<Array<Var>> params = NullOpt);

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
   * \note For function parameters, this function returns NullOpt.
   */
  Optional<Expr> LookupBinding(const Var& var);

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
   * \brief Create a new var with specified struct_info if the original var's shape or type does
   * not match with the specified ones.
   * \param var The var to be updated.
   * \param struct_info The struct info to be updated.
   * \return The var filled with struct_info
   */
  Var WithStructInfo(Var var, StructInfo struct_info);

  /*! \brief Internal block builder to emit bindings during rewriting. */
  BlockBuilder builder_;

  /*! \brief Remap a var to a new var in use-site. */
  std::unordered_map<Id, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;

 private:
  using TSelf = ExprMutator;
  using VisitBindingVTable =
      tvm::NodeFunctor<void(const ObjectRef& n, ExprMutator* self, const VarBindingNode* binding)>;
  // initialize the vtable.
  static VisitBindingVTable InitVisitBindingVTable();
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_EXPR_FUNCTOR_H_
