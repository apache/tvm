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
 * \file src/relax/expr_functor.cc
 * \brief A wrapper around ExprFunctor which functionally updates the AST.
 *
 * ExprMutator uses memoization and self return in order to amortize
 * the cost of using functional updates.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/op.h>

// functions to be overriden.
#define RELAX_VISIT_BINDING_DISPATCH(OP)                                        \
  vtable.template set_dispatch<OP>(                                             \
      [](const ffi::ObjectRef& n, TSelf* self, const VarBindingNode* binding) { \
        self->VisitBinding_(binding, static_cast<const OP*>(n.get()));          \
      });

#define RELAX_VAR_BINDING_DISPATCH_IMPL(Type)                                           \
  Type::VisitBindingVTable Type::InitVisitBindingVTable() {                             \
    VisitBindingVTable vtable;                                                          \
    RELAX_VISIT_BINDING_DISPATCH(ConstantNode);                                         \
    RELAX_VISIT_BINDING_DISPATCH(TupleNode);                                            \
    RELAX_VISIT_BINDING_DISPATCH(VarNode);                                              \
    RELAX_VISIT_BINDING_DISPATCH(DataflowVarNode);                                      \
    RELAX_VISIT_BINDING_DISPATCH(ShapeExprNode);                                        \
    RELAX_VISIT_BINDING_DISPATCH(ExternFuncNode);                                       \
    RELAX_VISIT_BINDING_DISPATCH(GlobalVarNode);                                        \
    RELAX_VISIT_BINDING_DISPATCH(FunctionNode);                                         \
    RELAX_VISIT_BINDING_DISPATCH(CallNode);                                             \
    RELAX_VISIT_BINDING_DISPATCH(SeqExprNode);                                          \
    RELAX_VISIT_BINDING_DISPATCH(IfNode);                                               \
    RELAX_VISIT_BINDING_DISPATCH(OpNode);                                               \
    RELAX_VISIT_BINDING_DISPATCH(TupleGetItemNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(PrimValueNode);                                        \
    RELAX_VISIT_BINDING_DISPATCH(StringImmNode);                                        \
    RELAX_VISIT_BINDING_DISPATCH(DataTypeImmNode);                                      \
    return vtable;                                                                      \
  }                                                                                     \
  void Type::VisitBinding_(const VarBindingNode* binding) {                             \
    static VisitBindingVTable vtable = InitVisitBindingVTable();                        \
    const Expr& value = binding->value;                                                 \
    TVM_FFI_ICHECK(value.defined()) << "Found null pointer node while traversing AST."; \
    TVM_FFI_ICHECK(vtable.can_dispatch(value))                                          \
        << "VisitVarBinding do not allow binding value type" << value->GetTypeKey();    \
    vtable(value, this, binding);                                                       \
  }

// functions to be overriden.
#define RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(OP)                                   \
  void ExprVisitor::VisitBinding_(const VarBindingNode* binding, const OP* value) { \
    this->VisitExpr(binding->value);                                                \
    this->VisitVarDef(binding->var);                                                \
  }

// functions to be overriden.
#define RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(OP)                                   \
  void ExprMutator::VisitBinding_(const VarBindingNode* binding, const OP* value) { \
    Expr new_value = this->VisitExpr(binding->value);                               \
    this->ReEmitBinding(binding, new_value);                                        \
  }

namespace tvm {
namespace relax {

// ==================
// ExprVisitor

void ExprVisitor::VisitExprDepTypeField(const Type& ty) {
  // recurse into type in case they depend on value
  // under the current scope.
  default_tyfield_visitor_.VisitType(ty);
}

ExprVisitor::DefaultTypeFieldVisitor::DefaultTypeFieldVisitor(ExprVisitor* parent)
    : parent_(parent) {}

void ExprVisitor::DefaultTypeFieldVisitor::VisitTypeExprField(const Expr& expr) {
  parent_->VisitExpr(expr);
}

void ExprVisitor::DefaultTypeFieldVisitor::VisitTypeExprField(const PrimExpr& expr) {
  parent_->VisitPrimExpr(expr);
}

void ExprVisitor::DefaultTypeFieldVisitor::VisitType_(const FuncTypeNode* op) {
  // Do not recurse into function type
  // as they won't contain ref to values in current scope.
}

void ExprVisitor::VisitExpr(const Expr& expr) { ExprFunctor::VisitExpr(expr); }

void ExprVisitor::VisitExpr_(const ConstantNode* op) {
  this->VisitSpan(op->span);
  // Constant's Type does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) {
  this->VisitSpan(op->span);
  // FuncType is not value-dep
}

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (Expr field : op->fields) {
    this->VisitExpr(field);
  }
  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

// Visit the use-site of a defined Var
void ExprVisitor::VisitExpr_(const VarNode* op) {
  this->VisitSpan(op->span);
  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

// Visit the use-site of a defined DataflowVar
void ExprVisitor::VisitExpr_(const DataflowVarNode* op) {
  VisitExpr_(static_cast<const VarNode*>(op));
}

void ExprVisitor::VisitExpr_(const FunctionNode* op) {
  this->VisitSpan(op->span);
  for (Var param : op->params) {
    this->VisitVarDef(param);
  }

  this->VisitExpr(op->body);
  // FuncType does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (Type ty_arg : op->ty_args) {
    this->VisitExprDepTypeField(ty_arg);
  }

  for (Expr arg : op->args) {
    this->VisitExpr(arg);
  }

  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);

  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

void ExprVisitor::VisitExpr_(const OpNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->tuple);

  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

void ExprVisitor::VisitExpr_(const ShapeExprNode* op) {
  for (PrimExpr val : op->values) {
    this->VisitPrimExpr(val);
  }
  this->VisitSpan(op->span);

  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

void ExprVisitor::VisitExpr_(const ExternFuncNode* op) {
  this->VisitSpan(op->span);
  // FuncType does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const SeqExprNode* op) {
  this->VisitSpan(op->span);
  for (BindingBlock block : op->blocks) {
    this->VisitBindingBlock(block);
  }
  this->VisitExpr(op->body);

  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
}

void ExprVisitor::VisitExpr_(const PrimValueNode* op) {
  this->VisitPrimExpr(op->value);
  if (auto* ty = op->ty.as<DependentTypeNode>()) {
    this->VisitExprDepTypeField(ffi::GetRef<Type>(ty));
  }
  this->VisitSpan(op->span);
}

void ExprVisitor::VisitExpr_(const StringImmNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const DataTypeImmNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitSpan(const Span& span) {}

void ExprVisitor::VisitPrimExpr(const PrimExpr& expr) {}

// implementations of binding visitor dispatch
RELAX_VAR_BINDING_DISPATCH_IMPL(ExprVisitor);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ConstantNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(TupleNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(VarNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(DataflowVarNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ShapeExprNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(ExternFuncNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(GlobalVarNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(FunctionNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(CallNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(SeqExprNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(IfNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(OpNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(TupleGetItemNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(PrimValueNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(StringImmNode);
RELAX_EXPR_VISITOR_VISIT_BINDING_IMPL(DataTypeImmNode);

void ExprVisitor::VisitBinding_(const MatchCastNode* binding) {
  this->VisitExpr(binding->value);
  this->VisitExprDepTypeField(binding->ty);
  this->VisitVarDef(binding->var);
}

void ExprVisitor::VisitBindingBlock_(const BindingBlockNode* block) {
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitBindingBlock_(const DataflowBlockNode* block) {
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
}

void ExprVisitor::VisitVarDef_(const DataflowVarNode* var) {
  VisitVarDef_(static_cast<const VarNode*>(var));
}

void ExprVisitor::VisitVarDef_(const VarNode* var) { this->VisitSpan(var->span); }

void ExprVisitor::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastNode>()) {
    VisitBinding_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
  }
}

void ExprVisitor::VisitBindingBlock(const BindingBlock& block) {
  if (const auto* node = block.as<DataflowBlockNode>()) {
    VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockNode>()) {
    VisitBindingBlock_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
  }
}

void ExprVisitor::VisitVarDef(const Var& var) {
  if (const auto* node = var.as<DataflowVarNode>()) {
    VisitVarDef_(node);
  } else if (const auto* node = var.as<VarNode>()) {
    VisitVarDef_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << var->GetTypeKey();
  }
}

class ExprApplyVisit : public ExprVisitor {
 public:
  explicit ExprApplyVisit(std::function<void(const Expr&)> f) : f_(f) {}

  void VisitExpr(const Expr& e) final {
    ExprVisitor::VisitExpr(e);
    f_(e);
  }

 private:
  std::function<void(const Expr&)> f_;
};

void PostOrderVisit(const Expr& e, std::function<void(const Expr&)> fvisit) {
  ExprApplyVisit(fvisit).VisitExpr(e);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.post_order_visit", [](Expr expr, ffi::Function f) {
    PostOrderVisit(expr, [f](const Expr& n) { f(n); });
  });
}

// ==================
// ExprMutatorBase

Type ExprMutatorBase::VisitExprDepTypeField(const Type& ty) {
  // recurse into type in case they depend on value
  // under the current scope.
  return default_tyfield_mutator_.VisitType(ty);
}

ExprMutatorBase::DefaultTypeFieldMutator::DefaultTypeFieldMutator(ExprMutatorBase* parent)
    : parent_(parent) {}

Expr ExprMutatorBase::DefaultTypeFieldMutator::VisitTypeExprField(const Expr& expr) {
  return parent_->VisitExpr(expr);
}

PrimExpr ExprMutatorBase::DefaultTypeFieldMutator::VisitTypeExprField(const PrimExpr& expr) {
  return parent_->VisitPrimExpr(expr);
}

Type ExprMutatorBase::DefaultTypeFieldMutator::VisitType_(const FuncTypeNode* op) {
  // Do not recurse into function type
  // as they won't contain ref to values in current scope.
  return ffi::GetRef<Type>(op);
}

Expr ExprMutatorBase::VisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

Expr ExprMutatorBase::VisitExpr_(const ConstantNode* op) {
  // Constant' type won't be affected by Expr/PrimExpr change.
  return ffi::GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const GlobalVarNode* op) {
  // FuncType won't be affected by Expr/PrimExpr change.
  return ffi::GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const TupleNode* op) {
  bool unchanged = true;
  tvm::ffi::Array<Expr> fields;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    unchanged &= new_field.same_as(field);
  }

  if (unchanged) {
    // If tuple's type change it means that
    // one of its fields' type will change
    // so un-changed already implies that type won't change
    return ffi::GetRef<Expr>(op);
  } else {
    // when there is a change return a new tuple node
    return Tuple(fields, op->span);
  }
}

// Visit the use-site of a defined Var
Expr ExprMutatorBase::VisitExpr_(const VarNode* op) {
  // type of var-use should remain stable
  // or the var itself will get replaced
  return ffi::GetRef<Expr>(op);
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutatorBase::VisitExpr_(const DataflowVarNode* op) {
  return VisitExpr_(static_cast<const VarNode*>(op));
}

Expr ExprMutatorBase::VisitExpr_(const FunctionNode* op) {
  // type of function is not value dependent
  // so no need to check ty field
  Expr body = this->VisitExpr(op->body);

  if (body.same_as(op->body)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return Function(op->params, body, op->ret_ty, op->is_pure, op->attrs);
  }
}

Expr ExprMutatorBase::VisitExpr_(const CallNode* call_node) {
  Expr new_op = this->VisitExpr(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  ffi::Array<Type> ty_args;
  for (Type ty_arg : call_node->ty_args) {
    Type new_ty_arg = this->VisitExprDepTypeField(ty_arg);
    ty_args.push_back(new_ty_arg);
    unchanged &= new_ty_arg.same_as(ty_arg);
  }

  tvm::ffi::Array<Expr> call_args;
  for (Expr arg : call_node->args) {
    Expr new_arg = this->VisitExpr(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  if (unchanged && VisitAndCheckTypeFieldUnchanged(call_node->ty)) {
    return ffi::GetRef<Expr>(call_node);
  } else {
    return Call(new_op, call_args, call_node->attrs, ty_args, call_node->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const IfNode* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitExpr(op->true_branch);
  Expr false_b = this->VisitExpr(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const OpNode* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const TupleGetItemNode* op) {
  auto t = this->VisitExpr(op->tuple);
  if (op->tuple.same_as(t)) {
    // type can be deterministically derived by tuple and index
    // if t does not change, then type won't change.
    return ffi::GetRef<Expr>(op);
  } else {
    return TupleGetItem(t, op->index, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const PrimValueNode* op) {
  auto value = this->VisitPrimExpr(op->value);
  if (op->value.same_as(value)) {
    // type can be deterministically derived by value
    // if value does not change, then type won't change.
    return ffi::GetRef<Expr>(op);
  }
  return PrimValue(value, op->span);
}

Expr ExprMutatorBase::VisitExpr_(const StringImmNode* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const DataTypeImmNode* op) { return ffi::GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const ShapeExprNode* op) {
  auto values = op->values.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

  if (values.same_as(op->values)) {
    // If values does not change, type won't change.
    return ffi::GetRef<Expr>(op);
  } else {
    return ShapeExpr(values, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const ExternFuncNode* op) {
  // Type of function remains value independent.
  return ffi::GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  ffi::Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  Expr body = this->VisitExpr(op->body);

  if (all_blocks_unchanged && body.same_as(op->body) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  }
  return SeqExpr(blocks, body);
}

BindingBlock ExprMutatorBase::VisitBindingBlock(const BindingBlock& block) {
  ffi::Array<Binding> bindings;
  if (const auto* node = block.as<BindingBlockNode>()) {
    for (auto binding : node->bindings) {
      if (auto var_binding = binding.as<VarBindingNode>()) {
        Expr new_value = this->VisitExpr(var_binding->value);
        bindings.push_back(VarBinding(var_binding->var, new_value));
      } else if (auto match_cast = binding.as<MatchCastNode>()) {
        Expr new_value = this->VisitExpr(match_cast->value);
        bindings.push_back(MatchCast(match_cast->var, new_value, match_cast->ty));
      } else {
        TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
      }
    }
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
  }

  if (block.as<DataflowBlockNode>()) {
    return DataflowBlock(bindings);
  } else {
    return BindingBlock(bindings);
  }
}

PrimExpr ExprMutatorBase::VisitPrimExpr(const PrimExpr& expr) { return expr; }

// ==================
// ExprMutator

Expr ExprMutator::VisitExpr(const Expr& expr) {
  return builder_->Normalize(ExprFunctor::VisitExpr(expr));
}

// Visit the use-site of a defined Var
Expr ExprMutator::VisitExpr_(const VarNode* op) {
  auto it = var_remap_.find(op->vid);
  if (it != var_remap_.end()) {
    return it->second;
  }

  // default case return self.
  return ffi::GetRef<Expr>(op);
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutator::VisitExpr_(const DataflowVarNode* op) {
  return VisitExpr_(static_cast<const VarNode*>(op));
}

Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::ffi::Array<Var> params;
  bool all_params_unchanged = true;
  for (Var param : op->params) {
    Var new_param = this->VisitVarDef(param);
    params.push_back(new_param);
    if (!param.same_as(new_param)) {
      var_remap_[param->vid] = new_param;
      all_params_unchanged = false;
    }
  }

  Expr body = this->VisitWithNewScope(op->body, params);

  if (all_params_unchanged && body.same_as(op->body)) {
    // No changes to the function, return the original object
    return ffi::GetRef<Expr>(op);
  } else if (IsBaseOf(GetType(body), op->ret_ty)) {
    // If the function was mutated into a form that can no longer
    // propagate shape information all the way to the return value, we
    // may keep the return type.  This is only allowed when the
    // body produces a return value that is the same as, or more
    // specific than, the pre-mutation type.  For example, if
    // the previous return value was `TensorType(shape=[16,16])`
    // but the body only produced `TensorType(ndim=2)`, we can
    // keep the more specific information.
    return Function(params, body, op->ret_ty, op->is_pure, op->attrs);
  } else {
    // If the function was mutated such that the body produces an
    // output that is incompatible with the original return struct
    // info, the original return type should not be used.  For
    // example, if the previous return value was
    // `TensorType(shape=[16,16])`, but the new return value is
    // `TensorType(shape=[8,8])`.
    return Function(params, body, std::nullopt, op->is_pure, op->attrs);
  }
}

Expr ExprMutator::VisitExpr_(const IfNode* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitWithInnerScope(op->true_branch);
  Expr false_b = this->VisitWithInnerScope(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutator::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  ffi::Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  builder_->BeginBindingBlock();
  Expr body = this->VisitExpr(op->body);
  BindingBlock prologue = builder_->EndBlock();
  if (!prologue->bindings.empty()) {
    blocks.push_back(prologue);
    all_blocks_unchanged = false;
  }

  if (all_blocks_unchanged && body.same_as(op->body) && VisitAndCheckTypeFieldUnchanged(op->ty)) {
    return ffi::GetRef<Expr>(op);
  } else {
    return SeqExpr(blocks, body);
  }
}

RELAX_VAR_BINDING_DISPATCH_IMPL(ExprMutator);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ConstantNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(TupleNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(VarNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(DataflowVarNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ShapeExprNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(ExternFuncNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(GlobalVarNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(FunctionNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(CallNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(SeqExprNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(IfNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(OpNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(TupleGetItemNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(PrimValueNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(StringImmNode);
RELAX_EXPR_MUTATOR_VISIT_BINDING_IMPL(DataTypeImmNode);

void ExprMutator::ReEmitBinding(const VarBindingNode* binding, Expr new_value) {
  Var new_var = this->VisitVarDef(binding->var);

  // fast path: re-emit binding if nothing changes
  if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
    builder_->EmitNormalized(ffi::GetRef<VarBinding>(binding));
    return;
  }

  auto new_ty = new_value->ty.as<Type>();

  TVM_FFI_CHECK(new_ty, InternalError)
      << "In binding of variable " << binding->var << ", the value " << new_value
      << " does not have Type.  "
      << "This typically occurs when ReEmitBinding is called without first calling Normalize.";

  Var temp = WithType(new_var, new_ty.value());
  if (!temp.same_as(new_var)) {
    new_var = temp;
  }

  this->var_remap_[binding->var->vid] = new_var;
  this->var_remap_[new_var->vid] = new_var;

  builder_->EmitNormalized(VarBinding(new_var, new_value));
}

void ExprMutator::VisitBinding_(const MatchCastNode* binding) {
  Expr new_value = this->VisitExpr(binding->value);
  Type new_ty = this->VisitExprDepTypeField(binding->ty);

  Var new_var = this->VisitVarDef(binding->var);

  MatchCast new_binding = [&]() -> MatchCast {
    if (new_var.same_as(binding->var) && new_value.same_as(binding->value) &&
        new_ty.same_as(binding->ty)) {
      // re-emit old binding if nothing changes
      return ffi::GetRef<MatchCast>(binding);
    } else {
      new_value = builder_->NormalizeArgument(new_value);
      new_var = WithType(new_var, new_ty);

      var_remap_[binding->var->vid] = new_var;
      var_remap_[new_var->vid] = new_var;

      return MatchCast(new_var, new_value, new_ty, binding->span);
    }
  }();

  builder_->EmitNormalized(new_binding);
  builder_->AddDefinitionToScope(new_binding->var);
}

BindingBlock ExprMutator::VisitBindingBlock_(const BindingBlockNode* block) {
  builder_->BeginBindingBlock();
  for (Binding binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

BindingBlock ExprMutator::VisitBindingBlock_(const DataflowBlockNode* block) {
  builder_->BeginDataflowBlock();
  for (auto binding : block->bindings) {
    this->VisitBinding(binding);
  }
  return builder_->EndBlock();
}

Var ExprMutator::VisitVarDef_(const DataflowVarNode* var) {
  Var output = VisitVarDef_(static_cast<const VarNode*>(var));
  // Because we delegate from DataflowVar visitor to Var visitor to
  // provide default behavior in subclasses, we may produce a Var
  // where we should produce a DataflowVar.
  if (!output->IsInstance<DataflowVarNode>()) {
    output = DataflowVar(output->vid, GetType(output), output->span);
  }
  return output;
}

Var ExprMutator::VisitVarDef_(const VarNode* var) {
  if (auto* ty_node = var->ty.as<DependentTypeNode>()) {
    Type ty = this->VisitExprDepTypeField(ffi::GetRef<Type>(ty_node));
    if (ty.same_as(var->ty)) {
      return ffi::GetRef<Var>(var);
    } else {
      return Var(var->vid, ty, var->span);
    }
  } else {
    return ffi::GetRef<Var>(var);
  }
}

void ExprMutator::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastNode>()) {
    VisitBinding_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
  }
}

BindingBlock ExprMutator::VisitBindingBlock(const BindingBlock& block) {
  BindingBlock ret;
  if (const auto* node = block.as<DataflowBlockNode>()) {
    ret = VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockNode>()) {
    ret = VisitBindingBlock_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
  }
  return ret;
}

Var ExprMutator::VisitVarDef(const Var& var) {
  Var ret;
  if (const auto* node = var.as<DataflowVarNode>()) {
    ret = VisitVarDef_(node);
  } else if (const auto* node = var.as<VarNode>()) {
    ret = VisitVarDef_(node);
  } else {
    TVM_FFI_THROW(TypeError) << "Invalid type: " << var->GetTypeKey();
  }
  return ret;
}

Expr ExprMutator::VisitWithNewScope(const Expr& expr, ffi::Optional<ffi::Array<Var>> params) {
  TVM_FFI_ICHECK(expr->IsInstance<SeqExprNode>())
      << "Normal form requires all new scope is stored as SeqExpr";

  PrimExpr constraint = IntImm::Bool(true);
  if (params.defined()) {
    auto non_negative_expressions =
        CollectNonNegativeExpressions(TupleType(params.value().Map(GetType)));
    for (const auto& expr : non_negative_expressions) {
      constraint = constraint && (expr >= 0);
    }
  }

  builder_->BeginScope(params);
  // Outer scope only includes TIR variables that can be inferred from
  // the function parameters.
  With<arith::ConstraintContext> context(builder_->GetAnalyzer(), constraint);
  builder_->BeginInnerScope();
  // Inner scope also includes any TIR variables that are defined by
  // MatchCast nodes, and are internal to the scope.
  Expr ret = this->VisitExpr(expr);

  builder_->EndScope();

  // Normalization (and the resulting Type inference) of the
  // expr occurs outside of the body's parameters, but inside the
  // function signature's scope.  This keeps variables that are
  // inferable based on the function signature, to allow callers to
  // propagate Type across the function.
  ret = builder_->Normalize(ret);
  builder_->EndScope();
  return ret;
}

Expr ExprMutator::VisitWithInnerScope(const Expr& expr) {
  TVM_FFI_ICHECK(expr->IsInstance<SeqExprNode>())
      << "Normal form requires all new scope is stored as SeqExpr";

  builder_->BeginInnerScope();
  Expr ret = this->VisitExpr(expr);
  builder_->EndScope();
  return ret;
}

ffi::Optional<Expr> ExprMutator::LookupBinding(const Var& var) {
  return builder_->LookupBinding(var);
}

Var ExprMutator::WithType(Var var, Type ty) {
  TVM_FFI_ICHECK(ty.defined());

  // TODO(relax-team) add TypeEqual check
  if (var->ty.defined()) {
    // use same-as as a quick path
    if (var->ty.same_as(ty) || ffi::StructuralEqual()(var->ty, ty)) {
      return var;
    } else {
      Var new_var = var.as<DataflowVarNode>() ? DataflowVar(var->vid, ty, var->span)
                                              : Var(var->vid, ty, var->span);
      return new_var;
    }
  } else {
    UpdateType(var, ty);
    return var;
  }
}

}  // namespace relax
}  // namespace tvm
