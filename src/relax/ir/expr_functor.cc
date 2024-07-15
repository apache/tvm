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
#include <tvm/ir/type_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>

// functions to be overriden.
#define RELAX_VISIT_BINDING_DISPATCH(OP)                                   \
  vtable.template set_dispatch<OP>(                                        \
      [](const ObjectRef& n, TSelf* self, const VarBindingNode* binding) { \
        self->VisitBinding_(binding, static_cast<const OP*>(n.get()));     \
      });

#define RELAX_VAR_BINDING_DISPATCH_IMPL(Type)                                        \
  Type::VisitBindingVTable Type::InitVisitBindingVTable() {                          \
    VisitBindingVTable vtable;                                                       \
    RELAX_VISIT_BINDING_DISPATCH(ConstantNode);                                      \
    RELAX_VISIT_BINDING_DISPATCH(TupleNode);                                         \
    RELAX_VISIT_BINDING_DISPATCH(VarNode);                                           \
    RELAX_VISIT_BINDING_DISPATCH(DataflowVarNode);                                   \
    RELAX_VISIT_BINDING_DISPATCH(ShapeExprNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(ExternFuncNode);                                    \
    RELAX_VISIT_BINDING_DISPATCH(GlobalVarNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(FunctionNode);                                      \
    RELAX_VISIT_BINDING_DISPATCH(CallNode);                                          \
    RELAX_VISIT_BINDING_DISPATCH(SeqExprNode);                                       \
    RELAX_VISIT_BINDING_DISPATCH(IfNode);                                            \
    RELAX_VISIT_BINDING_DISPATCH(OpNode);                                            \
    RELAX_VISIT_BINDING_DISPATCH(TupleGetItemNode);                                  \
    RELAX_VISIT_BINDING_DISPATCH(PrimValueNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(StringImmNode);                                     \
    RELAX_VISIT_BINDING_DISPATCH(DataTypeImmNode);                                   \
    return vtable;                                                                   \
  }                                                                                  \
  void Type::VisitBinding_(const VarBindingNode* binding) {                          \
    static VisitBindingVTable vtable = InitVisitBindingVTable();                     \
    const Expr& value = binding->value;                                              \
    ICHECK(value.defined()) << "Found null pointer node while traversing AST.";      \
    ICHECK(vtable.can_dispatch(value))                                               \
        << "VisitVarBinding do not allow binding value type" << value->GetTypeKey(); \
    vtable(value, this, binding);                                                    \
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

void ExprVisitor::VisitExprDepStructInfoField(const StructInfo& struct_info) {
  // recurse into struct info in case they depend on value
  // under the current scope.
  default_struct_info_field_visitor_.VisitStructInfo(struct_info);
}

ExprVisitor::DefaultStructInfoFieldVisitor::DefaultStructInfoFieldVisitor(ExprVisitor* parent)
    : parent_(parent) {}

void ExprVisitor::DefaultStructInfoFieldVisitor::VisitStructInfoExprField(const Expr& expr) {
  parent_->VisitExpr(expr);
}

void ExprVisitor::DefaultStructInfoFieldVisitor::VisitStructInfoExprField(const PrimExpr& expr) {
  parent_->VisitPrimExpr(expr);
}

void ExprVisitor::DefaultStructInfoFieldVisitor::VisitStructInfo_(const FuncStructInfoNode* op) {
  // Do not recurse into function struct info
  // as they won't contain ref to values in current scope.
}

void ExprVisitor::VisitExpr(const Expr& expr) { ExprFunctor::VisitExpr(expr); }

void ExprVisitor::VisitExpr_(const ConstantNode* op) {
  this->VisitSpan(op->span);
  // Constant's StructInfo does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const GlobalVarNode* op) {
  this->VisitSpan(op->span);
  // FuncStructInfo is not value-dep
}

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  this->VisitSpan(op->span);
  for (Expr field : op->fields) {
    this->VisitExpr(field);
  }
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

// Visit the use-site of a defined Var
void ExprVisitor::VisitExpr_(const VarNode* op) {
  this->VisitSpan(op->span);
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
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
  // FuncStructInfo does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->op);

  for (StructInfo sinfo_arg : op->sinfo_args) {
    this->VisitExprDepStructInfoField(sinfo_arg);
  }

  for (Expr arg : op->args) {
    this->VisitExpr(arg);
  }

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const OpNode* op) { this->VisitSpan(op->span); }

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->tuple);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const ShapeExprNode* op) {
  for (PrimExpr val : op->values) {
    this->VisitPrimExpr(val);
  }
  this->VisitSpan(op->span);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const ExternFuncNode* op) {
  this->VisitSpan(op->span);
  // FuncStructInfo does not depend on Expr.
}

void ExprVisitor::VisitExpr_(const SeqExprNode* op) {
  this->VisitSpan(op->span);
  for (BindingBlock block : op->blocks) {
    this->VisitBindingBlock(block);
  }
  this->VisitExpr(op->body);

  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
}

void ExprVisitor::VisitExpr_(const PrimValueNode* op) {
  this->VisitPrimExpr(op->value);
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
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
  this->VisitExprDepStructInfoField(binding->struct_info);
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
    LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
  }
}

void ExprVisitor::VisitBindingBlock(const BindingBlock& block) {
  if (const auto* node = block.as<DataflowBlockNode>()) {
    VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockNode>()) {
    VisitBindingBlock_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
  }
}

void ExprVisitor::VisitVarDef(const Var& var) {
  if (const auto* node = var.as<DataflowVarNode>()) {
    VisitVarDef_(node);
  } else if (const auto* node = var.as<VarNode>()) {
    VisitVarDef_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << var->GetTypeKey();
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

TVM_REGISTER_GLOBAL("relax.analysis.post_order_visit").set_body_typed([](Expr expr, PackedFunc f) {
  PostOrderVisit(expr, [f](const Expr& n) { f(n); });
});

// ==================
// ExprMutatorBase

StructInfo ExprMutatorBase::VisitExprDepStructInfoField(const StructInfo& struct_info) {
  // recurse into struct info in case they depend on value
  // under the current scope.
  return default_struct_info_field_mutator_.VisitStructInfo(struct_info);
}

ExprMutatorBase::DefaultStructInfoFieldMutator::DefaultStructInfoFieldMutator(
    ExprMutatorBase* parent)
    : parent_(parent) {}

Expr ExprMutatorBase::DefaultStructInfoFieldMutator::VisitStructInfoExprField(const Expr& expr) {
  return parent_->VisitExpr(expr);
}

PrimExpr ExprMutatorBase::DefaultStructInfoFieldMutator::VisitStructInfoExprField(
    const PrimExpr& expr) {
  return parent_->VisitPrimExpr(expr);
}

StructInfo ExprMutatorBase::DefaultStructInfoFieldMutator::VisitStructInfo_(
    const FuncStructInfoNode* op) {
  // Do not recurse into function struct info
  // as they won't contain ref to values in current scope.
  return GetRef<StructInfo>(op);
}

Expr ExprMutatorBase::VisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

Expr ExprMutatorBase::VisitExpr_(const ConstantNode* op) {
  // Constant' struct info won't be affected by Expr/PrimExpr change.
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const GlobalVarNode* op) {
  // FuncStructInfo won't be affected by Expr/PrimExpr change.
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const TupleNode* op) {
  bool unchanged = true;
  tvm::Array<Expr> fields;
  for (Expr field : op->fields) {
    Expr new_field = this->VisitExpr(field);
    fields.push_back(new_field);
    unchanged &= new_field.same_as(field);
  }

  if (unchanged) {
    // If tuple's struct info change it means that
    // one of its fields' struct info will change
    // so un-changed already implies that struct info won't change
    return GetRef<Expr>(op);
  } else {
    // when there is a change return a new tuple node
    return Tuple(fields, op->span);
  }
}

// Visit the use-site of a defined Var
Expr ExprMutatorBase::VisitExpr_(const VarNode* op) {
  // struct info of var-use should remain stable
  // or the var itself will get replaced
  return GetRef<Expr>(op);
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutatorBase::VisitExpr_(const DataflowVarNode* op) {
  return VisitExpr_(static_cast<const VarNode*>(op));
}

Expr ExprMutatorBase::VisitExpr_(const FunctionNode* op) {
  // struct info of function is not value dependent
  // so no need to check struct_info field
  Expr body = this->VisitExpr(op->body);

  if (body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(op->params, body, op->ret_struct_info, op->is_pure, op->attrs);
  }
}

Expr ExprMutatorBase::VisitExpr_(const CallNode* call_node) {
  Expr new_op = this->VisitExpr(call_node->op);
  bool unchanged = call_node->op.same_as(new_op);

  Array<StructInfo> sinfo_args;
  for (StructInfo sinfo_arg : call_node->sinfo_args) {
    StructInfo new_sinfo_arg = this->VisitExprDepStructInfoField(sinfo_arg);
    sinfo_args.push_back(new_sinfo_arg);
    unchanged &= new_sinfo_arg.same_as(sinfo_arg);
  }

  tvm::Array<Expr> call_args;
  for (Expr arg : call_node->args) {
    Expr new_arg = this->VisitExpr(arg);
    call_args.push_back(new_arg);
    unchanged &= new_arg.same_as(arg);
  }

  if (unchanged && VisitAndCheckStructInfoFieldUnchanged(call_node->struct_info_)) {
    return GetRef<Expr>(call_node);
  } else {
    return Call(new_op, call_args, call_node->attrs, sinfo_args, call_node->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const IfNode* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitExpr(op->true_branch);
  Expr false_b = this->VisitExpr(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const OpNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const TupleGetItemNode* op) {
  auto t = this->VisitExpr(op->tuple);
  if (op->tuple.same_as(t)) {
    // struct info can be deterministically derived by tuple and index
    // if t does not change, then struct info won't change.
    return GetRef<Expr>(op);
  } else {
    return TupleGetItem(t, op->index, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const PrimValueNode* op) {
  auto value = this->VisitPrimExpr(op->value);
  if (op->value.same_as(value)) {
    // struct info can be deterministically derived by value
    // if value does not change, then struct info won't change.
    return GetRef<Expr>(op);
  }
  return PrimValue(value, op->span);
}

Expr ExprMutatorBase::VisitExpr_(const StringImmNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const DataTypeImmNode* op) { return GetRef<Expr>(op); }

Expr ExprMutatorBase::VisitExpr_(const ShapeExprNode* op) {
  auto values = op->values.Map([this](const PrimExpr& e) { return this->VisitPrimExpr(e); });

  if (values.same_as(op->values)) {
    // If values does not change, struct info won't change.
    return GetRef<Expr>(op);
  } else {
    return ShapeExpr(values, op->span);
  }
}

Expr ExprMutatorBase::VisitExpr_(const ExternFuncNode* op) {
  // StructInfo of function remains value independent.
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  Array<BindingBlock> blocks;
  for (auto block : op->blocks) {
    BindingBlock new_block = this->VisitBindingBlock(block);
    if (!new_block->bindings.empty()) {
      blocks.push_back(new_block);
    }
    all_blocks_unchanged &= block.same_as(new_block);
  }

  Expr body = this->VisitExpr(op->body);

  if (all_blocks_unchanged && body.same_as(op->body) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
    return GetRef<Expr>(op);
  }
  return SeqExpr(blocks, body);
}

BindingBlock ExprMutatorBase::VisitBindingBlock(const BindingBlock& block) {
  Array<Binding> bindings;
  if (const auto* node = block.as<BindingBlockNode>()) {
    for (auto binding : node->bindings) {
      if (auto var_binding = binding.as<VarBindingNode>()) {
        Expr new_value = this->VisitExpr(var_binding->value);
        bindings.push_back(VarBinding(var_binding->var, new_value));
      } else if (auto match_cast = binding.as<MatchCastNode>()) {
        Expr new_value = this->VisitExpr(match_cast->value);
        bindings.push_back(MatchCast(match_cast->var, new_value, match_cast->struct_info));
      } else {
        LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
      }
    }
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
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
  return GetRef<Expr>(op);
}

// Visit the use-site of a defined DataflowVar
Expr ExprMutator::VisitExpr_(const DataflowVarNode* op) {
  return VisitExpr_(static_cast<const VarNode*>(op));
}

Expr ExprMutator::VisitExpr_(const FunctionNode* op) {
  tvm::Array<Var> params;
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
    return GetRef<Expr>(op);
  } else if (IsBaseOf(GetStructInfo(body), op->ret_struct_info)) {
    // If the function was mutated into a form that can no longer
    // propagate shape information all the way to the return value, we
    // may keep the return struct info.  This is only allowed when the
    // body produces a return value that is the same as, or more
    // specific than, the pre-mutation struct info.  For example, if
    // the previous return value was `TensorStructInfo(shape=[16,16])`
    // but the body only produced `TensorStructInfo(ndim=2)`, we can
    // keep the more specific information.
    return Function(params, body, op->ret_struct_info, op->is_pure, op->attrs);
  } else {
    // If the function was mutated such that the body produces an
    // output that is incompatible with the original return struct
    // info, the original return struct info should not be used.  For
    // example, if the previous return value was
    // `TensorStructInfo(shape=[16,16])`, but the new return value is
    // `TensorStructInfo(shape=[8,8])`.
    return Function(params, body, NullOpt, op->is_pure, op->attrs);
  }
}

Expr ExprMutator::VisitExpr_(const IfNode* op) {
  Expr guard = this->VisitExpr(op->cond);
  Expr true_b = this->VisitWithInnerScope(op->true_branch);
  Expr false_b = this->VisitWithInnerScope(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}

Expr ExprMutator::VisitExpr_(const SeqExprNode* op) {
  bool all_blocks_unchanged = true;
  Array<BindingBlock> blocks;
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

  if (all_blocks_unchanged && body.same_as(op->body) &&
      VisitAndCheckStructInfoFieldUnchanged(op->struct_info_)) {
    return GetRef<Expr>(op);
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
    builder_->EmitNormalized(GetRef<VarBinding>(binding));
    return;
  }

  auto new_sinfo = new_value->struct_info_.as<StructInfo>();

  ICHECK(new_sinfo)
      << "InternalError: "
      << "In binding of variable " << binding->var << ", the value " << new_value
      << " does not have StructInfo.  "
      << "This typically occurs when ReEmitBinding is called without first calling Normalize.";

  Var temp = WithStructInfo(new_var, new_sinfo.value());
  if (!temp.same_as(new_var)) {
    new_var = temp;
  }

  this->var_remap_[binding->var->vid] = new_var;
  this->var_remap_[new_var->vid] = new_var;

  builder_->EmitNormalized(VarBinding(new_var, new_value));
}

void ExprMutator::VisitBinding_(const MatchCastNode* binding) {
  Expr new_value = this->VisitExpr(binding->value);
  StructInfo new_struct_info = this->VisitExprDepStructInfoField(binding->struct_info);

  Var new_var = this->VisitVarDef(binding->var);

  MatchCast new_binding = [&]() -> MatchCast {
    if (new_var.same_as(binding->var) && new_value.same_as(binding->value) &&
        new_struct_info.same_as(binding->struct_info)) {
      // re-emit old binding if nothing changes
      return GetRef<MatchCast>(binding);
    } else {
      new_value = builder_->NormalizeArgument(new_value);
      new_var = WithStructInfo(new_var, new_struct_info);

      var_remap_[binding->var->vid] = new_var;
      var_remap_[new_var->vid] = new_var;

      return MatchCast(new_var, new_value, new_struct_info, binding->span);
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
    output = DataflowVar(output->vid, GetStructInfo(output), output->span);
  }
  return output;
}

Var ExprMutator::VisitVarDef_(const VarNode* var) {
  if (auto* sinfo = var->struct_info_.as<StructInfoNode>()) {
    StructInfo struct_info = this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
    if (struct_info.same_as(var->struct_info_)) {
      return GetRef<Var>(var);
    } else {
      return Var(var->vid, struct_info, var->span);
    }
  } else {
    return GetRef<Var>(var);
  }
}

void ExprMutator::VisitBinding(const Binding& binding) {
  if (const auto* node = binding.as<VarBindingNode>()) {
    VisitBinding_(node);
  } else if (const auto* node = binding.as<MatchCastNode>()) {
    VisitBinding_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
  }
}

BindingBlock ExprMutator::VisitBindingBlock(const BindingBlock& block) {
  BindingBlock ret;
  if (const auto* node = block.as<DataflowBlockNode>()) {
    ret = VisitBindingBlock_(node);
  } else if (const auto* node = block.as<BindingBlockNode>()) {
    ret = VisitBindingBlock_(node);
  } else {
    LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
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
    LOG(FATAL) << "TypeError: Invalid type: " << var->GetTypeKey();
  }
  return ret;
}

Expr ExprMutator::VisitWithNewScope(const Expr& expr, Optional<Array<Var>> params) {
  ICHECK(expr->IsInstance<SeqExprNode>())
      << "Normal form requires all new scope is stored as SeqExpr";

  PrimExpr constraint = Bool(true);
  if (params.defined()) {
    auto non_negative_expressions =
        CollectNonNegativeExpressions(TupleStructInfo(params.value().Map(GetStructInfo)));
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

  // Normalization (and the resulting StructInfo inference) of the
  // expr occurs outside of the body's parameters, but inside the
  // function signature's scope.  This keeps variables that are
  // inferable based on the function signature, to allow callers to
  // propagate StructInfo across the function.
  ret = builder_->Normalize(ret);
  builder_->EndScope();
  return ret;
}

Expr ExprMutator::VisitWithInnerScope(const Expr& expr) {
  ICHECK(expr->IsInstance<SeqExprNode>())
      << "Normal form requires all new scope is stored as SeqExpr";

  builder_->BeginInnerScope();
  Expr ret = this->VisitExpr(expr);
  builder_->EndScope();
  return ret;
}

Optional<Expr> ExprMutator::LookupBinding(const Var& var) { return builder_->LookupBinding(var); }

Var ExprMutator::WithStructInfo(Var var, StructInfo struct_info) {
  ICHECK(struct_info.defined());

  // TODO(relax-team) add StructInfoEqual check
  if (var->struct_info_.defined()) {
    // use same-as as a quick path
    if (var->struct_info_.same_as(struct_info) ||
        StructuralEqual()(var->struct_info_, struct_info)) {
      return var;
    } else {
      Var new_var = var.as<DataflowVarNode>() ? DataflowVar(var->vid, struct_info, var->span)
                                              : Var(var->vid, struct_info, var->span);
      return new_var;
    }
  } else {
    UpdateStructInfo(var, struct_info);
    return var;
  }
}

}  // namespace relax
}  // namespace tvm
