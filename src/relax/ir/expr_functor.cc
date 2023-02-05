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
  this->VisitSpan(op->span);
  if (auto* sinfo = op->struct_info_.as<StructInfoNode>()) {
    this->VisitExprDepStructInfoField(GetRef<StructInfo>(sinfo));
  }
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

void ExprVisitor::VisitVarDef_(const DataflowVarNode* var) { this->VisitSpan(var->span); }

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
  // struct info of var-use should remain stable
  // or the var itself will get replaced
  return GetRef<Expr>(op);
}

Expr ExprMutatorBase::VisitExpr_(const FunctionNode* op) {
  // struct info of function is not value dependent
  // so no need to check struct_info field
  Expr body = this->VisitExpr(op->body);

  if (body.same_as(op->body)) {
    return GetRef<Expr>(op);
  } else {
    return Function(op->params, body, op->ret_struct_info, op->attrs);
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

}  // namespace relax
}  // namespace tvm
