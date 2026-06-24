/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <unordered_set>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  IdNode::RegisterReflection();
  CallNode::RegisterReflection();
  TupleNode::RegisterReflection();
  TupleGetItemNode::RegisterReflection();
  ShapeExprNode::RegisterReflection();
  VarNode::RegisterReflection();
  BindingNode::RegisterReflection();
  DataflowVarNode::RegisterReflection();
  ConstantNode::RegisterReflection();
  PrimValueNode::RegisterReflection();
  StringImmNode::RegisterReflection();
  DataTypeImmNode::RegisterReflection();
  MatchCastNode::RegisterReflection();
  VarBindingNode::RegisterReflection();
  BindingBlockNode::RegisterReflection();
  DataflowBlockNode::RegisterReflection();
  SeqExprNode::RegisterReflection();
  IfNode::RegisterReflection();
  FunctionNode::RegisterReflection();
  ExternFuncNode::RegisterReflection();
}

Id::Id(ffi::String name_hint) {
  ffi::ObjectPtr<IdNode> n = ffi::make_object<IdNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

Call::Call(Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<Type> ty_args, Span span) {
  TVM_FFI_CHECK(op.defined(), ValueError) << "Call expects a defined operator";
  TVM_FFI_CHECK(!op->ty.defined() || op->ty->IsInstance<FuncTypeNode>(), ValueError)
      << "Call expects its operator to have FuncType, "
      << "but operator " << op << ", which was called with arguments " << args << ", has type "
      << op->ty;

  ffi::ObjectPtr<CallNode> n = ffi::make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->ty_args = std::move(ty_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

Call WithFields(Call call, ffi::Optional<Expr> opt_op, ffi::Optional<ffi::Array<Expr>> opt_args,
                ffi::Optional<Attrs> opt_attrs, ffi::Optional<ffi::Array<Type>> opt_ty_args,
                ffi::Optional<Span> opt_span) {
  // Collect new values for fields.
  Expr op = opt_op.value_or(call->op);
  ffi::Array<Expr> args = opt_args.value_or(call->args);
  Attrs attrs = opt_attrs.value_or(call->attrs);
  ffi::Array<Type> ty_args = opt_ty_args.value_or(call->ty_args);
  Span span = opt_span.value_or(call->span);

  TVM_FFI_CHECK(op.defined(), ValueError) << "Call expects a defined operator";

  // Check if anything changed.
  bool unchanged = op.same_as(call->op) && attrs.same_as(call->attrs) && span.same_as(call->span);
  if (unchanged) {
    if (args.size() == call->args.size()) {
      for (size_t i = 0; i < args.size(); i++) {
        unchanged &= args[i].same_as(call->args[i]);
      }
    } else {
      unchanged = false;
    }
  }
  if (unchanged) {
    if (ty_args.size() == call->ty_args.size()) {
      for (size_t i = 0; i < ty_args.size(); i++) {
        unchanged &= ty_args[i].same_as(call->ty_args[i]);
      }
    } else {
      unchanged = false;
    }
  }

  if (!unchanged) {
    // If call is only references, update it in place. Otherwise copy and update.
    CallNode* cow_call_node = call.CopyOnWrite();
    cow_call_node->op = op;
    cow_call_node->args = args;
    cow_call_node->attrs = attrs;
    cow_call_node->ty_args = ty_args;
    cow_call_node->span = span;
  }
  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Call",
                        [](Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<Type> ty_args,
                           Span span) { return Call(op, args, attrs, ty_args, span); });
}

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ffi::ObjectPtr<IfNode> n = ffi::make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}

If WithFields(If if_expr, ffi::Optional<Expr> opt_cond, ffi::Optional<Expr> opt_true_branch,
              ffi::Optional<Expr> opt_false_branch, ffi::Optional<Span> opt_span) {
  Expr cond = opt_cond.value_or(if_expr->cond);
  Expr true_branch = opt_true_branch.value_or(if_expr->true_branch);
  Expr false_branch = opt_false_branch.value_or(if_expr->false_branch);
  Span span = opt_span.value_or(if_expr->span);

  bool unchanged = cond.same_as(if_expr->cond) && true_branch.same_as(if_expr->true_branch) &&
                   false_branch.same_as(if_expr->false_branch) && span.same_as(if_expr->span);

  if (!unchanged) {
    IfNode* cow_if_node = if_expr.CopyOnWrite();
    cow_if_node->cond = cond;
    cow_if_node->true_branch = true_branch;
    cow_if_node->false_branch = false_branch;
    cow_if_node->span = span;
  }
  return if_expr;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.If", [](Expr cond, Expr true_branch, Expr false_branch, Span span) {
    return If(cond, true_branch, false_branch, span);
  });
}

Tuple::Tuple(tvm::ffi::Array<Expr> fields, Span span) {
  ffi::Optional<Type> tuple_ty = [&]() -> ffi::Optional<Type> {
    ffi::Array<Type> field_ty;
    for (const auto& field : fields) {
      if (field->ty.defined()) {
        field_ty.push_back(GetType(field));
      } else {
        return std::nullopt;
      }
    }
    return TupleType(field_ty);
  }();

  ffi::ObjectPtr<TupleNode> n = ffi::make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  if (tuple_ty.defined()) {
    n->ty = tuple_ty.value();
  }
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.Tuple", [](tvm::ffi::Array<Expr> fields, Span span) { return Tuple(fields, span); });
}

Tuple WithFields(Tuple tuple, ffi::Optional<ffi::Array<Expr>> opt_fields,
                 ffi::Optional<Span> opt_span) {
  ffi::Array<Expr> fields = opt_fields.value_or(tuple->fields);
  Span span = opt_span.value_or(tuple->span);

  bool all_fields_unchanged = true;
  if (fields.size() == tuple->fields.size()) {
    for (size_t i = 0; i < fields.size(); i++) {
      all_fields_unchanged &= fields[i].same_as(tuple->fields[i]);
    }
  } else {
    all_fields_unchanged = false;
  }

  all_fields_unchanged = all_fields_unchanged && span.same_as(tuple->span);
  if (!all_fields_unchanged) {
    TupleNode* cow_tuple_node = tuple.CopyOnWrite();
    cow_tuple_node->fields = fields;
    cow_tuple_node->span = span;
  }
  return tuple;
}

TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  TVM_FFI_ICHECK_GE(index, 0) << "Index out of bounds: Tuple " << tuple
                              << " cannot be accessed with negative index " << index;
  ffi::ObjectPtr<TupleGetItemNode> n = ffi::make_object<TupleGetItemNode>();

  if (auto* tuple_info = tuple->ty.as<TupleTypeNode>()) {
    TVM_FFI_ICHECK_LT(index, tuple_info->fields.size())
        << "Index out of bounds: Tuple " << tuple << " is of size " << tuple_info->fields.size()
        << ", and cannot be accessed with index " << index;
    auto ty = tuple_info->fields[index];
    n->ty = ty;
  }
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem WithFields(TupleGetItem tuple_get_item, ffi::Optional<Expr> opt_tuple,
                        ffi::Optional<int64_t> opt_index, ffi::Optional<Span> opt_span) {
  Expr tuple = opt_tuple.value_or(tuple_get_item->tuple);
  int64_t index = opt_index.value_or(tuple_get_item->index);
  Span span = opt_span.value_or(tuple_get_item->span);

  bool unchanged = tuple.same_as(tuple_get_item->tuple) && (index == tuple_get_item->index) &&
                   span.same_as(tuple_get_item->span);
  if (!unchanged) {
    TupleGetItemNode* cow_tuple_get_item_node = tuple_get_item.CopyOnWrite();
    cow_tuple_get_item_node->index = static_cast<int>(index);
    cow_tuple_get_item_node->span = span;
  }
  return tuple_get_item;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.TupleGetItem", [](Expr tuple, int index, Span span) {
    return TupleGetItem(tuple, index, span);
  });
}

ShapeExpr::ShapeExpr(ffi::Array<PrimExpr> values, Span span) {
  ffi::ObjectPtr<ShapeExprNode> n = ffi::make_object<ShapeExprNode>();

  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(PrimType::Int(64), value);
    }
    TVM_FFI_ICHECK(value.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64))
        << "the value in ShapeType can only have dtype of int64";
    return value;
  });
  n->span = span;
  n->ty = ShapeType(values, span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ShapeExpr", [](ffi::Array<PrimExpr> values, Span span) {
    return ShapeExpr(values, span);
  });
}

Var::Var(Id vid, ffi::Optional<Type> ty_annotation, Span span) {
  ffi::ObjectPtr<VarNode> n = ffi::make_object<VarNode>();
  n->vid = std::move(vid);
  if (ty_annotation.defined()) {
    n->ty = std::move(ty_annotation.value());
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

VarNode* Var::CopyOnWrite() {
  // The `TVM_DEFINE_OBJECT_REF_COW_METHOD` cannot be used for
  // Var, because it is the base class for `DataflowBlock`.
  // If the `TVM_DEFINE_OBJECT_REF_COW_METHOD` were used, the
  // automatic implementation would erroneously convert from a
  // `DataflowBlock` to a `Var`.
  TVM_FFI_ICHECK(data_ != nullptr);
  if (!data_.unique()) {
    ffi::ObjectPtr<VarNode> node;
    if (auto dataflow_var = as<DataflowVarNode>()) {
      node = ffi::make_object<DataflowVarNode>(*dataflow_var);
    } else {
      node = ffi::make_object<VarNode>(*(operator->()));
    }
    ffi::ObjectPtr<ffi::Object>(std::move(node)).swap(data_);
  }
  return static_cast<VarNode*>(data_.get());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.Var", [](ffi::String name_hint, ffi::Optional<Type> ty_annotation,
                           Span span) { return Var(name_hint, ty_annotation, span); })
      .def("relax.VarFromId", [](Id vid, ffi::Optional<Type> ty_annotation, Span span) {
        return Var(vid, ty_annotation, span);
      });
}

DataflowVar::DataflowVar(Id vid, ffi::Optional<Type> ty_annotation, Span span) {
  ffi::ObjectPtr<DataflowVarNode> n = ffi::make_object<DataflowVarNode>();
  n->vid = std::move(vid);
  if (ty_annotation.defined()) {
    n->ty = std::move(ty_annotation.value());
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.DataflowVar",
           [](ffi::String name_hint, ffi::Optional<Type> ty_annotation, Span span) {
             return DataflowVar(name_hint, ty_annotation, span);
           })
      .def("relax.DataflowVarFromId", [](Id vid, ffi::Optional<Type> ty_annotation, Span span) {
        return DataflowVar(vid, ty_annotation, span);
      });
}

Constant::Constant(runtime::Tensor data, ffi::Optional<Type> ty_annotation, Span span) {
  ffi::ObjectPtr<ConstantNode> n = ffi::make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);

  // set type.
  ffi::Array<PrimExpr> values;
  auto shape_tuple = n->data.Shape();
  for (size_t dim = 0; dim < shape_tuple.size(); ++dim) {
    values.push_back(IntImm::Int64(shape_tuple[dim]));
  }
  if (ty_annotation.defined()) {
    n->ty = ty_annotation.value();
  } else {
    TensorType tinfo(ShapeExpr(values), PrimType(n->data.DataType()), VDevice(), span);
    n->ty = tinfo;
  }

  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Constant",
                        [](runtime::Tensor data, ffi::Optional<Type> ty_annotation = std::nullopt,
                           Span span = Span()) { return Constant(data, ty_annotation, span); });
}

PrimValue::PrimValue(PrimExpr value, Span span) {
  ffi::ObjectPtr<PrimValueNode> n = ffi::make_object<PrimValueNode>();
  n->ty = PrimType(value.ty());
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

PrimValue PrimValue::Int64(int64_t value, Span span) {
  return PrimValue(IntImm::Int64(value), span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.PrimValue",
                        [](PrimExpr value, Span span) { return PrimValue(value, span); });
}

StringImm::StringImm(ffi::String value, Span span) {
  ffi::ObjectPtr<StringImmNode> n = ffi::make_object<StringImmNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  n->ty = ObjectType();
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
}

DataTypeImm::DataTypeImm(DLDataType value, Span span) {
  ffi::ObjectPtr<DataTypeImmNode> n = ffi::make_object<DataTypeImmNode>();
  n->value = value;
  n->span = std::move(span);
  n->ty = ObjectType();
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataTypeImm",
                        [](DLDataType value, Span span) { return DataTypeImm(value, span); });
}

MatchCast::MatchCast(Var var, Expr value, Type ty, Span span) {
  ffi::ObjectPtr<MatchCastNode> n = ffi::make_object<MatchCastNode>();
  TVM_FFI_ICHECK(var.defined()) << "MatchCast requires var to be defined";
  n->var = std::move(var);
  n->value = std::move(value);
  n->ty = std::move(ty);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.MatchCast", [](Var var, Expr value, Type ty, Span span) {
    return MatchCast(var, value, ty, span);
  });
}

VarBinding::VarBinding(Var var, Expr value, Span span) {
  ffi::ObjectPtr<VarBindingNode> n = ffi::make_object<VarBindingNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.VarBinding", [](Var var, Expr value, Span span) {
    return VarBinding(var, value, span);
  });
}

bool VarBindingNode::SEqual(const VarBindingNode* other,
                            ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
  if (value->IsInstance<FunctionNode>()) {
    // Recursive function definitions may reference the bound variable
    // within the value being bound.  In these cases, the
    // var comparison must occur first to define the var, to ensure it is
    // defined at point of use.
    return equal(var, other->var, true, "var") && equal(value, other->value, false, "value");
  } else {
    // In all other cases, visit the bound value before the variable
    // it is bound to, in order to provide better error messages.
    return equal(value, other->value, false, "value") && equal(var, other->var, true, "var");
  }
}

int64_t VarBindingNode::SHash(int64_t init_hash,
                              ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
  int64_t hash_value = init_hash;
  if (value->IsInstance<FunctionNode>()) {
    hash_value = hash(var, hash_value, true);
    hash_value = hash(value, hash_value, false);
  } else {
    hash_value = hash(value, hash_value, false);
    hash_value = hash(var, hash_value, true);
  }
  return hash_value;
}

BindingBlock::BindingBlock(ffi::Array<Binding> bindings, Span span) {
  ffi::ObjectPtr<BindingBlockNode> n = ffi::make_object<BindingBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

BindingBlockNode* BindingBlock::CopyOnWrite() {
  // The `TVM_DEFINE_OBJECT_REF_COW_METHOD` cannot be used for
  // BindingBlock, because it is the base class for `DataflowBlock`.
  // If the `TVM_DEFINE_OBJECT_REF_COW_METHOD` were used, the
  // automatic implementation would erroneously convert from a
  // `DataflowBlock` to a `BindingBlock`.
  TVM_FFI_ICHECK(data_ != nullptr);
  if (!data_.unique()) {
    ffi::ObjectPtr<BindingBlockNode> node;
    if (auto dataflow_block = as<DataflowBlockNode>()) {
      node = ffi::make_object<DataflowBlockNode>(*dataflow_block);
    } else {
      node = ffi::make_object<BindingBlockNode>(*(operator->()));
    }
    ffi::ObjectPtr<ffi::Object>(std::move(node)).swap(data_);
  }
  return static_cast<BindingBlockNode*>(data_.get());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.BindingBlock", [](ffi::Array<Binding> bindings, Span span) {
    return BindingBlock(bindings, span);
  });
}

DataflowBlock::DataflowBlock(ffi::Array<Binding> bindings, Span span) {
  ffi::ObjectPtr<DataflowBlockNode> n = ffi::make_object<DataflowBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataflowBlock", [](ffi::Array<Binding> bindings, Span span) {
    return DataflowBlock(bindings, span);
  });
}

SeqExpr::SeqExpr(Expr body) {
  if (auto seq = body.as<SeqExpr>()) {
    *this = seq.value();
  } else {
    *this = SeqExpr(ffi::Array<BindingBlock>{}, body);
  }
}

SeqExpr::SeqExpr(ffi::Array<BindingBlock> blocks, Expr body, Span span) {
  ffi::ObjectPtr<SeqExprNode> n = ffi::make_object<SeqExprNode>();
  n->blocks = std::move(blocks);
  n->body = std::move(body);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.SeqExpr", [](ffi::Array<BindingBlock> blocks, Expr body, Span span) {
    return SeqExpr(blocks, body, span);
  });
}

Function::Function(ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty, bool is_pure,
                   DictAttrs attrs, Span span) {
  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  ffi::Array<Type> param_ty;

  for (const Var& param : params) {
    TVM_FFI_ICHECK(param->ty.defined()) << "relax.Function requires params to contain ty";
    param_ty.push_back(GetType(param));
  }

  ffi::Optional<Type> body_ty;

  if (body->ty.defined()) {
    body_ty = GetType(body);
  }

  TVM_FFI_ICHECK(body_ty.defined() || ret_ty.defined())
      << "Function must be constructed with either "
      << "an explicit type for the return type, "
      << "or a normalized body with type.";

  // Use the body's type if there is no explicit return type,
  // or if the body may provide a more granular return type.
  bool use_body_ty =
      !ret_ty.defined() || (body_ty && ret_ty && IsBaseOf(ret_ty.value(), body_ty.value()));

  if (use_body_ty) {
    // MatchCast nodes within the body may introduce new symbolic
    // variables.  These are in-scope for the function body, but not
    // for the function's return type.  When hoisting the body's type
    // to the function return type, symbolic variables may only be
    // used if they were defined by the function's parameters.
    auto f_shape_var_map = [&] {
      auto tir_vars = DefinableTIRVarsInType(TupleType(params.Map(GetType)));
      std::unordered_set<tirx::Var> lookup(tir_vars.begin(), tir_vars.end());
      return [lookup = std::move(lookup)](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
        if (lookup.count(var)) {
          return var;
        } else {
          return std::nullopt;
        }
      };
    }();
    ret_ty = EraseToWellDefined(body_ty.value(), f_shape_var_map);
  }

  FuncType func_ty(param_ty, ret_ty.value(), is_pure);

  // set the fields
  ffi::ObjectPtr<FunctionNode> n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_ty = ret_ty.value();
  n->is_pure = is_pure;
  n->ty = std::move(func_ty);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Function",
                        [](ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty,
                           bool is_pure, DictAttrs attrs, Span span) {
                          return Function(params, body, ret_ty, is_pure, attrs, span);
                        });
}

Function Function::CreateEmpty(ffi::Array<Var> params, Type ret_ty, bool is_pure, DictAttrs attrs,
                               Span span) {
  ffi::Array<Type> param_ty;
  for (const Var& param : params) {
    TVM_FFI_ICHECK(param->ty.defined()) << "relax.Function requires params to contain ty.";
    param_ty.push_back(GetType(param));
  }

  FuncType finfo(param_ty, ret_ty, is_pure);

  // A dummy body, to ensure that the empty function is still well-formed.
  Expr body = [&]() -> Expr {
    Var output("output", ret_ty);
    Call expr(ExternFunc("_dummy_function", FuncType({}, ret_ty)), {});

    return SeqExpr({BindingBlock({VarBinding(output, expr)})}, output);
  }();

  // set the fields
  ffi::ObjectPtr<FunctionNode> n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->is_pure = is_pure;
  n->ty = std::move(finfo);
  n->ret_ty = std::move(ret_ty);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  return Function(std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.FunctionCreateEmpty", [](ffi::Array<Var> params, Type ret_ty,
                                                        bool is_pure, DictAttrs attrs, Span span) {
    return Function::CreateEmpty(params, ret_ty, is_pure, attrs, span);
  });
}

// Special opaque derivation function for ExternFunc
// Take look at ty_args to figure out the return Type.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto infer_by_ty_args = [](const Call& call, const BlockBuilder& ctx) -> Type {
    TVM_FFI_ICHECK(call->ty_args.defined()) << "ty_args field of CallNode should always be defined";
    if (call->ty_args.empty()) {
      return ObjectType();
    } else if (call->ty_args.size() == 1) {
      return call->ty_args[0];
    } else {
      return TupleType(call->ty_args);
    }
  };
  refl::GlobalDef().def("tvm.relax.type.infer_by_ty_args", infer_by_ty_args);
}

// Get the derive function.
FuncType GetExternFuncType() {
  EnvFunc fn = EnvFunc::Get("tvm.relax.type.infer_by_ty_args");
  TypeDeriveFunc derive;
  derive = fn;
  return FuncType::OpaqueFunc(derive);
}

ExternFunc::ExternFunc(ffi::String global_symbol, Span span)
    : ExternFunc(global_symbol, GetExternFuncType(), span) {}

ExternFunc::ExternFunc(ffi::String global_symbol, Type ty, Span span) {
  TVM_FFI_ICHECK(ty.as<FuncTypeNode>())
      << "ExternFunc must have FuncType, "
      << "but declaration of '" << global_symbol << "' received " << ty;

  ffi::ObjectPtr<ExternFuncNode> n = ffi::make_object<ExternFuncNode>();
  n->global_symbol = std::move(global_symbol);
  n->span = span;
  n->ty = ty;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ExternFunc",
                        [](ffi::String global_symbol, ffi::Optional<Type> ty, Span span) {
                          if (ty.defined()) {
                            return ExternFunc(global_symbol, ty.value(), span);
                          } else {
                            return ExternFunc(global_symbol, span);
                          }
                        });
}

Expr GetShapeOf(const Expr& expr) {
  // default case, to be normalized.
  TVM_FFI_ICHECK(expr->ty.defined()) << "GetShapeOf can only be applied to normalized expr";
  auto* tinfo = GetTypeAs<TensorTypeNode>(expr);

  TVM_FFI_ICHECK(tinfo != nullptr) << "ShapeOf can only be applied to expr with TensorType";
  if (tinfo->shape.defined()) return tinfo->shape.value();

  static const Op& op = Op::Get("relax.shape_of");
  // default case, call shape of, eagerly normalize the expr.
  relax::Call call_shape_of(op, {expr}, {}, {});
  UpdateType(call_shape_of, ShapeType(tinfo->ndim));
  return call_shape_of;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.GetShapeOf", [](const Expr& expr) { return GetShapeOf(expr); })
      .def("relax.FuncWithAttr",
           [](BaseFunc func, ffi::String key, ffi::ObjectRef value) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttr(std::move(func).as_or_throw<relax::Function>(), key, value);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithAttrs",
           [](BaseFunc func, ffi::Map<ffi::String, ffi::Any> attr_map) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttrs(std::move(func).as_or_throw<relax::Function>(), attr_map);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithoutAttr", [](BaseFunc func, ffi::String key) -> ffi::Optional<Function> {
        if (func->IsInstance<relax::FunctionNode>()) {
          return WithoutAttr(std::move(func).as_or_throw<relax::Function>(), key);
        }
        return std::nullopt;
      });
}

}  // namespace relax
}  // namespace tvm
