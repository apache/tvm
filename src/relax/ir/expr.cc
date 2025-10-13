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
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/type.h>

#include <unordered_set>

namespace tvm {
namespace relax {

using tvm::ReprPrinter;

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
  ObjectPtr<IdNode> n = ffi::make_object<IdNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

Call::Call(Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<StructInfo> sinfo_args,
           Span span) {
  CHECK(!op->struct_info_.defined() || op->struct_info_->IsInstance<FuncStructInfoNode>())
      << "ValueError: "
      << "Call expects its operator to have FuncStructInfo, "
      << "but operator " << op << ", which was called with arguments " << args
      << ", has struct info " << op->struct_info_;

  ObjectPtr<CallNode> n = ffi::make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->sinfo_args = std::move(sinfo_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

Call WithFields(Call call, ffi::Optional<Expr> opt_op, ffi::Optional<ffi::Array<Expr>> opt_args,
                ffi::Optional<Attrs> opt_attrs,
                ffi::Optional<ffi::Array<StructInfo>> opt_sinfo_args,
                ffi::Optional<Span> opt_span) {
  // Collect new values for fields.
  Expr op = opt_op.value_or(call->op);
  ffi::Array<Expr> args = opt_args.value_or(call->args);
  Attrs attrs = opt_attrs.value_or(call->attrs);
  ffi::Array<StructInfo> sinfo_args = opt_sinfo_args.value_or(call->sinfo_args);
  Span span = opt_span.value_or(call->span);

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
    if (sinfo_args.size() == call->sinfo_args.size()) {
      for (size_t i = 0; i < sinfo_args.size(); i++) {
        unchanged &= sinfo_args[i].same_as(call->sinfo_args[i]);
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
    cow_call_node->sinfo_args = sinfo_args;
    cow_call_node->span = span;
  }
  return call;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Call", [](Expr op, ffi::Array<Expr> args, Attrs attrs,
                                         ffi::Array<StructInfo> sinfo_args, Span span) {
    return Call(op, args, attrs, sinfo_args, span);
  });
}

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = ffi::make_object<IfNode>();
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
  ffi::Optional<StructInfo> tuple_sinfo = [&]() -> ffi::Optional<StructInfo> {
    ffi::Array<StructInfo> field_sinfo;
    for (const auto& field : fields) {
      if (field->struct_info_.defined()) {
        field_sinfo.push_back(GetStructInfo(field));
      } else {
        return std::nullopt;
      }
    }
    return TupleStructInfo(field_sinfo);
  }();

  ObjectPtr<TupleNode> n = ffi::make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  n->struct_info_ = tuple_sinfo;
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
  CHECK_GE(index, 0) << "Index out of bounds: Tuple " << tuple
                     << " cannot be accessed with negative index " << index;
  ObjectPtr<TupleGetItemNode> n = ffi::make_object<TupleGetItemNode>();

  if (auto* tuple_info = tuple->struct_info_.as<TupleStructInfoNode>()) {
    CHECK_LT(index, tuple_info->fields.size())
        << "Index out of bounds: Tuple " << tuple << " is of size " << tuple_info->fields.size()
        << ", and cannot be accessed with index " << index;
    auto sinfo = tuple_info->fields[index];
    n->struct_info_ = sinfo;
  }
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem WithFields(TupleGetItem tuple_get_item, ffi::Optional<Expr> opt_tuple,
                        ffi::Optional<Integer> opt_index, ffi::Optional<Span> opt_span) {
  Expr tuple = opt_tuple.value_or(tuple_get_item->tuple);
  Integer index = opt_index.value_or(tuple_get_item->index);
  Span span = opt_span.value_or(tuple_get_item->span);

  bool unchanged = tuple.same_as(tuple_get_item->tuple) && (index == tuple_get_item->index) &&
                   span.same_as(tuple_get_item->span);
  if (!unchanged) {
    TupleGetItemNode* cow_tuple_get_item_node = tuple_get_item.CopyOnWrite();
    cow_tuple_get_item_node->tuple = tuple;
    cow_tuple_get_item_node->index = index.IntValue();
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
  ObjectPtr<ShapeExprNode> n = ffi::make_object<ShapeExprNode>();

  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(DataType::Int(64), value);
    }
    ICHECK(value.dtype() == DataType::Int(64))
        << "the value in ShapeStructInfo can only have dtype of int64";
    return value;
  });
  n->span = span;
  n->struct_info_ = ShapeStructInfo(values, span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ShapeExpr", [](ffi::Array<PrimExpr> values, Span span) {
    return ShapeExpr(values, span);
  });
}

Var::Var(Id vid, ffi::Optional<StructInfo> struct_info_annotation, Span span) {
  ObjectPtr<VarNode> n = ffi::make_object<VarNode>();
  n->vid = std::move(vid);
  n->struct_info_ = std::move(struct_info_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

VarNode* Var::CopyOnWrite() {
  // The `TVM_DEFINE_OBJECT_REF_COW_METHOD` cannot be used for
  // Var, because it is the base class for `DataflowBlock`.
  // If the `TVM_DEFINE_OBJECT_REF_COW_METHOD` were used, the
  // automatic implementation would erroneously convert from a
  // `DataflowBlock` to a `Var`.
  ICHECK(data_ != nullptr);
  if (!data_.unique()) {
    ObjectPtr<VarNode> node;
    if (auto dataflow_var = as<DataflowVarNode>()) {
      node = ffi::make_object<DataflowVarNode>(*dataflow_var);
    } else {
      node = ffi::make_object<VarNode>(*(operator->()));
    }
    ObjectPtr<Object>(std::move(node)).swap(data_);
  }
  return static_cast<VarNode*>(data_.get());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.Var", [](ffi::String name_hint, ffi::Optional<StructInfo> struct_info_annotation,
                           Span span) { return Var(name_hint, struct_info_annotation, span); })
      .def("relax.VarFromId", [](Id vid, ffi::Optional<StructInfo> struct_info_annotation,
                                 Span span) { return Var(vid, struct_info_annotation, span); });
}

DataflowVar::DataflowVar(Id vid, ffi::Optional<StructInfo> struct_info_annotation, Span span) {
  ObjectPtr<DataflowVarNode> n = ffi::make_object<DataflowVarNode>();
  n->vid = std::move(vid);
  n->struct_info_ = std::move(struct_info_annotation);
  n->span = std::move(span);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.DataflowVar",
           [](ffi::String name_hint, ffi::Optional<StructInfo> struct_info_annotation, Span span) {
             return DataflowVar(name_hint, struct_info_annotation, span);
           })
      .def("relax.DataflowVarFromId",
           [](Id vid, ffi::Optional<StructInfo> struct_info_annotation, Span span) {
             return DataflowVar(vid, struct_info_annotation, span);
           });
}

Constant::Constant(runtime::Tensor data, ffi::Optional<StructInfo> struct_info_annotation,
                   Span span) {
  ObjectPtr<ConstantNode> n = ffi::make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);

  // set struct info.
  ffi::Array<PrimExpr> values;
  auto shape_tuple = n->data.Shape();
  for (size_t dim = 0; dim < shape_tuple.size(); ++dim) {
    values.push_back(IntImm(DataType::Int(64), shape_tuple[dim]));
  }
  if (struct_info_annotation.defined()) {
    n->struct_info_ = struct_info_annotation.value();
  } else {
    TensorStructInfo tinfo(ShapeExpr(values), n->data.DataType(), VDevice(), span);
    n->struct_info_ = tinfo;
  }

  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.Constant",
      [](runtime::Tensor data, ffi::Optional<StructInfo> struct_info_annotation = std::nullopt,
         Span span = Span()) { return Constant(data, struct_info_annotation, span); });
}

PrimValue::PrimValue(PrimExpr value, Span span) {
  ObjectPtr<PrimValueNode> n = ffi::make_object<PrimValueNode>();
  n->struct_info_ = PrimStructInfo(value);
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

PrimValue PrimValue::Int64(int64_t value, Span span) {
  return PrimValue(IntImm(DataType::Int(64), value), span);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.PrimValue",
                        [](PrimExpr value, Span span) { return PrimValue(value, span); });
}

StringImm::StringImm(ffi::String value, Span span) {
  ObjectPtr<StringImmNode> n = ffi::make_object<StringImmNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  n->struct_info_ = ObjectStructInfo();
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.StringImm",
                        [](ffi::String value, Span span) { return StringImm(value, span); });
}

DataTypeImm::DataTypeImm(DataType value, Span span) {
  ObjectPtr<DataTypeImmNode> n = ffi::make_object<DataTypeImmNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  n->struct_info_ = ObjectStructInfo();
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataTypeImm",
                        [](DataType value, Span span) { return DataTypeImm(value, span); });
}

MatchCast::MatchCast(Var var, Expr value, StructInfo struct_info, Span span) {
  ObjectPtr<MatchCastNode> n = ffi::make_object<MatchCastNode>();
  ICHECK(var.defined()) << "MatchCast requires var to be defined";
  n->var = std::move(var);
  n->value = std::move(value);
  n->struct_info = std::move(struct_info);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.MatchCast",
                        [](Var var, Expr value, StructInfo struct_info, Span span) {
                          return MatchCast(var, value, struct_info, span);
                        });
}

VarBinding::VarBinding(Var var, Expr value, Span span) {
  ObjectPtr<VarBindingNode> n = ffi::make_object<VarBindingNode>();
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

uint64_t VarBindingNode::SHash(uint64_t init_hash,
                               ffi::TypedFunction<uint64_t(AnyView, uint64_t, bool)> hash) const {
  uint64_t hash_value = init_hash;
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
  ObjectPtr<BindingBlockNode> n = ffi::make_object<BindingBlockNode>();
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
  ICHECK(data_ != nullptr);
  if (!data_.unique()) {
    ObjectPtr<BindingBlockNode> node;
    if (auto dataflow_block = as<DataflowBlockNode>()) {
      node = ffi::make_object<DataflowBlockNode>(*dataflow_block);
    } else {
      node = ffi::make_object<BindingBlockNode>(*(operator->()));
    }
    ObjectPtr<Object>(std::move(node)).swap(data_);
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
  ObjectPtr<DataflowBlockNode> n = ffi::make_object<DataflowBlockNode>();
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
  ObjectPtr<SeqExprNode> n = ffi::make_object<SeqExprNode>();
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

Function::Function(ffi::Array<Var> params, Expr body, ffi::Optional<StructInfo> ret_struct_info,
                   bool is_pure, DictAttrs attrs, Span span) {
  if (!attrs.defined()) {
    attrs = DictAttrs();
  }

  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  ffi::Array<StructInfo> param_sinfo;

  for (const Var& param : params) {
    CHECK(param->struct_info_.defined())
        << "relax.Function requires params to contain struct_info_";
    param_sinfo.push_back(GetStructInfo(param));
  }

  ffi::Optional<StructInfo> body_sinfo;

  if (body->struct_info_.defined()) {
    body_sinfo = GetStructInfo(body);
  }

  CHECK(body_sinfo.defined() || ret_struct_info.defined())
      << "Function must be constructed with either "
      << "an explicit struct info for the return type, "
      << "or a normalized body with struct info.";

  // Use the body's struct info if there is no explicit return type,
  // or if the body may provide a more granular return type.
  bool use_body_struct_info =
      !ret_struct_info.defined() ||
      (body_sinfo && ret_struct_info && IsBaseOf(ret_struct_info.value(), body_sinfo.value()));

  if (use_body_struct_info) {
    // MatchCast nodes within the body may introduce new symbolic
    // variables.  These are in-scope for the function body, but not
    // for the function's return type.  When hoisting the body's type
    // to the function return type, symbolic variables may only be
    // used if they were defined by the function's parameters.
    auto f_shape_var_map = [&] {
      auto tir_vars = DefinableTIRVarsInStructInfo(TupleStructInfo(params.Map(GetStructInfo)));
      std::unordered_set<tir::Var> lookup(tir_vars.begin(), tir_vars.end());
      return [lookup = std::move(lookup)](const tir::Var& var) -> ffi::Optional<PrimExpr> {
        if (lookup.count(var)) {
          return var;
        } else {
          return std::nullopt;
        }
      };
    }();
    ret_struct_info = EraseToWellDefined(body_sinfo.value(), f_shape_var_map);
  }

  FuncStructInfo func_sinfo(param_sinfo, ret_struct_info.value(), is_pure);

  // set the fields
  ObjectPtr<FunctionNode> n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_struct_info = ret_struct_info.value();
  n->is_pure = is_pure;
  n->struct_info_ = std::move(func_sinfo);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.Function", [](ffi::Array<Var> params, Expr body,
                                             ffi::Optional<StructInfo> ret_struct_info,
                                             bool is_pure, DictAttrs attrs, Span span) {
    return Function(params, body, ret_struct_info, is_pure, attrs, span);
  });
}

Function Function::CreateEmpty(ffi::Array<Var> params, StructInfo ret_struct_info, bool is_pure,
                               DictAttrs attrs, Span span) {
  ffi::Array<StructInfo> param_sinfo;
  for (const Var& param : params) {
    ICHECK(param->struct_info_.defined())
        << "relax.Function requires params to contain struct_info_.";
    param_sinfo.push_back(GetStructInfo(param));
  }

  FuncStructInfo finfo(param_sinfo, ret_struct_info, is_pure);

  // A dummy body, to ensure that the empty function is still well-formed.
  Expr body = [&]() -> Expr {
    Var output("output", ret_struct_info);
    Call expr(ExternFunc("_dummy_function", FuncStructInfo({}, ret_struct_info)), {});

    return SeqExpr({BindingBlock({VarBinding(output, expr)})}, output);
  }();

  // set the fields
  ObjectPtr<FunctionNode> n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->is_pure = is_pure;
  n->struct_info_ = std::move(finfo);
  n->ret_struct_info = std::move(ret_struct_info);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  return Function(std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.FunctionCreateEmpty", [](ffi::Array<Var> params, StructInfo ret_struct_info,
                                      bool is_pure, DictAttrs attrs, Span span) {
        return Function::CreateEmpty(params, ret_struct_info, is_pure, attrs, span);
      });
}

// Special opaque derivation function for ExternFunc
// Take look at sinfo_args to figure out the return StructInfo.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tvm.relax.struct_info.infer_by_sinfo_args",
                        [](const Call& call, const BlockBuilder& ctx) -> StructInfo {
                          ICHECK(call->sinfo_args.defined())
                              << "sinfo_args field of CallNode should always be defined";
                          if (call->sinfo_args.empty()) {
                            return ObjectStructInfo();
                          } else if (call->sinfo_args.size() == 1) {
                            return call->sinfo_args[0];
                          } else {
                            return TupleStructInfo(call->sinfo_args);
                          }
                        });
}

// Get the derive function.
FuncStructInfo GetExternFuncStructInfo() {
  EnvFunc fn = EnvFunc::Get("tvm.relax.struct_info.infer_by_sinfo_args");
  StructInfoDeriveFunc derive;
  derive = fn;
  return FuncStructInfo::OpaqueFunc(derive);
}

ExternFunc::ExternFunc(ffi::String global_symbol, Span span)
    : ExternFunc(global_symbol, GetExternFuncStructInfo(), span) {}

ExternFunc::ExternFunc(ffi::String global_symbol, StructInfo struct_info, Span span) {
  CHECK(struct_info.as<FuncStructInfoNode>())
      << "ExternFunc must have FuncStructInfo, "
      << "but declaration of '" << global_symbol << "' received " << struct_info;

  ObjectPtr<ExternFuncNode> n = ffi::make_object<ExternFuncNode>();
  n->global_symbol = std::move(global_symbol);
  n->span = span;
  n->struct_info_ = struct_info;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ExternFunc", [](ffi::String global_symbol,
                                               ffi::Optional<StructInfo> struct_info, Span span) {
    if (struct_info.defined()) {
      return ExternFunc(global_symbol, struct_info.value(), span);
    } else {
      return ExternFunc(global_symbol, span);
    }
  });
}

Expr GetShapeOf(const Expr& expr) {
  // default case, to be normalized.
  ICHECK(expr->struct_info_.defined()) << "GetShapeOf can only be applied to normalized expr";
  auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(expr);

  ICHECK(tinfo != nullptr) << "ShapeOf can only be applied to expr with TensorStructInfo";
  if (tinfo->shape.defined()) return tinfo->shape.value();

  static const Op& op = Op::Get("relax.shape_of");
  // default case, call shape of, eagerly normalize the expr.
  relax::Call call_shape_of(op, {expr}, {}, {});
  UpdateStructInfo(call_shape_of, ShapeStructInfo(tinfo->ndim));
  return call_shape_of;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.GetShapeOf", [](const Expr& expr) { return GetShapeOf(expr); })
      .def("relax.FuncWithAttr",
           [](BaseFunc func, ffi::String key, ObjectRef value) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttr(Downcast<relax::Function>(std::move(func)), key, value);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithAttrs",
           [](BaseFunc func, ffi::Map<ffi::String, ffi::Any> attr_map) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionNode>()) {
               return WithAttrs(Downcast<relax::Function>(std::move(func)), attr_map);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithoutAttr", [](BaseFunc func, ffi::String key) -> ffi::Optional<Function> {
        if (func->IsInstance<relax::FunctionNode>()) {
          return WithoutAttr(Downcast<relax::Function>(std::move(func)), key);
        }
        return std::nullopt;
      });
}

}  // namespace relax
}  // namespace tvm
