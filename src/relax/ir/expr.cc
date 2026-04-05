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
#include <tvm/ffi/ir/text/ast.h>
#include <tvm/ffi/ir/text/printer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/type.h>

#include <unordered_set>

#include "script_print_utils.h"

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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  refl::GlobalDef()
      .def("relax._shape_args", [](ShapeExpr node) -> ffi::Array<ffi::ObjectRef> {
        ffi::Array<ffi::ObjectRef> wrapper;
        wrapper.push_back(node->values);
        return wrapper;
      })
      .def("relax._dtype_str", [](DataTypeImm node) -> ffi::String {
        DataType dt = node->value;
        return dt.is_void() ? ffi::String("void")
                            : ffi::DLDataTypeToString(static_cast<DLDataType>(dt));
      })
      .def("relax._dataflow_outputs",
          [](DataflowBlock block) -> ::tvm::ffi::Function {
            return ::tvm::ffi::Function::FromTyped(
                [](::tvm::ffi::ObjectRef obj, text::IRPrinter printer, text::DefaultFrame frame) {
                  DataflowBlock blk = ::tvm::Downcast<DataflowBlock>(obj);
                  ffi::List<text::ExprAST> outputs;
                  for (const auto& b : blk->bindings) {
                    if (!b->var->IsInstance<DataflowVarNode>()) {
                      ffi::Optional<text::ExprAST> var_expr = printer->VarGet(b->var);
                      if (var_expr.has_value()) {
                        outputs.push_back(var_expr.value());
                      }
                    }
                  }
                  if (!outputs.empty()) {
                    text::ExprAST callee = text::ExprAttr(text::IdAST("R"), "output");
                    frame->stmts.push_back(
                        text::ExprStmtAST(text::ExprCall(callee, std::move(outputs))));
                  }
                });
          });
}

Id::Id(ffi::String name_hint) {
  ObjectPtr<IdNode> n = ffi::make_object<IdNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

Call::Call(Expr op, ffi::Array<Expr> args, Attrs attrs, ffi::Array<StructInfo> sinfo_args,
           Span span) {
  TVM_FFI_CHECK(!op->struct_info_.defined() || op->struct_info_->IsInstance<FuncStructInfoNode>(),
                ValueError)
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
  TVM_FFI_ICHECK_GE(index, 0) << "Index out of bounds: Tuple " << tuple
                              << " cannot be accessed with negative index " << index;
  ObjectPtr<TupleGetItemNode> n = ffi::make_object<TupleGetItemNode>();

  if (auto* tuple_info = tuple->struct_info_.as<TupleStructInfoNode>()) {
    TVM_FFI_ICHECK_LT(index, tuple_info->fields.size())
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
    TVM_FFI_ICHECK(value.dtype() == DataType::Int(64))
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
  TVM_FFI_ICHECK(data_ != nullptr);
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
  TVM_FFI_ICHECK(var.defined()) << "MatchCast requires var to be defined";
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
  TVM_FFI_ICHECK(data_ != nullptr);
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
    TVM_FFI_ICHECK(param->struct_info_.defined())
        << "relax.Function requires params to contain struct_info_";
    param_sinfo.push_back(GetStructInfo(param));
  }

  ffi::Optional<StructInfo> body_sinfo;

  if (body->struct_info_.defined()) {
    body_sinfo = GetStructInfo(body);
  }

  TVM_FFI_ICHECK(body_sinfo.defined() || ret_struct_info.defined())
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
      std::unordered_set<tirx::Var> lookup(tir_vars.begin(), tir_vars.end());
      return [lookup = std::move(lookup)](const tirx::Var& var) -> ffi::Optional<PrimExpr> {
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
    TVM_FFI_ICHECK(param->struct_info_.defined())
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
                          TVM_FFI_ICHECK(call->sinfo_args.defined())
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
  TVM_FFI_ICHECK(struct_info.as<FuncStructInfoNode>())
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
  TVM_FFI_ICHECK(expr->struct_info_.defined())
      << "GetShapeOf can only be applied to normalized expr";
  auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(expr);

  TVM_FFI_ICHECK(tinfo != nullptr) << "ShapeOf can only be applied to expr with TensorStructInfo";
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

// ---- __ffi_text_print__ overrides ----

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // DataflowBlock: with R.dataflow(): bindings + R.output(...)
  // Must use __ffi_text_print__ (not trait) to preserve output vars in parent scope.
  // The With trait's FramePop removes all vars, causing the SeqExpr body return to fail.
  refl::TypeAttrDef<DataflowBlockNode>().def(
      "__ffi_text_print__",
      [](DataflowBlock node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        text::DefaultFrame frame;
        printer->FramePush(frame);
        ffi::List<text::StmtAST> body;
        for (int i = 0; i < static_cast<int>(node->bindings.size()); ++i) {
          text::NodeAST s = printer->operator()(ffi::Any(node->bindings[i]),
                                           path->Attr("bindings")->ArrayItem(i))
                          .cast<text::NodeAST>();
          if (auto* block = s.as<text::StmtBlockASTObj>()) {
            for (const auto& st : block->stmts) body.push_back(st);
          } else if (s->IsInstance<text::StmtASTObj>()) {
            body.push_back(Downcast<text::StmtAST>(s));
          } else if (s->IsInstance<text::ExprASTObj>()) {
            body.push_back(text::ExprStmtAST(Downcast<text::ExprAST>(s)));
          }
        }
        // R.output() for non-DataflowVar bindings
        ffi::List<text::ExprAST> outputs;
        for (const auto& b : node->bindings) {
          if (!b->var->IsInstance<DataflowVarNode>()) {
            if (auto var_doc = printer->VarGet(b->var)) {
              outputs.push_back(var_doc.value());
            }
          }
        }
        if (!outputs.empty()) {
          body.push_back(text::ExprStmtAST(text::ExprCall(Relax("output"), std::move(outputs))));
        }
        // Pop frame BUT re-register output vars in the PARENT frame so they're
        // accessible to the enclosing SeqExpr's return statement.
        ffi::List<ffi::Any> output_vars;
        for (const auto& b : node->bindings) {
          if (!b->var->IsInstance<DataflowVarNode>()) {
            output_vars.push_back(b->var);
          }
        }
        // Save the var info before popping
        ffi::Map<ffi::Any, text::VarInfo> saved_info;
        for (const auto& ov : output_vars) {
          auto it = printer->obj2info.find(ov);
          if (it != printer->obj2info.end()) {
            saved_info.Set(ov, (*it).second);
          }
        }
        printer->FramePop();
        // Re-register output vars in parent frame
        if (!printer->frames.empty()) {
          ffi::ObjectRef parent_frame = printer->frames.back().cast<ffi::ObjectRef>();
          for (const auto& kv : saved_info) {
            printer->obj2info.Set(kv.first, kv.second);
            // Add to parent frame's var list
            auto frame_vars = printer->frame_vars[parent_frame].cast<ffi::List<ffi::Any>>();
            frame_vars.push_back(kv.first);
          }
        }

        text::ExprAST ctx = text::ExprCall(Relax("dataflow"), {});
        return text::WithAST(ffi::Optional<text::ExprAST>(), ctx, body);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // relax.Function: @R.function with struct_info annotations and prologue
  refl::TypeAttrDef<FunctionNode>().def(
      "__ffi_text_print__",
      [](Function func, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        // Step 1. Determine function name
        ffi::String func_name = FindFuncName(func, printer, path);
        text::IdAST name = text::IdAST(func_name);
        bool at_top_level = AtTopLevelInModule(path);
        bool has_global_symbol = func->attrs.defined() &&
                                 func->attrs->dict.count("global_symbol");

        // Step 2. Build decorator: @R.function with optional kwargs
        ffi::List<ffi::String> dec_keys;
        ffi::List<text::ExprAST> dec_values;
        // pure=False when function is impure
        if (!func->is_pure) {
          dec_keys.push_back(ffi::String("pure"));
          dec_values.push_back(text::LiteralAST::Bool(false));
        }
        // private=True when at top level (or standalone) without global_symbol
        // V1 marks standalone functions and top-level module functions as private
        // when they lack a global_symbol attribute.
        bool is_standalone = !path->step.defined() || path->depth <= 1;
        if ((at_top_level || is_standalone) && !has_global_symbol) {
          dec_keys.push_back(ffi::String("private"));
          dec_values.push_back(text::LiteralAST::Bool(true));
        }
        ffi::List<text::ExprAST> decorators;
        if (dec_keys.size() > 0) {
          decorators.push_back(
              text::ExprCallKw(Relax("function"), {}, std::move(dec_keys), std::move(dec_values)));
        } else {
          decorators.push_back(Relax("function"));
        }

        // Step 3. Push frame, define symbolic TIR vars, then print params.
        text::DefaultFrame frame;
        printer->FramePush(frame);
        int n = func->params.size();

        // Step 3a. Collect and define symbolic TIR vars from ALL struct_info
        // in the function: params, match_cast bindings, and var bindings.
        // This ensures all TIR vars are defined at function level (matching V1).
        // Without this, TIR vars first seen inside a DataflowBlock's match_cast
        // would be scoped inside the dataflow block, causing undefined-var errors
        // when they're referenced outside (e.g., in the return statement).
        {
          std::vector<tirx::Var> tir_vars;
          std::unordered_set<const Object*> seen;
          // Collect from params
          for (int i = 0; i < n; ++i) {
            Var var = func->params[i];
            if (var->struct_info_.defined()) {
              StructInfo si = Downcast<StructInfo>(var->struct_info_.value());
              CollectTIRVarsFromStructInfo(si, &tir_vars, &seen);
            }
          }
          // Collect from body bindings (match_cast struct_info and var struct_info).
          // Include FuncStructInfo vars: inner functions may reference TIR vars
          // from the outer scope (e.g., symbolic shape dims shared between outer
          // and inner function params).  The `seen` set prevents duplicates.
          SeqExpr body_seq = Downcast<SeqExpr>(func->body);
          for (const auto& block : body_seq->blocks) {
            for (const auto& binding : block->bindings) {
              if (const auto* mc = binding.as<MatchCastNode>()) {
                CollectTIRVarsFromStructInfo(mc->struct_info, &tir_vars, &seen);
              }
              if (binding->var->struct_info_.defined()) {
                StructInfo si = Downcast<StructInfo>(binding->var->struct_info_.value());
                CollectTIRVarsFromStructInfo(si, &tir_vars, &seen);
              }
            }
          }
          // Also collect from the return expression's struct_info if applicable
          if (body_seq->body->struct_info_.defined()) {
            StructInfo si = Downcast<StructInfo>(body_seq->body->struct_info_.value());
            CollectTIRVarsFromStructInfo(si, &tir_vars, &seen);
          }
          for (const auto& tir_var : tir_vars) {
            // Skip vars with empty name_hint (synthetic vars from FuncStructInfo etc.)
            if (tir_var->name_hint.empty()) continue;
            if (!printer->VarGet(tir_var).has_value()) {
              printer->VarDef(tir_var->name_hint, tir_var, frame);
              text::ExprAST var_id = printer->VarGet(tir_var).value();
              std::string dtype_str = DType2Str(tir_var->dtype);
              // Match V1's PrintVarCreation: add is_size_var=True kwarg for SizeVar
              if (tir_var->IsInstance<tirx::SizeVarNode>()) {
                frame->stmts.push_back(text::AssignAST(
                    var_id,
                    text::ExprCallKw(TIR(dtype_str), {},
                               {ffi::String("is_size_var")}, {text::LiteralAST::Bool(true)}),
                    ffi::Optional<text::ExprAST>()));
              } else {
                frame->stmts.push_back(
                    text::AssignAST(var_id, text::ExprCall(TIR(dtype_str), {}), ffi::Optional<text::ExprAST>()));
              }
            }
          }
        }

        // Step 3b. Print params (tirx::Vars in struct_info shapes are now defined)
        // Enable stringify_vars for param annotations ONLY for top-level/standalone
        // functions. Inner (nested) functions should use resolved var references
        // since TIR vars are already in scope from the outer function (matching V1).
        bool should_stringify = at_top_level || is_standalone;
        g_printing_func_annotation = should_stringify;
        ffi::List<text::AssignAST> params;
        for (int i = 0; i < n; ++i) {
          Var var = func->params[i];
          text::AccessPath var_p = path->Attr("params")->ArrayItem(i);
          printer->VarDef(var->vid->name_hint, var, frame);
          text::ExprAST var_id = printer->VarGet(var).value();
          ffi::Optional<text::ExprAST> annotation;
          if (var->struct_info_.defined()) {
            annotation = Print(printer, var->struct_info_.value(),
                                var_p->Attr("struct_info_"));
          }
          params.push_back(text::AssignAST(var_id, ffi::Optional<text::ExprAST>(), annotation));
        }
        g_printing_func_annotation = false;

        // Step 4. Print attributes (filter global_symbol when it matches func name)
        // V1 filters global_symbol for top-level functions (both in-module and standalone)
        // when the symbol matches the function name being used.
        bool should_filter_global_symbol = has_global_symbol &&
            func->attrs->dict.at("global_symbol").cast<ffi::String>() == func_name &&
            (at_top_level || is_standalone);
        if (func->attrs.defined() && !func->attrs->dict.empty()) {
          if (should_filter_global_symbol) {
            // global_symbol matches func name: filter it out
            ffi::Map<ffi::String, ffi::Any> filtered;
            for (const auto& kv : func->attrs->dict) {
              if (kv.first != "global_symbol") {
                filtered.Set(kv.first, kv.second);
              }
            }
            if (!filtered.empty()) {
              frame->stmts.push_back(
                  text::ExprStmtAST(text::ExprCall(Relax("func_attr"),
                                       {Print(printer, DictAttrs(filtered),
                                              path->Attr("attrs"))})));
            }
          } else {
            frame->stmts.push_back(
                text::ExprStmtAST(text::ExprCall(Relax("func_attr"),
                                     {Print(printer, func->attrs, path->Attr("attrs"))})));
          }
        }

        // Step 5. Print body: inline SeqExpr handling.
        // We must NOT delegate to the SeqExpr Seq trait because the trait's
        // PrintBody/PrintSeq pushes its own frame for the blocks. When a
        // DataflowBlock's With trait FramePop removes vars defined inside
        // the block, the subsequent `ret` resolution can no longer find the
        // body Var, producing "Undefined variable: v".
        // Instead, we print each block inline and resolve the return var
        // while DataflowBlock vars are still in scope.
        ffi::List<text::StmtAST> body;
        SeqExpr seq = Downcast<SeqExpr>(func->body);
        for (int i = 0; i < static_cast<int>(seq->blocks.size()); ++i) {
          text::NodeAST block_ast = printer->operator()(
              ffi::Any(seq->blocks[i]),
              path->Attr("body")->Attr("blocks")->ArrayItem(i)).cast<text::NodeAST>();
          if (auto* sb = block_ast.as<text::StmtBlockASTObj>()) {
            for (const auto& s : sb->stmts) body.push_back(s);
          } else if (block_ast->IsInstance<text::StmtASTObj>()) {
            body.push_back(Downcast<text::StmtAST>(block_ast));
          }
        }
        // Resolve the return var BEFORE FramePop, while all block vars
        // (including DataflowBlock vars) are still accessible.
        text::ExprAST ret_expr = Print(printer, seq->body, path->Attr("body")->Attr("body"));
        body.push_back(text::ReturnAST(ret_expr));

        ffi::List<text::StmtAST> all_body;
        for (const auto& s : frame->stmts) all_body.push_back(s);
        for (const auto& s : body) all_body.push_back(s);

        // Step 6. Print return type from FuncStructInfo->ret (matching V1)
        // Must be done BEFORE FramePop so that TIR Vars defined in the
        // function scope (e.g. N, N_1) are still accessible for name resolution.
        // Only stringify vars for top-level/standalone functions (matching V1).
        ffi::Optional<text::ExprAST> ret_type;
        if (const auto* func_sinfo = func->struct_info_.as<FuncStructInfoNode>()) {
          g_printing_func_annotation = should_stringify;
          ret_type = Print(printer, func_sinfo->ret, path->Attr("struct_info_")->Attr("ret"));
          g_printing_func_annotation = false;
        }
        printer->FramePop();
        text::FunctionAST func_ast(name, params, decorators, ret_type, all_body);

        // When printing standalone (not in module context), add header comments
        // matching V1's HeaderWrapper behavior for relax functions.
        if (is_standalone && !at_top_level) {
          ffi::List<text::StmtAST> result;
          result.push_back(text::CommentAST(
              ffi::Optional<ffi::String>(ffi::String("from tvm.script import tirx as T"))));
          result.push_back(text::CommentAST(
              ffi::Optional<ffi::String>(ffi::String("from tvm.script import relax as R"))));
          result.push_back(text::CommentAST(ffi::Optional<ffi::String>()));
          result.push_back(func_ast);
          return text::StmtBlockAST(result);
        }
        return func_ast;
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // Tuple: (a, b, ...) or R.tuple() for empty
  refl::TypeAttrDef<TupleNode>().def(
      "__ffi_text_print__",
      [](Tuple node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        if (node->fields.empty()) {
          return text::ExprCall(Relax("tuple"), {});
        }
        ffi::List<text::ExprAST> elts;
        for (int i = 0; i < static_cast<int>(node->fields.size()); ++i) {
          elts.push_back(Print(printer, node->fields[i], path->Attr("fields")->ArrayItem(i)));
        }
        return text::TupleAST({}, std::move(elts));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // Constant: R.const(value, dtype) for scalars, metadata[...] placeholder for tensors
  // For DTensorStructInfo: R.dist.const(value, struct_info_ann)
  refl::TypeAttrDef<ConstantNode>().def(
      "__ffi_text_print__",
      [](Constant node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        if (ffi::Optional<text::ExprAST> s = SpecialScalar(node->data, path->Attr("data"))) {
          // Check if DTensorStructInfo -> use R.dist.const
          if (node->struct_info_.defined() &&
              node->struct_info_.value()
                  .as<distributed::DTensorStructInfoNode>()) {
            text::ExprAST ann = Print(printer, node->struct_info_.value(),
                                path->Attr("struct_info_"));
            return text::ExprCall(Relax("dist.const"), {s.value(), ann});
          }
          return text::ExprCall(Relax("const"),
                          {s.value(), text::LiteralAST::Str(DType2Str(DataType(node->data->dtype)))});
        }
        // Non-scalar: emit R.const(0, dtype) as a lossy placeholder.
        // V2 does not have a metadata registry like V1, so we cannot
        // faithfully round-trip non-scalar constants yet.
        return text::ExprCall(Relax("const"),
                        {text::LiteralAST::Int(0),
                         text::LiteralAST::Str(DType2Str(DataType(node->data->dtype)))});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // relax.If: __ffi_text_print__ override
  // Matches V1's PrintIfExpr: prints If with branches that use ExprStmt (not return).
  // When used standalone (not inside VarBinding), the branches just have ExprStmts.
  refl::TypeAttrDef<IfNode>().def(
      "__ffi_text_print__",
      [](If node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        text::ExprAST cond = Print(printer, node->cond, path->Attr("cond"));
        ffi::List<text::StmtAST> then_branch = PrintSeqExprBody(
            node->true_branch, path->Attr("true_branch"), printer);
        ffi::List<text::StmtAST> else_branch = PrintSeqExprBody(
            node->false_branch, path->Attr("false_branch"), printer);
        return text::IfAST(cond, then_branch, else_branch);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // relax.VarBinding: __ffi_text_print__ override
  // For If values in VarBinding, the Assign trait drops the assignment (returns IfAST
  // directly). To fix this, we register __ffi_text_print__ on VarBinding to intercept
  // If values and produce IfAST with assignments in branches. For non-If values,
  // we replicate the Assign trait behavior exactly.
  refl::TypeAttrDef<VarBindingNode>().def(
      "__ffi_text_print__",
      [](VarBinding node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        if (const auto* if_node = node->value.as<IfNode>()) {
          // --- If case: produce IfAST with assignment in each branch ---
          text::IdAST lhs = printer->VarDef(node->var->vid->name_hint, node->var,
                                       ffi::Optional<ffi::ObjectRef>{});
          ffi::Optional<text::ExprAST> ann;
          if (node->var->struct_info_.defined()) {
            ann = Print(printer, node->var->struct_info_.value(),
                        path->Attr("var")->Attr("struct_info_"));
          }
          text::ExprAST cond = Print(printer, if_node->cond, path->Attr("value")->Attr("cond"));
          auto make_branch = [&](const SeqExpr& seq, const text::AccessPath& seq_path)
              -> ffi::List<text::StmtAST> {
            ffi::List<text::StmtAST> stmts;
            for (int i = 0; i < static_cast<int>(seq->blocks.size()); ++i) {
              text::NodeAST block_ast = printer->operator()(
                  ffi::Any(seq->blocks[i]),
                  seq_path->Attr("blocks")->ArrayItem(i)).cast<text::NodeAST>();
              if (auto* sb = block_ast.as<text::StmtBlockASTObj>()) {
                for (const auto& s : sb->stmts) stmts.push_back(s);
              } else if (block_ast->IsInstance<text::StmtASTObj>()) {
                stmts.push_back(Downcast<text::StmtAST>(block_ast));
              }
            }
            text::ExprAST ret_expr = Print(printer, seq->body, seq_path->Attr("body"));
            stmts.push_back(text::AssignAST(lhs, ret_expr, ann));
            return stmts;
          };
          ffi::List<text::StmtAST> then_branch = make_branch(
              if_node->true_branch, path->Attr("value")->Attr("true_branch"));
          ffi::List<text::StmtAST> else_branch = make_branch(
              if_node->false_branch, path->Attr("value")->Attr("false_branch"));
          return text::IfAST(cond, then_branch, else_branch);
        }
        // --- Non-If case: replicate Assign trait behavior ---
        // Define LHS variable
        text::IdAST lhs = printer->VarDef(node->var->vid->name_hint, node->var,
                                     ffi::Optional<ffi::ObjectRef>{});
        // Type annotation from var's struct_info_ (matching V1 behavior).
        // V1 always emits struct_info annotations on intermediate relax vars.
        ffi::Optional<text::ExprAST> ann;
        if (node->var->struct_info_.defined()) {
          ann = Print(printer, node->var->struct_info_.value(),
                      path->Attr("var")->Attr("struct_info_"));
        }
        // Print RHS
        ffi::Any rhs_result = printer->operator()(ffi::Any(node->value), path->Attr("value"));
        text::NodeAST rhs_node = rhs_result.cast<text::NodeAST>();
        // Handle Function RHS
        if (auto* func = rhs_node.as<text::FunctionASTObj>()) {
          return text::FunctionAST(lhs, func->args, func->decorators, func->return_type, func->body);
        }
        // Normal expression RHS: produce AssignAST with struct_info annotation
        if (rhs_node->IsInstance<text::ExprASTObj>()) {
          return text::AssignAST(lhs, Downcast<text::ExprAST>(rhs_node), ann);
        }
        // Statement-level RHS (StmtBlock, etc.): return directly
        return rhs_node;
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // relax.MatchCast: __ffi_text_print__ override
  // Prints: m = T.int64() ; n = T.int64() ; _: type_ann = R.match_cast(value, struct_info)
  // This defines symbolic shape variables used later (e.g. n, m from R.Tensor((n,m), ...)).
  refl::TypeAttrDef<MatchCastNode>().def(
      "__ffi_text_print__",
      [](MatchCast node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        ffi::List<text::StmtAST> stmts;
        // Step 1. Collect TIR vars from the match struct_info and define them
        {
          std::vector<tirx::Var> tir_vars;
          std::unordered_set<const Object*> seen;
          CollectTIRVarsFromStructInfo(node->struct_info, &tir_vars, &seen);
          for (const auto& tir_var : tir_vars) {
            if (!printer->VarGet(tir_var).has_value()) {
              printer->VarDef(tir_var->name_hint, tir_var, ffi::Optional<ffi::ObjectRef>{});
              text::ExprAST var_id = printer->VarGet(tir_var).value();
              std::string dtype_str = DType2Str(tir_var->dtype);
              stmts.push_back(
                  text::AssignAST(var_id, text::ExprCall(TIR(dtype_str), {}), ffi::Optional<text::ExprAST>()));
            }
          }
        }
        // Step 2. Build RHS: R.match_cast(value, struct_info)
        text::ExprAST val = Print(printer, node->value, path->Attr("value"));
        text::ExprAST si = Print(printer, node->struct_info, path->Attr("struct_info"));
        text::ExprAST rhs = text::ExprCall(Relax("match_cast"), {val, si});
        // Step 3. Define LHS variable
        text::IdAST lhs = printer->VarDef(node->var->vid->name_hint, node->var,
                                     ffi::Optional<ffi::ObjectRef>{});
        // Type annotation from var's struct info
        ffi::Optional<text::ExprAST> ann;
        if (node->var->struct_info_.defined()) {
          ann = Print(printer, node->var->struct_info_.value(),
                      path->Attr("var")->Attr("struct_info_"));
        }
        stmts.push_back(text::AssignAST(lhs, rhs, ann));
        if (stmts.size() == 1) {
          return stmts[0];
        }
        return text::StmtBlockAST(std::move(stmts));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // relax.Call: __ffi_text_print__ override
  // Maps Op-based calls to R.name(args...) syntax matching V1.
  // Handles ExternFunc, Op, Var, and GlobalVar ops.
  // Special cases: assert_op, hint_on_device, to_vdevice, call_tir family.
  refl::TypeAttrDef<CallNode>().def(
      "__ffi_text_print__",
      [](Call call, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        text::ExprAST prefix(ffi::UnsafeInit{});
        ffi::List<text::ExprAST> args;
        ffi::List<ffi::String> kw_keys;
        ffi::List<text::ExprAST> kw_vals;

        // Determine the op name for special-case checks
        std::string op_name;
        if (const auto* op_node = call->op.as<OpNode>()) {
          op_name = op_node->name;
        }

        // Step 1. Determine callee prefix
        if (const auto* ef = call->op.as<ExternFuncNode>()) {
          prefix = Relax("call_packed");
          args.push_back(text::LiteralAST::Str(ef->global_symbol));
        } else if (const auto* op_node = call->op.as<OpNode>()) {
          std::string name = op_node->name;
          if (name.rfind("relax.", 0) == 0) {
            prefix = Relax(name.substr(6));
          } else {
            prefix = text::IdAST(name);
          }
        } else if (call->op->IsInstance<VarNode>()) {
          prefix = Print(printer, call->op, path->Attr("op"));
        } else if (call->op->IsInstance<GlobalVarNode>()) {
          // Check if the GlobalVar has been registered (e.g. from module prologue)
          if (auto bound = printer->VarGet(call->op)) {
            prefix = bound.value();
          } else {
            // Fallback: search by name_hint in obj2info (same as tirx Call printer)
            GlobalVar op_gv = Downcast<GlobalVar>(call->op);
            bool found = false;
            for (const auto& kv : printer->obj2info) {
              if (const auto* gv_node = kv.first.as<GlobalVarNode>()) {
                if (gv_node->name_hint == op_gv->name_hint) {
                  prefix = kv.second->creator().cast<text::ExprAST>();
                  found = true;
                  break;
                }
              }
            }
            if (!found) {
              prefix = Print(printer, call->op, path->Attr("op"));
            }
          }
        } else {
          prefix = Print(printer, call->op, path->Attr("op"));
        }

        // ---- Special case: assert_op ----
        // V1 prints: R.assert_op(cond, *format_args, format=format_str)
        // args[0]=cond, args[1]=format_str, args[2:]=format_args
        if (op_name == "relax.assert_op" && call->args.size() >= 2) {
          args.push_back(Print(printer, call->args[0], path->Attr("args")->ArrayItem(0)));
          text::ExprAST format_str = Print(printer, call->args[1], path->Attr("args")->ArrayItem(1));
          for (int i = 2, n = call->args.size(); i < n; ++i) {
            args.push_back(Print(printer, call->args[i], path->Attr("args")->ArrayItem(i)));
          }
          kw_keys.push_back(ffi::String("format"));
          kw_vals.push_back(format_str);
          return text::CallAST(prefix, std::move(args), std::move(kw_keys), std::move(kw_vals));
        }

        // ---- Special case: print ----
        // V1 prints: R.print(*format_args, format=format_str)
        // args[0]=format_str, args[1:]=format_args
        // The format string must be a keyword arg to avoid round-trip failure
        // (otherwise it's interpreted as a positional arg and a new default
        // format string is added).
        if (op_name == "relax.print" && call->args.size() >= 1) {
          text::ExprAST format_str = Print(printer, call->args[0], path->Attr("args")->ArrayItem(0));
          for (int i = 1, n = call->args.size(); i < n; ++i) {
            args.push_back(Print(printer, call->args[i], path->Attr("args")->ArrayItem(i)));
          }
          kw_keys.push_back(ffi::String("format"));
          kw_vals.push_back(format_str);
          return text::CallAST(prefix, std::move(args), std::move(kw_keys), std::move(kw_vals));
        }

        // ---- Special case: hint_on_device ----
        // V1 prints: R.hint_on_device(expr, R.device(device_type=N, index=M), memory_scope)
        if (op_name == "relax.hint_on_device") {
          args.push_back(Print(printer, call->args[0], path->Attr("args")->ArrayItem(0)));
          if (call->attrs.defined()) {
            if (const auto* attrs = call->attrs.as<HintOnDeviceAttrs>()) {
              ffi::List<ffi::String> dev_keys;
              ffi::List<text::ExprAST> dev_vals;
              dev_keys.push_back(ffi::String("device_type"));
              dev_vals.push_back(text::LiteralAST::Int(attrs->device_type));
              dev_keys.push_back(ffi::String("index"));
              dev_vals.push_back(text::LiteralAST::Int(attrs->index));
              args.push_back(text::ExprCallKw(Relax("device"), {}, std::move(dev_keys),
                                        std::move(dev_vals)));
              args.push_back(text::LiteralAST::Str(std::string(attrs->memory_scope)));
            }
          }
          return text::CallAST(prefix, std::move(args), {}, {});
        }

        // ---- Special case: to_vdevice ----
        // V1 prints: R.to_vdevice(expr, dst_vdevice="kind:index:scope")
        if (op_name == "relax.to_vdevice") {
          args.push_back(Print(printer, call->args[0], path->Attr("args")->ArrayItem(0)));
          if (call->attrs.defined()) {
            if (const auto* attrs = call->attrs.as<ToVDeviceAttrs>()) {
              VDevice vdev = attrs->dst_vdevice;
              kw_keys.push_back(ffi::String("dst_vdevice"));
              // Use the pre-registered VDevice string from module.cc (kind:index:scope)
              if (auto opt_str = printer->VarGet(vdev)) {
                kw_vals.push_back(opt_str.value());
              } else {
                // Fallback: compute from target info
                std::string dev_kind = vdev->target.defined()
                    ? std::string(vdev->target->kind->name)
                    : "unknown";
                kw_vals.push_back(text::LiteralAST::Str(
                    dev_kind + ":" + std::to_string(vdev->vdevice_id) + ":" +
                    std::string(vdev->memory_scope)));
              }
            }
          }
          return text::CallAST(prefix, std::move(args), std::move(kw_keys), std::move(kw_vals));
        }

        // ---- Special case: call_tir family ----
        // V1 prints: R.call_tir(callee, input_tuple, out_sinfo=..., [tir_vars=...])
        // Only args[0] (callee) and args[1] (input tuple) are positional.
        // args[2] (tir_vars) is a kwarg. out_sinfo comes from sinfo_args.
        {
          bool is_call_tir_family = (op_name == "relax.call_tir" ||
                                     op_name == "relax.call_tir_inplace" ||
                                     op_name == "relax.call_dps_packed" ||
                                     op_name == "relax.call_tir_with_grad" ||
                                     op_name == "relax.dist.call_tir_local_view");
          if (is_call_tir_family) {
            // Positional: callee (args[0]) and input tuple (args[1])
            args.push_back(Print(printer, call->args[0], path->Attr("args")->ArrayItem(0)));
            args.push_back(Print(printer, call->args[1], path->Attr("args")->ArrayItem(1)));
            // out_sinfo from sinfo_args[0]
            // Also detect if any out_sinfo is DTensorStructInfo to choose dist.call_tir
            bool is_dtensor = false;
            if (call->sinfo_args.size() > 0) {
              StructInfo o_sinfo = Downcast<StructInfo>(call->sinfo_args[0]);
              text::AccessPath o_sinfo_p = path->Attr("sinfo_args")->ArrayItem(0);
              kw_keys.push_back(ffi::String("out_sinfo"));
              if (const auto* o = o_sinfo.as<TupleStructInfoNode>()) {
                ffi::List<text::ExprAST> fields;
                text::AccessPath fields_p = o_sinfo_p->Attr("fields");
                for (int i = 0, l = o->fields.size(); i < l; ++i) {
                  if (o->fields[i].as<distributed::DTensorStructInfoNode>()) {
                    is_dtensor = true;
                  }
                  fields.push_back(Print(printer, o->fields[i], fields_p->ArrayItem(i)));
                }
                kw_vals.push_back(text::ListAST({}, std::move(fields)));
              } else {
                if (o_sinfo.as<distributed::DTensorStructInfoNode>()) {
                  is_dtensor = true;
                }
                kw_vals.push_back(Print(printer, o_sinfo, o_sinfo_p));
              }
            }
            // call_tir_inplace: inplace_indices kwarg
            if (op_name == "relax.call_tir_inplace") {
              if (const auto* attrs = call->attrs.as<CallTIRInplaceAttrs>()) {
                kw_keys.push_back(ffi::String("inplace_indices"));
                ffi::List<text::ExprAST> index_fields;
                for (const auto& idx : attrs->inplace_indices) {
                  index_fields.push_back(text::LiteralAST::Int(idx.IntValue()));
                }
                kw_vals.push_back(text::ListAST({}, std::move(index_fields)));
              }
            }
            // call_tir_with_grad: te_grad_name, te_grad_kwargs
            if (op_name == "relax.call_tir_with_grad") {
              if (const auto* attrs = call->attrs.as<CallTIRWithGradAttrs>()) {
                kw_keys.push_back(ffi::String("te_grad_name"));
                kw_vals.push_back(text::LiteralAST::Str(std::string(attrs->te_grad_name)));
                if (!attrs->te_grad_kwargs.empty()) {
                  kw_keys.push_back(ffi::String("te_grad_kwargs"));
                  kw_vals.push_back(Print(printer, attrs->te_grad_kwargs,
                                          path->Attr("attrs")->Attr("te_grad_kwargs")));
                }
              }
            }
            // tir_vars: args[2] as kwarg (not for call_dps_packed)
            if (call->args.size() >= 3 && op_name != "relax.call_dps_packed") {
              kw_keys.push_back(ffi::String("tir_vars"));
              kw_vals.push_back(Print(printer, call->args[2], path->Attr("args")->ArrayItem(2)));
            }
            // Choose the right call variant:
            // - dist.call_tir_local_view stays as is
            // - call_tir_with_grad stays as is
            // - call_dps_packed stays as is
            // - call_tir_inplace stays as is
            // - call_tir: if DTensor sinfo detected, use dist.call_tir instead
            if (op_name == "relax.dist.call_tir_local_view") {
              prefix = Relax("dist.call_tir_local_view");
            } else if (op_name == "relax.call_tir_with_grad") {
              prefix = Relax("call_tir_with_grad");
            } else if (op_name == "relax.call_dps_packed") {
              prefix = Relax("call_dps_packed");
            } else if (op_name == "relax.call_tir_inplace") {
              prefix = Relax("call_tir_inplace");
            } else if (is_dtensor) {
              prefix = Relax("dist.call_tir");
            } else {
              prefix = Relax("call_tir");
            }
            return text::CallAST(prefix, std::move(args), std::move(kw_keys), std::move(kw_vals));
          }
        }

        // Step 2. Print args (non-special-case ops)
        for (int i = 0, n = call->args.size(); i < n; ++i) {
          args.push_back(Print(printer, call->args[i], path->Attr("args")->ArrayItem(i)));
        }

        // Step 3. Print attrs as kwargs
        if (call->attrs.defined()) {
          if (call->op->IsInstance<ExternFuncNode>()) {
            kw_keys.push_back(ffi::String("attrs_type_key"));
            kw_vals.push_back(text::LiteralAST::Str(call->attrs->GetTypeKey()));
          }
          if (const auto* dict_attrs = call->attrs.as<DictAttrsNode>()) {
            // Sort attrs by key for deterministic output
            std::vector<std::pair<ffi::String, ffi::Any>> sorted;
            for (const auto& kv : dict_attrs->dict) {
              sorted.push_back(kv);
            }
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& kv : sorted) {
              kw_keys.push_back(kv.first);
              kw_vals.push_back(
                  Print(printer, kv.second, path->Attr("attrs")->Attr(kv.first)));
            }
          } else if (call->attrs.defined()) {
            // Non-DictAttrs: use reflection to iterate fields
            const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(call->attrs->type_index());
            ffi::reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo* fi) {
              ffi::String fname(fi->name.data, fi->name.size);
              if (fname == "span") return;  // skip span field
              ffi::Any field_val = ffi::reflection::FieldGetter(fi)(call->attrs);
              kw_keys.push_back(fname);
              // Special-case DataType fields: the raw DLDataType value doesn't
              // round-trip through the generic printer (it falls to "handle").
              // Convert to a string literal matching V1 behavior.
              if (field_val.type_index() == ffi::TypeIndex::kTVMFFIDataType) {
                DLDataType dt = field_val.cast<DLDataType>();
                kw_vals.push_back(text::LiteralAST::Str(DType2Str(runtime::DataType(dt))));
              } else {
                kw_vals.push_back(
                    Print(printer, std::move(field_val), path->Attr("attrs")->Attr(fname)));
              }
            });
          }
        }

        // Step 4. Print sinfo_args
        // (call_tir family already returned above, so no duplication here)
        if (call->sinfo_args.size() > 0) {
          text::AccessPath sinfo_p = path->Attr("sinfo_args");
          ffi::List<text::ExprAST> sinfo_docs;
          for (int i = 0, n = call->sinfo_args.size(); i < n; ++i) {
            sinfo_docs.push_back(Print(printer, call->sinfo_args[i], sinfo_p->ArrayItem(i)));
          }
          kw_keys.push_back(ffi::String("sinfo_args"));
          kw_vals.push_back(text::TupleAST({}, std::move(sinfo_docs)));
        }

        if (!kw_keys.empty()) {
          return text::CallAST(prefix, std::move(args), std::move(kw_keys), std::move(kw_vals));
        }
        return text::CallAST(prefix, std::move(args), {}, {});
      });
}

}  // namespace relax
}  // namespace tvm
