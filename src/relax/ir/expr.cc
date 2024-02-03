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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

using tvm::ReprPrinter;
using tvm::runtime::Optional;

TVM_REGISTER_NODE_TYPE(IdNode);

Id::Id(String name_hint) {
  ObjectPtr<IdNode> n = make_object<IdNode>();
  n->name_hint = std::move(name_hint);
  data_ = std::move(n);
}

Call::Call(Expr op, Array<Expr> args, Attrs attrs, Array<StructInfo> sinfo_args, Span span) {
  CHECK(!op->struct_info_.defined() || op->struct_info_->IsInstance<FuncStructInfoNode>())
      << "ValueError: "
      << "Call expects its operator to have FuncStructInfo, "
      << "but operator " << op << ", which was called with arguments " << args
      << ", has struct info " << op->struct_info_;

  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->sinfo_args = std::move(sinfo_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

Call WithFields(Call call, Optional<Expr> opt_op, Optional<Array<Expr>> opt_args,
                Optional<Attrs> opt_attrs, Optional<Array<StructInfo>> opt_sinfo_args,
                Optional<Span> opt_span) {
  // Collect new values for fields.
  Expr op = opt_op.value_or(call->op);
  Array<Expr> args = opt_args.value_or(call->args);
  Attrs attrs = opt_attrs.value_or(call->attrs);
  Array<StructInfo> sinfo_args = opt_sinfo_args.value_or(call->sinfo_args);
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

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relax.Call")
    .set_body_typed([](Expr op, Array<Expr> args, Attrs attrs, Array<StructInfo> sinfo_args,
                       Span span) { return Call(op, args, attrs, sinfo_args, span); });

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}

If WithFields(If if_expr, Optional<Expr> opt_cond, Optional<Expr> opt_true_branch,
              Optional<Expr> opt_false_branch, Optional<Span> opt_span) {
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

TVM_REGISTER_NODE_TYPE(IfNode);

TVM_REGISTER_GLOBAL("relax.If")
    .set_body_typed([](Expr cond, Expr true_branch, Expr false_branch, Span span) {
      return If(cond, true_branch, false_branch, span);
    });

Tuple::Tuple(tvm::Array<relay::Expr> fields, Span span) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleNode);

TVM_REGISTER_GLOBAL("relax.Tuple").set_body_typed([](tvm::Array<relay::Expr> fields, Span span) {
  return Tuple(fields, span);
});

Tuple WithFields(Tuple tuple, Optional<Array<Expr>> opt_fields, Optional<Span> opt_span) {
  Array<Expr> fields = opt_fields.value_or(tuple->fields);
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
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();

  if (auto* tuple_info = tuple->struct_info_.as<TupleStructInfoNode>()) {
    CHECK_LT(index, tuple_info->fields.size())
        << "Index out of bounds: Tuple " << tuple << " is of size " << tuple_info->fields.size()
        << ", and cannot be accessed with index " << index;
    auto sinfo = tuple_info->fields[index];
    n->struct_info_ = sinfo;
    n->checked_type_ = GetStaticType(sinfo);
  }
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem WithFields(TupleGetItem tuple_get_item, Optional<Expr> opt_tuple,
                        Optional<Integer> opt_index, Optional<Span> opt_span) {
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

TVM_REGISTER_NODE_TYPE(TupleGetItemNode);

TVM_REGISTER_GLOBAL("relax.TupleGetItem").set_body_typed([](Expr tuple, int index, Span span) {
  return TupleGetItem(tuple, index, span);
});

TVM_REGISTER_NODE_TYPE(ShapeExprNode);

ShapeExpr::ShapeExpr(Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeExprNode> n = make_object<ShapeExprNode>();

  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(DataType::Int(64), value);
    }
    ICHECK(value.dtype() == DataType::Int(64))
        << "the value in ShapeStructInfo can only have dtype of int64";
    return value;
  });
  n->span = span;
  n->checked_type_ = ShapeType(values.size());
  n->struct_info_ = ShapeStructInfo(values, span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ShapeExpr").set_body_typed([](Array<PrimExpr> values, Span span) {
  return ShapeExpr(values, span);
});

TVM_REGISTER_NODE_TYPE(VarNode);

Var::Var(Id vid, Optional<StructInfo> struct_info_annotation, Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  if (struct_info_annotation) {
    n->checked_type_ = GetStaticType(struct_info_annotation.value());
  }
  n->struct_info_ = std::move(struct_info_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.Var")
    .set_body_typed([](String name_hint, Optional<StructInfo> struct_info_annotation, Span span) {
      return Var(name_hint, struct_info_annotation, span);
    });

TVM_REGISTER_GLOBAL("relax.VarFromId")
    .set_body_typed([](Id vid, Optional<StructInfo> struct_info_annotation, Span span) {
      return Var(vid, struct_info_annotation, span);
    });

TVM_REGISTER_NODE_TYPE(DataflowVarNode);

DataflowVar::DataflowVar(Id vid, Optional<StructInfo> struct_info_annotation, Span span) {
  ObjectPtr<DataflowVarNode> n = make_object<DataflowVarNode>();
  n->vid = std::move(vid);
  if (struct_info_annotation) {
    n->checked_type_ = GetStaticType(struct_info_annotation.value());
  }
  n->struct_info_ = std::move(struct_info_annotation);
  n->span = std::move(span);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowVar")
    .set_body_typed([](String name_hint, Optional<StructInfo> struct_info_annotation, Span span) {
      return DataflowVar(name_hint, struct_info_annotation, span);
    });

TVM_REGISTER_GLOBAL("relax.DataflowVarFromId")
    .set_body_typed([](Id vid, Optional<StructInfo> struct_info_annotation, Span span) {
      return DataflowVar(vid, struct_info_annotation, span);
    });

Constant::Constant(runtime::NDArray data, Optional<StructInfo> struct_info_annotation, Span span) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);

  // set struct info.
  Array<PrimExpr> values;
  auto shape_tuple = n->data.Shape();
  for (size_t dim = 0; dim < shape_tuple.size(); ++dim) {
    values.push_back(IntImm(DataType::Int(64), shape_tuple[dim]));
  }
  if (struct_info_annotation.defined()) {
    n->struct_info_ = struct_info_annotation.value();
    n->checked_type_ = GetStaticType(struct_info_annotation.value());
  } else {
    TensorStructInfo tinfo(ShapeExpr(values), n->data.DataType(), VDevice(), span);
    n->struct_info_ = tinfo;
    n->checked_type_ = DynTensorType(tinfo->ndim, tinfo->dtype);
  }

  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ConstantNode);

TVM_REGISTER_GLOBAL("relax.Constant")
    .set_body_typed([](runtime::NDArray data, Optional<StructInfo> struct_info_annotation = NullOpt,
                       Span span = Span()) {
      return Constant(data, struct_info_annotation, span);
    });

PrimValue::PrimValue(PrimExpr value, Span span) {
  ObjectPtr<PrimValueNode> n = make_object<PrimValueNode>();
  n->checked_type_ = PrimType(value.dtype());
  n->struct_info_ = PrimStructInfo(value);
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

PrimValue PrimValue::Int64(int64_t value, Span span) {
  return PrimValue(IntImm(DataType::Int(64), value), span);
}

TVM_REGISTER_NODE_TYPE(PrimValueNode);

TVM_REGISTER_GLOBAL("relax.PrimValue").set_body_typed([](PrimExpr value, Span span) {
  return PrimValue(value, span);
});

StringImm::StringImm(String value, Span span) {
  ObjectPtr<StringImmNode> n = make_object<StringImmNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  // use the base structinfo for now
  // we can choose to introduce more fine-grained struct info later if necessary.
  n->checked_type_ = ObjectType();
  n->struct_info_ = ObjectStructInfo();
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(StringImmNode);

TVM_REGISTER_GLOBAL("relax.StringImm").set_body_typed([](String value, Span span) {
  return StringImm(value, span);
});

DataTypeImm::DataTypeImm(DataType value, Span span) {
  ObjectPtr<DataTypeImmNode> n = make_object<DataTypeImmNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  // use the base structinfo for now
  // we can choose to introduce more fine-grained struct info later if necessary.
  n->checked_type_ = ObjectType();
  n->struct_info_ = ObjectStructInfo();
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DataTypeImmNode);

TVM_REGISTER_GLOBAL("relax.DataTypeImm").set_body_typed([](DataType value, Span span) {
  return DataTypeImm(value, span);
});

TVM_REGISTER_NODE_TYPE(MatchCastNode);

MatchCast::MatchCast(Var var, Expr value, StructInfo struct_info, Span span) {
  ObjectPtr<MatchCastNode> n = make_object<MatchCastNode>();
  ICHECK(var.defined()) << "MatchCast requires var to be defined";
  n->var = std::move(var);
  n->value = std::move(value);
  n->struct_info = std::move(struct_info);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.MatchCast")
    .set_body_typed([](Var var, Expr value, StructInfo struct_info, Span span) {
      return MatchCast(var, value, struct_info, span);
    });

TVM_REGISTER_NODE_TYPE(VarBindingNode);

VarBinding::VarBinding(Var var, Expr value, Span span) {
  ObjectPtr<VarBindingNode> n = make_object<VarBindingNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.VarBinding").set_body_typed([](Var var, Expr value, Span span) {
  return VarBinding(var, value, span);
});

TVM_REGISTER_NODE_TYPE(BindingBlockNode);

BindingBlock::BindingBlock(Array<Binding> bindings, Span span) {
  ObjectPtr<BindingBlockNode> n = make_object<BindingBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.BindingBlock").set_body_typed([](Array<Binding> bindings, Span span) {
  return BindingBlock(bindings, span);
});

TVM_REGISTER_NODE_TYPE(DataflowBlockNode);

DataflowBlock::DataflowBlock(Array<Binding> bindings, Span span) {
  ObjectPtr<DataflowBlockNode> n = make_object<DataflowBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowBlock").set_body_typed([](Array<Binding> bindings, Span span) {
  return DataflowBlock(bindings, span);
});

TVM_REGISTER_NODE_TYPE(SeqExprNode);

SeqExpr::SeqExpr(Array<BindingBlock> blocks, Expr body, Span span) {
  ObjectPtr<SeqExprNode> n = make_object<SeqExprNode>();
  n->blocks = std::move(blocks);
  n->body = std::move(body);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.SeqExpr")
    .set_body_typed([](Array<BindingBlock> blocks, Expr body, Span span) {
      return SeqExpr(blocks, body, span);
    });

TVM_REGISTER_NODE_TYPE(FunctionNode);

Function::Function(Array<Var> params, Expr body, Optional<StructInfo> ret_struct_info, bool is_pure,
                   DictAttrs attrs, Span span) {
  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  Array<StructInfo> param_sinfo;

  for (const Var& param : params) {
    CHECK(param->struct_info_.defined())
        << "relax.Function requires params to contain struct_info_";
    param_sinfo.push_back(GetStructInfo(param));
  }

  Optional<StructInfo> body_sinfo;

  if (body->struct_info_.defined()) {
    body_sinfo = GetStructInfo(body);
  }

  if (ret_struct_info.defined()) {
    // allow body to override ret if body is more fine-grained.
    if (body_sinfo.defined()) {
      if (IsBaseOf(ret_struct_info.value(), body_sinfo.value())) {
        ret_struct_info = body_sinfo;
      }
    }
  } else {
    CHECK(body_sinfo.defined())
        << "Function do not have a return signature and body is not normalized";
    ret_struct_info = body_sinfo;
  }

  FuncStructInfo func_sinfo(param_sinfo, ret_struct_info.value(), is_pure);

  // set the fields
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_struct_info = std::move(ret_struct_info.value());
  n->is_pure = is_pure;
  n->checked_type_ = GetStaticType(func_sinfo);
  n->struct_info_ = std::move(func_sinfo);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.Function")
    .set_body_typed([](Array<Var> params, Expr body, Optional<StructInfo> ret_struct_info,
                       bool is_pure, DictAttrs attrs, Span span) {
      return Function(params, body, ret_struct_info, is_pure, attrs, span);
    });

Function Function::CreateEmpty(Array<Var> params, StructInfo ret_struct_info, bool is_pure,
                               DictAttrs attrs, Span span) {
  Array<StructInfo> param_sinfo;
  for (const Var& param : params) {
    ICHECK(param->checked_type_.defined())
        << "relax.Function requires params to contain checked_type_.";
    param_sinfo.push_back(GetStructInfo(param));
  }

  FuncStructInfo finfo(param_sinfo, ret_struct_info, is_pure);

  // set the fields
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = Expr();
  n->is_pure = is_pure;
  n->checked_type_ = GetStaticType(finfo);
  n->struct_info_ = std::move(finfo);
  n->ret_struct_info = std::move(ret_struct_info);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  return Function(std::move(n));
}

TVM_REGISTER_GLOBAL("relax.FunctionCreateEmpty")
    .set_body_typed([](Array<Var> params, StructInfo ret_struct_info, bool is_pure, DictAttrs attrs,
                       Span span) {
      return Function::CreateEmpty(params, ret_struct_info, is_pure, attrs, span);
    });

// Special opaque derivation function for ExternFunc
// Take look at sinfo_args to figure out the return StructInfo.
TVM_REGISTER_GLOBAL("tvm.relax.struct_info.infer_by_sinfo_args")
    .set_body_typed([](const Call& call, const BlockBuilder& ctx) -> StructInfo {
      ICHECK(call->sinfo_args.defined()) << "sinfo_args field of CallNode should always be defined";
      if (call->sinfo_args.empty()) {
        return ObjectStructInfo();
      } else if (call->sinfo_args.size() == 1) {
        return call->sinfo_args[0];
      } else {
        return TupleStructInfo(call->sinfo_args);
      }
    });

// Get the derive function.
FuncStructInfo GetExternFuncStructInfo() {
  EnvFunc fn = EnvFunc::Get("tvm.relax.struct_info.infer_by_sinfo_args");
  StructInfoDeriveFunc derive;
  derive = fn;
  return FuncStructInfo::OpaqueFunc(derive);
}

TVM_REGISTER_NODE_TYPE(ExternFuncNode);

ExternFunc::ExternFunc(String global_symbol, Span span) {
  ObjectPtr<ExternFuncNode> n = make_object<ExternFuncNode>();
  n->global_symbol = std::move(global_symbol);
  n->span = span;
  static auto sinfo = GetExternFuncStructInfo();
  n->struct_info_ = sinfo;
  n->checked_type_ = GetStaticType(sinfo);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ExternFunc").set_body_typed([](String global_symbol, Span span) {
  return ExternFunc(global_symbol, span);
});

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

TVM_REGISTER_GLOBAL("relax.GetShapeOf").set_body_typed([](const Expr& expr) {
  return GetShapeOf(expr);
});

TVM_REGISTER_GLOBAL("relax.FuncWithAttr")
    .set_body_typed([](BaseFunc func, String key, ObjectRef value) -> Optional<Function> {
      if (func->IsInstance<relax::FunctionNode>()) {
        return WithAttr(Downcast<relax::Function>(std::move(func)), key, value);
      }
      return NullOpt;
    });

TVM_REGISTER_GLOBAL("relax.FuncWithAttrs")
    .set_body_typed([](BaseFunc func, Map<String, ObjectRef> attr_map) -> Optional<Function> {
      if (func->IsInstance<relax::FunctionNode>()) {
        return WithAttrs(Downcast<relax::Function>(std::move(func)), attr_map);
      }
      return NullOpt;
    });

TVM_REGISTER_GLOBAL("relax.FuncWithoutAttr")
    .set_body_typed([](BaseFunc func, String key) -> Optional<Function> {
      if (func->IsInstance<relax::FunctionNode>()) {
        return WithoutAttr(Downcast<relax::Function>(std::move(func)), key);
      }
      return NullOpt;
    });

}  // namespace relax
}  // namespace tvm
