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
 * \file src/tvm/relay/ir/expr.cc
 * \brief The expression AST nodes of Relay.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

using tvm::ReprPrinter;
using namespace tvm::runtime;

Constant ConstantNode::make(runtime::NDArray data) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  return Constant(n);
}

TVM_REGISTER_NODE_TYPE(ConstantNode);

TVM_REGISTER_GLOBAL("relay._make.Constant")
.set_body_typed(ConstantNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ConstantNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const ConstantNode*>(ref.get());
    const PackedFunc* fprint = Registry::Get("relay._constant_repr");
    CHECK(fprint) << "unable to find printing function for constants";
    std::string data = (*fprint)(GetRef<Constant>(node));
    p->stream << "Constant(" << data << ")";
  });

TensorType ConstantNode::tensor_type() const {
  auto dtype = DataType(data->dtype);
  Array<tvm::PrimExpr> shape;
  for (int i = 0; i < data->ndim; i++) {
    CHECK_LE(data->shape[i], std::numeric_limits<int32_t>::max());
    CHECK_GE(data->shape[i], std::numeric_limits<int32_t>::min());
    shape.push_back(
        tvm::IntImm(DataType::Int(32), data->shape[i]));
  }

  return TensorType(shape, dtype);
}

Tuple TupleNode::make(tvm::Array<relay::Expr> fields) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  return Tuple(n);
}

TVM_REGISTER_NODE_TYPE(TupleNode);

TVM_REGISTER_GLOBAL("relay._make.Tuple")
.set_body_typed(TupleNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const TupleNode*>(ref.get());
    p->stream << "Tuple(" << node->fields << ")";
  });


Var VarNode::make(Id vid, Type type_annotation) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  return Var(n);
}

Var VarNode::make(std::string name_hint, Type type_annotation) {
  ObjectPtr<IdNode> n = make_object<IdNode>();
  n->name_hint = std::move(name_hint);
  return VarNode::make(Id(n), type_annotation);
}

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_REGISTER_GLOBAL("relay._make.Var")
.set_body_typed(static_cast<Var (*)(std::string, Type)>(VarNode::make));

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<VarNode>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const VarNode*>(ref.get());
    p->stream << "Var(" << node->name_hint();
    if (node->type_annotation.defined()) {
      p->stream << ", ty=";
      p->Print(node->type_annotation);
    }
    p->stream << ")";
  });

Function FunctionNode::make(tvm::Array<Var> params,
                            Expr body,
                            Type ret_type,
                            tvm::Array<TypeVar> type_params,
                            tvm::Attrs attrs) {
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  CHECK(params.defined());
  CHECK(type_params.defined());
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->type_params = std::move(type_params);
  n->attrs = std::move(attrs);
  return Function(n);
}

FuncType FunctionNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    Type param_type = (param->type_annotation.defined()) ? param->type_annotation
      : IncompleteType(Kind::kType);
    param_types.push_back(param_type);
  }

  Type ret_type = (this->ret_type.defined()) ? this->ret_type
    : IncompleteType(Kind::kType);
  return FuncType(param_types, ret_type, this->type_params, {});
}

bool FunctionNode::IsPrimitive() const {
  ObjectRef res = FunctionGetAttr(GetRef<Function>(this), attr::kPrimitive);
  const tir::IntImmNode* pval = res.as<tir::IntImmNode>();
  return pval && pval->value != 0;
}

bool FunctionNode::IsMarkedInline() const {
  ObjectRef res = FunctionGetAttr(GetRef<Function>(this), attr::kInline);
  const tir::IntImmNode* pval = res.as<tir::IntImmNode>();
  return pval && pval->value != 0;
}

Function FunctionNode::SetParams(const tvm::Map<Var, Constant>& parameters) const {
  return FunctionSetAttr(GetRef<Function>(this), attr::kParams, parameters);
}

TVM_REGISTER_GLOBAL("relay._expr.FunctionSetParams")
.set_body_typed(
  [](const Function& func, const tvm::Map<Var, Constant>& parameters) {
    return func->SetParams(parameters);
});

tvm::Map<Var, Constant> FunctionNode::GetParams() const {
  auto node_ref = FunctionGetAttr(GetRef<Function>(this), attr::kParams);
  return Downcast<tvm::Map<Var, Constant>>(node_ref);
}

TVM_REGISTER_GLOBAL("relay._expr.FunctionGetParams")
.set_body_typed([](const Function& func) {
  return func->GetParams();
});

bool FunctionNode::UseDefaultCompiler() const {
  ObjectRef res = FunctionGetAttr(GetRef<Function>(this), attr::kCompiler);
  const tir::StringImmNode* pval = res.as<tir::StringImmNode>();
  return pval == nullptr || pval->value == "default";
}

ObjectRef FunctionGetAttr(const Function& func, const std::string& key) {
  if (!func->attrs.defined()) { return ObjectRef(); }

  const DictAttrsNode* dict_attrs = func->attrs.as<DictAttrsNode>();
  CHECK(dict_attrs);
  auto it = dict_attrs->dict.find(key);
  if (it != dict_attrs->dict.end()) {
    return (*it).second;
  } else {
    return ObjectRef();
  }
}

Function FunctionSetAttr(const Function& func, const std::string& key, const ObjectRef& data) {
  const DictAttrsNode* dattrs = func->attrs.as<DictAttrsNode>();
  Attrs func_attrs;
  if (dattrs) {
    Map<std::string, ObjectRef> dict = dattrs->dict;
    dict.Set(key, data);
    func_attrs = DictAttrsNode::make(dict);
  } else {
    Map<std::string, ObjectRef> dict = {{key, data}};
    func_attrs = DictAttrsNode::make(dict);
  }

  return FunctionNode::make(
    func->params,
    func->body,
    func->ret_type,
    func->type_params,
    func_attrs);
}

TVM_REGISTER_NODE_TYPE(FunctionNode);

TVM_REGISTER_GLOBAL("relay._make.Function")
.set_body_typed(FunctionNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<FunctionNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const FunctionNode*>(ref.get());
  p->stream << "FunctionNode(" << node->params << ", " << node->ret_type
            << ", " << node->body << ", " << node->type_params << ", "
            << node->attrs << ")";
});

Call CallNode::make(Expr op, Array<Expr> args, Attrs attrs,
                    Array<Type> type_args) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  return Call(n);
}

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relay._make.Call")
.set_body_typed(CallNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<CallNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const CallNode*>(ref.get());
  p->stream << "CallNode(" << node->op << ", " << node->args << ", "
            << node->attrs << ", " << node->type_args << ")";
  });

Let LetNode::make(Var var, Expr value, Expr body) {
  ObjectPtr<LetNode> n = make_object<LetNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->body = std::move(body);
  return Let(n);
}

TVM_REGISTER_NODE_TYPE(LetNode);

TVM_REGISTER_GLOBAL("relay._make.Let")
.set_body_typed(LetNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LetNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const LetNode*>(ref.get());
  p->stream << "LetNode(" << node->var << ", " << node->value
            << ", " << node->body << ")";
});

If IfNode::make(Expr cond, Expr true_branch, Expr false_branch) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  return If(n);
}

TVM_REGISTER_NODE_TYPE(IfNode);

TVM_REGISTER_GLOBAL("relay._make.If")
.set_body_typed(IfNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<IfNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const IfNode*>(ref.get());
  p->stream << "IfNode(" << node->cond << ", " << node->true_branch
            << ", " << node->false_branch << ")";
});

TupleGetItem TupleGetItemNode::make(Expr tuple, int index) {
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  return TupleGetItem(n);
}

TVM_REGISTER_NODE_TYPE(TupleGetItemNode);

TVM_REGISTER_GLOBAL("relay._make.TupleGetItem")
.set_body_typed(TupleGetItemNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<TupleGetItemNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const TupleGetItemNode*>(ref.get());
  p->stream << "TupleGetItemNode(" << node->tuple << ", " << node->index << ")";
});

RefCreate RefCreateNode::make(Expr value) {
  ObjectPtr<RefCreateNode> n = make_object<RefCreateNode>();
  n->value = std::move(value);
  return RefCreate(n);
}

TVM_REGISTER_NODE_TYPE(RefCreateNode);

TVM_REGISTER_GLOBAL("relay._make.RefCreate")
.set_body_typed(RefCreateNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RefCreateNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const RefCreateNode*>(ref.get());
  p->stream << "RefCreateNode(" << node->value << ")";
});

RefRead RefReadNode::make(Expr ref) {
  ObjectPtr<RefReadNode> n = make_object<RefReadNode>();
  n->ref = std::move(ref);
  return RefRead(n);
}

TVM_REGISTER_NODE_TYPE(RefReadNode);

TVM_REGISTER_GLOBAL("relay._make.RefRead")
.set_body_typed(RefReadNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RefReadNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const RefReadNode*>(ref.get());
  p->stream << "RefReadNode(" << node->ref << ")";
});

RefWrite RefWriteNode::make(Expr ref, Expr value) {
  ObjectPtr<RefWriteNode> n = make_object<RefWriteNode>();
  n->ref = std::move(ref);
  n->value = std::move(value);
  return RefWrite(n);
}

TVM_REGISTER_NODE_TYPE(RefWriteNode);

TVM_REGISTER_GLOBAL("relay._make.RefWrite")
.set_body_typed(RefWriteNode::make);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RefWriteNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const RefWriteNode*>(ref.get());
  p->stream << "RefWriteNode(" << node->ref << ", " << node->value << ")";
});

TVM_REGISTER_GLOBAL("relay._expr.TempExprRealize")
.set_body_typed([](TempExpr temp) {
  return temp->Realize();
});

TVM_REGISTER_GLOBAL("relay._expr.FunctionSetAttr")
.set_body_typed(
  [](Function func, std::string name, ObjectRef ref) {
    return FunctionSetAttr(func, name, ref);
});

TVM_REGISTER_GLOBAL("relay._expr.FunctionGetAttr")
.set_body_typed(
  [](Function func, std::string name) {
    return FunctionGetAttr(func, name);
});

TVM_REGISTER_GLOBAL("relay._make.Any")
.set_body_typed([]() { return Any::make(); });

}  // namespace relay
}  // namespace tvm
