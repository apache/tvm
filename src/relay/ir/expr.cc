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
 * \file src/relay/ir/expr.cc
 * \brief The expression AST nodes of Relay.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

using tvm::ReprPrinter;
using namespace tvm::runtime;

Constant::Constant(runtime::NDArray data, Span span) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ConstantNode);

TVM_REGISTER_GLOBAL("relay.ir.Constant").set_body_typed([](runtime::NDArray data) {
  return Constant(data);
});

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
    shape.push_back(tvm::IntImm(DataType::Int(32), data->shape[i]));
  }

  return TensorType(shape, dtype);
}

Tuple::Tuple(tvm::Array<relay::Expr> fields, Span span) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleNode);

TVM_REGISTER_GLOBAL("relay.ir.Tuple").set_body_typed([](tvm::Array<relay::Expr> fields) {
  return Tuple(fields);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleNode*>(ref.get());
      p->stream << "Tuple(" << node->fields << ")";
    });

Var::Var(Id vid, Type type_annotation, Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(VarNode);

TVM_REGISTER_GLOBAL("relay.ir.Var").set_body_typed([](String str, Type type_annotation) {
  return Var(str, type_annotation);
});

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

Call::Call(Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args, Span span) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relay.ir.Call")
    .set_body_typed([](Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args) {
      return Call(op, args, attrs, type_args);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CallNode*>(ref.get());
      p->stream << "CallNode(" << node->op << ", " << node->args << ", " << node->attrs << ", "
                << node->type_args << ")";
    });

Let::Let(Var var, Expr value, Expr body, Span span) {
  ObjectPtr<LetNode> n = make_object<LetNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(LetNode);

TVM_REGISTER_GLOBAL("relay.ir.Let").set_body_typed([](Var var, Expr value, Expr body) {
  return Let(var, value, body);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const LetNode*>(ref.get());
      p->stream << "LetNode(" << node->var << ", " << node->value << ", " << node->body << ")";
    });

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(IfNode);

TVM_REGISTER_GLOBAL("relay.ir.If")
    .set_body_typed([](Expr cond, Expr true_branch, Expr false_branch) {
      return If(cond, true_branch, false_branch);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IfNode*>(ref.get());
      p->stream << "IfNode(" << node->cond << ", " << node->true_branch << ", "
                << node->false_branch << ")";
    });

TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleGetItemNode);

TVM_REGISTER_GLOBAL("relay.ir.TupleGetItem").set_body_typed([](Expr tuple, int index) {
  return TupleGetItem(tuple, index);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleGetItemNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleGetItemNode*>(ref.get());
      p->stream << "TupleGetItemNode(" << node->tuple << ", " << node->index << ")";
    });

RefCreate::RefCreate(Expr value, Span span) {
  ObjectPtr<RefCreateNode> n = make_object<RefCreateNode>();
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(RefCreateNode);

TVM_REGISTER_GLOBAL("relay.ir.RefCreate").set_body_typed([](Expr value) {
  return RefCreate(value);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefCreateNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefCreateNode*>(ref.get());
      p->stream << "RefCreateNode(" << node->value << ")";
    });

RefRead::RefRead(Expr ref, Span span) {
  ObjectPtr<RefReadNode> n = make_object<RefReadNode>();
  n->ref = std::move(ref);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(RefReadNode);

TVM_REGISTER_GLOBAL("relay.ir.RefRead").set_body_typed([](Expr ref) { return RefRead(ref); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefReadNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefReadNode*>(ref.get());
      p->stream << "RefReadNode(" << node->ref << ")";
    });

RefWrite::RefWrite(Expr ref, Expr value, Span span) {
  ObjectPtr<RefWriteNode> n = make_object<RefWriteNode>();
  n->ref = std::move(ref);
  n->value = std::move(value);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(RefWriteNode);

TVM_REGISTER_GLOBAL("relay.ir.RefWrite").set_body_typed([](Expr ref, Expr value) {
  return RefWrite(ref, value);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefWriteNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefWriteNode*>(ref.get());
      p->stream << "RefWriteNode(" << node->ref << ", " << node->value << ")";
    });

TVM_REGISTER_GLOBAL("relay.ir.TempExprRealize").set_body_typed([](TempExpr temp) {
  return temp->Realize();
});

TVM_REGISTER_GLOBAL("relay.ir.Any").set_body_typed([]() { return Any(); });

}  // namespace relay
}  // namespace tvm
