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
#include <tvm/runtime/registry.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

ExprDoc ExprDocNode::Attr(String attr) const { return AttrAccessDoc(GetRef<ExprDoc>(this), attr); }

ExprDoc ExprDocNode::operator[](Array<Doc> indices) const {
  return IndexDoc(GetRef<ExprDoc>(this), indices);
}

ExprDoc ExprDocNode::Call(Array<ExprDoc, void> args) const {
  return CallDoc(GetRef<ExprDoc>(this), args, {}, {});
}

ExprDoc ExprDocNode::Call(Array<ExprDoc, void> args, Array<String, void> kwargs_keys,
                          Array<ExprDoc, void> kwargs_values) const {
  return CallDoc(GetRef<ExprDoc>(this), args, kwargs_keys, kwargs_values);
}

LiteralDoc::LiteralDoc(ObjectRef value) {
  ObjectPtr<LiteralDocNode> n = make_object<LiteralDocNode>();
  n->value = value;
  this->data_ = std::move(n);
}

IdDoc::IdDoc(String name) {
  ObjectPtr<IdDocNode> n = make_object<IdDocNode>();
  n->name = name;
  this->data_ = std::move(n);
}

AttrAccessDoc::AttrAccessDoc(ExprDoc value, String name) {
  ObjectPtr<AttrAccessDocNode> n = make_object<AttrAccessDocNode>();
  n->value = value;
  n->name = name;
  this->data_ = std::move(n);
}

IndexDoc::IndexDoc(ExprDoc value, Array<Doc> indices) {
  ObjectPtr<IndexDocNode> n = make_object<IndexDocNode>();
  n->value = value;
  n->indices = indices;
  this->data_ = std::move(n);
}

CallDoc::CallDoc(ExprDoc callee, Array<ExprDoc> args, Array<String> kwargs_keys,
                 Array<ExprDoc> kwargs_values) {
  ObjectPtr<CallDocNode> n = make_object<CallDocNode>();
  n->callee = callee;
  n->args = args;
  n->kwargs_keys = kwargs_keys;
  n->kwargs_values = kwargs_values;
  this->data_ = std::move(n);
}

OperationDoc::OperationDoc(OperationDocNode::Kind kind, Array<ExprDoc> operands) {
  ObjectPtr<OperationDocNode> n = make_object<OperationDocNode>();
  n->kind = kind;
  n->operands = operands;
  this->data_ = std::move(n);
}

LambdaDoc::LambdaDoc(Array<IdDoc> args, ExprDoc body) {
  ObjectPtr<LambdaDocNode> n = make_object<LambdaDocNode>();
  n->args = args;
  n->body = body;
  this->data_ = std::move(n);
}

TupleDoc::TupleDoc(Array<ExprDoc> elements) {
  ObjectPtr<TupleDocNode> n = make_object<TupleDocNode>();
  n->elements = elements;
  this->data_ = std::move(n);
}

ListDoc::ListDoc(Array<ExprDoc> elements) {
  ObjectPtr<ListDocNode> n = make_object<ListDocNode>();
  n->elements = elements;
  this->data_ = std::move(n);
}

DictDoc::DictDoc(Array<ExprDoc> keys, Array<ExprDoc> values) {
  ObjectPtr<DictDocNode> n = make_object<DictDocNode>();
  n->keys = keys;
  n->values = values;
  this->data_ = std::move(n);
}

SliceDoc::SliceDoc(Optional<ExprDoc> start, Optional<ExprDoc> stop, Optional<ExprDoc> step) {
  ObjectPtr<SliceDocNode> n = make_object<SliceDocNode>();
  n->start = start;
  n->stop = stop;
  n->step = step;
  this->data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DocNode);

TVM_REGISTER_NODE_TYPE(ExprDocNode);
TVM_REGISTER_GLOBAL("script.printer.ExprDocAttr").set_body_method<ExprDoc>(&ExprDocNode::Attr);
TVM_REGISTER_GLOBAL("script.printer.ExprDocIndex")
    .set_body_method<ExprDoc>(&ExprDocNode::operator[]);
TVM_REGISTER_GLOBAL("script.printer.ExprDocCall")
    .set_body_method<ExprDoc, ExprDocNode, ExprDoc, Array<ExprDoc>, Array<String>, Array<ExprDoc>>(
        &ExprDocNode::Call);

TVM_REGISTER_NODE_TYPE(LiteralDocNode);
TVM_REGISTER_GLOBAL("script.printer.LiteralDocNone").set_body_typed(LiteralDoc::None);
TVM_REGISTER_GLOBAL("script.printer.LiteralDocInt").set_body_typed(LiteralDoc::Int);
TVM_REGISTER_GLOBAL("script.printer.LiteralDocBoolean").set_body_typed(LiteralDoc::Boolean);
TVM_REGISTER_GLOBAL("script.printer.LiteralDocFloat").set_body_typed(LiteralDoc::Float);
TVM_REGISTER_GLOBAL("script.printer.LiteralDocStr").set_body_typed(LiteralDoc::Str);

TVM_REGISTER_NODE_TYPE(IdDocNode);
TVM_REGISTER_GLOBAL("script.printer.IdDoc").set_body_typed([](String name) { return IdDoc(name); });

TVM_REGISTER_NODE_TYPE(AttrAccessDocNode);
TVM_REGISTER_GLOBAL("script.printer.AttrAccessDoc").set_body_typed([](ExprDoc value, String attr) {
  return AttrAccessDoc(value, attr);
});

TVM_REGISTER_NODE_TYPE(IndexDocNode);
TVM_REGISTER_GLOBAL("script.printer.IndexDoc")
    .set_body_typed([](ExprDoc value, Array<Doc> indices) { return IndexDoc(value, indices); });

TVM_REGISTER_NODE_TYPE(CallDocNode);
TVM_REGISTER_GLOBAL("script.printer.CallDoc")
    .set_body_typed([](ExprDoc callee,             //
                       Array<ExprDoc> args,        //
                       Array<String> kwargs_keys,  //
                       Array<ExprDoc> kwargs_values) {
      return CallDoc(callee, args, kwargs_keys, kwargs_values);
    });

TVM_REGISTER_NODE_TYPE(OperationDocNode);
TVM_REGISTER_GLOBAL("script.printer.OperationDoc")
    .set_body_typed([](int32_t kind, Array<ExprDoc> operands) {
      return OperationDoc(OperationDocNode::Kind(kind), operands);
    });

TVM_REGISTER_NODE_TYPE(LambdaDocNode);
TVM_REGISTER_GLOBAL("script.printer.LambdaDoc").set_body_typed([](Array<IdDoc> args, ExprDoc body) {
  return LambdaDoc(args, body);
});

TVM_REGISTER_NODE_TYPE(TupleDocNode);
TVM_REGISTER_GLOBAL("script.printer.TupleDoc").set_body_typed([](Array<ExprDoc> elements) {
  return TupleDoc(elements);
});

TVM_REGISTER_NODE_TYPE(ListDocNode);
TVM_REGISTER_GLOBAL("script.printer.ListDoc").set_body_typed([](Array<ExprDoc> elements) {
  return ListDoc(elements);
});

TVM_REGISTER_NODE_TYPE(DictDocNode);
TVM_REGISTER_GLOBAL("script.printer.DictDoc")
    .set_body_typed([](Array<ExprDoc> keys, Array<ExprDoc> values) {
      return DictDoc(keys, values);
    });

TVM_REGISTER_NODE_TYPE(SliceDocNode);
TVM_REGISTER_GLOBAL("script.printer.SliceDoc")
    .set_body_typed([](Optional<ExprDoc> start, Optional<ExprDoc> stop, Optional<ExprDoc> step) {
      return SliceDoc(start, stop, step);
    });
}  // namespace printer
}  // namespace script
}  // namespace tvm
