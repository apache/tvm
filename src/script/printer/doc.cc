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
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/logging.h>
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
  return CallDoc(GetRef<ExprDoc>(this), args, Array<String>(), Array<ExprDoc>());
}

ExprDoc ExprDocNode::Call(Array<ExprDoc, void> args, Array<String, void> kwargs_keys,
                          Array<ExprDoc, void> kwargs_values) const {
  return CallDoc(GetRef<ExprDoc>(this), args, kwargs_keys, kwargs_values);
}

ExprDoc ExprDoc::operator[](Array<Doc> indices) const { return (*get())[indices]; }

StmtBlockDoc::StmtBlockDoc(Array<StmtDoc> stmts) {
  ObjectPtr<StmtBlockDocNode> n = make_object<StmtBlockDocNode>();
  n->stmts = stmts;
  this->data_ = std::move(n);
}

LiteralDoc::LiteralDoc(ObjectRef value, const Optional<ObjectPath>& object_path) {
  ObjectPtr<LiteralDocNode> n = make_object<LiteralDocNode>();
  n->value = value;
  if (object_path.defined()) {
    n->source_paths.push_back(object_path.value());
  }
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

AssignDoc::AssignDoc(ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
  CHECK(rhs.defined() || annotation.defined())
      << "ValueError: At least one of rhs and annotation needs to be non-null for AssignDoc.";
  CHECK(lhs->IsInstance<IdDocNode>() || annotation == nullptr)
      << "ValueError: annotation can only be nonnull if lhs is an identifier.";

  ObjectPtr<AssignDocNode> n = make_object<AssignDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->annotation = annotation;
  this->data_ = std::move(n);
}

IfDoc::IfDoc(ExprDoc predicate, Array<StmtDoc> then_branch, Array<StmtDoc> else_branch) {
  CHECK(!then_branch.empty() || !else_branch.empty())
      << "ValueError: At least one of the then branch or else branch needs to be non-empty.";

  ObjectPtr<IfDocNode> n = make_object<IfDocNode>();
  n->predicate = predicate;
  n->then_branch = then_branch;
  n->else_branch = else_branch;
  this->data_ = std::move(n);
}

WhileDoc::WhileDoc(ExprDoc predicate, Array<StmtDoc> body) {
  ObjectPtr<WhileDocNode> n = make_object<WhileDocNode>();
  n->predicate = predicate;
  n->body = body;
  this->data_ = std::move(n);
}

ForDoc::ForDoc(ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ForDocNode> n = make_object<ForDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = make_object<ScopeDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(ExprDoc rhs, Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = make_object<ScopeDocNode>();
  n->lhs = NullOpt;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ExprStmtDoc::ExprStmtDoc(ExprDoc expr) {
  ObjectPtr<ExprStmtDocNode> n = make_object<ExprStmtDocNode>();
  n->expr = expr;
  this->data_ = std::move(n);
}

AssertDoc::AssertDoc(ExprDoc test, Optional<ExprDoc> msg) {
  ObjectPtr<AssertDocNode> n = make_object<AssertDocNode>();
  n->test = test;
  n->msg = msg;
  this->data_ = std::move(n);
}

ReturnDoc::ReturnDoc(ExprDoc value) {
  ObjectPtr<ReturnDocNode> n = make_object<ReturnDocNode>();
  n->value = value;
  this->data_ = std::move(n);
}

FunctionDoc::FunctionDoc(IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators,
                         Optional<ExprDoc> return_type, Array<StmtDoc> body) {
  ObjectPtr<FunctionDocNode> n = make_object<FunctionDocNode>();
  n->name = name;
  n->args = args;
  n->decorators = decorators;
  n->return_type = return_type;
  n->body = body;
  this->data_ = std::move(n);
}

ClassDoc::ClassDoc(IdDoc name, Array<ExprDoc> decorators, Array<StmtDoc> body) {
  ObjectPtr<ClassDocNode> n = make_object<ClassDocNode>();
  n->name = name;
  n->decorators = decorators;
  n->body = body;
  this->data_ = std::move(n);
}

CommentDoc::CommentDoc(String comment) {
  ObjectPtr<CommentDocNode> n = make_object<CommentDocNode>();
  n->comment = comment;
  this->data_ = std::move(n);
}

DocStringDoc::DocStringDoc(String docs) {
  ObjectPtr<DocStringDocNode> n = make_object<DocStringDocNode>();
  n->comment = docs;
  this->data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DocNode);
TVM_REGISTER_GLOBAL("script.printer.DocSetSourcePaths")
    .set_body_typed([](Doc doc, Array<ObjectPath> source_paths) {
      doc->source_paths = source_paths;
    });

TVM_REGISTER_NODE_TYPE(ExprDocNode);
TVM_REGISTER_GLOBAL("script.printer.ExprDocAttr")
    .set_body_method<ExprDoc, ExprDocNode, ExprDoc, String>(&ExprDocNode::Attr);
TVM_REGISTER_GLOBAL("script.printer.ExprDocIndex")
    .set_body_method<ExprDoc>(&ExprDocNode::operator[]);
TVM_REGISTER_GLOBAL("script.printer.ExprDocCall")
    .set_body_method<ExprDoc, ExprDocNode, ExprDoc, Array<ExprDoc>, Array<String>, Array<ExprDoc>>(
        &ExprDocNode::Call);

TVM_REGISTER_NODE_TYPE(StmtDocNode);
TVM_REGISTER_GLOBAL("script.printer.StmtDocSetComment")
    .set_body_typed([](StmtDoc doc, Optional<String> comment) { doc->comment = comment; });

TVM_REGISTER_NODE_TYPE(StmtBlockDocNode);
TVM_REGISTER_GLOBAL("script.printer.StmtBlockDoc").set_body_typed([](Array<StmtDoc> stmts) {
  return StmtBlockDoc(stmts);
});

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

TVM_REGISTER_NODE_TYPE(AssignDocNode);
TVM_REGISTER_GLOBAL("script.printer.AssignDoc")
    .set_body_typed([](ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
      return AssignDoc(lhs, rhs, annotation);
    });

TVM_REGISTER_NODE_TYPE(IfDocNode);
TVM_REGISTER_GLOBAL("script.printer.IfDoc")
    .set_body_typed([](ExprDoc predicate, Array<StmtDoc> then_branch, Array<StmtDoc> else_branch) {
      return IfDoc(predicate, then_branch, else_branch);
    });

TVM_REGISTER_NODE_TYPE(WhileDocNode);
TVM_REGISTER_GLOBAL("script.printer.WhileDoc")
    .set_body_typed([](ExprDoc predicate, Array<StmtDoc> body) {
      return WhileDoc(predicate, body);
    });

TVM_REGISTER_NODE_TYPE(ForDocNode);
TVM_REGISTER_GLOBAL("script.printer.ForDoc")
    .set_body_typed([](ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
      return ForDoc(lhs, rhs, body);
    });

TVM_REGISTER_NODE_TYPE(ScopeDocNode);
TVM_REGISTER_GLOBAL("script.printer.ScopeDoc")
    .set_body_typed([](Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body) {
      return ScopeDoc(lhs, rhs, body);
    });

TVM_REGISTER_NODE_TYPE(ExprStmtDocNode);
TVM_REGISTER_GLOBAL("script.printer.ExprStmtDoc").set_body_typed([](ExprDoc expr) {
  return ExprStmtDoc(expr);
});

TVM_REGISTER_NODE_TYPE(AssertDocNode);
TVM_REGISTER_GLOBAL("script.printer.AssertDoc")
    .set_body_typed([](ExprDoc test, Optional<ExprDoc> msg = NullOpt) {
      return AssertDoc(test, msg);
    });

TVM_REGISTER_NODE_TYPE(ReturnDocNode);
TVM_REGISTER_GLOBAL("script.printer.ReturnDoc").set_body_typed([](ExprDoc value) {
  return ReturnDoc(value);
});

TVM_REGISTER_NODE_TYPE(FunctionDocNode);
TVM_REGISTER_GLOBAL("script.printer.FunctionDoc")
    .set_body_typed([](IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators,
                       Optional<ExprDoc> return_type, Array<StmtDoc> body) {
      return FunctionDoc(name, args, decorators, return_type, body);
    });

TVM_REGISTER_NODE_TYPE(ClassDocNode);
TVM_REGISTER_GLOBAL("script.printer.ClassDoc")
    .set_body_typed([](IdDoc name, Array<ExprDoc> decorators, Array<StmtDoc> body) {
      return ClassDoc(name, decorators, body);
    });

TVM_REGISTER_NODE_TYPE(CommentDocNode);
TVM_REGISTER_GLOBAL("script.printer.CommentDoc").set_body_typed([](String comment) {
  return CommentDoc(comment);
});

TVM_REGISTER_NODE_TYPE(DocStringDocNode);
TVM_REGISTER_GLOBAL("script.printer.DocStringDoc").set_body_typed([](String docs) {
  return DocStringDoc(docs);
});

}  // namespace printer
}  // namespace script
}  // namespace tvm
