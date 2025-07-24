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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/doc.h>

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK({
  DocNode::RegisterReflection();
  ExprDocNode::RegisterReflection();
  StmtDocNode::RegisterReflection();
  StmtBlockDocNode::RegisterReflection();
  LiteralDocNode::RegisterReflection();
  IdDocNode::RegisterReflection();
  AttrAccessDocNode::RegisterReflection();
  IndexDocNode::RegisterReflection();
  CallDocNode::RegisterReflection();
  OperationDocNode::RegisterReflection();
  LambdaDocNode::RegisterReflection();
  TupleDocNode::RegisterReflection();
  ListDocNode::RegisterReflection();
  DictDocNode::RegisterReflection();
  SliceDocNode::RegisterReflection();
  AssignDocNode::RegisterReflection();
  IfDocNode::RegisterReflection();
  WhileDocNode::RegisterReflection();
  ForDocNode::RegisterReflection();
  ScopeDocNode::RegisterReflection();
  ExprStmtDocNode::RegisterReflection();
  AssertDocNode::RegisterReflection();
  ReturnDocNode::RegisterReflection();
  FunctionDocNode::RegisterReflection();
  ClassDocNode::RegisterReflection();
  CommentDocNode::RegisterReflection();
  DocStringDocNode::RegisterReflection();
});

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
  n->lhs = std::nullopt;
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
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.DocSetSourcePaths",
      [](Doc doc, Array<ObjectPath> source_paths) { doc->source_paths = source_paths; });
});

TVM_REGISTER_NODE_TYPE(ExprDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("script.printer.ExprDocAttr", &ExprDocNode::Attr)
      .def_method("script.printer.ExprDocIndex", &ExprDocNode::operator[])
      .def_method(
          "script.printer.ExprDocCall",
          [](ExprDoc doc, Array<ExprDoc> args, Array<String> kwargs_keys,
             Array<ExprDoc> kwargs_values) { return doc->Call(args, kwargs_keys, kwargs_values); });
});

TVM_REGISTER_NODE_TYPE(StmtDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.StmtDocSetComment",
                        [](StmtDoc doc, Optional<String> comment) { doc->comment = comment; });
});

TVM_REGISTER_NODE_TYPE(StmtBlockDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.StmtBlockDoc",
                        [](Array<StmtDoc> stmts) { return StmtBlockDoc(stmts); });
});

TVM_REGISTER_NODE_TYPE(LiteralDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.printer.LiteralDocNone", LiteralDoc::None)
      .def("script.printer.LiteralDocInt", LiteralDoc::Int)
      .def("script.printer.LiteralDocBoolean", LiteralDoc::Boolean)
      .def("script.printer.LiteralDocFloat", LiteralDoc::Float)
      .def("script.printer.LiteralDocStr", LiteralDoc::Str);
});

TVM_REGISTER_NODE_TYPE(IdDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IdDoc", [](String name) { return IdDoc(name); });
});

TVM_REGISTER_NODE_TYPE(AttrAccessDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.AttrAccessDoc",
                        [](ExprDoc value, String attr) { return AttrAccessDoc(value, attr); });
});

TVM_REGISTER_NODE_TYPE(IndexDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IndexDoc",
                        [](ExprDoc value, Array<Doc> indices) { return IndexDoc(value, indices); });
});

TVM_REGISTER_NODE_TYPE(CallDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.CallDoc", [](ExprDoc callee,             //
                                                     Array<ExprDoc> args,        //
                                                     Array<String> kwargs_keys,  //
                                                     Array<ExprDoc> kwargs_values) {
    return CallDoc(callee, args, kwargs_keys, kwargs_values);
  });
});

TVM_REGISTER_NODE_TYPE(OperationDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.OperationDoc", [](int32_t kind, Array<ExprDoc> operands) {
    return OperationDoc(OperationDocNode::Kind(kind), operands);
  });
});

TVM_REGISTER_NODE_TYPE(LambdaDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.LambdaDoc",
                        [](Array<IdDoc> args, ExprDoc body) { return LambdaDoc(args, body); });
});

TVM_REGISTER_NODE_TYPE(TupleDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.TupleDoc",
                        [](Array<ExprDoc> elements) { return TupleDoc(elements); });
});

TVM_REGISTER_NODE_TYPE(ListDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ListDoc",
                        [](Array<ExprDoc> elements) { return ListDoc(elements); });
});

TVM_REGISTER_NODE_TYPE(DictDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.DictDoc", [](Array<ExprDoc> keys, Array<ExprDoc> values) {
    return DictDoc(keys, values);
  });
});

TVM_REGISTER_NODE_TYPE(SliceDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.SliceDoc",
                        [](Optional<ExprDoc> start, Optional<ExprDoc> stop,
                           Optional<ExprDoc> step) { return SliceDoc(start, stop, step); });
});

TVM_REGISTER_NODE_TYPE(AssignDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.AssignDoc",
                        [](ExprDoc lhs, Optional<ExprDoc> rhs, Optional<ExprDoc> annotation) {
                          return AssignDoc(lhs, rhs, annotation);
                        });
});

TVM_REGISTER_NODE_TYPE(IfDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IfDoc", [](ExprDoc predicate, Array<StmtDoc> then_branch,
                                                   Array<StmtDoc> else_branch) {
    return IfDoc(predicate, then_branch, else_branch);
  });
});

TVM_REGISTER_NODE_TYPE(WhileDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.WhileDoc", [](ExprDoc predicate, Array<StmtDoc> body) {
    return WhileDoc(predicate, body);
  });
});

TVM_REGISTER_NODE_TYPE(ForDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ForDoc", [](ExprDoc lhs, ExprDoc rhs, Array<StmtDoc> body) {
    return ForDoc(lhs, rhs, body);
  });
});

TVM_REGISTER_NODE_TYPE(ScopeDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ScopeDoc",
                        [](Optional<ExprDoc> lhs, ExprDoc rhs, Array<StmtDoc> body) {
                          return ScopeDoc(lhs, rhs, body);
                        });
});

TVM_REGISTER_NODE_TYPE(ExprStmtDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ExprStmtDoc",
                        [](ExprDoc expr) { return ExprStmtDoc(expr); });
});

TVM_REGISTER_NODE_TYPE(AssertDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.AssertDoc",
      [](ExprDoc test, Optional<ExprDoc> msg = std::nullopt) { return AssertDoc(test, msg); });
});

TVM_REGISTER_NODE_TYPE(ReturnDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ReturnDoc", [](ExprDoc value) { return ReturnDoc(value); });
});

TVM_REGISTER_NODE_TYPE(FunctionDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.FunctionDoc",
                        [](IdDoc name, Array<AssignDoc> args, Array<ExprDoc> decorators,
                           Optional<ExprDoc> return_type, Array<StmtDoc> body) {
                          return FunctionDoc(name, args, decorators, return_type, body);
                        });
});

TVM_REGISTER_NODE_TYPE(ClassDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ClassDoc",
                        [](IdDoc name, Array<ExprDoc> decorators, Array<StmtDoc> body) {
                          return ClassDoc(name, decorators, body);
                        });
});

TVM_REGISTER_NODE_TYPE(CommentDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.CommentDoc",
                        [](String comment) { return CommentDoc(comment); });
});

TVM_REGISTER_NODE_TYPE(DocStringDocNode);
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.DocStringDoc",
                        [](String docs) { return DocStringDoc(docs); });
});

}  // namespace printer
}  // namespace script
}  // namespace tvm
