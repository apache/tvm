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

TVM_FFI_STATIC_INIT_BLOCK() {
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
}

ExprDoc ExprDocNode::Attr(ffi::String attr) const {
  return AttrAccessDoc(ffi::GetRef<ExprDoc>(this), attr);
}

ExprDoc ExprDocNode::operator[](ffi::Array<Doc> indices) const {
  return IndexDoc(ffi::GetRef<ExprDoc>(this), indices);
}

ExprDoc ExprDocNode::Call(ffi::Array<ExprDoc, void> args) const {
  return CallDoc(ffi::GetRef<ExprDoc>(this), args, ffi::Array<ffi::String>(),
                 ffi::Array<ExprDoc>());
}

ExprDoc ExprDocNode::Call(ffi::Array<ExprDoc, void> args, ffi::Array<ffi::String, void> kwargs_keys,
                          ffi::Array<ExprDoc, void> kwargs_values) const {
  return CallDoc(ffi::GetRef<ExprDoc>(this), args, kwargs_keys, kwargs_values);
}

ExprDoc ExprDoc::operator[](ffi::Array<Doc> indices) const { return (*get())[indices]; }

StmtBlockDoc::StmtBlockDoc(ffi::Array<StmtDoc> stmts) {
  ObjectPtr<StmtBlockDocNode> n = ffi::make_object<StmtBlockDocNode>();
  n->stmts = stmts;
  this->data_ = std::move(n);
}

LiteralDoc::LiteralDoc(ffi::Any value, const ffi::Optional<AccessPath>& object_path) {
  ObjectPtr<LiteralDocNode> n = ffi::make_object<LiteralDocNode>();
  n->value = value;
  if (object_path.defined()) {
    n->source_paths.push_back(object_path.value());
  }
  this->data_ = std::move(n);
}

IdDoc::IdDoc(ffi::String name) {
  ObjectPtr<IdDocNode> n = ffi::make_object<IdDocNode>();
  n->name = name;
  this->data_ = std::move(n);
}

AttrAccessDoc::AttrAccessDoc(ExprDoc value, ffi::String name) {
  ObjectPtr<AttrAccessDocNode> n = ffi::make_object<AttrAccessDocNode>();
  n->value = value;
  n->name = name;
  this->data_ = std::move(n);
}

IndexDoc::IndexDoc(ExprDoc value, ffi::Array<Doc> indices) {
  ObjectPtr<IndexDocNode> n = ffi::make_object<IndexDocNode>();
  n->value = value;
  n->indices = indices;
  this->data_ = std::move(n);
}

CallDoc::CallDoc(ExprDoc callee, ffi::Array<ExprDoc> args, ffi::Array<ffi::String> kwargs_keys,
                 ffi::Array<ExprDoc> kwargs_values) {
  ObjectPtr<CallDocNode> n = ffi::make_object<CallDocNode>();
  n->callee = callee;
  n->args = args;
  n->kwargs_keys = kwargs_keys;
  n->kwargs_values = kwargs_values;
  this->data_ = std::move(n);
}

OperationDoc::OperationDoc(OperationDocNode::Kind kind, ffi::Array<ExprDoc> operands) {
  ObjectPtr<OperationDocNode> n = ffi::make_object<OperationDocNode>();
  n->kind = kind;
  n->operands = operands;
  this->data_ = std::move(n);
}

LambdaDoc::LambdaDoc(ffi::Array<IdDoc> args, ExprDoc body) {
  ObjectPtr<LambdaDocNode> n = ffi::make_object<LambdaDocNode>();
  n->args = args;
  n->body = body;
  this->data_ = std::move(n);
}

TupleDoc::TupleDoc(ffi::Array<ExprDoc> elements) {
  ObjectPtr<TupleDocNode> n = ffi::make_object<TupleDocNode>();
  n->elements = elements;
  this->data_ = std::move(n);
}

ListDoc::ListDoc(ffi::Array<ExprDoc> elements) {
  ObjectPtr<ListDocNode> n = ffi::make_object<ListDocNode>();
  n->elements = elements;
  this->data_ = std::move(n);
}

DictDoc::DictDoc(ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values) {
  ObjectPtr<DictDocNode> n = ffi::make_object<DictDocNode>();
  n->keys = keys;
  n->values = values;
  this->data_ = std::move(n);
}

SliceDoc::SliceDoc(ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                   ffi::Optional<ExprDoc> step) {
  ObjectPtr<SliceDocNode> n = ffi::make_object<SliceDocNode>();
  n->start = start;
  n->stop = stop;
  n->step = step;
  this->data_ = std::move(n);
}

AssignDoc::AssignDoc(ExprDoc lhs, ffi::Optional<ExprDoc> rhs, ffi::Optional<ExprDoc> annotation) {
  CHECK(rhs.defined() || annotation.defined())
      << "ValueError: At least one of rhs and annotation needs to be non-null for AssignDoc.";
  CHECK(lhs->IsInstance<IdDocNode>() || annotation == nullptr)
      << "ValueError: annotation can only be nonnull if lhs is an identifier.";

  ObjectPtr<AssignDocNode> n = ffi::make_object<AssignDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->annotation = annotation;
  this->data_ = std::move(n);
}

IfDoc::IfDoc(ExprDoc predicate, ffi::Array<StmtDoc> then_branch, ffi::Array<StmtDoc> else_branch) {
  CHECK(!then_branch.empty() || !else_branch.empty())
      << "ValueError: At least one of the then branch or else branch needs to be non-empty.";

  ObjectPtr<IfDocNode> n = ffi::make_object<IfDocNode>();
  n->predicate = predicate;
  n->then_branch = then_branch;
  n->else_branch = else_branch;
  this->data_ = std::move(n);
}

WhileDoc::WhileDoc(ExprDoc predicate, ffi::Array<StmtDoc> body) {
  ObjectPtr<WhileDocNode> n = ffi::make_object<WhileDocNode>();
  n->predicate = predicate;
  n->body = body;
  this->data_ = std::move(n);
}

ForDoc::ForDoc(ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) {
  ObjectPtr<ForDocNode> n = ffi::make_object<ForDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = ffi::make_object<ScopeDocNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ScopeDoc::ScopeDoc(ExprDoc rhs, ffi::Array<StmtDoc> body) {
  ObjectPtr<ScopeDocNode> n = ffi::make_object<ScopeDocNode>();
  n->lhs = std::nullopt;
  n->rhs = rhs;
  n->body = body;
  this->data_ = std::move(n);
}

ExprStmtDoc::ExprStmtDoc(ExprDoc expr) {
  ObjectPtr<ExprStmtDocNode> n = ffi::make_object<ExprStmtDocNode>();
  n->expr = expr;
  this->data_ = std::move(n);
}

AssertDoc::AssertDoc(ExprDoc test, ffi::Optional<ExprDoc> msg) {
  ObjectPtr<AssertDocNode> n = ffi::make_object<AssertDocNode>();
  n->test = test;
  n->msg = msg;
  this->data_ = std::move(n);
}

ReturnDoc::ReturnDoc(ExprDoc value) {
  ObjectPtr<ReturnDocNode> n = ffi::make_object<ReturnDocNode>();
  n->value = value;
  this->data_ = std::move(n);
}

FunctionDoc::FunctionDoc(IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<ExprDoc> decorators,
                         ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body) {
  ObjectPtr<FunctionDocNode> n = ffi::make_object<FunctionDocNode>();
  n->name = name;
  n->args = args;
  n->decorators = decorators;
  n->return_type = return_type;
  n->body = body;
  this->data_ = std::move(n);
}

ClassDoc::ClassDoc(IdDoc name, ffi::Array<ExprDoc> decorators, ffi::Array<StmtDoc> body) {
  ObjectPtr<ClassDocNode> n = ffi::make_object<ClassDocNode>();
  n->name = name;
  n->decorators = decorators;
  n->body = body;
  this->data_ = std::move(n);
}

CommentDoc::CommentDoc(ffi::String comment) {
  ObjectPtr<CommentDocNode> n = ffi::make_object<CommentDocNode>();
  n->comment = comment;
  this->data_ = std::move(n);
}

DocStringDoc::DocStringDoc(ffi::String docs) {
  ObjectPtr<DocStringDocNode> n = ffi::make_object<DocStringDocNode>();
  n->comment = docs;
  this->data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.DocSetSourcePaths",
      [](Doc doc, ffi::Array<AccessPath> source_paths) { doc->source_paths = source_paths; });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("script.printer.ExprDocAttr", &ExprDocNode::Attr)
      .def_method("script.printer.ExprDocIndex", &ExprDocNode::operator[])
      .def_method("script.printer.ExprDocCall",
                  [](ExprDoc doc, ffi::Array<ExprDoc> args, ffi::Array<ffi::String> kwargs_keys,
                     ffi::Array<ExprDoc> kwargs_values) {
                    return doc->Call(args, kwargs_keys, kwargs_values);
                  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.StmtDocSetComment",
      [](StmtDoc doc, ffi::Optional<ffi::String> comment) { doc->comment = comment; });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.StmtBlockDoc",
                        [](ffi::Array<StmtDoc> stmts) { return StmtBlockDoc(stmts); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.printer.LiteralDocNone", LiteralDoc::None)
      .def("script.printer.LiteralDocInt", LiteralDoc::Int)
      .def("script.printer.LiteralDocBoolean", LiteralDoc::Boolean)
      .def("script.printer.LiteralDocFloat", LiteralDoc::Float)
      .def("script.printer.LiteralDocStr", LiteralDoc::Str);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IdDoc", [](ffi::String name) { return IdDoc(name); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.AttrAccessDoc",
                        [](ExprDoc value, ffi::String attr) { return AttrAccessDoc(value, attr); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.IndexDoc", [](ExprDoc value, ffi::Array<Doc> indices) {
    return IndexDoc(value, indices);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.CallDoc", [](ExprDoc callee,                       //
                                                     ffi::Array<ExprDoc> args,             //
                                                     ffi::Array<ffi::String> kwargs_keys,  //
                                                     ffi::Array<ExprDoc> kwargs_values) {
    return CallDoc(callee, args, kwargs_keys, kwargs_values);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.OperationDoc",
                        [](int32_t kind, ffi::Array<ExprDoc> operands) {
                          return OperationDoc(OperationDocNode::Kind(kind), operands);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.LambdaDoc",
                        [](ffi::Array<IdDoc> args, ExprDoc body) { return LambdaDoc(args, body); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.TupleDoc",
                        [](ffi::Array<ExprDoc> elements) { return TupleDoc(elements); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ListDoc",
                        [](ffi::Array<ExprDoc> elements) { return ListDoc(elements); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.DictDoc",
      [](ffi::Array<ExprDoc> keys, ffi::Array<ExprDoc> values) { return DictDoc(keys, values); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.SliceDoc",
                        [](ffi::Optional<ExprDoc> start, ffi::Optional<ExprDoc> stop,
                           ffi::Optional<ExprDoc> step) { return SliceDoc(start, stop, step); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.AssignDoc", [](ExprDoc lhs, ffi::Optional<ExprDoc> rhs,
                                                       ffi::Optional<ExprDoc> annotation) {
    return AssignDoc(lhs, rhs, annotation);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.IfDoc",
      [](ExprDoc predicate, ffi::Array<StmtDoc> then_branch, ffi::Array<StmtDoc> else_branch) {
        return IfDoc(predicate, then_branch, else_branch);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.WhileDoc", [](ExprDoc predicate, ffi::Array<StmtDoc> body) {
    return WhileDoc(predicate, body);
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.ForDoc",
      [](ExprDoc lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) { return ForDoc(lhs, rhs, body); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ScopeDoc",
                        [](ffi::Optional<ExprDoc> lhs, ExprDoc rhs, ffi::Array<StmtDoc> body) {
                          return ScopeDoc(lhs, rhs, body);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ExprStmtDoc",
                        [](ExprDoc expr) { return ExprStmtDoc(expr); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "script.printer.AssertDoc",
      [](ExprDoc test, ffi::Optional<ExprDoc> msg = std::nullopt) { return AssertDoc(test, msg); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ReturnDoc", [](ExprDoc value) { return ReturnDoc(value); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.FunctionDoc",
                        [](IdDoc name, ffi::Array<AssignDoc> args, ffi::Array<ExprDoc> decorators,
                           ffi::Optional<ExprDoc> return_type, ffi::Array<StmtDoc> body) {
                          return FunctionDoc(name, args, decorators, return_type, body);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.ClassDoc",
                        [](IdDoc name, ffi::Array<ExprDoc> decorators, ffi::Array<StmtDoc> body) {
                          return ClassDoc(name, decorators, body);
                        });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.CommentDoc",
                        [](ffi::String comment) { return CommentDoc(comment); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("script.printer.DocStringDoc",
                        [](ffi::String docs) { return DocStringDoc(docs); });
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
