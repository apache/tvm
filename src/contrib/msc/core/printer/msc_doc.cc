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
 * \file src/contrib/msc/core/printer/msc_doc.cc
 */

#include "msc_doc.h"

#include <utility>

namespace tvm {
namespace contrib {
namespace msc {

DeclareDoc::DeclareDoc(Optional<ExprDoc> type, ExprDoc variable, Array<ExprDoc> init_args,
                       bool use_constructor) {
  ObjectPtr<DeclareDocNode> n = make_object<DeclareDocNode>();
  n->type = type;
  n->variable = variable;
  n->init_args = init_args;
  n->use_constructor = use_constructor;
  this->data_ = std::move(n);
}

StrictListDoc::StrictListDoc(ListDoc list, bool allow_empty) {
  ObjectPtr<StrictListDocNode> n = make_object<StrictListDocNode>();
  n->list = list;
  n->allow_empty = allow_empty;
  this->data_ = std::move(n);
}

PointerDoc::PointerDoc(String name) {
  ObjectPtr<PointerDocNode> n = make_object<PointerDocNode>();
  n->name = name;
  this->data_ = std::move(n);
}

StructDoc::StructDoc(IdDoc name, Array<ExprDoc> decorators, Array<StmtDoc> body) {
  ObjectPtr<StructDocNode> n = make_object<StructDocNode>();
  n->name = name;
  n->decorators = decorators;
  n->body = body;
  this->data_ = std::move(n);
}

ConstructorDoc::ConstructorDoc(IdDoc name, Array<AssignDoc> args, Array<StmtDoc> body) {
  ObjectPtr<ConstructorDocNode> n = make_object<ConstructorDocNode>();
  n->name = name;
  n->args = args;
  n->body = body;
  this->data_ = std::move(n);
}

SwitchDoc::SwitchDoc(Array<ExprDoc> predicates, Array<Array<StmtDoc>> branchs,
                     Array<StmtDoc> default_branch) {
  ObjectPtr<SwitchDocNode> n = make_object<SwitchDocNode>();
  n->predicates = predicates;
  n->branchs = branchs;
  n->default_branch = default_branch;
  this->data_ = std::move(n);
}

LambdaDoc::LambdaDoc(IdDoc name, Array<AssignDoc> args, Array<ExprDoc> refs, Array<StmtDoc> body) {
  ObjectPtr<LambdaDocNode> n = make_object<LambdaDocNode>();
  n->name = name;
  n->args = args;
  n->refs = refs;
  n->body = body;
  this->data_ = std::move(n);
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
