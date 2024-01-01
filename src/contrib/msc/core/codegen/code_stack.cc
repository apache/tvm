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
 * \file src/contrib/msc/core/codegen/code_stack.cc
 */

#include "code_stack.h"

namespace tvm {
namespace contrib {
namespace msc {

const Array<Doc> BaseStack::GetDocs() const {
  ICHECK(blocks_.size() == 1) << "Has incomplete blocks, please check";
  return TopBlock();
}

void BaseStack::Line(const Doc& doc) { PushDoc(doc); }

void BaseStack::Line(const String& line) { Line(IdDoc(line)); }

void BaseStack::Comment(const String& comment, bool attach) {
  if (attach) {
    const auto& doc = TopDoc();
    ICHECK(doc->IsInstance<StmtDocNode>()) << "Only stmt doc support attach comments";
    const auto& stmt = Downcast<StmtDoc>(doc);
    stmt->comment = comment;
  } else {
    PushDoc(CommentDoc(comment));
  }
}

void BaseStack::Declare(const String& type, const String& variable, size_t len,
                        bool use_constructor) {
  PushDoc(DocUtils::ToDeclare(type, variable, len, use_constructor));
}

void BaseStack::DeclareArgBase(const ExprDoc& value) {
  const auto& declare = PopCheckedDoc<DeclareDoc, DeclareDocNode>();
  Array<ExprDoc> init_args = declare->init_args;
  init_args.push_back(value);
  PushDoc(DeclareDoc(declare->type, declare->variable, init_args, declare->use_constructor));
}

void BaseStack::FuncDef(const String& func_name, const String& ret_type) {
  if (ret_type.size() > 0) {
    PushDoc(FunctionDoc(IdDoc(func_name), Array<AssignDoc>(), Array<ExprDoc>(), IdDoc(ret_type),
                        Array<StmtDoc>()));
  } else {
    PushDoc(FunctionDoc(IdDoc(func_name), Array<AssignDoc>(), Array<ExprDoc>(), NullOpt,
                        Array<StmtDoc>()));
  }
}

void BaseStack::FuncArg(const String& arg, const String& annotation, const String& value) {
  const auto& func = PopCheckedDoc<FunctionDoc, FunctionDocNode>();
  Array<AssignDoc> args = func->args;
  args.push_back(DocUtils::ToAssign(arg, value, annotation));
  PushDoc(FunctionDoc(func->name, args, func->decorators, func->return_type, func->body));
}

void BaseStack::FuncDecorator(const String& decorator) {
  const auto& func = PopCheckedDoc<FunctionDoc, FunctionDocNode>();
  Array<ExprDoc> decorators = func->decorators;
  decorators.push_back(IdDoc(decorator));
  PushDoc(FunctionDoc(func->name, func->args, decorators, func->return_type, func->body));
}

void BaseStack::FuncStart() {
  ICHECK(TopDoc()->IsInstance<FunctionDocNode>()) << "FunctionDoc is not saved";
  BlockStart();
}

void BaseStack::FuncEnd() {
  const auto& block = PopBlock();
  const auto& func = PopCheckedDoc<FunctionDoc, FunctionDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(FunctionDoc(func->name, func->args, func->decorators, func->return_type, body));
}

void BaseStack::ClassDef(const String& class_name) {
  PushDoc(ClassDoc(IdDoc(class_name), Array<ExprDoc>(), Array<StmtDoc>()));
}

void BaseStack::ClassDecorator(const String& decorator) {
  const auto& class_doc = PopCheckedDoc<ClassDoc, ClassDocNode>();
  Array<ExprDoc> decorators = class_doc->decorators;
  decorators.push_back(IdDoc(decorator));
  PushDoc(ClassDoc(class_doc->name, decorators, class_doc->body));
}

void BaseStack::ClassStart() {
  ICHECK(TopDoc()->IsInstance<ClassDocNode>()) << "ClassDoc is not saved";
  BlockStart();
}

void BaseStack::ClassEnd() {
  const auto& block = PopBlock();
  const auto& class_doc = PopCheckedDoc<ClassDoc, ClassDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(ClassDoc(class_doc->name, class_doc->decorators, body));
}

void BaseStack::StructStart(const String& struct_name) {
  PushDoc(StructDoc(IdDoc(struct_name), Array<ExprDoc>(), Array<StmtDoc>()));
  BlockStart();
}

void BaseStack::StructEnd() {
  const auto& block = PopBlock();
  const auto& struct_doc = PopCheckedDoc<StructDoc, StructDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(StructDoc(struct_doc->name, struct_doc->decorators, body));
}

void BaseStack::ConstructorDef(const String& constructor_name) {
  PushDoc(ConstructorDoc(IdDoc(constructor_name), Array<AssignDoc>(), Array<StmtDoc>()));
}

void BaseStack::ConstructorArg(const String& arg, const String& annotation, const String& value) {
  const auto& func = PopCheckedDoc<ConstructorDoc, ConstructorDocNode>();
  Array<AssignDoc> args = func->args;
  args.push_back(DocUtils::ToAssign(arg, value, annotation));
  PushDoc(ConstructorDoc(func->name, args, func->body));
}

void BaseStack::ConstructorStart() {
  ICHECK(TopDoc()->IsInstance<ConstructorDocNode>()) << "ConstructorDoc is not saved";
  BlockStart();
}

void BaseStack::ConstructorEnd() {
  const auto& block = PopBlock();
  const auto& func = PopCheckedDoc<ConstructorDoc, ConstructorDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(ConstructorDoc(func->name, func->args, body));
}

void BaseStack::LambdaDef(const String& lambda_name) {
  PushDoc(LambdaDoc(IdDoc(lambda_name), Array<AssignDoc>(), Array<ExprDoc>(), Array<StmtDoc>()));
}

void BaseStack::LambdaArg(const String& arg, const String& annotation, const String& value) {
  const auto& lambda = PopCheckedDoc<LambdaDoc, LambdaDocNode>();
  Array<AssignDoc> args = lambda->args;
  args.push_back(DocUtils::ToAssign(arg, value, annotation));
  PushDoc(LambdaDoc(lambda->name, args, lambda->refs, lambda->body));
}

void BaseStack::LambdaRef(const String& ref) {
  const auto& lambda = PopCheckedDoc<LambdaDoc, LambdaDocNode>();
  Array<ExprDoc> refs = lambda->refs;
  refs.push_back(IdDoc(ref));
  PushDoc(LambdaDoc(lambda->name, lambda->args, refs, lambda->body));
}

void BaseStack::LambdaStart() {
  ICHECK(TopDoc()->IsInstance<LambdaDocNode>()) << "LambdaDoc is not saved";
  BlockStart();
}

void BaseStack::LambdaEnd(const String& ret_val) {
  if (ret_val.size() > 0) {
    PushDoc(ReturnDoc(IdDoc(ret_val)));
  }
  const auto& block = PopBlock();
  const auto& lambda = PopCheckedDoc<LambdaDoc, LambdaDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(LambdaDoc(lambda->name, lambda->args, lambda->refs, body));
}

void BaseStack::LambdaEnd(const ExprDoc& ret_val) {
  PushDoc(ReturnDoc(ret_val));
  LambdaEnd("");
}

void BaseStack::FuncCall(const String& callee, Optional<ExprDoc> assign_to,
                         Optional<ExprDoc> caller) {
  if (!caller.defined()) {
    PushDoc(CallDoc(IdDoc(callee), Array<ExprDoc>(), Array<String>(), Array<ExprDoc>()));
  } else {
    const auto& new_access = AttrAccessDoc(caller.value(), callee);
    PushDoc(CallDoc(new_access, Array<ExprDoc>(), Array<String>(), Array<ExprDoc>()));
  }
  if (assign_to.defined()) {
    const auto& last_call = PopCheckedDoc<CallDoc, CallDocNode>();
    if (assign_to.value()->IsInstance<DeclareDocNode>()) {
      const auto& declare = Downcast<DeclareDoc>(assign_to.value());
      PushDoc(AssignDoc(declare->variable, last_call, declare->type));
    } else {
      const auto& declare = DocUtils::ToDeclare("", assign_to.value());
      PushDoc(AssignDoc(declare->variable, last_call, declare->type));
    }
  }
}

void BaseStack::FuncCall(const String& callee, const String& assign_to, const String& caller) {
  Optional<ExprDoc> assign_doc;
  if (assign_to.size() == 0) {
    assign_doc = NullOpt;
  } else {
    assign_doc = IdDoc(assign_to);
  }
  Optional<ExprDoc> caller_doc;
  if (caller.size() == 0) {
    caller_doc = NullOpt;
  } else {
    caller_doc = IdDoc(caller);
  }
  FuncCall(callee, assign_doc, caller_doc);
}

void BaseStack::MethodCall(const String& callee, bool new_line) {
  const auto& host = PopDoc();
  if (host->IsInstance<ExprDocNode>()) {
    const auto& v_callee = callee + (new_line ? DocSymbol::NextLine() : "");
    FuncCall(v_callee, NullOpt, Downcast<ExprDoc>(host));
  } else if (const auto* a_node = host.as<AssignDocNode>()) {
    ICHECK(a_node->rhs.defined()) << "Can not find rhs for inplace host";
    FuncCall(callee, DeclareDoc(a_node->annotation, a_node->lhs, Array<ExprDoc>(), true),
             a_node->rhs);
  } else {
    LOG(FATAL) << "Unexpected host type for inplace " << host->GetTypeKey();
  }
}

void BaseStack::InplaceStart(const String& callee, Optional<ExprDoc> assign_to,
                             Optional<ExprDoc> caller) {
  FuncCall(callee, assign_to, caller);
}

void BaseStack::InplaceStart(const String& callee, const String& assign_to, const String& caller) {
  FuncCall(callee, assign_to, caller);
}

void BaseStack::InplaceEnd() {
  const auto& last = PopDoc();
  // get args and kwargs
  if (last->IsInstance<CallDocNode>()) {
    CallArgBase(Downcast<CallDoc>(last));
  } else if (const auto* assign = last.as<AssignDocNode>()) {
    const auto& call = Downcast<CallDoc>(assign->rhs);
    ICHECK(assign->lhs->IsInstance<IdDocNode>())
        << "assign lhs should be IdDoc, get " << assign->lhs->GetTypeKey();
    const auto& key = Downcast<IdDoc>(assign->lhs)->name;
    CallArgBase(call, key);
  } else {
    LOG(FATAL) << "Unexpected last type for call arg " << last->GetTypeKey();
  }
}

void BaseStack::PopNest(const String& key) {
  const auto& last = PopDoc();
  if (last->IsInstance<CallDocNode>()) {
    CallArgBase(Downcast<CallDoc>(last), key);
  } else {
    LOG(FATAL) << "Unexpected nest type " << last->GetTypeKey();
  }
}

void BaseStack::CallArgBase(const ExprDoc& value, const String& key) {
  const auto& last = PopDoc();
  Array<ExprDoc> args;
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  // get args and kwargs
  if (const auto* call = last.as<CallDocNode>()) {
    args = call->args;
    kwargs_keys = call->kwargs_keys;
    kwargs_values = call->kwargs_values;
  } else if (const auto* assign = last.as<AssignDocNode>()) {
    const auto& call = Downcast<CallDoc>(assign->rhs);
    args = call->args;
    kwargs_keys = call->kwargs_keys;
    kwargs_values = call->kwargs_values;
  } else {
    LOG(FATAL) << "Unexpected last type for call arg " << last->GetTypeKey();
  }
  // push args or kwargs
  if (key.size() == 0) {
    ICHECK(kwargs_keys.size() == 0) << "kwargs followed by args " << value;
    args.push_back(value);
  } else {
    kwargs_keys.push_back(key);
    kwargs_values.push_back(value);
  }
  // push doc
  if (const auto* call = last.as<CallDocNode>()) {
    PushDoc(CallDoc(call->callee, args, kwargs_keys, kwargs_values));
  } else if (const auto* assign = last.as<AssignDocNode>()) {
    const auto& call = Downcast<CallDoc>(assign->rhs);
    const auto& new_call = CallDoc(call->callee, args, kwargs_keys, kwargs_values);
    PushDoc(AssignDoc(assign->lhs, new_call, assign->annotation));
  } else {
    LOG(FATAL) << "Unexpected last type for call arg " << last->GetTypeKey();
  }
}

void BaseStack::ConditionIf(const String& predicate) {
  Array<StmtDoc> else_branch{ExprStmtDoc(IdDoc("pass"))};
  PushDoc(IfDoc(IdDoc(predicate), Array<StmtDoc>(), else_branch));
  BlockStart();
}

void BaseStack::ConditionElse() {
  const auto& block = PopBlock();
  const auto& if_doc = PopCheckedDoc<IfDoc, IfDocNode>();
  PushDoc(IfDoc(if_doc->predicate, DocUtils::ToStmts(block), Array<StmtDoc>()));
  BlockStart();
}

void BaseStack::ConditionEnd() {
  const auto& block = PopBlock();
  const auto& if_doc = PopCheckedDoc<IfDoc, IfDocNode>();
  const auto& branch = DocUtils::ToStmts(block);
  if (if_doc->then_branch.size() == 0) {
    PushDoc(IfDoc(if_doc->predicate, branch, Array<StmtDoc>()));
  } else {
    PushDoc(IfDoc(if_doc->predicate, if_doc->then_branch, branch));
  }
}

void BaseStack::ForEnd() {
  const auto& block = PopBlock();
  const auto& for_doc = PopCheckedDoc<ForDoc, ForDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(ForDoc(for_doc->lhs, for_doc->rhs, body));
}

void BaseStack::WhileStart(const String& predicate) {
  PushDoc(WhileDoc(IdDoc(predicate), Array<StmtDoc>()));
  BlockStart();
}

void BaseStack::WhileEnd() {
  const auto& block = PopBlock();
  const auto& while_doc = PopCheckedDoc<WhileDoc, WhileDocNode>();
  const auto& body = DocUtils::ToStmts(block);
  PushDoc(WhileDoc(while_doc->predicate, body));
}

void BaseStack::SwitchStart(const String& predicate) {
  Array<ExprDoc> predicates;
  predicates.push_back(IdDoc(predicate));
  PushDoc(SwitchDoc(predicates, Array<Array<StmtDoc>>(), Array<StmtDoc>()));
  BlockStart();
}

void BaseStack::SwitchCase(const String& predicate) {
  const auto& block = PopBlock();
  const auto& switch_doc = PopCheckedDoc<SwitchDoc, SwitchDocNode>();
  auto branchs = switch_doc->branchs;
  branchs.push_back(DocUtils::ToStmts(block));
  if (predicate.size() == 0) {
    Array<StmtDoc> default_branch{ExprStmtDoc(IdDoc("pass"))};
    PushDoc(SwitchDoc(switch_doc->predicates, branchs, default_branch));
  } else {
    auto predicates = switch_doc->predicates;
    predicates.push_back(IdDoc(predicate));
    PushDoc(SwitchDoc(predicates, branchs, switch_doc->default_branch));
  }
  BlockStart();
}

void BaseStack::SwitchEnd() {
  const auto& block = PopBlock();
  const auto& switch_doc = PopCheckedDoc<SwitchDoc, SwitchDocNode>();
  if (switch_doc->default_branch.size() > 0) {
    PushDoc(SwitchDoc(switch_doc->predicates, switch_doc->branchs, DocUtils::ToStmts(block)));
  } else {
    auto branchs = switch_doc->branchs;
    branchs.push_back(DocUtils::ToStmts(block));
    PushDoc(SwitchDoc(switch_doc->predicates, branchs, switch_doc->default_branch));
  }
}

void BaseStack::BlockStart() {
  Array<Doc> block;
  blocks_.push(block);
}

void BaseStack::BlockEnd(bool block_docs) {
  const auto& docs = PopBlock();
  if (block_docs) {
    PushDoc(DocUtils::ToStmtBlock(docs));
  } else {
    for (const auto& d : docs) {
      PushDoc(d);
    }
  }
}

void BaseStack::ScopeStart(const String& scope_def, const String& scope_ref) {
  if (scope_ref.size() > 0) {
    PushDoc(ScopeDoc(IdDoc(scope_ref), IdDoc(scope_def), Array<StmtDoc>()));
  } else {
    PushDoc(ScopeDoc(NullOpt, IdDoc(scope_def), Array<StmtDoc>()));
  }
  BlockStart();
}

void BaseStack::ScopeEnd() {
  const auto& block = PopBlock();
  const auto& scope = PopCheckedDoc<ScopeDoc, ScopeDocNode>();
  PushDoc(ScopeDoc(scope->lhs, scope->rhs, DocUtils::ToStmts(block)));
}

bool BaseStack::HasBlock() const { return blocks_.size() > 0; }

const Array<Doc> BaseStack::TopBlock() const {
  ICHECK(HasBlock()) << "No block found";
  return blocks_.top();
}

const Array<Doc> BaseStack::PopBlock() {
  const auto& block = TopBlock();
  blocks_.pop();
  return block;
}

bool BaseStack::HasDoc() {
  if (!HasBlock()) {
    return false;
  }
  return TopBlock().size() > 0;
}

const Doc BaseStack::TopDoc() {
  ICHECK(HasDoc()) << "No doc or block found";
  return TopBlock().back();
}

const Doc BaseStack::PopDoc() {
  const auto& doc = TopDoc();
  blocks_.top().pop_back();
  return doc;
}

template <typename TDoc, typename TDocNode>
const TDoc BaseStack::PopCheckedDoc() {
  ICHECK(HasDoc() && TopDoc()->IsInstance<TDocNode>())
      << "Last doc(" << TopDoc()->GetTypeKey() << ") is not expected type "
      << TDocNode::TypeIndex2Key(TDocNode::RuntimeTypeIndex());
  return Downcast<TDoc>(PopDoc());
}

void BaseStack::PushDoc(const Doc& doc) {
  ICHECK(HasBlock()) << "No block found";
  blocks_.top().push_back(doc);
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
