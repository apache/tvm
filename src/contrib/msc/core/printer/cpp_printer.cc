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
 * \file src/contrib/msc/core/printer/cpp_printer.cc
 */

#include "cpp_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

void CppPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  bool defined = false;
  if (!value.defined()) {
    output_ << "nullptr";
    defined = true;
  } else if (const auto* int_imm = value.as<IntImmNode>()) {
    if (int_imm->dtype.is_bool()) {
      output_ << (int_imm->value ? "true" : "false");
      defined = true;
    }
  }
  if (!defined) {
    MSCBasePrinter::PrintTypedDoc(doc);
  }
}

void CppPrinter::PrintTypedDoc(const IndexDoc& doc) {
  ICHECK(doc->indices.size() == 1) << "CppPrinter only support 1 size indices";
  PrintDoc(doc->value, false);
  output_ << "[";
  PrintDoc(doc->indices[0], false);
  output_ << "]";
}

void CppPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value, false);
  if (StringUtils::EndsWith(doc->name, DocSymbol::NextLine())) {
    const auto& v_name = StringUtils::Replace(doc->name, DocSymbol::NextLine(), "");
    if (!doc->value->IsInstance<PointerDocNode>()) {
      IncreaseIndent();
      PrintDoc(IdDoc("."));
      DecreaseIndent();
    }
    output_ << v_name;
  } else {
    if (!doc->value->IsInstance<PointerDocNode>()) {
      output_ << ".";
    }
    output_ << doc->name;
  }
}

void CppPrinter::PrintTypedDoc(const CallDoc& doc) {
  EnterEndlineScope(false);
  PrintDoc(doc->callee, false);
  output_ << "(";
  PrintJoinedDocs(doc->args);
  ICHECK_EQ(doc->kwargs_keys.size(), doc->kwargs_values.size())
      << "CallDoc should have equal number of elements in kwargs_keys and kwargs_values.";
  if (doc->args.size() > 0 && doc->kwargs_keys.size() > 0) {
    output_ << ", ";
  }
  PrintJoinedDocs(doc->kwargs_values);
  output_ << ")";
  ExitEndlineScope();
  Endline();
}

void CppPrinter::PrintTypedDoc(const AssignDoc& doc) {
  ICHECK(doc->lhs.defined()) << "lhs should be given for assign";
  if (doc->annotation.defined()) {
    if (!IsEmptyDoc(doc->annotation.value())) {
      PrintDoc(doc->annotation.value(), false);
      output_ << " ";
    }
  }
  PrintDoc(doc->lhs, false);
  if (doc->rhs.defined()) {
    output_ << " = ";
    EnterEndlineScope(false);
    PrintDoc(doc->rhs.value(), false);
    ExitEndlineScope();
    Endline();
  }
}

void CppPrinter::PrintTypedDoc(const IfDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "if (";
  PrintDoc(doc->predicate, false);
  output_ << ") {";
  PrintIndentedBlock(doc->then_branch);
  if (!doc->else_branch.empty()) {
    NewLine();
    output_ << "} else {";
    PrintIndentedBlock(doc->else_branch);
  }
  NewLine();
  output_ << "}";
}

void CppPrinter::PrintTypedDoc(const WhileDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "while (";
  PrintDoc(doc->predicate, false);
  output_ << ") {";
  PrintIndentedBlock(doc->body);
  NewLine();
  output_ << "}";
}

void CppPrinter::PrintTypedDoc(const ForDoc& doc) {
  MaybePrintComment(doc, true);
  if (doc->rhs->IsInstance<TupleDocNode>()) {
    const auto& tuple = Downcast<TupleDoc>(doc->rhs);
    ICHECK_EQ(tuple->elements.size(), 2) << "For with tuple should has 2 elements";
    output_ << "for (size_t ";
    PrintDoc(doc->lhs, false);
    output_ << " = ";
    PrintDoc(tuple->elements[0], false);
    output_ << "; ";
    PrintDoc(doc->lhs, false);
    output_ << " < ";
    PrintDoc(tuple->elements[1], false);
    output_ << "; ";
    PrintDoc(doc->lhs, false);
    output_ << "++";
  } else {
    output_ << "for (const auto& ";
    PrintDoc(doc->lhs, false);
    output_ << " : ";
    PrintDoc(doc->rhs, false);
  }
  output_ << ") {";
  PrintIndentedBlock(doc->body);
  NewLine();
  output_ << "}";
}

void CppPrinter::PrintTypedDoc(const ScopeDoc& doc) {
  MaybePrintComment(doc, true);
  ICHECK(doc->rhs.defined()) << "rhs should be given for scope";
  PrintDoc(doc->rhs, false);
  PrintIndentedBlock(doc->body);
}

void CppPrinter::PrintTypedDoc(const FunctionDoc& doc) {
  MaybePrintComment(doc, true);
  for (const AssignDoc& arg_doc : doc->args) {
    ICHECK(arg_doc->comment == nullptr) << "Function arg cannot have comment attached to them.";
  }
  if (doc->return_type.defined()) {
    if (!IsEmptyDoc(doc->return_type.value())) {
      PrintDoc(doc->return_type.value(), false);
      output_ << " ";
    }
  } else {
    output_ << "void ";
  }
  PrintDoc(doc->name, false);
  output_ << "(";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ")";
  if (doc->decorators.size() > 0) {
    output_ << " ";
    PrintJoinedDocs(doc->decorators, " ");
  }
  if (doc->body.size() > 0) {
    output_ << " {";
    PrintIndentedBlock(doc->body);
    if (doc->return_type.defined()) {
      if (!IsEmptyDoc(doc->return_type.value())) {
        Endline();
      }
    }
    NewLine();
    output_ << "}";
  } else {
    Endline();
  }
  NewLine(false);
}

void CppPrinter::PrintTypedDoc(const ClassDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "class ";
  PrintDoc(doc->name, false);
  output_ << " {";
  for (const StmtDoc& d : doc->body) {
    PrintDoc(d);
  }
  output_ << "}";
  Endline();
  output_ << "  // class ";
  PrintDoc(doc->name, false);
  NewLine(false);
}

void CppPrinter::PrintTypedDoc(const CommentDoc& doc) {
  if (doc->comment.defined()) {
    output_ << "// " << doc->comment.value();
  }
}

void CppPrinter::PrintTypedDoc(const DeclareDoc& doc) {
  if (doc->type.defined()) {
    PrintDoc(doc->type.value(), false);
    output_ << " ";
  }
  PrintDoc(doc->variable, false);
  if (doc->init_args.size() > 0) {
    if (doc->use_constructor) {
      output_ << "(";
      PrintJoinedDocs(doc->init_args, ", ");
      output_ << ")";
    } else {
      output_ << "{";
      PrintJoinedDocs(doc->init_args, ", ");
      output_ << "}";
    }
  }
  Endline();
}

void CppPrinter::PrintTypedDoc(const PointerDoc& doc) { output_ << doc->name << "->"; }

void CppPrinter::PrintTypedDoc(const StrictListDoc& doc) {
  if (doc->allow_empty || doc->list->elements.size() > 0) {
    PrintDoc(doc->list, false);
  } else {
    output_ << "{}";
  }
}

void CppPrinter::PrintTypedDoc(const StructDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "struct ";
  PrintDoc(doc->name, false);
  output_ << " {";
  IncreaseIndent();
  for (const StmtDoc& d : doc->body) {
    PrintDoc(d);
  }
  DecreaseIndent();
  NewLine(false);
  output_ << "}";
  Endline();
  output_ << "  // struct ";
  PrintDoc(doc->name, false);
  NewLine(false);
}

void CppPrinter::PrintTypedDoc(const ConstructorDoc& doc) {
  MaybePrintComment(doc, true);
  for (const AssignDoc& arg_doc : doc->args) {
    ICHECK(arg_doc->comment == nullptr) << "Constructor arg cannot have comment attached to them.";
  }
  PrintDoc(doc->name, false);
  output_ << "(";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ")";
  if (doc->body.size() > 0) {
    output_ << " {";
    PrintIndentedBlock(doc->body);
    NewLine();
    output_ << "}";
  } else {
    Endline();
  }
  NewLine(false);
}

void CppPrinter::PrintTypedDoc(const LambdaDoc& doc) {
  MaybePrintComment(doc, true);
  for (const AssignDoc& arg_doc : doc->args) {
    ICHECK(arg_doc->comment == nullptr) << "Function arg cannot have comment attached to them.";
  }
  output_ << "auto ";
  PrintDoc(doc->name, false);
  output_ << " = [";
  PrintJoinedDocs(doc->refs, ", ");
  output_ << "](";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ")";
  if (doc->body.size() > 0) {
    output_ << " {";
    PrintIndentedBlock(doc->body);
    Endline();
    NewLine();
    output_ << "};";
  } else {
    Endline();
  }
  NewLine(false);
}

void CppPrinter::PrintTypedDoc(const SwitchDoc& doc) {
  MaybePrintComment(doc, true);
  ICHECK_EQ(doc->predicates.size(), doc->branchs.size())
      << "predicates " << doc->predicates.size() << " mismatch with branchs "
      << doc->branchs.size();
  for (size_t i = 0; i < doc->predicates.size(); i++) {
    if (i == 0) {
      output_ << "if (";
    } else {
      NewLine();
      output_ << "} else if (";
    }
    PrintDoc(doc->predicates[i], false);
    output_ << ") {";
    PrintIndentedBlock(doc->branchs[i]);
  }
  if (!doc->default_branch.empty()) {
    NewLine();
    output_ << "} else {";
    PrintIndentedBlock(doc->default_branch);
  }
  NewLine();
  output_ << "}";
}

bool CppPrinter::IsEmptyDoc(const ExprDoc& doc) {
  if (!doc->IsInstance<IdDocNode>()) {
    return false;
  }
  const auto& id_doc = Downcast<IdDoc>(doc);
  return id_doc->name == DocSymbol::Empty();
}

void CppPrinter::PrintIndentedBlock(const Array<StmtDoc>& docs) {
  IncreaseIndent();
  for (const StmtDoc& d : docs) {
    PrintDoc(d);
  }
  DecreaseIndent();
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
