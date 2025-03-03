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
 * \file src/contrib/msc/core/printer/python_printer.cc
 */

#include "python_printer.h"

#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

void PythonPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  bool defined = false;
  if (!value.defined()) {
    output_ << "None";
    defined = true;
  } else if (const auto* int_imm = value.as<IntImmNode>()) {
    if (int_imm->dtype.is_bool()) {
      output_ << (int_imm->value ? "True" : "False");
      defined = true;
    }
  }
  if (!defined) {
    MSCBasePrinter::PrintTypedDoc(doc);
  }
}

void PythonPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value, false);
  output_ << "." << doc->name;
}

void PythonPrinter::PrintTypedDoc(const IndexDoc& doc) {
  PrintDoc(doc->value, false);
  if (doc->indices.size() == 0) {
    output_ << "[()]";
  } else {
    output_ << "[";
    PrintJoinedDocs(doc->indices, ", ");
    output_ << "]";
  }
}

void PythonPrinter::PrintTypedDoc(const CallDoc& doc) {
  PrintDoc(doc->callee, false);
  output_ << "(";
  PrintJoinedDocs(doc->args);
  ICHECK_EQ(doc->kwargs_keys.size(), doc->kwargs_values.size())
      << "CallDoc should have equal number of elements in kwargs_keys and kwargs_values.";
  if (doc->args.size() > 0 && doc->kwargs_keys.size() > 0) {
    output_ << ", ";
  }
  for (size_t i = 0; i < doc->kwargs_keys.size(); i++) {
    output_ << doc->kwargs_keys[i] << "=";
    PrintDoc(doc->kwargs_values[i], false);
    output_ << (i == doc->kwargs_keys.size() - 1 ? "" : ", ");
  }
  output_ << ")";
}

void PythonPrinter::PrintTypedDoc(const AssignDoc& doc) {
  if (const auto* tuple_doc = doc->lhs.as<TupleDocNode>()) {
    PrintJoinedDocs(tuple_doc->elements, ", ");
  } else {
    PrintDoc(doc->lhs, false);
  }

  if (doc->annotation) {
    output_ << ": ";
    PrintDoc(doc->annotation.value(), false);
  }
  if (doc->rhs) {
    output_ << " = ";
    if (const auto* tuple_doc = doc->rhs.as<TupleDocNode>()) {
      if (tuple_doc->elements.size() > 1) {
        PrintJoinedDocs(tuple_doc->elements, ", ");
      } else {
        PrintDoc(doc->rhs.value(), false);
      }
    } else {
      PrintDoc(doc->rhs.value(), false);
    }
  }
  MaybePrintComment(doc);
}

void PythonPrinter::PrintTypedDoc(const IfDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "if ";
  PrintDoc(doc->predicate, false);
  output_ << ":";

  PrintIndentedBlock(doc->then_branch);

  if (!doc->else_branch.empty()) {
    NewLine();
    output_ << "else:";
    PrintIndentedBlock(doc->else_branch);
  }
}

void PythonPrinter::PrintTypedDoc(const ForDoc& doc) {
  MaybePrintComment(doc, true);
  if (doc->rhs->IsInstance<TupleDocNode>()) {
    const auto& tuple = Downcast<TupleDoc>(doc->rhs);
    ICHECK_EQ(tuple->elements.size(), 2) << "For with tuple should has 2 elements";
    output_ << "for ";
    PrintDoc(doc->lhs, false);
    output_ << " in range(";
    PrintDoc(tuple->elements[0], false);
    output_ << ", ";
    PrintDoc(tuple->elements[1], false);
    output_ << "):";
  } else {
    output_ << "for ";
    PrintDoc(doc->lhs, false);
    output_ << " in ";
    PrintDoc(doc->rhs, false);
    output_ << ":";
  }
  PrintIndentedBlock(doc->body);
}

void PythonPrinter::PrintTypedDoc(const ScopeDoc& doc) {
  MaybePrintComment(doc, true);
  output_ << "with ";
  PrintDoc(doc->rhs, false);
  if (doc->lhs != nullptr) {
    output_ << " as ";
    PrintDoc(doc->lhs.value(), false);
  }
  output_ << ":";

  PrintIndentedBlock(doc->body);
}

void PythonPrinter::PrintTypedDoc(const FunctionDoc& doc) {
  for (const AssignDoc& arg_doc : doc->args) {
    ICHECK(arg_doc->comment == nullptr) << "Function arg cannot have comment attached to them.";
  }

  PrintDecorators(doc->decorators);

  output_ << "def ";
  PrintDoc(doc->name, false);

  output_ << "(";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ")";

  if (doc->return_type.defined()) {
    output_ << " -> ";
    PrintDoc(doc->return_type.value(), false);
  }

  output_ << ":";

  if (doc->comment.defined()) {
    IncreaseIndent();
    MaybePrintComment(doc, true);
    DecreaseIndent();
  }
  PrintIndentedBlock(doc->body);
  NewLine(false);
}

void PythonPrinter::PrintTypedDoc(const ClassDoc& doc) {
  PrintDecorators(doc->decorators);

  output_ << "class ";
  PrintDoc(doc->name, false);
  output_ << ":";

  MaybePrintComment(doc, true);
  PrintIndentedBlock(doc->body);
}

void PythonPrinter::PrintTypedDoc(const CommentDoc& doc) {
  if (doc->comment.defined()) {
    output_ << "# " << doc->comment.value();
  }
}

void PythonPrinter::PrintTypedDoc(const StrictListDoc& doc) {
  if (doc->allow_empty || doc->list->elements.size() > 0) {
    PrintDoc(doc->list, false);
  } else {
    output_ << "None";
  }
}

void PythonPrinter::PrintTypedDoc(const SwitchDoc& doc) {
  MaybePrintComment(doc, true);
  ICHECK_EQ(doc->predicates.size(), doc->branchs.size())
      << "predicates " << doc->predicates.size() << " mismatch with branchs "
      << doc->branchs.size();
  for (size_t i = 0; i < doc->predicates.size(); i++) {
    if (i == 0) {
      output_ << "if ";
    } else {
      NewLine();
      output_ << "elif ";
    }
    PrintDoc(doc->predicates[i], false);
    output_ << ":";
    PrintIndentedBlock(doc->branchs[i]);
  }
  if (!doc->default_branch.empty()) {
    NewLine();
    output_ << "else:";
    PrintIndentedBlock(doc->default_branch);
  }
}

void PythonPrinter::MaybePrintComment(const StmtDoc& stmt, bool multi_lines) {
  if (stmt->comment.defined() && multi_lines) {
    NewLine();
    output_ << "\"\"\"";
    for (const auto& l : StringUtils::Split(stmt->comment.value(), "\n")) {
      PrintDoc(IdDoc(l));
    }
    NewLine();
    output_ << "\"\"\"";
    NewLine();
  } else {
    MSCBasePrinter::MaybePrintComment(stmt, multi_lines);
  }
}

void PythonPrinter::PrintIndentedBlock(const Array<StmtDoc>& docs) {
  IncreaseIndent();
  for (const StmtDoc& d : docs) {
    PrintDoc(d);
  }
  if (docs.empty()) {
    NewLine() << "pass";
  }
  DecreaseIndent();
}

void PythonPrinter::PrintDecorators(const Array<ExprDoc>& decorators) {
  for (const ExprDoc& decorator : decorators) {
    output_ << "@";
    PrintDoc(decorator, false);
    NewLine();
  }
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
