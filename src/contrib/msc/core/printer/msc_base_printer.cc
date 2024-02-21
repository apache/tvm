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
 * \file src/contrib/msc/core/printer/msc_base_printer.cc
 */

#include "msc_base_printer.h"

#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

void MSCBasePrinter::PrintDoc(const Doc& doc, bool new_line) {
  if (new_line) {
    NewLine();
    lines_++;
  }
  if (auto doc_node = doc.as<LiteralDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<IdDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<AttrAccessDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<IndexDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<CallDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ListDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<TupleDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<DictDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<SliceDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<StmtBlockDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<AssignDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<IfDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<WhileDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ForDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ScopeDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ExprStmtDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<AssertDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ReturnDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<FunctionDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ClassDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<CommentDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<DeclareDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<StrictListDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<PointerDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<StructDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<ConstructorDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<SwitchDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else if (auto doc_node = doc.as<LambdaDoc>()) {
    PrintTypedDoc(doc_node.value());
  } else {
    LOG(FATAL) << "Do not know how to print " << doc->GetTypeKey();
    throw;
  }
}

void MSCBasePrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  if (!value.defined()) {
    output_ << "\"\"";
  } else if (const auto* int_imm = value.as<IntImmNode>()) {
    output_ << int_imm->value;
  } else if (const auto* float_imm = value.as<FloatImmNode>()) {
    output_.precision(config_.float_precision);
    if (std::isinf(float_imm->value) || std::isnan(float_imm->value)) {
      output_ << '"' << float_imm->value << '"';
    } else {
      output_ << float_imm->value;
    }
  } else if (const auto* string_obj = value.as<StringObj>()) {
    output_ << "\"" << tvm::support::StrEscape(string_obj->data, string_obj->size) << "\"";
  } else {
    LOG(FATAL) << "TypeError: Unsupported literal value type: " << value->GetTypeKey();
  }
}

void MSCBasePrinter::PrintTypedDoc(const IdDoc& doc) { output_ << doc->name; }

void MSCBasePrinter::PrintTypedDoc(const ListDoc& doc) {
  output_ << "[";
  PrintJoinedDocs(doc->elements);
  output_ << "]";
}

void MSCBasePrinter::PrintTypedDoc(const TupleDoc& doc) {
  output_ << "(";
  if (doc->elements.size() == 1) {
    PrintDoc(doc->elements[0]);
    output_ << ",";
  } else {
    PrintJoinedDocs(doc->elements);
  }
  output_ << ")";
}

void MSCBasePrinter::PrintTypedDoc(const ReturnDoc& doc) {
  output_ << "return ";
  PrintDoc(doc->value, false);
  MaybePrintComment(doc);
}

void MSCBasePrinter::PrintTypedDoc(const StmtBlockDoc& doc) {
  for (const StmtDoc& stmt : doc->stmts) {
    NewLine();
    PrintDoc(stmt);
  }
}

void MSCBasePrinter::PrintTypedDoc(const ExprStmtDoc& doc) {
  PrintDoc(doc->expr, false);
  MaybePrintComment(doc);
}

void MSCBasePrinter::MaybePrintComment(const StmtDoc& stmt, bool multi_lines) {
  if (stmt->comment.defined()) {
    if (multi_lines) {
      for (const auto& l : StringUtils::Split(stmt->comment.value(), "\n")) {
        PrintDoc(CommentDoc(l));
      }
    } else {
      PrintDoc(CommentDoc(stmt->comment.value()), false);
    }
  }
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
