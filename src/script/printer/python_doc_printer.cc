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

#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include "../../support/str_escape.h"
#include "./base_doc_printer.h"

namespace tvm {
namespace script {
namespace printer {

class PythonDocPrinter : public DocPrinter {
 public:
  explicit PythonDocPrinter(int indent_spaces = 4) : DocPrinter(indent_spaces) {}

 protected:
  using DocPrinter::PrintDoc;

  void PrintTypedDoc(const LiteralDoc& doc) final;
  void PrintTypedDoc(const IdDoc& doc) final;
  void PrintTypedDoc(const AttrAccessDoc& doc) final;
  void PrintTypedDoc(const IndexDoc& doc) final;
  void PrintTypedDoc(const OperationDoc& doc) final;
  void PrintTypedDoc(const CallDoc& doc) final;
  void PrintTypedDoc(const LambdaDoc& doc) final;
  void PrintTypedDoc(const ListDoc& doc) final;
  void PrintTypedDoc(const DictDoc& doc) final;
  void PrintTypedDoc(const TupleDoc& doc) final;
  void PrintTypedDoc(const SliceDoc& doc) final;

 private:
  template <typename DocType>
  void PrintJoinedDocs(const Array<DocType>& docs, const std::string& separator) {
    bool is_first = true;
    for (auto& doc : docs) {
      if (is_first) {
        is_first = false;
      } else {
        output_ << separator;
      }
      PrintDoc(doc);
    }
  }
};

void PythonDocPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  if (!value.defined()) {
    output_ << "None";
  } else if (const auto* int_imm = value.as<IntImmNode>()) {
    if (int_imm->dtype.is_bool()) {
      output_ << (int_imm->value ? "True" : "False");
    } else {
      output_ << int_imm->value;
    }
  } else if (const auto* float_imm = value.as<FloatImmNode>()) {
    // TODO(yelite): Make float number printing roundtrippable
    output_.precision(17);
    output_ << float_imm->value;
  } else if (const auto* string_obj = value.as<StringObj>()) {
    output_ << "\"" << support::StrEscape(string_obj->data, string_obj->size) << "\"";
  } else {
    LOG(FATAL) << "TypeError: Unsupported literal value type: " << value->GetTypeKey();
  }
}

void PythonDocPrinter::PrintTypedDoc(const IdDoc& doc) { output_ << doc->name; }

void PythonDocPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  PrintDoc(doc->value);
  output_ << "." << doc->name;
}

void PythonDocPrinter::PrintTypedDoc(const IndexDoc& doc) {
  PrintDoc(doc->value);
  if (doc->indices.size() == 0) {
    output_ << "[()]";
  } else {
    output_ << "[";
    PrintJoinedDocs(doc->indices, ", ");
    output_ << "]";
  }
}

const std::string OperatorToString(OperationDocNode::Kind operation_kind) {
  static const std::vector<std::string> op_kind2str = []() {
    using OpKind = OperationDocNode::Kind;
    std::map<OpKind, std::string> raw_table = {
        {OpKind::kUSub, "-"},       //
        {OpKind::kInvert, "~"},     //
        {OpKind::kAdd, "+"},        //
        {OpKind::kSub, "-"},        //
        {OpKind::kMult, "*"},       //
        {OpKind::kDiv, "/"},        //
        {OpKind::kFloorDiv, "//"},  //
        {OpKind::kMod, "%"},        //
        {OpKind::kPow, "**"},       //
        {OpKind::kLShift, "<<"},    //
        {OpKind::kRShift, ">>"},    //
        {OpKind::kBitAnd, "&"},     //
        {OpKind::kBitOr, "|"},      //
        {OpKind::kBitXor, "^"},     //
        {OpKind::kLt, "<"},         //
        {OpKind::kLtE, "<="},       //
        {OpKind::kEq, "=="},        //
        {OpKind::kNotEq, "!="},     //
        {OpKind::kGt, ">"},         //
        {OpKind::kGtE, ">="},       //
    };

    std::vector<std::string> table;
    table.resize(static_cast<int>(OperationDocNode::Kind::kSpecialEnd) + 1);

    for (const auto& kv : raw_table) {
      table[static_cast<int>(kv.first)] = kv.second;
    }

    return table;
  }();

  auto op_index = static_cast<int>(operation_kind);
  ICHECK_LT(op_index, op_kind2str.size());
  const std::string str = op_kind2str[op_index];
  ICHECK(!str.empty()) << "OperationDocNode::Kind " << static_cast<int>(operation_kind)
                       << " cannot be converted to operator token in Python directly.";
  return str;
}

void PythonDocPrinter::PrintTypedDoc(const OperationDoc& doc) {
  using OpKind = OperationDocNode::Kind;
  if (doc->kind < OpKind::kUnaryEnd) {
    // Unary Operators
    ICHECK_EQ(doc->operands.size(), 1);
    output_ << OperatorToString(doc->kind);
    PrintDoc(doc->operands[0]);
  } else if (doc->kind < OpKind::kBinaryEnd) {
    // Binary Operator
    ICHECK_EQ(doc->operands.size(), 2);
    PrintDoc(doc->operands[0]);
    output_ << " " << OperatorToString(doc->kind) << " ";
    PrintDoc(doc->operands[1]);
  } else if (doc->kind == OpKind::kIfThenElse) {
    ICHECK_EQ(doc->operands.size(), 3)
        << "ValueError: IfThenElse requires 3 operands, but got " << doc->operands.size();
    PrintDoc(doc->operands[1]);
    output_ << " if ";
    PrintDoc(doc->operands[0]);
    output_ << " else ";
    PrintDoc(doc->operands[2]);
  } else {
    LOG(FATAL) << "Unknown OperationDocNode::Kind " << static_cast<int>(doc->kind);
    throw;
  }
}

void PythonDocPrinter::PrintTypedDoc(const CallDoc& doc) {
  PrintDoc(doc->callee);

  output_ << "(";

  // Print positional args
  bool is_first = true;
  for (const ExprDoc& arg : doc->args) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    PrintDoc(arg);
  }

  // Print keyword args
  ICHECK_EQ(doc->kwargs_keys.size(), doc->kwargs_values.size())
      << "CallDoc should have equal number of elements in kwargs_keys and kwargs_values.";
  for (size_t i = 0; i < doc->kwargs_keys.size(); i++) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    const String& keyword = doc->kwargs_keys[i];
    output_ << keyword;
    output_ << "=";
    PrintDoc(doc->kwargs_values[i]);
  }

  output_ << ")";
}

void PythonDocPrinter::PrintTypedDoc(const LambdaDoc& doc) {
  output_ << "lambda ";
  PrintJoinedDocs(doc->args, ", ");
  output_ << ": ";
  PrintDoc(doc->body);
}

void PythonDocPrinter::PrintTypedDoc(const ListDoc& doc) {
  output_ << "[";
  PrintJoinedDocs(doc->elements, ", ");
  output_ << "]";
}

void PythonDocPrinter::PrintTypedDoc(const TupleDoc& doc) {
  output_ << "(";
  if (doc->elements.size() == 1) {
    PrintDoc(doc->elements[0]);
    output_ << ",";
  } else {
    PrintJoinedDocs(doc->elements, ", ");
  }
  output_ << ")";
}

void PythonDocPrinter::PrintTypedDoc(const DictDoc& doc) {
  ICHECK_EQ(doc->keys.size(), doc->values.size())
      << "DictDoc should have equal number of elements in keys and values.";
  output_ << "{";
  size_t idx = 0;
  for (const ExprDoc& key : doc->keys) {
    if (idx > 0) {
      output_ << ", ";
    }
    PrintDoc(key);
    output_ << ": ";
    PrintDoc(doc->values[idx]);
    idx++;
  }
  output_ << "}";
}

void PythonDocPrinter::PrintTypedDoc(const SliceDoc& doc) {
  if (doc->start != nullptr) {
    PrintDoc(doc->start.value());
  }
  output_ << ":";
  if (doc->stop != nullptr) {
    PrintDoc(doc->stop.value());
  }
  if (doc->step != nullptr) {
    output_ << ":";
    PrintDoc(doc->step.value());
  }
}

String DocToPythonScript(Doc doc, int indent_spaces) {
  PythonDocPrinter printer(indent_spaces);
  printer.Append(doc);
  return printer.GetString();
}

TVM_REGISTER_GLOBAL("script.printer.DocToPythonScript").set_body_typed(DocToPythonScript);

}  // namespace printer
}  // namespace script
}  // namespace tvm
