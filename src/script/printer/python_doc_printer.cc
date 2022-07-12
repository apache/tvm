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

#include <regex>

#include "../../support/str_escape.h"
#include "./base_doc_printer.h"
#include "tvm/runtime/logging.h"

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

/*
 * This function checks whether an input string is a valid
 * identifier. Invalid identifier name can make the result
 * still parsable but into a different IR tree. So we want
 * to fail as soon as possible.
 */
bool IsValidPythonIdentifier(const std::string& id) {
  // This regex is just an approximation of the Python identifier
  // rule. This doesn't exclude the reserved keywords. But it should
  // be good enough for roundtrippable TVMScript printing and parsing.
  static const std::regex id_pattern(R"(^[^\d\W]\w*$)");
  return std::regex_match(id, id_pattern);
}

void PythonDocPrinter::PrintTypedDoc(const IdDoc& doc) {
  CHECK(IsValidPythonIdentifier(doc->name))
      << "ValueError: " << doc->name << " is not a valid identifier.";
  output_ << doc->name;
}

void PythonDocPrinter::PrintTypedDoc(const AttrAccessDoc& doc) {
  CHECK(IsValidPythonIdentifier(doc->attr))
      << "ValueError: " << doc->attr << " is not a valid attribute.";
  PrintDoc(doc->value);
  output_ << "." << doc->attr;
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

const char* OperatorToString(OperationDocNode::Kind operation_kind) {
  constexpr int OP_STR_TABLE_SIZE = static_cast<int>(OperationDocNode::Kind::kSpecialEnd) + 1;
  static const std::array<const char*, OP_STR_TABLE_SIZE> OP_STR_TABLE = []() {
    using OpKind = OperationDocNode::Kind;
    std::array<const char*, OP_STR_TABLE_SIZE> table;
    auto set_op = [&table](auto op, const char* str) { table[static_cast<int>(op)] = str; };

    set_op(OpKind::kUSub, "-");
    set_op(OpKind::kInvert, "~");
    set_op(OpKind::kAdd, "+");
    set_op(OpKind::kSub, "-");
    set_op(OpKind::kMult, "*");
    set_op(OpKind::kDiv, "/");
    set_op(OpKind::kFloorDiv, "//");
    set_op(OpKind::kMod, "%");
    set_op(OpKind::kPow, "**");
    set_op(OpKind::kLShift, "<<");
    set_op(OpKind::kRShift, ">>");
    set_op(OpKind::kBitAnd, "&");
    set_op(OpKind::kBitOr, "|");
    set_op(OpKind::kBitXor, "^");
    set_op(OpKind::kLt, "<");
    set_op(OpKind::kLtE, "<=");
    set_op(OpKind::kEq, "==");
    set_op(OpKind::kNotEq, "!=");
    set_op(OpKind::kGt, ">");
    set_op(OpKind::kGtE, ">=");

    return table;
  }();

  auto op_index = static_cast<int>(operation_kind);
  ICHECK_LT(op_index, OP_STR_TABLE_SIZE);
  const char* str = OP_STR_TABLE[static_cast<int>(operation_kind)];
  if (str == nullptr) {
    LOG(FATAL) << "OperationDocNode::Kind " << static_cast<int>(operation_kind)
               << " cannot be converted to operator token in Python directly.";
    throw;
  }
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
  } else if (doc->kind == OpKind::kAssert) {
    // Special Operator: Assert
    output_ << "assert ";
    PrintDoc(doc->operands[0]);
    if (doc->operands.size() > 1) {
      output_ << ", ";
      PrintDoc(doc->operands[1]);
    }
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
  for (size_t i = 0; i < doc->kwargs_keys.size(); i++) {
    if (is_first) {
      is_first = false;
    } else {
      output_ << ", ";
    }
    const String& keyword = doc->kwargs_keys[i];
    CHECK(IsValidPythonIdentifier(keyword))
        << "ValueError: " << keyword << " is not a valid name for keyword parameter.";
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
