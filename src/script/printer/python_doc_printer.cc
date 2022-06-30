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
#include <tvm/script/printer/doc_printer.h>

namespace tvm {
namespace script {
namespace printer {

namespace {

/*!
 * \brief Print a Python literal string
 *
 * \param string the string to be printed
 * \param out the output stream
 */
void PrintLiteralString(const String& string, std::ostringstream& out) {
  // TODO: Escape and smart quote (choose ' or " automatically)
  out << "\"" << string << "\"";
}

/*!
 * \brief Print a tvm::ir::PrimExpr as Python literal
 *
 * This only supports IntImm and FloatImm with size of 64 bits
 *
 * \param expr the PrimExpr to be printed
 * \param out the output stream
 */
void PrintLiteralPrimExpr(const PrimExpr& expr, std::ostringstream& out) {
  const DataType& dtype = expr->dtype;

  if (dtype == DataType::Int(64)) {
    out << Downcast<IntImm>(expr)->value;
  } else if (dtype == DataType::Float(64)) {
    // TODO: make the float printing roundtrippable 
    std::ostringstream number_value;
    number_value.precision(17);
    number_value << Downcast<FloatImm>(expr)->value;
    out << number_value.str();
  } else if (dtype == DataType::Bool()) {
    out << (Downcast<IntImm>(expr)->value ? "True" : "False");
  } else {
    LOG(FATAL) << "Cannot print value with dtype " << dtype << " as literal expression";
  }
}

}  // namespace

class PythonDocPrinter : public DocPrinter {
 public:
  PythonDocPrinter(const DocPrinterOptions& options) : DocPrinter(options) {}

 protected:
  using DocPrinter::PrintDoc;

  void PrintTypedDoc(const LiteralDoc& doc) final;
};

void PythonDocPrinter::PrintTypedDoc(const LiteralDoc& doc) {
  const ObjectRef& value = doc->value;
  if (!value.defined()) {
    output_ << "None";
  } else if (const auto* expr_node = value.as<PrimExprNode>()) {
    PrintLiteralPrimExpr(GetRef<PrimExpr>(expr_node), output_);
  } else if (const auto* string_obj = value.as<StringObj>()) {
    PrintLiteralString(GetRef<String>(string_obj), output_);
  } else {
    LOG(FATAL) << "Unsupported literal value type " << value->GetTypeKey();
  }
}

std::unique_ptr<DocPrinter> GetPythonDocPrinter(const DocPrinterOptions& options) {
  return std::make_unique<PythonDocPrinter>(options);
}

TVM_REGISTER_GLOBAL("script.printer.PrintDocAsPython")
    .set_body_typed([](Doc doc, int indent_spaces = 4) {
      PythonDocPrinter printer({.indent_spaces = indent_spaces});
      printer.Append(doc);
      return printer.GetString();
    });

}  // namespace printer
}  // namespace script
}  // namespace tvm
