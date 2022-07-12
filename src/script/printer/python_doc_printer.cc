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

String DocToPythonScript(Doc doc, int indent_spaces) {
  PythonDocPrinter printer(indent_spaces);
  printer.Append(doc);
  return printer.GetString();
}

TVM_REGISTER_GLOBAL("script.printer.DocToPythonScript").set_body_typed(DocToPythonScript);

}  // namespace printer
}  // namespace script
}  // namespace tvm
