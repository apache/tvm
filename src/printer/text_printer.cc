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
 * \file text_printer.cc
 * \brief Printer to print out the unified IR text format
 *        that can be parsed by a parser.
 */

#include "text_printer.h"

#include <tvm/tir/function.h>

#include <string>

namespace tvm {

static const char* kSemVer = "0.0.5";

Doc TextPrinter::PrintMod(const IRModule& mod) {
  Doc doc;
  int counter = 0;
  // type definitions
  for (const auto& kv : mod->type_definitions) {
    if (counter++ != 0) {
      doc << Doc::NewLine();
    }
    doc << relay_text_printer_.Print(kv.second);
    doc << Doc::NewLine();
  }
  // functions
  for (const auto& kv : mod->functions) {
    if (kv.second.as<relay::FunctionNode>()) {
      relay_text_printer_.dg_ =
          relay::DependencyGraph::Create(&relay_text_printer_.arena_, kv.second);
    }
    if (counter++ != 0) {
      doc << Doc::NewLine();
    }
    if (kv.second.as<relay::FunctionNode>()) {
      std::ostringstream os;
      os << "def @" << kv.first->name_hint;
      doc << relay_text_printer_.PrintFunc(Doc::Text(os.str()), kv.second);
    } else if (kv.second.as<tir::PrimFuncNode>()) {
      doc << tir_text_printer_.PrintPrimFunc(Downcast<tir::PrimFunc>(kv.second));
    }
    doc << Doc::NewLine();
  }
  return doc;
}

String PrettyPrint(const ObjectRef& node) {
  Doc doc;
  doc << TextPrinter(false, nullptr, false).PrintFinal(node);
  return doc.str();
}

String AsText(const ObjectRef& node, bool show_meta_data,
              runtime::TypedPackedFunc<String(ObjectRef)> annotate) {
  Doc doc;
  doc << "#[version = \"" << kSemVer << "\"]" << Doc::NewLine();
  runtime::TypedPackedFunc<std::string(ObjectRef)> ftyped = nullptr;
  if (annotate != nullptr) {
    ftyped = runtime::TypedPackedFunc<std::string(ObjectRef)>(
        [&annotate](const ObjectRef& expr) -> std::string { return annotate(expr); });
  }
  doc << TextPrinter(show_meta_data, ftyped).PrintFinal(node);
  return doc.str();
}

TVM_REGISTER_GLOBAL("ir.PrettyPrint").set_body_typed(PrettyPrint);

TVM_REGISTER_GLOBAL("ir.AsText").set_body_typed(AsText);

}  // namespace tvm
