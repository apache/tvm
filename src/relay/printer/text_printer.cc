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

#include "./text_printer.h"

#include <tvm/tir/function.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace relay {

static const char* kSemVer = "0.0.5";

Doc TextPrinter::PrintMod(const IRModule& mod) {
  Doc doc;
  int counter = 0;

  // We'll print in alphabetical order to make a/b diffs easier to work with.

  // type definitions
  std::vector<GlobalTypeVar> tyvars;
  for (const auto& kv : mod->type_definitions) {
    tyvars.emplace_back(kv.first);
  }
  std::sort(tyvars.begin(), tyvars.end(),
            [](const GlobalTypeVar& left, const GlobalTypeVar& right) {
              return left->name_hint < right->name_hint;
            });
  for (const auto& tyvar : tyvars) {
    if (counter++ != 0) {
      doc << Doc::NewLine();
    }
    doc << relay_text_printer_.Print(mod->type_definitions[tyvar]);
    doc << Doc::NewLine();
  }

  // functions
  std::vector<GlobalVar> vars;
  for (const auto& kv : mod->functions) {
    vars.emplace_back(kv.first);
  }
  std::sort(vars.begin(), vars.end(), [](const GlobalVar& left, const GlobalVar& right) {
    return left->name_hint < right->name_hint;
  });
  for (const auto& var : vars) {
    const BaseFunc& base_func = mod->functions[var];
    if (base_func.as<relay::FunctionNode>()) {
      relay_text_printer_.dg_ =
          relay::DependencyGraph::Create(&relay_text_printer_.arena_, base_func);
    }
    if (counter++ != 0) {
      doc << Doc::NewLine();
    }
    if (base_func.as<relay::FunctionNode>()) {
      std::ostringstream os;
      os << "def @" << var->name_hint;
      doc << relay_text_printer_.PrintFunc(Doc::Text(os.str()), base_func);
    } else if (base_func.as<tir::PrimFuncNode>()) {
      doc << "@" << var->name_hint;
      doc << " = " << tir_text_printer_.PrintPrimFunc(Downcast<tir::PrimFunc>(base_func));
    }
    doc << Doc::NewLine();
  }

#if TVM_LOG_DEBUG
  // attributes
  // TODO(mbs): Make this official, including support from parser.
  if (mod->attrs.defined() && !mod->attrs->dict.empty()) {
    std::vector<String> keys;
    for (const auto& kv : mod->attrs->dict) {
      keys.emplace_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    doc << "attributes {" << Doc::NewLine();
    for (const auto& key : keys) {
      doc << "  '" << key << "' = " << PrettyPrint(mod->attrs->dict[key]) << Doc::NewLine();
    }
    doc << "}" << Doc::NewLine();
  }
#endif

  return doc;
}

String PrettyPrint(const ObjectRef& node) {
  Doc doc;
  doc << TextPrinter(/*show_meta_data=*/false, nullptr, false).PrintFinal(node);
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

TVM_REGISTER_GLOBAL("relay.ir.PrettyPrint").set_body_typed(PrettyPrint);
TVM_REGISTER_GLOBAL("relay.ir.AsText").set_body_typed(AsText);

}  // namespace relay
}  // namespace tvm
