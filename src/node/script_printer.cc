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
#include <tvm/ir/expr.h>
#include <tvm/node/repr_printer.h>
#include <tvm/node/script_printer.h>
#include <tvm/runtime/registry.h>

#include <regex>

namespace tvm {

TVMScriptPrinter::FType& TVMScriptPrinter::vtable() {
  static FType inst;
  return inst;
}

std::string TVMScriptPrinter::Script(const ObjectRef& node, const Optional<PrinterConfig>& cfg) {
  if (!TVMScriptPrinter::vtable().can_dispatch(node)) {
    return AsLegacyRepr(node);
  }
  return TVMScriptPrinter::vtable()(node, cfg.value_or(PrinterConfig()));
}

bool IsIdentifier(const std::string& name) {
  static const std::regex kValidIdentifier("^[a-zA-Z_][a-zA-Z0-9_]*$");
  return std::regex_match(name, kValidIdentifier);
}

PrinterConfig::PrinterConfig(Map<String, ObjectRef> config_dict) {
  runtime::ObjectPtr<PrinterConfigNode> n = make_object<PrinterConfigNode>();
  if (auto v = config_dict.Get("name")) {
    n->binding_names.push_back(Downcast<String>(v));
  }
  if (auto v = config_dict.Get("show_meta")) {
    n->show_meta = Downcast<runtime::BoxBool>(v)->value;
  }
  if (auto v = config_dict.Get("ir_prefix")) {
    n->ir_prefix = Downcast<String>(v);
  }
  if (auto v = config_dict.Get("tir_prefix")) {
    n->tir_prefix = Downcast<String>(v);
  }
  if (auto v = config_dict.Get("relax_prefix")) {
    n->relax_prefix = Downcast<String>(v);
  }
  if (auto v = config_dict.Get("module_alias")) {
    n->module_alias = Downcast<String>(v);
  }
  if (auto v = config_dict.Get("buffer_dtype")) {
    n->buffer_dtype = DataType(runtime::String2DLDataType(Downcast<String>(v)));
  }
  if (auto v = config_dict.Get("int_dtype")) {
    n->int_dtype = DataType(runtime::String2DLDataType(Downcast<String>(v)));
  }
  if (auto v = config_dict.Get("float_dtype")) {
    n->float_dtype = DataType(runtime::String2DLDataType(Downcast<String>(v)));
  }
  if (auto v = config_dict.Get("verbose_expr")) {
    n->verbose_expr = Downcast<runtime::BoxBool>(v)->value;
  }
  if (auto v = config_dict.Get("indent_spaces")) {
    n->indent_spaces = Downcast<runtime::BoxInt>(v)->value;
  }
  if (auto v = config_dict.Get("print_line_numbers")) {
    n->print_line_numbers = Downcast<runtime::BoxBool>(v)->value;
  }
  if (auto v = config_dict.Get("num_context_lines")) {
    n->num_context_lines = Downcast<runtime::BoxInt>(v)->value;
  }
  if (auto v = config_dict.Get("path_to_underline")) {
    n->path_to_underline = Downcast<Optional<Array<ObjectPath>>>(v).value_or(Array<ObjectPath>());
  }
  if (auto v = config_dict.Get("path_to_annotate")) {
    n->path_to_annotate =
        Downcast<Optional<Map<ObjectPath, String>>>(v).value_or(Map<ObjectPath, String>());
  }
  if (auto v = config_dict.Get("obj_to_underline")) {
    n->obj_to_underline = Downcast<Optional<Array<ObjectRef>>>(v).value_or(Array<ObjectRef>());
  }
  if (auto v = config_dict.Get("obj_to_annotate")) {
    n->obj_to_annotate =
        Downcast<Optional<Map<ObjectRef, String>>>(v).value_or(Map<ObjectRef, String>());
  }
  if (auto v = config_dict.Get("syntax_sugar")) {
    n->syntax_sugar = Downcast<runtime::BoxBool>(v)->value;
  }
  if (auto v = config_dict.Get("show_object_address")) {
    n->show_object_address = Downcast<runtime::BoxBool>(v)->value;
  }

  // Checking prefixes if they are valid Python identifiers.
  CHECK(IsIdentifier(n->ir_prefix)) << "Invalid `ir_prefix`: " << n->ir_prefix;
  CHECK(IsIdentifier(n->tir_prefix)) << "Invalid `tir_prefix`: " << n->tir_prefix;
  CHECK(IsIdentifier(n->relax_prefix)) << "Invalid `relax_prefix`: " << n->relax_prefix;
  CHECK(n->module_alias.empty() || IsIdentifier(n->module_alias))
      << "Invalid `module_alias`: " << n->module_alias;

  this->data_ = std::move(n);
}

Array<String> PrinterConfigNode::GetBuiltinKeywords() {
  Array<String> result{this->ir_prefix, this->tir_prefix, this->relax_prefix};
  if (!this->module_alias.empty()) {
    result.push_back(this->module_alias);
  }
  return result;
}

TVM_REGISTER_NODE_TYPE(PrinterConfigNode);
TVM_REGISTER_GLOBAL("node.PrinterConfig").set_body_typed([](Map<String, ObjectRef> config_dict) {
  return PrinterConfig(config_dict);
});
TVM_REGISTER_GLOBAL("node.TVMScriptPrinterScript").set_body_typed(TVMScriptPrinter::Script);

}  // namespace tvm
