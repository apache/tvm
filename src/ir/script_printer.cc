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
#include <tvm/ffi/extra/dataclass.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/repr.h>
#include <tvm/script/printer/config.h>

#include <algorithm>

namespace tvm {

using AccessPath = ffi::reflection::AccessPath;

TVM_FFI_STATIC_INIT_BLOCK() { PrinterConfigNode::RegisterReflection(); }

TVMScriptPrinter::FType& TVMScriptPrinter::vtable() {
  static FType inst;
  return inst;
}

std::string TVMScriptPrinter::Script(const ffi::ObjectRef& node,
                                     const ffi::Optional<PrinterConfig>& cfg) {
  if (!TVMScriptPrinter::vtable().can_dispatch(node)) {
    // Fall back to ffi::ReprPrint for types not registered with TVMScriptPrinter.
    return std::string(ffi::ReprPrint(ffi::Any(node)));
  }
  return TVMScriptPrinter::vtable()(node, cfg.value_or(PrinterConfig()));
}

bool IsIdentifier(const std::string& name) {
  // Python identifiers follow the regex: "^[a-zA-Z_][a-zA-Z0-9_]*$"
  // `std::regex` would cause a symbol conflict with PyTorch, we avoids to use it in the codebase.
  //
  // We convert the regex into following conditions:
  // 1. The name is not empty.
  // 2. The first character is either an alphabet or an underscore.
  // 3. The rest of the characters are either an alphabet, a digit or an underscore.
  return name.size() > 0 &&                            //
         (std::isalpha(name[0]) || name[0] == '_') &&  //
         std::all_of(name.begin() + 1, name.end(),
                     [](char c) { return std::isalnum(c) || c == '_'; });
}

PrinterConfig::PrinterConfig(ffi::Map<ffi::String, Any> config_dict) {
  ffi::ObjectPtr<PrinterConfigNode> n = ffi::make_object<PrinterConfigNode>();
  if (auto v = config_dict.Get("name")) {
    n->binding_names.push_back(Downcast<ffi::String>(v.value()));
  }
  if (auto v = config_dict.Get("show_meta")) {
    n->show_meta = v.value().cast<bool>();
  }
  if (auto v = config_dict.Get("ir_prefix")) {
    n->ir_prefix = Downcast<ffi::String>(v.value());
  }
  if (auto v = config_dict.Get("module_alias")) {
    n->module_alias = Downcast<ffi::String>(v.value());
  }
  if (auto v = config_dict.Get("int_dtype")) {
    n->int_dtype = DataType(ffi::StringToDLDataType(Downcast<ffi::String>(v.value())));
  }
  if (auto v = config_dict.Get("float_dtype")) {
    n->float_dtype = DataType(ffi::StringToDLDataType(Downcast<ffi::String>(v.value())));
  }
  if (auto v = config_dict.Get("verbose_expr")) {
    n->verbose_expr = v.value().cast<bool>();
  }
  if (auto v = config_dict.Get("indent_spaces")) {
    n->indent_spaces = v.value().cast<int>();
  }
  if (auto v = config_dict.Get("print_line_numbers")) {
    n->print_line_numbers = v.value().cast<bool>();
  }
  if (auto v = config_dict.Get("num_context_lines")) {
    n->num_context_lines = v.value().cast<int>();
  }
  if (auto v = config_dict.Get("path_to_underline")) {
    n->path_to_underline =
        Downcast<ffi::Optional<ffi::Array<AccessPath>>>(v).value_or(ffi::Array<AccessPath>());
  }
  if (auto v = config_dict.Get("path_to_annotate")) {
    n->path_to_annotate = Downcast<ffi::Optional<ffi::Map<AccessPath, ffi::String>>>(v).value_or(
        ffi::Map<AccessPath, ffi::String>());
  }
  if (auto v = config_dict.Get("obj_to_underline")) {
    n->obj_to_underline = Downcast<ffi::Optional<ffi::Array<ffi::ObjectRef>>>(v).value_or(
        ffi::Array<ffi::ObjectRef>());
  }
  if (auto v = config_dict.Get("obj_to_annotate")) {
    n->obj_to_annotate = Downcast<ffi::Optional<ffi::Map<ffi::ObjectRef, ffi::String>>>(v).value_or(
        ffi::Map<ffi::ObjectRef, ffi::String>());
  }
  if (auto v = config_dict.Get("syntax_sugar")) {
    n->syntax_sugar = v.value().cast<bool>();
  }
  if (auto v = config_dict.Get("show_object_address")) {
    n->show_object_address = v.value().cast<bool>();
  }
  // Dialect-specific keys are stored in extra_config with dotted-name keys.
  // String-typed dialect keys passed through directly.
  for (const char* key : {"tirx.prefix", "relax.prefix"}) {
    if (auto v = config_dict.Get(key)) {
      n->extra_config.Set(ffi::String(key), v.value());
    }
  }
  // "tirx.buffer_dtype" is passed as a DLDataType string from Python; convert to DataType.
  if (auto v = config_dict.Get("tirx.buffer_dtype")) {
    DataType dt(ffi::StringToDLDataType(Downcast<ffi::String>(v.value())));
    n->extra_config.Set(ffi::String("tirx.buffer_dtype"), ffi::Any(dt));
  }
  // Boolean dialect keys.
  if (auto v = config_dict.Get("relax.show_all_struct_info")) {
    n->extra_config.Set(ffi::String("relax.show_all_struct_info"), v.value());
  }
  if (auto v = config_dict.Get("extra_config")) {
    auto extra = Downcast<ffi::Map<ffi::String, ffi::Any>>(v.value());
    for (auto kv : extra) {
      n->extra_config.Set(kv.first, kv.second);
    }
  }

  // Checking prefixes if they are valid Python identifiers.
  TVM_FFI_ICHECK(IsIdentifier(std::string(n->ir_prefix)))
      << "Invalid `ir_prefix`: " << n->ir_prefix;
  ffi::String tir_prefix = n->GetExtraConfig<ffi::String>("tirx.prefix", "T");
  ffi::String relax_prefix = n->GetExtraConfig<ffi::String>("relax.prefix", "R");
  TVM_FFI_ICHECK(IsIdentifier(std::string(tir_prefix))) << "Invalid `tirx.prefix`: " << tir_prefix;
  TVM_FFI_ICHECK(IsIdentifier(std::string(relax_prefix)))
      << "Invalid `relax.prefix`: " << relax_prefix;
  TVM_FFI_ICHECK(n->module_alias.empty() || IsIdentifier(std::string(n->module_alias)))
      << "Invalid `module_alias`: " << n->module_alias;

  this->data_ = std::move(n);
}

ffi::Array<ffi::String> PrinterConfigNode::GetBuiltinKeywords() {
  ffi::String tir_prefix = GetExtraConfig<ffi::String>("tirx.prefix", "T");
  ffi::String relax_prefix = GetExtraConfig<ffi::String>("relax.prefix", "R");
  ffi::Array<ffi::String> result{this->ir_prefix, tir_prefix, relax_prefix};
  if (!this->module_alias.empty()) {
    result.push_back(this->module_alias);
  }
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("node.PrinterConfig",
           [](ffi::Map<ffi::String, Any> config_dict) { return PrinterConfig(config_dict); })
      .def("node.TVMScriptPrinterScript", TVMScriptPrinter::Script);
}

}  // namespace tvm
