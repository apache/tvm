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
#include <tvm/script/printer/config.h>
#include <tvm/script/printer/printer.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace {

ffi::Map<ffi::String, ffi::String>& NamespaceAliases() {
  static ffi::Map<ffi::String, ffi::String> aliases;
  return aliases;
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

}  // namespace

namespace script {
namespace printer {

void RegisterNamespaceAlias(const ffi::String& key, const ffi::String& default_alias) {
  TVM_FFI_ICHECK(!NamespaceAliases().count(key)) << "Duplicate printer dialect prefix: " << key;
  NamespaceAliases().Set(key, default_alias);
}

const ffi::Map<ffi::String, ffi::String>& GetNamespaceAliases() { return NamespaceAliases(); }

}  // namespace printer
}  // namespace script

PrinterConfig::PrinterConfig(ffi::Map<ffi::String, Any> config_dict) {
  ffi::ObjectPtr<PrinterConfigNode> n = ffi::make_object<PrinterConfigNode>();
  std::unordered_set<ffi::String> core_keys;
  auto get = [&](const char* key) {
    core_keys.insert(key);
    return config_dict.Get(key);
  };
  if (auto v = get("name")) {
    n->binding_names.push_back(v.value().as_or_throw<ffi::String>());
  }
  if (auto v = get("show_meta")) {
    n->show_meta = v.value().cast<bool>();
  }
  if (auto v = get("ir_prefix")) {
    n->ir_prefix = v.value().as_or_throw<ffi::String>();
  }
  if (auto v = get("module_alias")) {
    n->module_alias = v.value().as_or_throw<ffi::String>();
  }
  if (auto v = get("buffer_dtype")) {
    n->buffer_dtype = ffi::StringToDLDataType(v.value().as_or_throw<ffi::String>());
  }
  if (auto v = get("int_dtype")) {
    n->extra_config.Set(
        "ir.int_dtype",
        ffi::DLDataTypeToString(ffi::StringToDLDataType(v.value().as_or_throw<ffi::String>())));
  }
  if (auto v = get("float_dtype")) {
    n->extra_config.Set(
        "ir.float_dtype",
        ffi::DLDataTypeToString(ffi::StringToDLDataType(v.value().as_or_throw<ffi::String>())));
  }
  if (auto v = get("verbose_expr")) {
    n->verbose_expr = v.value().cast<bool>();
  }
  if (auto v = get("indent_spaces")) {
    n->indent_spaces = v.value().cast<int>();
  }
  if (auto v = get("print_line_numbers")) {
    n->print_line_numbers = v.value().cast<bool>();
  }
  if (auto v = get("num_context_lines")) {
    n->num_context_lines = v.value().cast<int>();
  }
  if (auto v = get("path_to_underline")) {
    n->path_to_underline =
        v.value().as_or_throw<ffi::Optional<ffi::Array<ffi::reflection::AccessPath>>>().value_or(
            ffi::Array<ffi::reflection::AccessPath>());
  }
  if (auto v = get("path_to_annotate")) {
    n->path_to_annotate =
        v.value()
            .as_or_throw<ffi::Optional<ffi::Map<ffi::reflection::AccessPath, ffi::String>>>()
            .value_or(ffi::Map<ffi::reflection::AccessPath, ffi::String>());
  }
  if (auto v = get("obj_to_underline")) {
    n->obj_to_underline =
        v.value().as_or_throw<ffi::Optional<ffi::Array<ffi::ObjectRef>>>().value_or(
            ffi::Array<ffi::ObjectRef>());
  }
  if (auto v = get("obj_to_annotate")) {
    n->obj_to_annotate =
        v.value().as_or_throw<ffi::Optional<ffi::Map<ffi::ObjectRef, ffi::String>>>().value_or(
            ffi::Map<ffi::ObjectRef, ffi::String>());
  }
  if (auto v = get("render_invisible_path_info")) {
    n->render_invisible_path_info = v.value().cast<bool>();
  }
  auto extra_config = get("extra_config");
  for (const auto& kv : config_dict) {
    if (!core_keys.count(kv.first)) {
      n->extra_config.Set(kv.first, kv.second);
    }
  }
  if (extra_config) {
    auto extra = extra_config.value().as_or_throw<ffi::Map<ffi::String, ffi::Any>>();
    for (auto kv : extra) {
      n->extra_config.Set(kv.first, kv.second);
    }
    if (auto render = extra.Get("render_invisible_path_info")) {
      n->render_invisible_path_info = render.value().cast<bool>();
    }
  }

  // Keep legacy constructor arguments as adapters; explicit extra_config values
  // take precedence. Store validated dtype strings with the existing defaults.
  for (const auto& [key, fallback] :
       {std::pair{"ir.int_dtype", "int32"}, std::pair{"ir.float_dtype", "void"}}) {
    ffi::String value = n->GetExtraConfig<ffi::String>(key, fallback);
    n->extra_config.Set(key, ffi::DLDataTypeToString(ffi::StringToDLDataType(value)));
  }

  // Validate all registered prefixes before translation assigns names.
  n->GetBuiltinKeywords();

  this->data_ = std::move(n);
}

ffi::Array<ffi::String> PrinterConfigNode::GetBuiltinKeywords() {
  TVM_FFI_ICHECK(IsIdentifier(std::string(ir_prefix))) << "Invalid `ir_prefix`: " << ir_prefix;
  TVM_FFI_ICHECK(module_alias.empty() || IsIdentifier(std::string(module_alias)))
      << "Invalid `module_alias`: " << module_alias;
  ffi::Array<ffi::String> result{ir_prefix};
  const auto& aliases = NamespaceAliases();
  std::vector<std::pair<ffi::String, ffi::String>> ordered(aliases.begin(), aliases.end());
  std::sort(ordered.begin(), ordered.end());
  for (const auto& [key, default_prefix] : ordered) {
    ffi::String prefix = GetExtraConfig<ffi::String>(key, default_prefix);
    TVM_FFI_ICHECK(IsIdentifier(std::string(prefix))) << "Invalid `" << key << "`: " << prefix;
    result.push_back(prefix);
  }
  if (!module_alias.empty()) {
    result.push_back(module_alias);
  }
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  PrinterConfigNode::RegisterReflection();
  ffi::reflection::GlobalDef().def(
      "node.PrinterConfig",
      [](ffi::Map<ffi::String, Any> config_dict) { return PrinterConfig(config_dict); });
}

}  // namespace tvm
