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

#include "name_transforms.h"

#include <tvm/runtime/registry.h>

#include <cctype>
#include <string>

namespace tvm {
namespace relay {
namespace backend {

std::string ToCFunctionStyle(const std::string& original_name) {
  ICHECK(!original_name.empty());

  int tvm_prefix_length = 3;
  std::string function_name("TVM");

  bool new_block = true;
  for (const char& symbol : original_name.substr(tvm_prefix_length)) {
    if (std::isalpha(symbol)) {
      if (new_block) {
        function_name.push_back(std::toupper(symbol));
        new_block = false;
      } else {
        function_name.push_back(std::tolower(symbol));
      }
    } else if (symbol == '_') {
      new_block = true;
    }
  }
  return function_name;
}

std::string ToCVariableStyle(const std::string& original_name) {
  ICHECK(!original_name.empty());

  std::string variable_name;
  variable_name.resize(original_name.size());

  std::transform(original_name.begin(), original_name.end(), variable_name.begin(), ::tolower);
  return variable_name;
}

std::string CombineNames(const Array<String>& names) {
  std::stringstream combine_stream;
  ICHECK(!names.empty());

  for (const String& name : names) {
    ICHECK(!name.empty());
    combine_stream << name << "_";
  }

  std::string combined_name = combine_stream.str();
  combined_name.pop_back();
  return combined_name;
}

std::string SanitiseName(const std::string& name) {
  ICHECK(!name.empty());

  auto multipleSeparators = [](char before, char after) {
    return before == '_' && before == after;
  };
  auto isNotAlnum = [](char c) { return !std::isalnum(c); };
  std::string sanitised_input = name;
  std::replace_if(sanitised_input.begin(), sanitised_input.end(), isNotAlnum, '_');

  sanitised_input.erase(
      std::unique(sanitised_input.begin(), sanitised_input.end(), multipleSeparators),
      sanitised_input.end());

  return sanitised_input;
}

TVM_REGISTER_GLOBAL("relay.backend.ToCFunctionStyle").set_body_typed(ToCFunctionStyle);
TVM_REGISTER_GLOBAL("relay.backend.ToCVariableStyle").set_body_typed(ToCVariableStyle);
TVM_REGISTER_GLOBAL("relay.backend.PrefixName").set_body_typed(PrefixName);
TVM_REGISTER_GLOBAL("relay.backend.PrefixGeneratedName").set_body_typed(PrefixGeneratedName);
TVM_REGISTER_GLOBAL("relay.backend.SanitiseName").set_body_typed(SanitiseName);

}  // namespace backend
}  // namespace relay
}  // namespace tvm
