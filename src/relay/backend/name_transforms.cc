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

#include <tvm/runtime/name_transforms.h>
#include <tvm/runtime/registry.h>

#include <cctype>
#include <string>

namespace tvm {
namespace relay {
namespace backend {

std::string ToCFunctionStyle(const std::string& original_name) {
  ICHECK(!original_name.empty()) << "Function name is empty";
  ICHECK_EQ(original_name.find("TVM"), 0) << "Function not TVM prefixed";

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
  ICHECK(!original_name.empty()) << "Variable name is empty";
  ICHECK_EQ(original_name.find("TVM"), 0) << "Variable not TVM prefixed";

  std::string variable_name;
  variable_name.resize(original_name.size());

  std::transform(original_name.begin(), original_name.end(), variable_name.begin(), ::tolower);
  return variable_name;
}

std::string ToCConstantStyle(const std::string& original_name) {
  ICHECK_EQ(original_name.find("TVM"), 0) << "Constant not TVM prefixed";
  std::string constant_name = ToCVariableStyle(original_name);

  std::transform(constant_name.begin(), constant_name.end(), constant_name.begin(), ::toupper);
  return constant_name;
}

std::string CombineNames(const Array<String>& names) {
  std::stringstream combine_stream;
  ICHECK(!names.empty()) << "Name segments empty";

  for (const String& name : names) {
    ICHECK(!name.empty()) << "Name segment is empty";
    combine_stream << name << "_";
  }

  std::string combined_name = combine_stream.str();
  combined_name.pop_back();
  return combined_name;
}

TVM_REGISTER_GLOBAL("relay.backend.ToCFunctionStyle").set_body_typed(ToCFunctionStyle);
TVM_REGISTER_GLOBAL("relay.backend.ToCVariableStyle").set_body_typed(ToCVariableStyle);
TVM_REGISTER_GLOBAL("relay.backend.ToCConstantStyle").set_body_typed(ToCConstantStyle);
TVM_REGISTER_GLOBAL("relay.backend.PrefixName").set_body_typed(PrefixName);
TVM_REGISTER_GLOBAL("relay.backend.PrefixGeneratedName").set_body_typed(PrefixGeneratedName);

}  // namespace backend
}  // namespace relay
}  // namespace tvm
