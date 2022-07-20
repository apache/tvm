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

#include "tvm/ir/name_supply.h"

#include <tvm/runtime/registry.h>

#include <utility>

namespace tvm {

NameSupply::NameSupply() : NameSupply("") {}

NameSupply::NameSupply(const String& prefix, std::unordered_map<std::string, int> name_map) {
  auto n = make_object<NameSupplyNode>(prefix);
  n->name_map = std::move(name_map);
  data_ = std::move(n);
}

NameSupplyNode::NameSupplyNode(const String& prefix) : prefix_(prefix) {}

String NameSupplyNode::ReserveName(const String& name, bool add_prefix) {
  String final_name = name;
  if (add_prefix) {
    final_name = prefix_module_name(name);
  }
  name_map[final_name] = 0;
  return final_name;
}

String NameSupplyNode::FreshName(const String& name, bool add_prefix) {
  String unique_name = name;
  if (add_prefix) {
    unique_name = prefix_module_name(name);
  }
  unique_name = GetUniqueName(unique_name);
  return unique_name;
}

bool NameSupplyNode::ContainsName(const String& name, bool add_prefix) {
  String unique_name = name;
  if (add_prefix) {
    unique_name = prefix_module_name(name);
  }

  return name_map.count(unique_name);
}

void NameSupplyNode::Clear() { name_map.clear(); }

String NameSupplyNode::prefix_module_name(const String& name) {
  if (prefix_.empty()) {
    return name;
  }

  std::stringstream ss;
  ICHECK(name.defined());
  ss << prefix_ << "_" << name;
  return ss.str();
}

std::string NameSupplyNode::GetUniqueName(std::string prefix) {
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] == '.') prefix[i] = '_';
  }
  auto it = name_map.find(prefix);
  if (it != name_map.end()) {
    while (true) {
      std::ostringstream os;
      os << prefix << (++it->second);
      std::string name = os.str();
      if (name_map.count(name) == 0) {
        prefix = name;
        break;
      }
    }
  }
  name_map[prefix] = 0;
  return prefix;
}

TVM_REGISTER_NODE_TYPE(NameSupplyNode);

TVM_REGISTER_GLOBAL("ir.NameSupply").set_body_typed([](String prefix) {
  return NameSupply(prefix);
});

TVM_REGISTER_GLOBAL("ir.NameSupply_FreshName")
    .set_body_method<NameSupply>(&NameSupplyNode::FreshName);

TVM_REGISTER_GLOBAL("ir.NameSupply_ReserveName")
    .set_body_method<NameSupply>(&NameSupplyNode::ReserveName);

TVM_REGISTER_GLOBAL("ir.NameSupply_ContainsName")
    .set_body_method<NameSupply>(&NameSupplyNode::ContainsName);

TVM_REGISTER_GLOBAL("ir.NameSupply_Clear").set_body_method<NameSupply>(&NameSupplyNode::Clear);

}  // namespace tvm
