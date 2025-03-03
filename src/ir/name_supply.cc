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
 * \file name_supply.cc
 * \brief NameSupply that can be used to generate unique variable names.
 */
#include "tvm/ir/name_supply.h"

#include <tvm/runtime/registry.h>

#include <utility>

namespace tvm {

NameSupply::NameSupply(const String& prefix, std::unordered_map<std::string, int> name_map) {
  auto n = make_object<NameSupplyNode>(prefix, std::move(name_map));
  data_ = std::move(n);
}

String NameSupplyNode::ReserveName(const String& name, bool add_prefix) {
  String final_name = name;
  if (add_prefix) {
    final_name = add_prefix_to_name(name);
  }
  name_map[final_name] = 0;
  return final_name;
}

String NameSupplyNode::FreshName(const String& name, bool add_prefix, bool add_underscore) {
  String unique_name = name;
  if (add_prefix) {
    unique_name = add_prefix_to_name(name);
  }
  unique_name = GetUniqueName(unique_name, add_underscore);
  return unique_name;
}

bool NameSupplyNode::ContainsName(const String& name, bool add_prefix) {
  String unique_name = name;
  if (add_prefix) {
    unique_name = add_prefix_to_name(name);
  }

  return name_map.count(unique_name);
}

String NameSupplyNode::add_prefix_to_name(const String& name) {
  if (prefix_.empty()) {
    return name;
  }

  std::ostringstream ss;
  ICHECK(name.defined());
  ss << prefix_ << "_" << name;
  return ss.str();
}

std::string NameSupplyNode::GetUniqueName(std::string name, bool add_underscore) {
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  auto it = name_map.find(name);
  if (it != name_map.end()) {
    auto new_name = name;
    while (!name_map.insert({new_name, 0}).second) {
      std::ostringstream os;
      os << name << (add_underscore ? "_" : "") << (++it->second);
      new_name = os.str();
    }
    return new_name;
  }
  name_map[name] = 0;
  return name;
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

}  // namespace tvm
