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
 * \file unique_name_supply.cc
 * \brief UniqueNameSupply that can be used to generate unique variable names.
 */
#include "tvm/ir/unique_name_supply.h"

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <sstream>
#include <utility>

namespace tvm {

UniqueNameSupply::UniqueNameSupply(const ffi::String& prefix,
                                   ffi::Map<ffi::String, int64_t> name_map) {
  auto n = ffi::make_object<UniqueNameSupplyNode>(prefix, std::move(name_map));
  data_ = std::move(n);
}

ffi::String UniqueNameSupplyNode::ReserveName(const ffi::String& name, bool add_prefix) {
  ffi::String final_name = name;
  if (add_prefix) {
    final_name = AddPrefixToName(name);
  }
  name_map.Set(final_name, 0);
  return final_name;
}

ffi::String UniqueNameSupplyNode::FreshName(const ffi::String& name, bool add_prefix,
                                            bool add_underscore) {
  ffi::String unique_name = name;
  if (unique_name.empty()) {
    unique_name = "v";
  }
  if (add_prefix) {
    unique_name = AddPrefixToName(unique_name);
  }
  return GetUniqueName(unique_name, add_underscore);
}

bool UniqueNameSupplyNode::ContainsName(const ffi::String& name, bool add_prefix) {
  ffi::String unique_name = name;
  if (add_prefix) {
    unique_name = AddPrefixToName(name);
  }
  return name_map.count(unique_name);
}

ffi::String UniqueNameSupplyNode::AddPrefixToName(const ffi::String& name) {
  if (prefix_.empty()) {
    return name;
  }

  std::ostringstream ss;
  ss << prefix_ << "_" << name;
  return ss.str();
}

std::string UniqueNameSupplyNode::GetUniqueName(std::string name, bool add_underscore) {
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  ffi::String name_key = name;
  auto it = name_map.find(name_key);
  if (it != name_map.end()) {
    auto new_name = name;
    int64_t suffix = (*it).second;
    while (name_map.count(ffi::String(new_name))) {
      std::ostringstream os;
      os << name << (add_underscore ? "_" : "") << (++suffix);
      new_name = os.str();
    }
    name_map.Set(name_key, suffix);
    name_map.Set(ffi::String(new_name), 0);
    return new_name;
  }
  name_map.Set(name_key, 0);
  return name;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  UniqueNameSupplyNode::RegisterReflection();
  refl::GlobalDef()
      .def("ir.UniqueNameSupply", [](ffi::String prefix) { return UniqueNameSupply(prefix); })
      .def_method("ir.UniqueNameSupply_FreshName", &UniqueNameSupplyNode::FreshName)
      .def_method("ir.UniqueNameSupply_ReserveName", &UniqueNameSupplyNode::ReserveName)
      .def_method("ir.UniqueNameSupply_ContainsName", &UniqueNameSupplyNode::ContainsName);
}

}  // namespace tvm
