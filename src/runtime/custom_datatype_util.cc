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

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

TVM_DLL std::string GetCustomTypeName(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_name");
  CHECK(f) << "Function not found";
  return (*f)(type_code).operator std::string();
}

TVM_DLL uint8_t GetCustomTypeCode(const std::string& type_name) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_code");
  CHECK(f) << "Function not found";
  return (*f)(type_name).operator int();
}

TVM_DLL bool GetCustomTypeRegistered(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_registered");
  CHECK(f) << "Function not found";
  return (*f)(type_code).operator bool();
}

}  // namespace runtime
}  // namespace tvm
