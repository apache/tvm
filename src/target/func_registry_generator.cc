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
 * Defines functions that generate FuncRegistry structs for C runtime.
 * \file func_registry_generator.cc
 */

#include "func_registry_generator.h"

#include <sstream>

namespace tvm {
namespace target {

std::string GenerateFuncRegistryNames(const Array<String>& function_names) {
  std::stringstream ss;

  unsigned char function_nums[sizeof(uint16_t)];
  *reinterpret_cast<uint16_t*>(function_nums) = function_names.size();
  for (auto f : function_nums) {
    ss << f;
  }

  for (auto f : function_names) {
    ss << f << '\0';
  }

  return ss.str();
}

}  // namespace target
}  // namespace tvm
