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
 * \file hexagon_module.cc
 * \brief The HexagonLibraryModuleNode
 */
#include "../hexagon_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

#include "../../dso_library.h"
#include "hexagon_buffer.h"
#include "hexagon_common.h"

namespace tvm {
namespace runtime {

Module HexagonModuleCreate(std::string data, std::string fmt,
                           std::unordered_map<std::string, FunctionInfo> fmap, std::string asm_str,
                           std::string obj_str, std::string ir_str, std::string bc_str,
                           const std::set<std::string>& packed_c_abi) {
  CHECK(fmt == "so") << "Invalid format provided when constructing Hexagon runtime module: " << fmt
                     << ". Valid formats are: 'so'.";
  auto n = make_object<DSOLibrary>();
  n->Init(data);
  return CreateModuleFromLibrary(n, hexagon::WrapPackedFunc);
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_hexagon").set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = make_object<DSOLibrary>();
  n->Init(args[0]);
  *rv = CreateModuleFromLibrary(n, hexagon::WrapPackedFunc);
});
}  // namespace runtime
}  // namespace tvm
