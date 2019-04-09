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
 *  Copyright (c) 2017 by Contributors
 * \file nnvm/compiler/packed_func_ext.h
 * \brief Extension to enable packed functionn for nnvm types
 */
#ifndef NNVM_COMPILER_PACKED_FUNC_EXT_H_
#define NNVM_COMPILER_PACKED_FUNC_EXT_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <nnvm/graph.h>
#include <nnvm/symbolic.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace nnvm {
namespace compiler {

using tvm::runtime::PackedFunc;

using AttrDict = std::unordered_map<std::string, std::string>;

/*!
 * \brief Get PackedFunction from global registry and
 *  report error if it does not exist
 * \param name The name of the function.
 * \return The created PackedFunc.
 */
inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = tvm::runtime::Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}
}  // namespace compiler
}  // namespace nnvm

// Enable the graph and symbol object exchange.
namespace tvm {
namespace runtime {

template<>
struct extension_type_info<nnvm::Symbol> {
  static const int code = 16;
};

template<>
struct extension_type_info<nnvm::Graph> {
  static const int code = 17;
};

template<>
struct extension_type_info<nnvm::compiler::AttrDict> {
  static const int code = 18;
};

}  // namespace runtime
}  // namespace tvm
#endif  // NNVM_COMPILER_PACKED_FUNC_EXT_H_
