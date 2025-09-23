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
 * \file node/serialization.cc
 * \brief Utilities to serialize TVM AST/IR objects.
 */
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>

namespace tvm {

std::string SaveJSON(Any n) {
  int indent = 2;
  ffi::json::Object metadata{{"tvm_version", TVM_VERSION}};
  ffi::json::Value jgraph = ffi::ToJSONGraph(n, metadata);
  return ffi::json::Stringify(jgraph, indent);
}

Any LoadJSON(std::string json_str) {
  ffi::json::Value jgraph = ffi::json::Parse(json_str);
  return ffi::FromJSONGraph(jgraph);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("node.SaveJSON", SaveJSON).def("node.LoadJSON", LoadJSON);
}
}  // namespace tvm
