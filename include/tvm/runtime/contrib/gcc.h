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
 * \file gcc.h
 * \brief Test an example runtime module to interpreting a json string.
 */
#ifndef TVM_RUNTIME_CONTRIB_GCC_H_
#define TVM_RUNTIME_CONTRIB_GCC_H_

#include <dmlc/logging.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

class ExampleJSonModule : public ModuleNode {
 public:
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void run(int id, const std::vector<int>& inputs);

  const char* type_key() const { return "examplejson"; }

  void SaveToBinary(dmlc::Stream* stream) final {
    // Write to a json string.
  }

  // Note this is a very simple json that only serves for demostration purpose.
  // Users usually have their own format and they can serialize it using the
  // SaveToBinary method and deserialize it using LoadFromFile.
  void ParseJson(const std::string& json);

  static Module LoadFromFile(const std::string& json, const std::string& format) {
    auto n = tvm::runtime::make_object<ExampleJSonModule>();
    n->ParseJson(json);
    return Module(n);
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {}
  std::string GetSource(const std::string& format = "") final { return ""; }

 private:
  std::string curr_subgraph_;
  // op -> inputs
  std::map<std::string, std::map<int, std::vector<int> >> graph_;
  std::vector<NDArray> data_entry_;
  // id -> op
  std::vector<std::string> op_id_;
};
#endif  // TVM_RUNTIME_CONTRIB_GCC_H_
}  // namespace runtime
}  // namespace tvm
