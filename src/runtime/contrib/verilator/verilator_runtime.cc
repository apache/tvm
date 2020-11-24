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
 * \file src/runtime/contrib/verilator/verilator_runtime.cc
 * \brief A simple JSON runtime for Verilator.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "verilator_device.h"
#include "verilator_kernel.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class VerilatorJSONRuntime : public JSONRuntimeBase {
 public:
  VerilatorJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                       const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "verilator_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    std::vector<int*> in_ptr;
    std::vector<int*> out_ptr;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      uint32_t eid = EntryID(input_nodes_[i], 0);
      int* data = static_cast<int*>(data_entry_[eid]->data);
      in_ptr.push_back(data);
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      int* data = static_cast<int*>(data_entry_[eid]->data);
      out_ptr.push_back(data);
    }
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("add" == op_name) {
          auto entry = node.GetInputs()[0];
          auto shape = nodes_[entry.id_].GetOpShape()[entry.index_];
          verilator_add(device_, in_ptr[0], in_ptr[1], out_ptr[0], shape[0], shape[1]);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

 private:
  void BuildEngine() {
    device_ = VerilatorAlloc();
    // reset for 10 cycles
    VerilatorReset(device_, 10);
  }

  /* The verilator handle. */
  VerilatorHandle device_{nullptr};
};

runtime::Module VerilatorJSONRuntimeCreate(String symbol_name, String graph_json,
                                           const Array<String>& const_names) {
  auto n = make_object<VerilatorJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.VerilatorJSONRuntimeCreate")
    .set_body_typed(VerilatorJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_verilator_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<VerilatorJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
