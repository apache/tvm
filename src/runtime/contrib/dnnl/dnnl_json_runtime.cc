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
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "dnnl.hpp"
#include "../../json/json_node.h"
#include "../../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  explicit DNNLJSONRuntime(const std::string& graph_json) : JSONRuntimeBase(graph_json) {}
  ~DNNLJSONRuntime() = default;

  void Run() override {
    // Invoke the engine and return the result
  }

  void Init() override {
    // Create a engine here
  }

  void BuildEngine() {
    for (size_t nid = 0; nid < this->nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "input") {
        // Handle inputs
      } else {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        // Handle kernel
        for (const auto& e : node.GetInputs()) {
          // uint32_t eid = this->EntryID(e);
          // shape/type for the i-th input
          // std::vector<int64_t> shape = node.GetShape()[e.index_];
          // DLDataType dltype = node.GetDataType()[e.index_];
        }
      }
    }
  }

  void Conv2d() {
  }

  void Dense() {
  }

  void BatchNorm() {
  }

  void Relu() {
  }

  // Macro for add, subtract, multiply...

 private:
  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
};

TVM_REGISTER_GLOBAL("runtime.ext.dnnl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  auto n = tvm::runtime::make_object<DNNLJSONRuntime>(args[0].operator std::string());
  *rv = Module(n);
});

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
