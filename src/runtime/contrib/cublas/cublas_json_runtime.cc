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
 * \file src/runtime/contrib/cublas/cublas_json_runtime.cc
 * \brief A simple JSON runtime for CUBLAS.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "cublas_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class CublasJSONRuntime : public JSONRuntimeBase {
 public:
  CublasJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                    const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  void Init(const Array<NDArray>& consts) override {}

  const char* type_key() const override { return "cublas_json"; }  // May be overridden

  void Run() override {
    auto* entry_ptr = tvm::contrib::CuBlasLtThreadEntry::ThreadLocal();

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        uint32_t output_eid = EntryID(outputs_[0]);
        auto out_ptr = data_entry_[output_eid];
        bool transa = false;
        bool transb = false;
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

        if (op_name.find("transposed") != std::string::npos) {
          transb = true;
        }

        if (op_name.find("relu") != std::string::npos) {
          epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
        } else if (op_name.find("gelu") != std::string::npos) {
          epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
        } else if (op_name.find("bias") != std::string::npos) {
          epilogue = CUBLASLT_EPILOGUE_BIAS;
        }

        auto get_inputs = [this](const JSONGraphNode& node, bool has_bias) {
          const DLTensor* bias = nullptr;
          if (has_bias) {
            bias = GetInput(node, 2);
          }
          return std::make_tuple(GetInput(node, 0), GetInput(node, 1), bias);
        };

        auto [a_ptr, b_ptr, bias_ptr] = get_inputs(node, epilogue != CUBLASLT_EPILOGUE_DEFAULT);

        tvm::contrib::CallCublasLt(entry_ptr->handle, stream, a_ptr, b_ptr, bias_ptr, out_ptr,
                                   transa, transb, epilogue);
      }
    }
  }

 private:
  const DLTensor* GetInput(const JSONGraphNode& node, const int idx) {
    ICHECK_LT(idx, node.GetInputs().size());
    auto eid = EntryID(node.GetInputs()[idx]);
    ICHECK(eid < data_entry_.size());
    return data_entry_[eid];
  }
};

runtime::Module CublasJSONRuntimeCreate(String symbol_name, String graph_json,
                                        const Array<String>& const_names) {
  auto n = make_object<CublasJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CublasJSONRuntimeCreate").set_body_typed(CublasJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cublas_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<CublasJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
