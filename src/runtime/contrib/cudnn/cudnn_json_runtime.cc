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
 * \file src/runtime/contrib/cudnn/cudnn_json_runtime.cc
 * \brief A simple JSON runtime for CUDNN.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <regex>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "cudnn_utils.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class cuDNNJSONRuntime : public JSONRuntimeBase {
 public:
  cuDNNJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                   const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  void Init(const Array<NDArray>& consts) override {}

  const char* type_key() const override { return "cudnn_json"; }  // May be overridden

  void Run() override {
    auto* entry_ptr = tvm::contrib::CuDNNThreadEntry::ThreadLocal();

    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    cudaStream_t stream = static_cast<cudaStream_t>((*func)().operator void*());

    auto getVecIntAttrFromVecStr = [this](const JSONGraphNode& node, const std::string& attrStr) {
      auto stringToInt = [](const std::string& str) { return std::stoi(str); };
      auto stringVec = node.GetAttr<std::vector<std::string>>(attrStr);
      std::vector<int> intVec(stringVec.size());
      std::transform(stringVec.begin(), stringVec.end(), intVec.begin(), stringToInt);
      return intVec;
    };

    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        uint32_t output_eid = EntryID(outputs_[0]);
        auto out_ptr = data_entry_[output_eid];

        auto attr_in_name = [this](const std::string& op_name, const std::string& attr_name) {
          return std::regex_search(op_name, std::regex(attr_name));
        };
        bool has_bias = attr_in_name(op_name, "bias");

        auto get_inputs = [this](const JSONGraphNode& node, bool has_bias) {
          const DLTensor* bias = nullptr;
          if (has_bias) {
            bias = GetInput(node, 2);
          }
          return std::make_tuple(GetInput(node, 0), GetInput(node, 1), bias);
        };

        auto [a_ptr, b_ptr, bias_ptr] = get_inputs(node, has_bias);

        int mode = CUDNN_CROSS_CORRELATION;  // always use cross-correlation
        int format = CUDNN_TENSOR_NHWC;
        int act = CUDNN_ACTIVATION_IDENTITY;  // identity activation by default
        // todo(leiwang1999): how to add algo selection support in warmup?
        int algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        int dims = b_ptr->ndim - 2;  // remove O and I dims
        int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
        double coef = 1.0;
        std::vector<int> padding = getVecIntAttrFromVecStr(node, "padding");
        std::vector<int> strides = getVecIntAttrFromVecStr(node, "strides");
        std::vector<int> dilation = getVecIntAttrFromVecStr(node, "dilation");
        std::string conv_dtype = node.GetAttr<std::vector<std::string>>("out_dtype")[0];
        std::string layout = node.GetAttr<std::vector<std::string>>("out_layout")[0];
        if (layout == "NCHW")
          format = CUDNN_TENSOR_NCHW;
        else if (layout == "NHWC")
          format = CUDNN_TENSOR_NHWC;
        else
          LOG(FATAL) << "Unsupported layout: " << layout;

        if (attr_in_name(op_name, "relu")) {
          act = CUDNN_ACTIVATION_RELU;
        } else if (attr_in_name(op_name, "relu6")) {
          act = CUDNN_ACTIVATION_CLIPPED_RELU;
          coef = 6.0;
        } else if (attr_in_name(op_name, "leaky_relu")) {
          act = CUDNN_ACTIVATION_RELU;
          coef = 0.1;
        }

        if (has_bias) {
          tvm::contrib::CallCudnnConvolutionBiasActivationForward(
              entry_ptr->handle, stream, mode, format, algo, dims, groups, act, coef,
              padding.data(), strides.data(), dilation.data(), a_ptr, b_ptr, out_ptr, bias_ptr,
              conv_dtype);
        } else
          tvm::contrib::CallCudnnConvolutionForward(
              entry_ptr->handle, stream, mode, format, algo, dims, groups, padding.data(),
              strides.data(), dilation.data(), a_ptr, b_ptr, out_ptr, conv_dtype);
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

runtime::Module cuDNNJSONRuntimeCreate(String symbol_name, String graph_json,
                                       const Array<String>& const_names) {
  auto n = make_object<cuDNNJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.cuDNNJSONRuntimeCreate").set_body_typed(cuDNNJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cudnn_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<cuDNNJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
