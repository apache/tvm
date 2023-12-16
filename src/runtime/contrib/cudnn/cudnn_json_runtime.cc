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

  void Init(const Array<NDArray>& consts) override {
    auto* entry_ptr = tvm::contrib::CuDNNThreadEntry::ThreadLocal();
    auto func = tvm::runtime::Registry::Get("runtime.get_cuda_stream");
    ICHECK(func != nullptr);
    stream = static_cast<cudaStream_t>((*func)().operator void*());

    auto attr_in_name = [](const std::string& op_name, const std::string& attr_name) {
      return op_name.find(attr_name) != std::string::npos;
    };

    auto vstr2vint = [](const JSONGraphNode& node, const std::string& attrStr) {
      auto string_to_int = [](const std::string& str) { return std::stoi(str); };
      auto string_vec = node.GetAttr<std::vector<std::string>>(attrStr);
      std::vector<int> int_vec(string_vec.size());
      std::transform(string_vec.begin(), string_vec.end(), int_vec.begin(), string_to_int);
      return int_vec;
    };
    // get some config from the graph
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const auto& node = nodes_[i];
      if (node.GetOpType() == "kernel") {
        op_name = node.GetOpName();
        std::vector<int> input_dims, kernel_dims, output_dims;
        auto input_node = nodes_[0];
        auto input_shapes = input_node.GetOpShape()[0];
        auto kernel_node = nodes_[1];
        auto kernel_shapes = kernel_node.GetOpShape()[0];
        auto output_shapes = node.GetOpShape()[0];
        for (const auto& _i : input_shapes) {
          input_dims.emplace_back(static_cast<int>(_i));
        }
        for (const auto& _i : kernel_shapes) {
          kernel_dims.emplace_back(static_cast<int>(_i));
        }
        for (const auto& _i : output_shapes) {
          output_dims.emplace_back(static_cast<int>(_i));
        }
        has_bias = attr_in_name(op_name, "bias");
        groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
        padding = vstr2vint(node, "padding");
        strides = vstr2vint(node, "strides");
        dilation = vstr2vint(node, "dilation");
        conv_dtype = node.GetAttr<std::vector<std::string>>("out_dtype")[0];
        std::string layout = node.GetAttr<std::vector<std::string>>("out_layout")[0];
        dims = layout.size() - 2;  // remove O and I dims

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
        this->handle = entry_ptr->handle;
        this->kernel_node = node;

        // find best algo
        TVMRetValue best_algo;

        tvm::contrib::FindAlgo(format, dims, groups, padding.data(), strides.data(),
                               dilation.data(), input_dims.data(), kernel_dims.data(),
                               output_dims.data(), conv_dtype, conv_dtype, false, &best_algo);

        this->algo = best_algo.operator int();
      }
    }
  }

  const char* type_key() const override { return "cudnn_json"; }  // May be overridden

  void Run() override {
    auto get_inputs = [this](const JSONGraphNode& node, bool has_bias) {
      const DLTensor* bias = nullptr;
      if (has_bias) {
        bias = GetInput(node, 2);
      }
      return std::make_tuple(GetInput(node, 0), GetInput(node, 1), bias);
    };

    auto [a_ptr, b_ptr, bias_ptr] = get_inputs(kernel_node, has_bias);
    uint32_t output_eid = EntryID(outputs_[0]);
    auto out_ptr = data_entry_[output_eid];

    if (this->has_bias) {
      tvm::contrib::ConvolutionBiasActivationForward(
          this->mode, this->format, this->algo, this->dims, this->groups, this->act, this->coef,
          this->padding.data(), this->strides.data(), this->dilation.data(), a_ptr, b_ptr, out_ptr,
          bias_ptr, this->conv_dtype);
    } else {
      tvm::contrib::ConvolutionForward(
          this->mode, this->format, this->algo, this->dims, this->groups, this->padding.data(),
          this->strides.data(), this->dilation.data(), a_ptr, b_ptr, out_ptr, this->conv_dtype);
    }
  }

 private:
  const DLTensor* GetInput(const JSONGraphNode& node, const int idx) {
    ICHECK_LT(idx, node.GetInputs().size());
    auto eid = EntryID(node.GetInputs()[idx]);
    ICHECK(eid < data_entry_.size());
    return data_entry_[eid];
  }
  /*conv op name*/
  std::string op_name;
  /*conv mode: CUDNN_CROSS_CORRELATION by default*/
  int mode = CUDNN_CROSS_CORRELATION;
  /*algo: by default we select the implicit gemm algo, will be tuned in the initial pass.*/
  int algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  /*if has bias*/
  bool has_bias = false;
  /*args for function call*/
  int act = CUDNN_ACTIVATION_IDENTITY;
  double coef = 1.0;
  int format = CUDNN_TENSOR_NHWC;
  int dims = 2;
  int groups = 1;
  std::vector<int> padding;
  std::vector<int> strides;
  std::vector<int> dilation;
  std::string conv_dtype;
  cudaStream_t stream;
  cudnnHandle_t handle;
  tvm::runtime::json::JSONGraphNode kernel_node;
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
