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
 * \file src/runtime/contrib/xnnpack/xnnpack_json_runtime.cc
 * \brief Minimal XNNPACK JSON runtime.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/tensor.h>
#include <xnnpack.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class XNNPACKJSONRuntime : public JSONRuntimeBase {
 public:
  XNNPACKJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                     const ffi::Array<ffi::String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  ~XNNPACKJSONRuntime() {
    if (runtime_ != nullptr) {
      xnn_delete_runtime(runtime_);
      runtime_ = nullptr;
    }
    if (subgraph_ != nullptr) {
      xnn_delete_subgraph(subgraph_);
      subgraph_ = nullptr;
    }
  }

  const char* kind() const override { return "xnnpack_json"; }

  void Init(const ffi::Array<Tensor>& consts) override {
    TVM_FFI_ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required constants.";

    SetupConstants(consts);

    const xnn_status status = xnn_initialize(nullptr);
    TVM_FFI_ICHECK_EQ(status, xnn_status_success)
        << "Failed to initialize XNNPACK runtime. xnn_initialize returned status " << status;

    // TODO(XNNPACK): XNNPACK may read XNN_EXTRA_BYTES past tensor bounds. Operator lowering must
    // ensure buffers passed to XNNPACK satisfy this padding contract.
    // TODO(XNNPACK): Static weight tensors passed into XNNPACK must outlive XNNPACK subgraphs,
    // runtimes, and operator objects that reference them.
    BuildRuntime();
  }

  void Run() override {
    TVM_FFI_ICHECK(runtime_ != nullptr) << "XNNPACK runtime has not been built.";
    TVM_FFI_ICHECK(input_eid_ < data_entry_.size());
    TVM_FFI_ICHECK(output_eid_ < data_entry_.size());

    const DLTensor* input = data_entry_[input_eid_];
    const DLTensor* output = data_entry_[output_eid_];
    ValidateTensor(input, input_shape_, "input");
    ValidateTensor(output, output_shape_, "output");

    const size_t input_bytes = NumElements(input_shape_) * sizeof(float);
    const size_t output_bytes = NumElements(output_shape_) * sizeof(float);
    input_buffer_.resize(input_bytes + XNN_EXTRA_BYTES);
    output_buffer_.resize(output_bytes + XNN_EXTRA_BYTES);
    std::memcpy(input_buffer_.data(), TensorData(input), input_bytes);
    std::memset(input_buffer_.data() + input_bytes, 0, XNN_EXTRA_BYTES);
    std::memset(output_buffer_.data(), 0, output_bytes + XNN_EXTRA_BYTES);

    CheckXNNStatus(
        xnn_reshape_external_value(runtime_, input_eid_, input_shape_.size(), input_shape_.data()),
        "xnn_reshape_external_value(input)");
    CheckXNNStatus(xnn_reshape_external_value(runtime_, output_eid_, output_shape_.size(),
                                              output_shape_.data()),
                   "xnn_reshape_external_value(output)");
    CheckXNNStatus(xnn_reshape_runtime(runtime_), "xnn_reshape_runtime");

    std::vector<xnn_external_value> external_values{
        {input_eid_, input_buffer_.data()},
        {output_eid_, output_buffer_.data()},
    };
    CheckXNNStatus(xnn_setup_runtime_v2(runtime_, external_values.size(), external_values.data()),
                   "xnn_setup_runtime_v2");
    CheckXNNStatus(xnn_invoke_runtime(runtime_), "xnn_invoke_runtime");

    std::memcpy(MutableTensorData(output), output_buffer_.data(), output_bytes);
  }

 private:
  static void CheckXNNStatus(xnn_status status, const char* call) {
    TVM_FFI_ICHECK_EQ(status, xnn_status_success) << call << " failed with status " << status;
  }

  static bool IsFloat32(const DLDataType& dtype) {
    return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
  }

  static size_t NumElements(const std::vector<size_t>& shape) {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1),
                           std::multiplies<size_t>());
  }

  static const void* TensorData(const DLTensor* tensor) {
    return static_cast<const void*>(static_cast<const uint8_t*>(tensor->data) +
                                    tensor->byte_offset);
  }

  static void* MutableTensorData(const DLTensor* tensor) {
    return static_cast<void*>(static_cast<uint8_t*>(tensor->data) + tensor->byte_offset);
  }

  static void ValidateTensor(const DLTensor* tensor, const std::vector<size_t>& expected_shape,
                             const char* name) {
    TVM_FFI_ICHECK(tensor != nullptr) << "Missing XNNPACK " << name << " tensor.";
    TVM_FFI_ICHECK_EQ(tensor->device.device_type, kDLCPU)
        << "XNNPACK " << name << " tensor must be on CPU.";
    TVM_FFI_ICHECK(IsFloat32(tensor->dtype)) << "XNNPACK " << name << " tensor must be float32.";
    TVM_FFI_ICHECK_EQ(static_cast<size_t>(tensor->ndim), expected_shape.size())
        << "XNNPACK " << name << " tensor rank mismatch.";

    for (size_t i = 0; i < expected_shape.size(); ++i) {
      TVM_FFI_ICHECK_EQ(static_cast<size_t>(tensor->shape[i]), expected_shape[i])
          << "XNNPACK " << name << " tensor shape mismatch at dim " << i << ".";
    }

    if (tensor->strides != nullptr) {
      int64_t expected_stride = 1;
      for (int i = tensor->ndim - 1; i >= 0; --i) {
        TVM_FFI_ICHECK_EQ(tensor->strides[i], expected_stride)
            << "XNNPACK " << name << " tensor must be compact.";
        expected_stride *= tensor->shape[i];
      }
    }
  }

  static std::vector<size_t> GetShape(const JSONGraphNode& node, uint32_t index) {
    auto shapes = node.GetOpShape();
    TVM_FFI_ICHECK_LT(index, shapes.size());
    std::vector<size_t> shape;
    for (int64_t dim : shapes[index]) {
      TVM_FFI_ICHECK_GT(dim, 0) << "XNNPACK only supports static positive shapes.";
      shape.push_back(static_cast<size_t>(dim));
    }
    return shape;
  }

  static void CheckDType(const JSONGraphNode& node, uint32_t index) {
    auto dtypes = node.GetOpDataType();
    TVM_FFI_ICHECK_LT(index, dtypes.size());
    TVM_FFI_ICHECK(IsFloat32(dtypes[index])) << "XNNPACK only supports float32 tensors.";
  }

  void BuildRuntime() {
    TVM_FFI_ICHECK_EQ(const_idx_.size(), 0U) << "XNNPACK ReLU does not use constants.";
    TVM_FFI_ICHECK_EQ(input_var_eid_.size(), 1U) << "XNNPACK ReLU expects one input.";
    TVM_FFI_ICHECK_EQ(outputs_.size(), 1U) << "XNNPACK ReLU expects one output.";

    const JSONGraphNodeEntry output_entry = outputs_[0];
    TVM_FFI_ICHECK_LT(output_entry.id_, nodes_.size());
    const JSONGraphNode& kernel_node = nodes_[output_entry.id_];
    TVM_FFI_ICHECK_EQ(kernel_node.GetOpType(), "kernel");
    TVM_FFI_ICHECK_EQ(kernel_node.GetOpName(), "xnnpack.relu");

    auto inputs = kernel_node.GetInputs();
    TVM_FFI_ICHECK_EQ(inputs.size(), 1U) << "xnnpack.relu expects exactly one input.";
    const JSONGraphNodeEntry input_entry = inputs[0];
    TVM_FFI_ICHECK_LT(input_entry.id_, nodes_.size());

    input_eid_ = EntryID(input_entry);
    output_eid_ = EntryID(output_entry);

    CheckDType(nodes_[input_entry.id_], input_entry.index_);
    CheckDType(kernel_node, output_entry.index_);
    input_shape_ = GetShape(nodes_[input_entry.id_], input_entry.index_);
    output_shape_ = GetShape(kernel_node, output_entry.index_);
    TVM_FFI_ICHECK(input_shape_ == output_shape_) << "XNNPACK ReLU input/output shapes must match.";

    CheckXNNStatus(xnn_create_subgraph(NumEntries(), 0, &subgraph_), "xnn_create_subgraph");

    uint32_t input_id = XNN_INVALID_VALUE_ID;
    CheckXNNStatus(xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, input_shape_.size(),
                                           input_shape_.data(), nullptr, input_eid_,
                                           XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id),
                   "xnn_define_tensor_value(input)");
    TVM_FFI_ICHECK_EQ(input_id, input_eid_);

    uint32_t output_id = XNN_INVALID_VALUE_ID;
    CheckXNNStatus(xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, output_shape_.size(),
                                           output_shape_.data(), nullptr, output_eid_,
                                           XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id),
                   "xnn_define_tensor_value(output)");
    TVM_FFI_ICHECK_EQ(output_id, output_eid_);

    xnn_unary_params params{};
    params.clamp.min = 0.0f;
    params.clamp.max = std::numeric_limits<float>::infinity();
    CheckXNNStatus(xnn_define_unary(subgraph_, xnn_unary_clamp, &params, input_id, output_id, 0),
                   "xnn_define_unary");

    CheckXNNStatus(xnn_create_runtime_v2(subgraph_, nullptr, 0, &runtime_),
                   "xnn_create_runtime_v2");
  }

  xnn_subgraph_t subgraph_{nullptr};
  xnn_runtime_t runtime_{nullptr};
  uint32_t input_eid_{0};
  uint32_t output_eid_{0};
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  std::vector<uint8_t> input_buffer_;
  std::vector<uint8_t> output_buffer_;
};

ffi::Module XNNPACKJSONRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                     const ffi::Array<ffi::String>& const_names) {
  auto n = tvm::ffi::make_object<XNNPACKJSONRuntime>(symbol_name, graph_json, const_names);
  return ffi::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.XNNPACKJSONRuntimeCreate", XNNPACKJSONRuntimeCreate)
      .def("ffi.Module.load_from_bytes.xnnpack_json",
           JSONRuntimeBase::LoadFromBytes<XNNPACKJSONRuntime>);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
