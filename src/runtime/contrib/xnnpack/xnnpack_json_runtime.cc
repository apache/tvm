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
#include <unordered_set>
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

    std::vector<xnn_external_value> external_values;
    external_values.reserve(external_tensors_.size());

    for (auto& entry : external_tensors_) {
      TVM_FFI_ICHECK_LT(entry.eid, data_entry_.size());
      const DLTensor* tensor = data_entry_[entry.eid];
      ValidateTensor(tensor, entry.shape, entry.name.c_str());

      const size_t bytes = NumElements(entry.shape) * sizeof(float);
      entry.buffer.resize(bytes + XNN_EXTRA_BYTES);
      if (entry.is_output) {
        std::memset(entry.buffer.data(), 0, bytes + XNN_EXTRA_BYTES);
      } else {
        std::memcpy(entry.buffer.data(), TensorData(tensor), bytes);
        std::memset(entry.buffer.data() + bytes, 0, XNN_EXTRA_BYTES);
      }

      CheckXNNStatus(
          xnn_reshape_external_value(runtime_, entry.eid, entry.shape.size(), entry.shape.data()),
          "xnn_reshape_external_value");
      external_values.push_back({entry.eid, entry.buffer.data()});
    }
    CheckXNNStatus(xnn_reshape_runtime(runtime_), "xnn_reshape_runtime");

    CheckXNNStatus(xnn_setup_runtime_v2(runtime_, external_values.size(), external_values.data()),
                   "xnn_setup_runtime_v2");
    CheckXNNStatus(xnn_invoke_runtime(runtime_), "xnn_invoke_runtime");

    for (auto& entry : external_tensors_) {
      if (!entry.is_output) continue;
      const size_t bytes = NumElements(entry.shape) * sizeof(float);
      std::memcpy(MutableTensorData(data_entry_[entry.eid]), entry.buffer.data(), bytes);
    }
  }

 private:
  struct ExternalTensor {
    uint32_t eid{0};
    std::vector<size_t> shape;
    std::string name;
    bool is_output{false};
    std::vector<uint8_t> buffer;
  };

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

  static std::vector<uint32_t> GetUIntArray(const JSONGraphNode& node, const std::string& key) {
    ffi::Array<int64_t> arr = node.GetAttr<ffi::Array<int64_t>>(key);
    std::vector<uint32_t> result;
    for (int64_t value : arr) {
      TVM_FFI_ICHECK_GE(value, 0);
      result.push_back(static_cast<uint32_t>(value));
    }
    return result;
  }

  static float GetFloatAttr(const JSONGraphNode& node, const std::string& key) {
    return static_cast<float>(node.GetAttr<double>(key));
  }

  static bool IsGraphOutput(const std::unordered_set<uint32_t>& output_eids, uint32_t eid) {
    return output_eids.find(eid) != output_eids.end();
  }

  void DefineTensor(uint32_t eid, const JSONGraphNode& node, uint32_t index, uint32_t flags,
                    const void* data = nullptr) {
    if (value_ids_[eid] != XNN_INVALID_VALUE_ID) return;
    CheckDType(node, index);
    std::vector<size_t> shape = GetShape(node, index);
    uint32_t id = XNN_INVALID_VALUE_ID;
    const uint32_t external_id = flags != 0 ? eid : XNN_INVALID_VALUE_ID;
    CheckXNNStatus(xnn_define_tensor_value(subgraph_, xnn_datatype_fp32, shape.size(), shape.data(),
                                           data, external_id, flags, &id),
                   "xnn_define_tensor_value");
    if (flags != 0) {
      TVM_FFI_ICHECK_EQ(id, eid);
    }
    value_ids_[eid] = id;
  }

  const void* PrepareConstant(uint32_t eid, const JSONGraphNode& node) {
    const DLTensor* tensor = data_entry_[eid];
    std::vector<size_t> shape = GetShape(node, 0);
    ValidateTensor(tensor, shape, "constant");
    const size_t bytes = NumElements(shape) * sizeof(float);
    constant_buffers_.emplace_back(bytes + XNN_EXTRA_BYTES);
    std::memcpy(constant_buffers_.back().data(), TensorData(tensor), bytes);
    std::memset(constant_buffers_.back().data() + bytes, 0, XNN_EXTRA_BYTES);
    return constant_buffers_.back().data();
  }

  void DefineGraphInputsAndConstants() {
    for (uint32_t eid : input_var_eid_) {
      const uint32_t nid = NodeIDFromEntryID(eid);
      DefineTensor(eid, nodes_[nid], 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);
      external_tensors_.push_back({eid, GetShape(nodes_[nid], 0), "input", false, {}});
    }

    for (uint32_t nid : const_idx_) {
      const uint32_t eid = EntryID(nid, 0);
      const void* data = PrepareConstant(eid, nodes_[nid]);
      DefineTensor(eid, nodes_[nid], 0, 0, data);
    }
  }

  uint32_t NodeIDFromEntryID(uint32_t eid) const {
    for (uint32_t nid = 0; nid + 1 < node_row_ptr_.size(); ++nid) {
      if (node_row_ptr_[nid] <= eid && eid < node_row_ptr_[nid + 1]) {
        return nid;
      }
    }
    TVM_FFI_THROW(InternalError) << "Cannot resolve JSON node id for entry id " << eid;
  }

  void DefineOutput(const JSONGraphNode& node, const JSONGraphNodeEntry& output_entry,
                    const std::unordered_set<uint32_t>& graph_output_eids) {
    const uint32_t eid = EntryID(output_entry);
    const uint32_t flags =
        IsGraphOutput(graph_output_eids, eid) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
    DefineTensor(eid, node, output_entry.index_, flags);
    if (flags != 0) {
      external_tensors_.push_back({eid, GetShape(node, output_entry.index_), "output", true, {}});
    }
  }

  void DefineUnary(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
                   uint32_t output_id) {
    TVM_FFI_ICHECK_EQ(inputs.size(), 1U);
    const std::string unary_op = node.GetAttr<ffi::String>("unary_op");
    const uint32_t input_id = value_ids_[EntryID(inputs[0])];

    if (unary_op == "clamp") {
      xnn_unary_params params{};
      params.clamp.min = GetFloatAttr(node, "activation_min");
      params.clamp.max = GetFloatAttr(node, "activation_max");
      CheckXNNStatus(xnn_define_unary(subgraph_, xnn_unary_clamp, &params, input_id, output_id, 0),
                     "xnn_define_unary(clamp)");
    } else if (unary_op == "sigmoid") {
      CheckXNNStatus(
          xnn_define_unary(subgraph_, xnn_unary_sigmoid, nullptr, input_id, output_id, 0),
          "xnn_define_unary(sigmoid)");
    } else {
      TVM_FFI_ICHECK_EQ(unary_op, "tanh");
      CheckXNNStatus(xnn_define_unary(subgraph_, xnn_unary_tanh, nullptr, input_id, output_id, 0),
                     "xnn_define_unary(tanh)");
    }
  }

  void DefineAdd(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
                 uint32_t output_id) {
    TVM_FFI_ICHECK_EQ(inputs.size(), 2U);
    xnn_binary_params params{};
    params.output_min = -std::numeric_limits<float>::max();
    params.output_max = std::numeric_limits<float>::max();
    CheckXNNStatus(
        xnn_define_binary(subgraph_, xnn_binary_add, &params, value_ids_[EntryID(inputs[0])],
                          value_ids_[EntryID(inputs[1])], output_id, XNN_FLAG_NO_BROADCAST),
        "xnn_define_binary(add)");
  }

  void DefineConv2D(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
                    uint32_t output_id) {
    const bool has_bias = static_cast<int64_t>(node.GetAttr<int64_t>("has_bias")) != 0;
    TVM_FFI_ICHECK_EQ(inputs.size(), has_bias ? 3U : 2U);
    auto padding = GetUIntArray(node, "padding");
    auto strides = GetUIntArray(node, "strides");
    auto dilation = GetUIntArray(node, "dilation");
    TVM_FFI_ICHECK_EQ(padding.size(), 4U);
    TVM_FFI_ICHECK_EQ(strides.size(), 2U);
    TVM_FFI_ICHECK_EQ(dilation.size(), 2U);

    std::vector<size_t> weight_shape = GetShape(nodes_[inputs[1].id_], inputs[1].index_);
    TVM_FFI_ICHECK_EQ(weight_shape.size(), 4U);
    const uint32_t bias_id = has_bias ? value_ids_[EntryID(inputs[2])] : XNN_INVALID_VALUE_ID;

    CheckXNNStatus(xnn_define_convolution_2d(
                       subgraph_, padding[0], padding[3], padding[2], padding[1], weight_shape[1],
                       weight_shape[2], strides[0], strides[1], dilation[0], dilation[1],
                       static_cast<uint32_t>(node.GetAttr<int64_t>("groups")), weight_shape[3],
                       weight_shape[0], GetFloatAttr(node, "activation_min"),
                       GetFloatAttr(node, "activation_max"), value_ids_[EntryID(inputs[0])],
                       value_ids_[EntryID(inputs[1])], bias_id, output_id, 0),
                   "xnn_define_convolution_2d");
  }

  void DefinePool2D(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
                    uint32_t output_id, bool is_max_pool) {
    TVM_FFI_ICHECK_EQ(inputs.size(), 1U);
    auto padding = GetUIntArray(node, "padding");
    auto pool_size = GetUIntArray(node, "pool_size");
    auto strides = GetUIntArray(node, "strides");
    TVM_FFI_ICHECK_EQ(padding.size(), 4U);
    TVM_FFI_ICHECK_EQ(pool_size.size(), 2U);
    TVM_FFI_ICHECK_EQ(strides.size(), 2U);

    if (is_max_pool) {
      auto dilation = GetUIntArray(node, "dilation");
      TVM_FFI_ICHECK_EQ(dilation.size(), 2U);
      CheckXNNStatus(xnn_define_max_pooling_2d(
                         subgraph_, padding[0], padding[3], padding[2], padding[1], pool_size[0],
                         pool_size[1], strides[0], strides[1], dilation[0], dilation[1],
                         GetFloatAttr(node, "activation_min"), GetFloatAttr(node, "activation_max"),
                         value_ids_[EntryID(inputs[0])], output_id, 0),
                     "xnn_define_max_pooling_2d");
    } else {
      CheckXNNStatus(
          xnn_define_average_pooling_2d(
              subgraph_, padding[0], padding[3], padding[2], padding[1], pool_size[0], pool_size[1],
              strides[0], strides[1], GetFloatAttr(node, "activation_min"),
              GetFloatAttr(node, "activation_max"), value_ids_[EntryID(inputs[0])], output_id, 0),
          "xnn_define_average_pooling_2d");
    }
  }

  void BuildRuntime() {
    CheckXNNStatus(xnn_create_subgraph(NumEntries(), 0, &subgraph_), "xnn_create_subgraph");
    value_ids_.assign(NumEntries(), XNN_INVALID_VALUE_ID);
    external_tensors_.clear();
    constant_buffers_.clear();

    std::unordered_set<uint32_t> graph_output_eids;
    for (const auto& output : outputs_) {
      graph_output_eids.insert(EntryID(output));
    }

    DefineGraphInputsAndConstants();

    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const JSONGraphNode& node = nodes_[nid];
      if (node.GetOpType() != "kernel") continue;
      TVM_FFI_ICHECK_EQ(node.GetNumOutput(), 1U);
      const JSONGraphNodeEntry output_entry(static_cast<int>(nid), 0);
      DefineOutput(node, output_entry, graph_output_eids);
      const uint32_t output_id = value_ids_[EntryID(output_entry)];

      auto inputs = node.GetInputs();
      for (const auto& input : inputs) {
        TVM_FFI_ICHECK_LT(EntryID(input), value_ids_.size());
        TVM_FFI_ICHECK_NE(value_ids_[EntryID(input)], XNN_INVALID_VALUE_ID)
            << "XNNPACK input value was not defined before its use.";
      }

      const std::string op_kind = node.GetAttr<ffi::String>("op_kind");
      if (op_kind == "unary") {
        DefineUnary(node, inputs, output_id);
      } else if (op_kind == "add") {
        DefineAdd(node, inputs, output_id);
      } else if (op_kind == "conv2d") {
        DefineConv2D(node, inputs, output_id);
      } else if (op_kind == "max_pool2d") {
        DefinePool2D(node, inputs, output_id, true);
      } else {
        TVM_FFI_ICHECK_EQ(op_kind, "avg_pool2d");
        DefinePool2D(node, inputs, output_id, false);
      }
    }

    CheckXNNStatus(xnn_create_runtime_v2(subgraph_, nullptr, 0, &runtime_),
                   "xnn_create_runtime_v2");
  }

  xnn_subgraph_t subgraph_{nullptr};
  xnn_runtime_t runtime_{nullptr};
  std::vector<uint32_t> value_ids_;
  std::vector<ExternalTensor> external_tensors_;
  std::vector<std::vector<uint8_t>> constant_buffers_;
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
