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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
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
                     const ffi::Array<ffi::String> const_names,
                     const std::string& options = DefaultOptionsString())
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        options_string_(options),
        options_(ParseOptions(options)) {}

  static std::string DefaultOptionsString() {
    return "use_weights_cache=0;use_workspace=0;profile=0;dont_spin_workers=0;"
           "transient_indirection_buffer=0;num_threads=1;";
  }

  ~XNNPACKJSONRuntime() {
    if (runtime_ != nullptr) {
      xnn_delete_runtime(runtime_);
      runtime_ = nullptr;
    }
    if (subgraph_ != nullptr) {
      xnn_delete_subgraph(subgraph_);
      subgraph_ = nullptr;
    }
#if defined(TVM_XNNPACK_HAS_WORKSPACE)
    if (workspace_ != nullptr) {
      xnn_release_workspace(workspace_);
      workspace_ = nullptr;
    }
#endif
#if defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
    if (weights_cache_ != nullptr) {
      xnn_delete_weights_cache(weights_cache_);
      weights_cache_ = nullptr;
    }
#endif
#if defined(TVM_XNNPACK_HAS_PTHREADPOOL_CREATE)
    if (threadpool_ != nullptr) {
      pthreadpool_destroy(threadpool_);
      threadpool_ = nullptr;
    }
#endif
  }

  const char* kind() const override { return "xnnpack_json"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) override {
    ffi::ObjectPtr<ffi::Object> sptr_to_self = ffi::GetObjectPtr<ffi::Object>(this);
    if (name == "get_profile_json") {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = ffi::String(this->GetProfileJSON());
      });
    }
    if (name == "get_runtime_options") {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = ffi::String(this->options_string_);
      });
    }
    return JSONRuntimeBase::GetFunction(name);
  }

  ffi::Bytes SaveToBytes() const override {
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(symbol_name_);
    stream.Write(graph_json_);
    std::vector<std::string> consts;
    for (const auto& it : const_names_) {
      consts.push_back(it);
    }
    stream.Write(consts);
    stream.Write(options_string_);
    return ffi::Bytes(std::move(result));
  }

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
  struct RuntimeOptions {
    bool use_weights_cache{false};
    bool use_workspace{false};
    bool profile{false};
    bool dont_spin_workers{false};
    bool transient_indirection_buffer{false};
    int64_t num_threads{1};
  };

  struct ExternalTensor {
    uint32_t eid{0};
    std::vector<size_t> shape;
    std::string name;
    bool is_output{false};
    std::vector<uint8_t> buffer;
  };

  static RuntimeOptions ParseOptions(const std::string& options) {
    RuntimeOptions parsed;
    size_t offset = 0;
    while (offset < options.size()) {
      size_t end = options.find(';', offset);
      if (end == std::string::npos) end = options.size();
      std::string item = options.substr(offset, end - offset);
      if (!item.empty()) {
        size_t equals = item.find('=');
        TVM_FFI_ICHECK(equals != std::string::npos) << "Malformed XNNPACK runtime option: " << item;
        const std::string key = item.substr(0, equals);
        const std::string value = item.substr(equals + 1);
        const bool bool_value = value == "1";
        if (key == "use_weights_cache") {
          parsed.use_weights_cache = bool_value;
        } else if (key == "use_workspace") {
          parsed.use_workspace = bool_value;
        } else if (key == "profile") {
          parsed.profile = bool_value;
        } else if (key == "dont_spin_workers") {
          parsed.dont_spin_workers = bool_value;
        } else if (key == "transient_indirection_buffer") {
          parsed.transient_indirection_buffer = bool_value;
        } else if (key == "num_threads") {
          parsed.num_threads = std::stoll(value);
        } else {
          TVM_FFI_THROW(ValueError) << "Unsupported XNNPACK runtime option: " << key;
        }
      }
      offset = end + 1;
    }
    TVM_FFI_ICHECK_GE(parsed.num_threads, 1) << "XNNPACK num_threads must be >= 1.";
    return parsed;
  }

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

  static std::string EscapeJSON(const std::string& value) {
    std::ostringstream os;
    for (char ch : value) {
      if (ch == '"' || ch == '\\') {
        os << '\\' << ch;
      } else if (ch == '\n') {
        os << "\\n";
      } else {
        os << ch;
      }
    }
    return os.str();
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

  uint32_t RuntimeFlags() const {
    uint32_t flags = 0;
    if (options_.profile) {
#if defined(TVM_XNNPACK_HAS_PROFILING) && defined(TVM_XNNPACK_HAS_BASIC_PROFILING_FLAG)
      flags |= XNN_FLAG_BASIC_PROFILING;
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK profiling was requested but is unavailable.";
#endif
    }
    if (options_.dont_spin_workers) {
#if defined(TVM_XNNPACK_HAS_DONT_SPIN_WORKERS_FLAG)
      flags |= XNN_FLAG_DONT_SPIN_WORKERS;
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK dont_spin_workers was requested but is unavailable.";
#endif
    }
    if (options_.transient_indirection_buffer) {
#if defined(TVM_XNNPACK_HAS_TRANSIENT_INDIRECTION_BUFFER_FLAG)
      flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
#else
      TVM_FFI_THROW(RuntimeError)
          << "XNNPACK transient_indirection_buffer was requested but is unavailable.";
#endif
    }
    return flags;
  }

  void CreateOptionalResources() {
    if (options_.num_threads > 1) {
#if defined(TVM_XNNPACK_HAS_PTHREADPOOL_CREATE)
      threadpool_ = pthreadpool_create(static_cast<size_t>(options_.num_threads));
      TVM_FFI_ICHECK(threadpool_ != nullptr) << "Failed to create XNNPACK pthreadpool.";
#else
      TVM_FFI_THROW(RuntimeError)
          << "XNNPACK num_threads > 1 was requested but pthreadpool is unavailable.";
#endif
    }

    if (options_.use_weights_cache) {
#if defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
      CheckXNNStatus(xnn_create_weights_cache(&weights_cache_), "xnn_create_weights_cache");
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK weights cache was requested but is unavailable.";
#endif
    }

    if (options_.use_workspace) {
#if defined(TVM_XNNPACK_HAS_WORKSPACE)
      CheckXNNStatus(xnn_create_workspace(&workspace_), "xnn_create_workspace");
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK workspace was requested but is unavailable.";
#endif
    }
  }

  void CreateRuntime() {
    const uint32_t flags = RuntimeFlags();
    CreateOptionalResources();

#if defined(TVM_XNNPACK_HAS_RUNTIME_V4)
    CheckXNNStatus(
        xnn_create_runtime_v4(subgraph_, weights_cache_, workspace_, threadpool_, flags, &runtime_),
        "xnn_create_runtime_v4");
#elif defined(TVM_XNNPACK_HAS_RUNTIME_V3) && defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
    TVM_FFI_ICHECK(!options_.use_workspace) << "XNNPACK workspace requires xnn_create_runtime_v4.";
    CheckXNNStatus(xnn_create_runtime_v3(subgraph_, weights_cache_, threadpool_, flags, &runtime_),
                   "xnn_create_runtime_v3");
#else
    TVM_FFI_ICHECK(!options_.use_weights_cache)
        << "XNNPACK weights cache requires xnn_create_runtime_v3 or newer.";
    TVM_FFI_ICHECK(!options_.use_workspace) << "XNNPACK workspace requires xnn_create_runtime_v4.";
    CheckXNNStatus(xnn_create_runtime_v2(subgraph_, threadpool_, flags, &runtime_),
                   "xnn_create_runtime_v2");
#endif

    if (options_.use_weights_cache) {
#if defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
      CheckXNNStatus(
          xnn_finalize_weights_cache(weights_cache_, xnn_weights_cache_finalization_kind_soft),
          "xnn_finalize_weights_cache");
#endif
    }
  }

  std::string GetProfileJSON() const {
    if (!options_.profile) return "[]";
#if defined(TVM_XNNPACK_HAS_PROFILING)
    if (runtime_ == nullptr) return "[]";

    size_t num_operators = 0;
    size_t bytes = 0;
    CheckXNNStatus(xnn_get_runtime_profiling_info(runtime_, xnn_profile_info_num_operators,
                                                  sizeof(num_operators), &num_operators, &bytes),
                   "xnn_get_runtime_profiling_info(num_operators)");
    if (num_operators == 0) return "[]";

    size_t names_size = 0;
    xnn_status status = xnn_get_runtime_profiling_info(runtime_, xnn_profile_info_operator_name, 0,
                                                       nullptr, &names_size);
    TVM_FFI_ICHECK(status == xnn_status_success || status == xnn_status_out_of_memory)
        << "xnn_get_runtime_profiling_info(operator_name) failed with status " << status;
    std::vector<char> names(names_size);
    CheckXNNStatus(xnn_get_runtime_profiling_info(runtime_, xnn_profile_info_operator_name,
                                                  names.size(), names.data(), &names_size),
                   "xnn_get_runtime_profiling_info(operator_name)");

    std::vector<uint64_t> timings(num_operators);
    CheckXNNStatus(
        xnn_get_runtime_profiling_info(runtime_, xnn_profile_info_operator_timing,
                                       timings.size() * sizeof(uint64_t), timings.data(), &bytes),
        "xnn_get_runtime_profiling_info(operator_timing)");

    std::vector<std::string> parsed_names;
    size_t start = 0;
    for (size_t i = 0; i < names.size() && parsed_names.size() < num_operators; ++i) {
      if (names[i] == '\0') {
        parsed_names.emplace_back(names.data() + start, i - start);
        start = i + 1;
      }
    }
    while (parsed_names.size() < num_operators) {
      parsed_names.push_back("");
    }

    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < num_operators; ++i) {
      if (i != 0) os << ",";
      os << "{\"name\":\"" << EscapeJSON(parsed_names[i]) << "\",\"time_ns\":" << timings[i] << "}";
    }
    os << "]";
    return os.str();
#else
    TVM_FFI_THROW(RuntimeError) << "XNNPACK profiling is unavailable.";
#endif
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

    CreateRuntime();
  }

  xnn_subgraph_t subgraph_{nullptr};
  xnn_runtime_t runtime_{nullptr};
#if defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
  xnn_weights_cache_t weights_cache_{nullptr};
#endif
#if defined(TVM_XNNPACK_HAS_WORKSPACE)
  xnn_workspace_t workspace_{nullptr};
#endif
#if defined(TVM_XNNPACK_HAS_PTHREADPOOL_CREATE)
  pthreadpool_t threadpool_{nullptr};
#endif
  std::string options_string_;
  RuntimeOptions options_;
  std::vector<uint32_t> value_ids_;
  std::vector<ExternalTensor> external_tensors_;
  std::vector<std::vector<uint8_t>> constant_buffers_;
};

ffi::Module XNNPACKJSONRuntimeCreate(const ffi::String& symbol_name, const ffi::String& graph_json,
                                     const ffi::Array<ffi::String>& const_names,
                                     const ffi::String& options) {
  auto n = tvm::ffi::make_object<XNNPACKJSONRuntime>(symbol_name, graph_json, const_names,
                                                     std::string(options));
  return ffi::Module(n);
}

ffi::Module XNNPACKJSONRuntimeLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  std::string symbol;
  std::string graph_json;
  std::vector<std::string> consts;
  std::string options;
  TVM_FFI_ICHECK(stream.Read(&symbol)) << "Loading symbol name failed";
  TVM_FFI_ICHECK(stream.Read(&graph_json)) << "Loading graph json failed";
  TVM_FFI_ICHECK(stream.Read(&consts)) << "Loading the const name list failed";
  if (!stream.Read(&options)) {
    options = XNNPACKJSONRuntime::DefaultOptionsString();
  }
  ffi::Array<ffi::String> const_names;
  for (const auto& it : consts) {
    const_names.push_back(it);
  }
  auto n = tvm::ffi::make_object<XNNPACKJSONRuntime>(symbol, graph_json, const_names, options);
  return ffi::Module(n);
}

ffi::Map<ffi::String, ffi::Any> XNNPACKJSONRuntimeGetCapabilities() {
  ffi::Map<ffi::String, ffi::Any> result;
  result.Set("runtime_v4", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_RUNTIME_V4)
                               1
#else
                               0
#endif
                               ));
  result.Set("runtime_v3", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_RUNTIME_V3)
                               1
#else
                               0
#endif
                               ));
  result.Set("weights_cache", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
                                  1
#else
                                  0
#endif
                                  ));
  result.Set("workspace", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_WORKSPACE)
                              1
#else
                              0
#endif
                              ));
  result.Set("profiling", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_PROFILING) && defined(TVM_XNNPACK_HAS_BASIC_PROFILING_FLAG)
                              1
#else
                              0
#endif
                              ));
  result.Set("pthreadpool", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_PTHREADPOOL_CREATE)
                                1
#else
                                0
#endif
                                ));
  result.Set("dont_spin_workers", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DONT_SPIN_WORKERS_FLAG)
                                      1
#else
                                      0
#endif
                                      ));
  result.Set("transient_indirection_buffer", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_TRANSIENT_INDIRECTION_BUFFER_FLAG)
                                                 1
#else
                                                 0
#endif
                                                 ));
  return result;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.XNNPACKJSONRuntimeCreate", XNNPACKJSONRuntimeCreate)
      .def("runtime.XNNPACKJSONRuntimeGetCapabilities", XNNPACKJSONRuntimeGetCapabilities)
      .def("ffi.Module.load_from_bytes.xnnpack_json", XNNPACKJSONRuntimeLoadFromBytes);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
