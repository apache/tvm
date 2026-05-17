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
#include <cmath>
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
           "transient_indirection_buffer=0;num_threads=1;precision=fp32;";
  }

  static std::string ValidateQuantizationMetadataJSON(
      const ffi::Map<ffi::String, ffi::Any>& metadata, const ffi::Array<ffi::Any>& shape) {
    const QuantizationMetadata parsed =
        ParseQuantizationMetadata(metadata, ShapeFromAnyArray(shape));
    return QuantizationMetadataToJSON(parsed);
  }

  static std::string QuantizedTensorDefinitionSmoke(const ffi::Map<ffi::String, ffi::Any>& metadata,
                                                    const ffi::Array<ffi::Any>& shape) {
    const QuantizationMetadata parsed =
        ParseQuantizationMetadata(metadata, ShapeFromAnyArray(shape));
    DefineQuantizedTensorForSmoke(parsed);
    return QuantizationMetadataToJSON(parsed);
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
    if (name == "get_quantization_metadata_json") {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        *rv = ffi::String(this->GetQuantizationMetadataJSON());
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
  }

  void Run() override {
    if (runtime_ == nullptr) {
      BuildRuntime();
    }

    std::vector<xnn_external_value> external_values;
    external_values.reserve(external_tensors_.size());

    for (auto& entry : external_tensors_) {
      TVM_FFI_ICHECK_LT(entry.eid, data_entry_.size());
      const DLTensor* tensor = data_entry_[entry.eid];
      ValidateTensor(tensor, entry.shape, entry.dtype, entry.name.c_str());

      const size_t bytes = NumElements(entry.shape) * entry.element_size;
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
      const size_t bytes = NumElements(entry.shape) * entry.element_size;
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
    std::string precision{"fp32"};
  };

  struct ExternalTensor {
    uint32_t eid{0};
    std::vector<size_t> shape;
    std::string name;
    DLDataType dtype{kDLFloat, 32, 1};
    size_t element_size{sizeof(float)};
    bool is_output{false};
    std::vector<uint8_t> buffer;
  };

  struct QuantizationMetadata {
    std::string dtype;
    std::string qscheme;
    std::vector<float> scale;
    int32_t zero_point{0};
    int64_t axis{-1};
    size_t channel_dim{0};
    std::string signedness;
    std::vector<size_t> shape;
    std::vector<float> padded_scale;
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
        } else if (key == "precision") {
          parsed.precision = value;
        } else {
          TVM_FFI_THROW(ValueError) << "Unsupported XNNPACK runtime option: " << key;
        }
      }
      offset = end + 1;
    }
    TVM_FFI_ICHECK_GE(parsed.num_threads, 1) << "XNNPACK num_threads must be >= 1.";
    TVM_FFI_ICHECK(parsed.precision == "fp32" || parsed.precision == "fp16_hint" ||
                   parsed.precision == "fp16_force")
        << "Unsupported XNNPACK precision: " << parsed.precision;
    return parsed;
  }

  static void CheckXNNStatus(xnn_status status, const char* call) {
    TVM_FFI_ICHECK_EQ(status, xnn_status_success) << call << " failed with status " << status;
  }

  void CheckRuntimeCreateStatus(xnn_status status, const char* call) const {
    TVM_FFI_ICHECK_EQ(status, xnn_status_success)
        << call << " failed with status " << status << " for XNNPACK precision '"
        << options_.precision
        << "'. If precision='fp16_force', this means XNNPACK could not create an FP16 runtime for "
           "the current graph, hardware, or linked XNNPACK build.";
  }

  static bool IsFloat32(const DLDataType& dtype) {
    return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
  }

  static bool IsInt8(const DLDataType& dtype) {
    return dtype.code == kDLInt && dtype.bits == 8 && dtype.lanes == 1;
  }

  static bool IsInt32(const DLDataType& dtype) {
    return dtype.code == kDLInt && dtype.bits == 32 && dtype.lanes == 1;
  }

  static size_t ElementSize(const DLDataType& dtype) {
    TVM_FFI_ICHECK_EQ(dtype.lanes, 1);
    return (dtype.bits + 7) / 8;
  }

  static std::string DTypeName(const DLDataType& dtype) {
    if (IsFloat32(dtype)) return "float32";
    if (IsInt8(dtype)) return "int8";
    if (IsInt32(dtype)) return "int32";
    std::ostringstream os;
    os << "code=" << static_cast<int>(dtype.code) << ",bits=" << static_cast<int>(dtype.bits);
    return os.str();
  }

  static int64_t AnyToInt64(const ffi::Any& value, const char* name) {
    if (auto opt = value.try_cast<int64_t>()) return opt.value();
    TVM_FFI_THROW(ValueError) << "XNNPACK quantization metadata field '" << name
                              << "' must be an integer.";
  }

  static double AnyToDouble(const ffi::Any& value, const char* name) {
    if (auto opt = value.try_cast<double>()) return opt.value();
    if (auto opt = value.try_cast<int64_t>()) return static_cast<double>(opt.value());
    TVM_FFI_THROW(ValueError) << "XNNPACK quantization metadata field '" << name
                              << "' must be numeric.";
  }

  static std::string AnyToString(const ffi::Any& value, const char* name) {
    if (auto opt = value.try_cast<ffi::String>()) return std::string(opt.value());
    TVM_FFI_THROW(ValueError) << "XNNPACK quantization metadata field '" << name
                              << "' must be a string.";
  }

  static ffi::Any RequiredField(const ffi::Map<ffi::String, ffi::Any>& metadata,
                                const std::string& key) {
    auto it = metadata.find(key);
    TVM_FFI_ICHECK(it != metadata.end()) << "Missing XNNPACK quantization metadata field: " << key;
    return (*it).second;
  }

  static std::vector<size_t> ShapeFromAnyArray(const ffi::Array<ffi::Any>& shape) {
    std::vector<size_t> result;
    for (const ffi::Any& dim : shape) {
      const int64_t value = AnyToInt64(dim, "shape");
      TVM_FFI_ICHECK_GT(value, 0) << "XNNPACK quantization metadata shape must be static positive.";
      result.push_back(static_cast<size_t>(value));
    }
    return result;
  }

  static std::vector<float> ScaleFromAny(const ffi::Any& value) {
    std::vector<float> result;
    if (auto opt_arr = value.try_cast<ffi::Array<ffi::Any>>()) {
      for (const ffi::Any& item : opt_arr.value()) {
        result.push_back(static_cast<float>(AnyToDouble(item, "scale")));
      }
    } else {
      result.push_back(static_cast<float>(AnyToDouble(value, "scale")));
    }
    return result;
  }

  static std::string ExpectedSignedness(const std::string& dtype) {
    if (dtype == "uint8") return "unsigned";
    if (dtype == "int8" || dtype == "int32") return "signed";
    TVM_FFI_THROW(ValueError) << "Unsupported XNNPACK quantized dtype: " << dtype;
  }

  static void CheckZeroPointRange(const std::string& dtype, int64_t zero_point) {
    if (dtype == "int8") {
      TVM_FFI_ICHECK_GE(zero_point, -128)
          << "XNNPACK int8 quantization zero_point must be in [-128, 127].";
      TVM_FFI_ICHECK_LE(zero_point, 127)
          << "XNNPACK int8 quantization zero_point must be in [-128, 127].";
    } else if (dtype == "uint8") {
      TVM_FFI_ICHECK_GE(zero_point, 0)
          << "XNNPACK uint8 quantization zero_point must be in [0, 255].";
      TVM_FFI_ICHECK_LE(zero_point, 255)
          << "XNNPACK uint8 quantization zero_point must be in [0, 255].";
    } else if (dtype == "int32") {
      TVM_FFI_ICHECK_GE(zero_point, std::numeric_limits<int32_t>::min());
      TVM_FFI_ICHECK_LE(zero_point, std::numeric_limits<int32_t>::max());
    } else {
      TVM_FFI_THROW(ValueError) << "Unsupported XNNPACK quantized dtype: " << dtype;
    }
  }

  static xnn_datatype QuantizedDatatype(const QuantizationMetadata& metadata) {
    if (metadata.qscheme == "per_tensor") {
      if (metadata.dtype == "int8") {
#if defined(TVM_XNNPACK_HAS_DATATYPE_QINT8)
        return xnn_datatype_qint8;
#else
        TVM_FFI_THROW(RuntimeError) << "XNNPACK qint8 datatype is unavailable.";
#endif
      }
      if (metadata.dtype == "uint8") {
#if defined(TVM_XNNPACK_HAS_DATATYPE_QUINT8)
        return xnn_datatype_quint8;
#else
        TVM_FFI_THROW(RuntimeError) << "XNNPACK quint8 datatype is unavailable.";
#endif
      }
      if (metadata.dtype == "int32") {
#if defined(TVM_XNNPACK_HAS_DATATYPE_QINT32)
        return xnn_datatype_qint32;
#else
        TVM_FFI_THROW(RuntimeError) << "XNNPACK qint32 datatype is unavailable.";
#endif
      }
    } else if (metadata.qscheme == "per_channel") {
      if (metadata.dtype == "int8") {
#if defined(TVM_XNNPACK_HAS_DATATYPE_QCINT8)
        return xnn_datatype_qcint8;
#else
        TVM_FFI_THROW(RuntimeError) << "XNNPACK qcint8 datatype is unavailable.";
#endif
      }
      if (metadata.dtype == "int32") {
#if defined(TVM_XNNPACK_HAS_DATATYPE_QCINT32)
        return xnn_datatype_qcint32;
#else
        TVM_FFI_THROW(RuntimeError) << "XNNPACK qcint32 datatype is unavailable.";
#endif
      }
    }
    TVM_FFI_THROW(ValueError) << "Unsupported XNNPACK quantization dtype/qscheme combination: "
                              << metadata.dtype << "/" << metadata.qscheme;
  }

  static std::string QuantizedDatatypeName(const QuantizationMetadata& metadata) {
    if (metadata.qscheme == "per_tensor") {
      if (metadata.dtype == "int8") return "xnn_datatype_qint8";
      if (metadata.dtype == "uint8") return "xnn_datatype_quint8";
      if (metadata.dtype == "int32") return "xnn_datatype_qint32";
    } else if (metadata.qscheme == "per_channel") {
      if (metadata.dtype == "int8") return "xnn_datatype_qcint8";
      if (metadata.dtype == "int32") return "xnn_datatype_qcint32";
    }
    return "unsupported";
  }

  static size_t ExtraQuantizationParams() {
#if defined(TVM_XNNPACK_HAS_EXTRA_QUANTIZATION_PARAMS)
    return XNN_EXTRA_QUANTIZATION_PARAMS;
#else
    return 0;
#endif
  }

  static QuantizationMetadata ParseQuantizationMetadata(
      const ffi::Map<ffi::String, ffi::Any>& metadata, std::vector<size_t> shape) {
    QuantizationMetadata parsed;
    parsed.dtype = AnyToString(RequiredField(metadata, "dtype"), "dtype");
    parsed.qscheme = AnyToString(RequiredField(metadata, "qscheme"), "qscheme");
    parsed.signedness = AnyToString(RequiredField(metadata, "signedness"), "signedness");
    parsed.shape = std::move(shape);

    TVM_FFI_ICHECK(parsed.qscheme == "none" || parsed.qscheme == "per_tensor" ||
                   parsed.qscheme == "per_channel")
        << "Unsupported XNNPACK quantization qscheme: " << parsed.qscheme;
    if (parsed.qscheme == "none") {
      return parsed;
    }

    parsed.scale = ScaleFromAny(RequiredField(metadata, "scale"));
    const int64_t zero_point = AnyToInt64(RequiredField(metadata, "zero_point"), "zero_point");
    CheckZeroPointRange(parsed.dtype, zero_point);
    parsed.zero_point = static_cast<int32_t>(zero_point);
    parsed.axis = AnyToInt64(RequiredField(metadata, "axis"), "axis");
    const int64_t channel_dim = AnyToInt64(RequiredField(metadata, "channel_dim"), "channel_dim");
    TVM_FFI_ICHECK_GE(channel_dim, 0) << "XNNPACK quantization channel_dim must be non-negative.";
    TVM_FFI_ICHECK_LT(static_cast<size_t>(channel_dim), parsed.shape.size())
        << "XNNPACK quantization channel_dim is out of range.";
    parsed.channel_dim = static_cast<size_t>(channel_dim);

    TVM_FFI_ICHECK_EQ(parsed.signedness, ExpectedSignedness(parsed.dtype))
        << "XNNPACK quantization signedness does not match dtype.";
    for (float scale : parsed.scale) {
      TVM_FFI_ICHECK(std::isfinite(scale) && scale > 0.0f)
          << "XNNPACK quantization scales must be finite and positive.";
    }

    if (parsed.qscheme == "per_tensor") {
      TVM_FFI_ICHECK_EQ(parsed.scale.size(), 1U)
          << "XNNPACK per-tensor quantization expects a scalar scale.";
      TVM_FFI_ICHECK(parsed.dtype == "int8" || parsed.dtype == "uint8" || parsed.dtype == "int32")
          << "Unsupported XNNPACK per-tensor quantized dtype: " << parsed.dtype;
    } else {
      TVM_FFI_ICHECK(parsed.dtype == "int8" || parsed.dtype == "int32")
          << "Unsupported XNNPACK per-channel quantized dtype: " << parsed.dtype;
      TVM_FFI_ICHECK_EQ(parsed.scale.size(), parsed.shape[parsed.channel_dim])
          << "XNNPACK per-channel quantization scale length must match channel_dim.";
      parsed.padded_scale = parsed.scale;
      parsed.padded_scale.resize(parsed.scale.size() + ExtraQuantizationParams(), 0.0f);
    }

    // Map Relax QDQ axis to XNNPACK channel_dim directly in Phase 5C-0. Quantized
    // layout rewrites are intentionally not implemented in this metadata-only phase.
    int64_t normalized_axis = parsed.axis;
    if (normalized_axis < 0) normalized_axis += static_cast<int64_t>(parsed.shape.size());
    TVM_FFI_ICHECK_GE(normalized_axis, 0) << "XNNPACK quantization axis is out of range.";
    TVM_FFI_ICHECK_LT(static_cast<size_t>(normalized_axis), parsed.shape.size())
        << "XNNPACK quantization axis is out of range.";
    TVM_FFI_ICHECK_EQ(static_cast<size_t>(normalized_axis), parsed.channel_dim)
        << "XNNPACK quantization axis must match channel_dim in Phase 5C-0.";

    (void)QuantizedDatatype(parsed);
#if defined(TVM_XNNPACK_HAS_VALIDATE_QUANTIZED_TENSOR)
    if (parsed.qscheme == "per_tensor") {
      CheckXNNStatus(
          xnn_validate_quantized_tensor(QuantizedDatatype(parsed), parsed.zero_point,
                                        parsed.scale[0], parsed.shape.size(), parsed.shape.data()),
          "xnn_validate_quantized_tensor");
    }
#endif
#if defined(TVM_XNNPACK_HAS_VALIDATE_CHANNELWISE_QUANTIZED_TENSOR)
    if (parsed.qscheme == "per_channel") {
      CheckXNNStatus(xnn_validate_channelwise_quantized_tensor(
                         QuantizedDatatype(parsed), parsed.zero_point, parsed.padded_scale.data(),
                         parsed.shape.size(), parsed.channel_dim, parsed.shape.data()),
                     "xnn_validate_channelwise_quantized_tensor");
    }
#endif
    return parsed;
  }

  static std::string QuantizationMetadataToJSON(const QuantizationMetadata& metadata) {
    std::ostringstream os;
    os << "{\"dtype\":\"" << EscapeJSON(metadata.dtype) << "\",";
    os << "\"qscheme\":\"" << EscapeJSON(metadata.qscheme) << "\",";
    os << "\"signedness\":\"" << EscapeJSON(metadata.signedness) << "\",";
    os << "\"axis\":" << metadata.axis << ",";
    os << "\"channel_dim\":" << metadata.channel_dim << ",";
    os << "\"zero_point\":" << metadata.zero_point << ",";
    os << "\"scale\":";
    if (metadata.scale.size() == 1) {
      os << metadata.scale[0];
    } else {
      os << "[";
      for (size_t i = 0; i < metadata.scale.size(); ++i) {
        if (i != 0) os << ",";
        os << metadata.scale[i];
      }
      os << "]";
    }
    os << ",\"xnn_datatype\":\"" << QuantizedDatatypeName(metadata) << "\",";
    os << "\"extra_quantization_params\":" << ExtraQuantizationParams() << ",";
    os << "\"padded_scale_length\":" << metadata.padded_scale.size() << "}";
    return os.str();
  }

  static void DefineQuantizedTensorForSmoke(const QuantizationMetadata& metadata) {
    TVM_FFI_ICHECK_NE(metadata.qscheme, "none")
        << "XNNPACK quantized tensor smoke test requires quantized metadata.";
    const xnn_status init_status = xnn_initialize(nullptr);
    TVM_FFI_ICHECK_EQ(init_status, xnn_status_success)
        << "Failed to initialize XNNPACK for quantized tensor smoke test.";

    xnn_subgraph_t subgraph = nullptr;
    CheckXNNStatus(xnn_create_subgraph(1, 0, &subgraph), "xnn_create_subgraph");
    auto delete_subgraph = [&subgraph]() {
      if (subgraph != nullptr) {
        xnn_delete_subgraph(subgraph);
        subgraph = nullptr;
      }
    };

    uint32_t id = XNN_INVALID_VALUE_ID;
    xnn_status status = xnn_status_invalid_parameter;
    if (metadata.qscheme == "per_tensor") {
#if defined(TVM_XNNPACK_HAS_DEFINE_QUANTIZED_TENSOR_VALUE)
      status = xnn_define_quantized_tensor_value(
          subgraph, QuantizedDatatype(metadata), metadata.zero_point, metadata.scale[0],
          metadata.shape.size(), metadata.shape.data(), nullptr, XNN_INVALID_VALUE_ID, 0, &id);
#else
      delete_subgraph();
      TVM_FFI_THROW(RuntimeError) << "XNNPACK quantized tensor definition API is unavailable.";
#endif
    } else {
#if defined(TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE_V2)
      status = xnn_define_channelwise_quantized_tensor_value_v2(
          subgraph, QuantizedDatatype(metadata), metadata.zero_point, metadata.padded_scale.data(),
          metadata.shape.size(), metadata.channel_dim, metadata.shape.data(), nullptr,
          XNN_INVALID_VALUE_ID, 0, &id);
#elif defined(TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE)
      TVM_FFI_ICHECK_EQ(metadata.zero_point, 0)
          << "XNNPACK channelwise quantized tensor definition without v2 requires zero_point=0.";
      status = xnn_define_channelwise_quantized_tensor_value(
          subgraph, QuantizedDatatype(metadata), metadata.padded_scale.data(),
          metadata.shape.size(), metadata.channel_dim, metadata.shape.data(), nullptr,
          XNN_INVALID_VALUE_ID, 0, &id);
#else
      delete_subgraph();
      TVM_FFI_THROW(RuntimeError)
          << "XNNPACK channelwise quantized tensor definition API is unavailable.";
#endif
    }
    delete_subgraph();
    CheckXNNStatus(status, "xnn_define_*quantized_tensor_value");
    TVM_FFI_ICHECK_NE(id, XNN_INVALID_VALUE_ID)
        << "XNNPACK quantized tensor smoke test did not define a value.";
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
                             const DLDataType& expected_dtype, const char* name) {
    TVM_FFI_ICHECK(tensor != nullptr) << "Missing XNNPACK " << name << " tensor.";
    TVM_FFI_ICHECK_EQ(tensor->device.device_type, kDLCPU)
        << "XNNPACK " << name << " tensor must be on CPU.";
    TVM_FFI_ICHECK(tensor->dtype.code == expected_dtype.code &&
                   tensor->dtype.bits == expected_dtype.bits &&
                   tensor->dtype.lanes == expected_dtype.lanes)
        << "XNNPACK " << name << " tensor dtype mismatch: expected " << DTypeName(expected_dtype)
        << ", got " << DTypeName(tensor->dtype) << ".";
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

  static DLDataType GetDType(const JSONGraphNode& node, uint32_t index) {
    auto dtypes = node.GetOpDataType();
    TVM_FFI_ICHECK_LT(index, dtypes.size());
    return dtypes[index];
  }

  static void CheckFloat32DType(const JSONGraphNode& node, uint32_t index) {
    DLDataType dtype = GetDType(node, index);
    TVM_FFI_ICHECK(IsFloat32(dtype)) << "XNNPACK float path only supports float32 tensors.";
  }

  static void CheckInt8DType(const JSONGraphNode& node, uint32_t index) {
    DLDataType dtype = GetDType(node, index);
    TVM_FFI_ICHECK(IsInt8(dtype)) << "XNNPACK QS8 path only supports int8 tensor boundaries.";
  }

  static void CheckInt32DType(const JSONGraphNode& node, uint32_t index) {
    DLDataType dtype = GetDType(node, index);
    TVM_FFI_ICHECK(IsInt32(dtype)) << "XNNPACK QS8 bias tensors must be int32.";
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

  static std::vector<size_t> GetSizeArray(const JSONGraphNode& node, const std::string& key) {
    ffi::Array<int64_t> arr = node.GetAttr<ffi::Array<int64_t>>(key);
    std::vector<size_t> result;
    for (int64_t value : arr) {
      TVM_FFI_ICHECK_GT(value, 0);
      result.push_back(static_cast<size_t>(value));
    }
    return result;
  }

  static float GetFloatAttr(const JSONGraphNode& node, const std::string& key) {
    return static_cast<float>(node.GetAttr<double>(key));
  }

  static std::vector<float> ParseFloatList(const std::string& value) {
    std::vector<float> result;
    size_t offset = 0;
    while (offset <= value.size()) {
      size_t comma = value.find(',', offset);
      if (comma == std::string::npos) comma = value.size();
      if (comma > offset) {
        result.push_back(static_cast<float>(std::stod(value.substr(offset, comma - offset))));
      }
      if (comma == value.size()) break;
      offset = comma + 1;
    }
    TVM_FFI_ICHECK(!result.empty()) << "XNNPACK qparam scale list must be non-empty.";
    return result;
  }

  QuantizationMetadata GetNodeQParams(const JSONGraphNode& node, const std::string& prefix,
                                      const std::vector<size_t>& shape, const std::string& dtype) {
    QuantizationMetadata metadata;
    metadata.dtype = dtype;
    metadata.qscheme = std::string(node.GetAttr<ffi::String>(prefix + "_qscheme"));
    metadata.scale = ParseFloatList(std::string(node.GetAttr<ffi::String>(prefix + "_scales")));
    metadata.zero_point = static_cast<int32_t>(node.GetAttr<int64_t>(prefix + "_zero_point"));
    metadata.axis = node.GetAttr<int64_t>(prefix + "_axis");
    int64_t channel_dim = node.GetAttr<int64_t>(prefix + "_channel_dim");
    if (channel_dim < 0) {
      channel_dim = metadata.axis < 0 ? metadata.axis + static_cast<int64_t>(shape.size())
                                      : metadata.axis;
    }
    metadata.channel_dim = static_cast<size_t>(channel_dim);
    metadata.signedness = "signed";
    metadata.shape = shape;
    metadata = ParseQuantizationMetadata(MetadataMap(metadata), shape);
    quantization_metadata_.push_back(metadata);
    return quantization_metadata_.back();
  }

  static ffi::Map<ffi::String, ffi::Any> MetadataMap(const QuantizationMetadata& metadata) {
    ffi::Map<ffi::String, ffi::Any> map;
    map.Set("dtype", metadata.dtype);
    map.Set("qscheme", metadata.qscheme);
    if (metadata.scale.size() == 1) {
      map.Set("scale", static_cast<double>(metadata.scale[0]));
    } else {
      ffi::Array<ffi::Any> scales;
      for (float scale : metadata.scale) scales.push_back(static_cast<double>(scale));
      map.Set("scale", scales);
    }
    map.Set("zero_point", static_cast<int64_t>(metadata.zero_point));
    map.Set("axis", metadata.axis);
    map.Set("channel_dim", static_cast<int64_t>(metadata.channel_dim));
    map.Set("signedness", metadata.signedness);
    return map;
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
    CheckFloat32DType(node, index);
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
    ValidateTensor(tensor, shape, GetDType(node, 0), "constant");
    const size_t bytes = NumElements(shape) * sizeof(float);
    constant_buffers_.emplace_back(bytes + XNN_EXTRA_BYTES);
    std::memcpy(constant_buffers_.back().data(), TensorData(tensor), bytes);
    std::memset(constant_buffers_.back().data() + bytes, 0, XNN_EXTRA_BYTES);
    return constant_buffers_.back().data();
  }

  const void* PrepareTypedConstant(uint32_t eid, const JSONGraphNode& node, uint32_t index) {
    const DLTensor* tensor = data_entry_[eid];
    std::vector<size_t> shape = GetShape(node, index);
    DLDataType dtype = GetDType(node, index);
    ValidateTensor(tensor, shape, dtype, "constant");
    const size_t bytes = NumElements(shape) * ElementSize(dtype);
    constant_buffers_.emplace_back(bytes + XNN_EXTRA_BYTES);
    std::memcpy(constant_buffers_.back().data(), TensorData(tensor), bytes);
    std::memset(constant_buffers_.back().data() + bytes, 0, XNN_EXTRA_BYTES);
    return constant_buffers_.back().data();
  }

  void DefineQuantizedTensor(uint32_t eid, const std::vector<size_t>& shape,
                             const QuantizationMetadata& metadata, uint32_t flags,
                             const void* data = nullptr) {
    if (value_ids_[eid] != XNN_INVALID_VALUE_ID) return;
    uint32_t id = XNN_INVALID_VALUE_ID;
    const uint32_t external_id = flags != 0 ? eid : XNN_INVALID_VALUE_ID;
    if (metadata.qscheme == "per_tensor") {
#if defined(TVM_XNNPACK_HAS_DEFINE_QUANTIZED_TENSOR_VALUE)
      CheckXNNStatus(
          xnn_define_quantized_tensor_value(
              subgraph_, QuantizedDatatype(metadata), metadata.zero_point, metadata.scale[0],
              shape.size(), shape.data(), data, external_id, flags, &id),
          "xnn_define_quantized_tensor_value");
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK quantized tensor definition API is unavailable.";
#endif
    } else {
#if defined(TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE_V2)
      CheckXNNStatus(
          xnn_define_channelwise_quantized_tensor_value_v2(
              subgraph_, QuantizedDatatype(metadata), metadata.zero_point,
              metadata.padded_scale.data(), shape.size(), metadata.channel_dim, shape.data(), data,
              external_id, flags, &id),
          "xnn_define_channelwise_quantized_tensor_value_v2");
#elif defined(TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE)
      TVM_FFI_ICHECK_EQ(metadata.zero_point, 0)
          << "XNNPACK channelwise quantized tensor definition without v2 requires zero_point=0.";
      CheckXNNStatus(
          xnn_define_channelwise_quantized_tensor_value(
              subgraph_, QuantizedDatatype(metadata), metadata.padded_scale.data(), shape.size(),
              metadata.channel_dim, shape.data(), data, external_id, flags, &id),
          "xnn_define_channelwise_quantized_tensor_value");
#else
      TVM_FFI_THROW(RuntimeError)
          << "XNNPACK channelwise quantized tensor definition API is unavailable.";
#endif
    }
    if (flags != 0) {
      TVM_FFI_ICHECK_EQ(id, eid);
    }
    value_ids_[eid] = id;
  }

  void DefineGraphInputsAndConstants() {
    for (uint32_t eid : input_var_eid_) {
      const uint32_t nid = NodeIDFromEntryID(eid);
      if (!IsFloat32(GetDType(nodes_[nid], 0))) continue;
      DefineTensor(eid, nodes_[nid], 0, XNN_VALUE_FLAG_EXTERNAL_INPUT);
      external_tensors_.push_back(
          {eid, GetShape(nodes_[nid], 0), "input", GetDType(nodes_[nid], 0), sizeof(float), false,
           {}});
    }

    for (uint32_t nid : const_idx_) {
      const uint32_t eid = EntryID(nid, 0);
      if (!IsFloat32(GetDType(nodes_[nid], 0))) continue;
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
      external_tensors_.push_back({eid, GetShape(node, output_entry.index_), "output",
                                   GetDType(node, output_entry.index_), sizeof(float), true, {}});
    }
  }

  uint32_t DefineQS8Output(const JSONGraphNode& node, const JSONGraphNodeEntry& output_entry,
                           const std::unordered_set<uint32_t>& graph_output_eids) {
    const uint32_t eid = EntryID(output_entry);
    const uint32_t flags =
        IsGraphOutput(graph_output_eids, eid) ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0;
    CheckInt8DType(node, output_entry.index_);
    std::vector<size_t> shape = GetShape(node, output_entry.index_);
    QuantizationMetadata qparams = GetNodeQParams(node, "output", shape, "int8");
    DefineQuantizedTensor(eid, shape, qparams, flags);
    if (flags != 0) {
      external_tensors_.push_back(
          {eid, shape, "output", GetDType(node, output_entry.index_), sizeof(int8_t), true, {}});
    }
    return value_ids_[eid];
  }

  void DefineQS8ExternalInput(const JSONGraphNode& node,
                              const std::vector<JSONGraphNodeEntry>& inputs, size_t input_index,
                              const std::string& qparam_prefix) {
    TVM_FFI_ICHECK_LT(input_index, inputs.size());
    const uint32_t input_eid = EntryID(inputs[input_index]);
    if (value_ids_[input_eid] != XNN_INVALID_VALUE_ID) return;
    const uint32_t input_nid = inputs[input_index].id_;
    CheckInt8DType(nodes_[input_nid], inputs[input_index].index_);
    std::vector<size_t> input_shape = GetShape(nodes_[input_nid], inputs[input_index].index_);
    QuantizationMetadata input_qparams = GetNodeQParams(node, qparam_prefix, input_shape, "int8");
    const bool is_external =
        std::find(input_var_eid_.begin(), input_var_eid_.end(), input_eid) != input_var_eid_.end();
    DefineQuantizedTensor(input_eid, input_shape, input_qparams,
                          is_external ? XNN_VALUE_FLAG_EXTERNAL_INPUT : 0);
    if (is_external && std::none_of(external_tensors_.begin(), external_tensors_.end(),
                                    [input_eid](const ExternalTensor& entry) {
                                      return entry.eid == input_eid;
                                    })) {
      external_tensors_.push_back(
          {input_eid, input_shape, "input", GetDType(nodes_[input_nid], inputs[input_index].index_),
           sizeof(int8_t), false, {}});
    }
  }

  void DefineQS8IslandInputs(const JSONGraphNode& node,
                             const std::vector<JSONGraphNodeEntry>& inputs) {
    const std::string op_kind = node.GetAttr<ffi::String>("op_kind");
    if (op_kind == "qs8_add") {
      TVM_FFI_ICHECK_EQ(inputs.size(), 2U);
      DefineQS8ExternalInput(node, inputs, 0, "lhs");
      DefineQS8ExternalInput(node, inputs, 1, "rhs");
    } else {
      TVM_FFI_ICHECK_EQ(inputs.size(), 1U);
      DefineQS8ExternalInput(node, inputs, 0, "input");
    }
  }

  void DefineQS8Inputs(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs) {
    TVM_FFI_ICHECK(inputs.size() == 2U || inputs.size() == 3U);
    const uint32_t input_eid = EntryID(inputs[0]);
    const uint32_t input_nid = inputs[0].id_;
    CheckInt8DType(nodes_[input_nid], inputs[0].index_);
    std::vector<size_t> input_shape = GetShape(nodes_[input_nid], inputs[0].index_);
    QuantizationMetadata input_qparams = GetNodeQParams(node, "input", input_shape, "int8");
    DefineQuantizedTensor(input_eid, input_shape, input_qparams, XNN_VALUE_FLAG_EXTERNAL_INPUT);
    if (std::none_of(external_tensors_.begin(), external_tensors_.end(),
                     [input_eid](const ExternalTensor& entry) { return entry.eid == input_eid; })) {
      external_tensors_.push_back(
          {input_eid, input_shape, "input", GetDType(nodes_[input_nid], inputs[0].index_),
           sizeof(int8_t), false, {}});
    }

    const uint32_t weight_eid = EntryID(inputs[1]);
    const uint32_t weight_nid = inputs[1].id_;
    CheckInt8DType(nodes_[weight_nid], inputs[1].index_);
    std::vector<size_t> weight_shape = GetShape(nodes_[weight_nid], inputs[1].index_);
    const void* weight_data =
        PrepareTypedConstant(weight_eid, nodes_[weight_nid], inputs[1].index_);
    QuantizationMetadata weight_qparams = GetNodeQParams(node, "weight", weight_shape, "int8");
    DefineQuantizedTensor(weight_eid, weight_shape, weight_qparams, 0, weight_data);

    if (inputs.size() == 3U) {
      const uint32_t bias_eid = EntryID(inputs[2]);
      const uint32_t bias_nid = inputs[2].id_;
      CheckInt32DType(nodes_[bias_nid], inputs[2].index_);
      std::vector<size_t> bias_shape = GetShape(nodes_[bias_nid], inputs[2].index_);
      const void* bias_data = PrepareTypedConstant(bias_eid, nodes_[bias_nid], inputs[2].index_);
      QuantizationMetadata bias_qparams = GetNodeQParams(node, "bias", bias_shape, "int32");
      DefineQuantizedTensor(bias_eid, bias_shape, bias_qparams, 0, bias_data);
    }
  }

  void DefineQS8DepthwiseInputs(const JSONGraphNode& node,
                                const std::vector<JSONGraphNodeEntry>& inputs) {
    TVM_FFI_ICHECK(inputs.size() == 2U || inputs.size() == 3U);
    const uint32_t input_eid = EntryID(inputs[0]);
    const uint32_t input_nid = inputs[0].id_;
    CheckInt8DType(nodes_[input_nid], inputs[0].index_);
    std::vector<size_t> input_shape = GetShape(nodes_[input_nid], inputs[0].index_);
    QuantizationMetadata input_qparams = GetNodeQParams(node, "input", input_shape, "int8");
    DefineQuantizedTensor(input_eid, input_shape, input_qparams, XNN_VALUE_FLAG_EXTERNAL_INPUT);
    if (std::none_of(external_tensors_.begin(), external_tensors_.end(),
                     [input_eid](const ExternalTensor& entry) { return entry.eid == input_eid; })) {
      external_tensors_.push_back(
          {input_eid, input_shape, "input", GetDType(nodes_[input_nid], inputs[0].index_),
           sizeof(int8_t), false, {}});
    }

    const uint32_t weight_eid = EntryID(inputs[1]);
    const uint32_t weight_nid = inputs[1].id_;
    CheckInt8DType(nodes_[weight_nid], inputs[1].index_);
    std::vector<size_t> hwoi_shape = GetShape(nodes_[weight_nid], inputs[1].index_);
    TVM_FFI_ICHECK_EQ(hwoi_shape.size(), 4U);
    TVM_FFI_ICHECK_EQ(hwoi_shape[3], 1U)
        << "XNNPACK QS8 depthwise currently requires depth_multiplier=1.";
    std::vector<size_t> xnn_shape = {1, hwoi_shape[0], hwoi_shape[1],
                                     hwoi_shape[2] * hwoi_shape[3]};
    const void* weight_data =
        PrepareTypedConstant(weight_eid, nodes_[weight_nid], inputs[1].index_);
    QuantizationMetadata weight_qparams = GetNodeQParams(node, "weight", xnn_shape, "int8");
    DefineQuantizedTensor(weight_eid, xnn_shape, weight_qparams, 0, weight_data);

    if (inputs.size() == 3U) {
      const uint32_t bias_eid = EntryID(inputs[2]);
      const uint32_t bias_nid = inputs[2].id_;
      CheckInt32DType(nodes_[bias_nid], inputs[2].index_);
      std::vector<size_t> bias_shape = GetShape(nodes_[bias_nid], inputs[2].index_);
      const void* bias_data = PrepareTypedConstant(bias_eid, nodes_[bias_nid], inputs[2].index_);
      QuantizationMetadata bias_qparams = GetNodeQParams(node, "bias", bias_shape, "int32");
      DefineQuantizedTensor(bias_eid, bias_shape, bias_qparams, 0, bias_data);
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

  void DefineQS8Add(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
                    uint32_t output_id) {
    TVM_FFI_ICHECK_EQ(inputs.size(), 2U);
    xnn_binary_params params{};
    params.output_min = GetFloatAttr(node, "activation_min");
    params.output_max = GetFloatAttr(node, "activation_max");
    CheckXNNStatus(
        xnn_define_binary(subgraph_, xnn_binary_add, &params, value_ids_[EntryID(inputs[0])],
                          value_ids_[EntryID(inputs[1])], output_id, XNN_FLAG_NO_BROADCAST),
        "xnn_define_binary(qs8_add)");
  }

  void DefineQS8Reshape(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
                        uint32_t output_id) {
#if defined(TVM_XNNPACK_HAS_STATIC_RESHAPE)
    TVM_FFI_ICHECK_EQ(inputs.size(), 1U);
    std::vector<size_t> new_shape = GetSizeArray(node, "new_shape");
    CheckXNNStatus(
        xnn_define_static_reshape(subgraph_, new_shape.size(), new_shape.data(),
                                  value_ids_[EntryID(inputs[0])], output_id, 0),
        "xnn_define_static_reshape");
#else
    TVM_FFI_THROW(RuntimeError) << "XNNPACK static reshape API is unavailable.";
#endif
  }

  void DefineQS8Copy(const std::vector<JSONGraphNodeEntry>& inputs, uint32_t output_id) {
#if defined(TVM_XNNPACK_HAS_COPY)
    TVM_FFI_ICHECK_EQ(inputs.size(), 1U);
    CheckXNNStatus(xnn_define_copy(subgraph_, value_ids_[EntryID(inputs[0])], output_id, 0),
                   "xnn_define_copy");
#else
    TVM_FFI_THROW(RuntimeError) << "XNNPACK copy API is unavailable.";
#endif
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

  void DefineQS8FullyConnected(const JSONGraphNode& node,
                               const std::vector<JSONGraphNodeEntry>& inputs,
                               uint32_t output_id) {
#if defined(TVM_XNNPACK_HAS_FULLY_CONNECTED)
    const bool has_bias = static_cast<int64_t>(node.GetAttr<int64_t>("has_bias")) != 0;
    TVM_FFI_ICHECK_EQ(inputs.size(), has_bias ? 3U : 2U);
    const uint32_t bias_id = has_bias ? value_ids_[EntryID(inputs[2])] : XNN_INVALID_VALUE_ID;
    uint32_t flags = 0;
#if defined(TVM_XNNPACK_HAS_TRANSPOSE_WEIGHTS_FLAG)
    flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
#else
    TVM_FFI_THROW(RuntimeError)
        << "XNNPACK fully_connected with Relax [input_channels, output_channels] weights "
           "requires XNN_FLAG_TRANSPOSE_WEIGHTS.";
#endif
    CheckXNNStatus(xnn_define_fully_connected(
                       subgraph_, GetFloatAttr(node, "activation_min"),
                       GetFloatAttr(node, "activation_max"), value_ids_[EntryID(inputs[0])],
                       value_ids_[EntryID(inputs[1])], bias_id, output_id, flags),
                   "xnn_define_fully_connected");
#else
    TVM_FFI_THROW(RuntimeError) << "XNNPACK fully_connected API is unavailable.";
#endif
  }

  void DefineQS8Conv2D(const JSONGraphNode& node, const std::vector<JSONGraphNodeEntry>& inputs,
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
                       weight_shape[2], strides[0], strides[1], dilation[0], dilation[1], 1,
                       weight_shape[3], weight_shape[0], GetFloatAttr(node, "activation_min"),
                       GetFloatAttr(node, "activation_max"), value_ids_[EntryID(inputs[0])],
                       value_ids_[EntryID(inputs[1])], bias_id, output_id, 0),
                   "xnn_define_convolution_2d(qs8)");
  }

  void DefineQS8DepthwiseConv2D(const JSONGraphNode& node,
                                const std::vector<JSONGraphNodeEntry>& inputs,
                                uint32_t output_id) {
#if defined(TVM_XNNPACK_HAS_DEPTHWISE_CONVOLUTION_2D)
    const bool has_bias = static_cast<int64_t>(node.GetAttr<int64_t>("has_bias")) != 0;
    TVM_FFI_ICHECK_EQ(inputs.size(), has_bias ? 3U : 2U);
    auto padding = GetUIntArray(node, "padding");
    auto strides = GetUIntArray(node, "strides");
    auto dilation = GetUIntArray(node, "dilation");
    std::vector<size_t> input_shape = GetShape(nodes_[inputs[0].id_], inputs[0].index_);
    std::vector<size_t> weight_shape = GetShape(nodes_[inputs[1].id_], inputs[1].index_);
    TVM_FFI_ICHECK_EQ(input_shape.size(), 4U);
    TVM_FFI_ICHECK_EQ(weight_shape.size(), 4U);
    const uint32_t input_channels = static_cast<uint32_t>(input_shape[3]);
    const uint32_t depth_multiplier = static_cast<uint32_t>(weight_shape[3]);
    const uint32_t bias_id = has_bias ? value_ids_[EntryID(inputs[2])] : XNN_INVALID_VALUE_ID;
    CheckXNNStatus(xnn_define_depthwise_convolution_2d(
                       subgraph_, padding[0], padding[3], padding[2], padding[1], weight_shape[0],
                       weight_shape[1], strides[0], strides[1], dilation[0], dilation[1],
                       depth_multiplier, input_channels, GetFloatAttr(node, "activation_min"),
                       GetFloatAttr(node, "activation_max"), value_ids_[EntryID(inputs[0])],
                       value_ids_[EntryID(inputs[1])], bias_id, output_id, 0),
                   "xnn_define_depthwise_convolution_2d(qs8)");
#else
    TVM_FFI_THROW(RuntimeError) << "XNNPACK depthwise convolution API is unavailable.";
#endif
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
    if (options_.precision == "fp16_hint") {
#if defined(TVM_XNNPACK_HAS_HINT_FP16_INFERENCE_FLAG)
      flags |= XNN_FLAG_HINT_FP16_INFERENCE;
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK precision='fp16_hint' was requested but "
                                     "XNN_FLAG_HINT_FP16_INFERENCE is unavailable.";
#endif
    } else if (options_.precision == "fp16_force") {
#if defined(TVM_XNNPACK_HAS_FORCE_FP16_INFERENCE_FLAG)
      flags |= XNN_FLAG_FORCE_FP16_INFERENCE;
#else
      TVM_FFI_THROW(RuntimeError) << "XNNPACK precision='fp16_force' was requested but "
                                     "XNN_FLAG_FORCE_FP16_INFERENCE is unavailable.";
#endif
    } else {
      TVM_FFI_ICHECK_EQ(options_.precision, "fp32");
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
    CheckRuntimeCreateStatus(
        xnn_create_runtime_v4(subgraph_, weights_cache_, workspace_, threadpool_, flags, &runtime_),
        "xnn_create_runtime_v4");
#elif defined(TVM_XNNPACK_HAS_RUNTIME_V3) && defined(TVM_XNNPACK_HAS_WEIGHTS_CACHE)
    TVM_FFI_ICHECK(!options_.use_workspace) << "XNNPACK workspace requires xnn_create_runtime_v4.";
    CheckRuntimeCreateStatus(
        xnn_create_runtime_v3(subgraph_, weights_cache_, threadpool_, flags, &runtime_),
        "xnn_create_runtime_v3");
#else
    TVM_FFI_ICHECK(!options_.use_weights_cache)
        << "XNNPACK weights cache requires xnn_create_runtime_v3 or newer.";
    TVM_FFI_ICHECK(!options_.use_workspace) << "XNNPACK workspace requires xnn_create_runtime_v4.";
    CheckRuntimeCreateStatus(xnn_create_runtime_v2(subgraph_, threadpool_, flags, &runtime_),
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

  std::string GetQuantizationMetadataJSON() const {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < quantization_metadata_.size(); ++i) {
      if (i != 0) os << ",";
      os << QuantizationMetadataToJSON(quantization_metadata_[i]);
    }
    os << "]";
    return os.str();
  }

  void BuildRuntime() {
    CheckXNNStatus(xnn_create_subgraph(NumEntries(), 0, &subgraph_), "xnn_create_subgraph");
    value_ids_.assign(NumEntries(), XNN_INVALID_VALUE_ID);
    external_tensors_.clear();
    constant_buffers_.clear();
    quantization_metadata_.clear();

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
      auto inputs = node.GetInputs();
      const std::string op_kind = node.GetAttr<ffi::String>("op_kind");
      uint32_t output_id = XNN_INVALID_VALUE_ID;
      if (op_kind == "qs8_fully_connected" || op_kind == "qs8_conv2d" ||
          op_kind == "qs8_depthwise_conv2d") {
        if (op_kind == "qs8_depthwise_conv2d") {
          DefineQS8DepthwiseInputs(node, inputs);
        } else {
          DefineQS8Inputs(node, inputs);
        }
        output_id = DefineQS8Output(node, output_entry, graph_output_eids);
      } else if (op_kind == "qs8_reshape" || op_kind == "qs8_max_pool2d" ||
                 op_kind == "qs8_avg_pool2d" || op_kind == "qs8_add" ||
                 op_kind == "qs8_copy") {
        DefineQS8IslandInputs(node, inputs);
        output_id = DefineQS8Output(node, output_entry, graph_output_eids);
      } else {
        DefineOutput(node, output_entry, graph_output_eids);
        output_id = value_ids_[EntryID(output_entry)];
      }

      for (const auto& input : inputs) {
        TVM_FFI_ICHECK_LT(EntryID(input), value_ids_.size());
        TVM_FFI_ICHECK_NE(value_ids_[EntryID(input)], XNN_INVALID_VALUE_ID)
            << "XNNPACK input value was not defined before its use.";
      }

      if (op_kind == "unary") {
        DefineUnary(node, inputs, output_id);
      } else if (op_kind == "add") {
        DefineAdd(node, inputs, output_id);
      } else if (op_kind == "conv2d") {
        DefineConv2D(node, inputs, output_id);
      } else if (op_kind == "qs8_fully_connected") {
        DefineQS8FullyConnected(node, inputs, output_id);
      } else if (op_kind == "qs8_conv2d") {
        DefineQS8Conv2D(node, inputs, output_id);
      } else if (op_kind == "qs8_depthwise_conv2d") {
        DefineQS8DepthwiseConv2D(node, inputs, output_id);
      } else if (op_kind == "qs8_reshape") {
        DefineQS8Reshape(node, inputs, output_id);
      } else if (op_kind == "qs8_copy") {
        DefineQS8Copy(inputs, output_id);
      } else if (op_kind == "qs8_max_pool2d") {
        DefinePool2D(node, inputs, output_id, true);
      } else if (op_kind == "qs8_avg_pool2d") {
        DefinePool2D(node, inputs, output_id, false);
      } else if (op_kind == "qs8_add") {
        DefineQS8Add(node, inputs, output_id);
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
  std::vector<QuantizationMetadata> quantization_metadata_;
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
  result.Set("fp16_hint", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_HINT_FP16_INFERENCE_FLAG)
                              1
#else
                              0
#endif
                              ));
  result.Set("fp16_force", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_FORCE_FP16_INFERENCE_FLAG)
                               1
#else
                               0
#endif
                               ));
  result.Set("datatype_fp16", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DATATYPE_FP16)
                                  1
#else
                                  0
#endif
                                  ));
  result.Set("datatype_qint8", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DATATYPE_QINT8)
                                   1
#else
                                   0
#endif
                                   ));
  result.Set("datatype_quint8", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DATATYPE_QUINT8)
                                    1
#else
                                    0
#endif
                                    ));
  result.Set("datatype_qint32", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DATATYPE_QINT32)
                                    1
#else
                                    0
#endif
                                    ));
  result.Set("datatype_qcint8", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DATATYPE_QCINT8)
                                    1
#else
                                    0
#endif
                                    ));
  result.Set("datatype_qcint32", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DATATYPE_QCINT32)
                                     1
#else
                                     0
#endif
                                     ));
  result.Set("extra_quantization_params", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_EXTRA_QUANTIZATION_PARAMS)
                                              XNN_EXTRA_QUANTIZATION_PARAMS
#else
                                              0
#endif
                                              ));
  result.Set("define_quantized_tensor_value", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DEFINE_QUANTIZED_TENSOR_VALUE)
                                                  1
#else
                                                  0
#endif
                                                  ));
  result.Set("define_channelwise_quantized_tensor_value", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE) || \
    defined(TVM_XNNPACK_HAS_DEFINE_CHANNELWISE_QUANTIZED_TENSOR_VALUE_V2)
                                                              1
#else
                                                              0
#endif
                                                              ));
  result.Set("validate_quantized_tensor", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_VALIDATE_QUANTIZED_TENSOR)
                                              1
#else
                                              0
#endif
                                              ));
  result.Set("validate_channelwise_quantized_tensor", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_VALIDATE_CHANNELWISE_QUANTIZED_TENSOR)
                                                          1
#else
                                                          0
#endif
                                                          ));
  result.Set("fully_connected", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_FULLY_CONNECTED)
                                            1
#else
                                            0
#endif
                                            ));
  result.Set("depthwise_convolution_2d", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_DEPTHWISE_CONVOLUTION_2D)
                                                     1
#else
                                                     0
#endif
                                                     ));
  result.Set("static_reshape", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_STATIC_RESHAPE)
                                      1
#else
                                      0
#endif
                                      ));
  result.Set("copy", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_COPY)
                             1
#else
                             0
#endif
                             ));
  result.Set("transpose_weights", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_TRANSPOSE_WEIGHTS_FLAG)
                                              1
#else
                                              0
#endif
                                              ));
  result.Set("fp32_static_weights", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_FP32_STATIC_WEIGHTS_FLAG)
                                        1
#else
                                        0
#endif
                                        ));
  result.Set("fp32_static_biases", static_cast<int64_t>(
#if defined(TVM_XNNPACK_HAS_FP32_STATIC_BIASES_FLAG)
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
      .def("runtime.XNNPACKJSONRuntimeValidateQuantizationMetadata",
           XNNPACKJSONRuntime::ValidateQuantizationMetadataJSON)
      .def("runtime.XNNPACKJSONRuntimeQuantizedTensorDefinitionSmoke",
           XNNPACKJSONRuntime::QuantizedTensorDefinitionSmoke)
      .def("ffi.Module.load_from_bytes.xnnpack_json", XNNPACKJSONRuntimeLoadFromBytes);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
