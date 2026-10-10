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
 * \file metadata.h
 * \brief Meta data related utilities
 */
#ifndef TVM_RUNTIME_METADATA_H_
#define TVM_RUNTIME_METADATA_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

inline ffi::String get_name_mangled(const ffi::String& module_name, const ffi::String& name) {
  std::stringstream ss;
  ss << module_name << "_" << name;
  return ss.str();
}

namespace launch_param {

/*! \brief A tag to specify whether or not dynamic shared memory is used */
constexpr const char* kUseDynamicSharedMemoryTag = "tirx.use_dyn_shared_memory";
/*! \brief A tag to specify whether or not use programatic dependent launch */
constexpr const char* kUseProgramaticDependentLaunch = "tirx.use_programtic_dependent_launch";
/*! \brief A tag to specify whether or not use cooperative launch */
constexpr const char* kUseCooperativeLaunch = "tirx.use_cooperative_launch";
/*! \brief A tag to launch with CUDA's statically required block dimension */
constexpr const char* kUseRequiredBlockDimension = "tirx.use_required_block_dimension";

}  // namespace launch_param

/*! \brief Extra tags for function arguments */
enum class ArgExtraTags : int { kNone = 0, kTensorMap = 1 };

/*! \brief function information needed by device */
class FunctionInfoObj : public ffi::Object {
 public:
  ffi::String name;
  ffi::Array<DLDataType> arg_types;
  ffi::Array<ffi::String> launch_param_tags;
  ffi::Array<ArgExtraTags> arg_extra_tags;
  ffi::Array<ffi::String> cuda_launch_fields;
  ffi::Map<ffi::String, int64_t> cuda_kernel_attrs;

  ffi::json::Value SaveToJSON() const {
    namespace json = ::tvm::ffi::json;
    json::Object obj;
    obj.Set("name", name);
    json::Array sarg_types;
    for (const auto& t : arg_types) {
      sarg_types.push_back(ffi::String(ffi::DLDataTypeToString(t)));
    }
    obj.Set("arg_types", std::move(sarg_types));
    {
      json::Array tags;
      for (const auto& s : launch_param_tags) tags.push_back(s);
      obj.Set("launch_param_tags", std::move(tags));
    }
    json::Array iarg_extra_tags;
    for (const auto& t : arg_extra_tags) {
      iarg_extra_tags.push_back(static_cast<int64_t>(t));
    }
    obj.Set("arg_extra_tags", std::move(iarg_extra_tags));
    if (!cuda_launch_fields.empty()) {
      json::Array fields;
      for (const auto& field : cuda_launch_fields) fields.push_back(field);
      obj.Set("cuda_launch_fields", std::move(fields));
    }
    if (!cuda_kernel_attrs.empty()) {
      json::Object kernel_attrs;
      for (const auto& [key, value] : cuda_kernel_attrs) kernel_attrs.Set(key, value);
      obj.Set("cuda_kernel_attrs", std::move(kernel_attrs));
    }
    return obj;
  }

  void LoadFromJSON(ffi::json::Object src) {
    namespace json = ::tvm::ffi::json;
    cuda_launch_fields = {};
    cuda_kernel_attrs = {};
    name = src.at("name").cast<ffi::String>();
    auto sarg_types_arr = src.at("arg_types").cast<json::Array>();
    arg_types = ffi::Array<DLDataType>();
    for (size_t i = 0; i < sarg_types_arr.size(); ++i) {
      arg_types.push_back(
          ffi::StringToDLDataType(std::string(sarg_types_arr[i].cast<ffi::String>())));
    }
    auto lt = src.find("launch_param_tags");
    if (lt != src.end()) {
      auto arr = (*lt).second.cast<json::Array>();
      launch_param_tags = ffi::Array<ffi::String>();
      for (const auto& elem : arr) launch_param_tags.push_back(elem.cast<ffi::String>());
    } else {
      auto tt = src.find("thread_axis_tags");
      if (tt != src.end()) {
        auto arr = (*tt).second.cast<json::Array>();
        launch_param_tags = ffi::Array<ffi::String>();
        for (const auto& elem : arr) launch_param_tags.push_back(elem.cast<ffi::String>());
      }
    }
    auto et = src.find("arg_extra_tags");
    if (et != src.end()) {
      auto earr = (*et).second.cast<json::Array>();
      arg_extra_tags = ffi::Array<ArgExtraTags>();
      for (size_t i = 0; i < earr.size(); ++i) {
        arg_extra_tags.push_back(static_cast<ArgExtraTags>(earr[i].cast<int64_t>()));
      }
    }
    if (auto it = src.find("cuda_launch_fields"); it != src.end()) {
      for (const auto& field : (*it).second.cast<json::Array>())
        cuda_launch_fields.push_back(field.cast<ffi::String>());
    }
    if (auto it = src.find("cuda_kernel_attrs"); it != src.end()) {
      for (const auto& [key, value] : (*it).second.cast<json::Object>())
        cuda_kernel_attrs.Set(key.cast<ffi::String>(), value.cast<int64_t>());
    }
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.FunctionInfo", FunctionInfoObj, ffi::Object);
};

class FunctionInfo : public ffi::ObjectRef {
 public:
  FunctionInfo(ffi::String name, ffi::Array<DLDataType> arg_types,
               ffi::Array<ffi::String> launch_param_tags, ffi::Array<ArgExtraTags> arg_extra_tags,
               ffi::Array<ffi::String> cuda_launch_fields = {},
               ffi::Map<ffi::String, int64_t> cuda_kernel_attrs = {}) {
    auto n = ffi::make_object<FunctionInfoObj>();
    n->name = std::move(name);
    n->arg_types = std::move(arg_types);
    n->launch_param_tags = std::move(launch_param_tags);
    n->arg_extra_tags = std::move(arg_extra_tags);
    n->cuda_launch_fields = std::move(cuda_launch_fields);
    n->cuda_kernel_attrs = std::move(cuda_kernel_attrs);
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FunctionInfo, ffi::ObjectRef, FunctionInfoObj);
};

}  // namespace runtime

namespace support {

// ---- Serializer for runtime::FunctionInfo ----
template <>
struct Serializer<runtime::FunctionInfo> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const runtime::FunctionInfo& info) {
    Serializer<ffi::String>::Write(strm, "__tvm_function_info_v2__");
    Serializer<ffi::String>::Write(strm, info->name);
    Serializer<ffi::Array<DLDataType>>::Write(strm, info->arg_types);
    Serializer<ffi::Array<ffi::String>>::Write(strm, info->launch_param_tags);
    Serializer<ffi::Array<runtime::ArgExtraTags>>::Write(strm, info->arg_extra_tags);
    Serializer<ffi::Array<ffi::String>>::Write(strm, info->cuda_launch_fields);
    Serializer<ffi::Map<ffi::String, int64_t>>::Write(strm, info->cuda_kernel_attrs);
  }

  static bool Read(Stream* strm, runtime::FunctionInfo* info) {
    auto n = ffi::make_object<runtime::FunctionInfoObj>();
    if (!Serializer<ffi::String>::Read(strm, &(n->name))) return false;
    bool version2 = n->name == "__tvm_function_info_v2__";
    if (version2 && !Serializer<ffi::String>::Read(strm, &(n->name))) return false;
    if (!Serializer<ffi::Array<DLDataType>>::Read(strm, &(n->arg_types))) return false;
    if (!Serializer<ffi::Array<ffi::String>>::Read(strm, &(n->launch_param_tags))) return false;
    if (!Serializer<ffi::Array<runtime::ArgExtraTags>>::Read(strm, &(n->arg_extra_tags)))
      return false;
    if (version2) {
      if (!Serializer<ffi::Array<ffi::String>>::Read(strm, &(n->cuda_launch_fields))) return false;
      if (!Serializer<ffi::Map<ffi::String, int64_t>>::Read(strm, &(n->cuda_kernel_attrs)))
        return false;
    }
    *info = runtime::FunctionInfo(std::move(n));
    return true;
  }
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_RUNTIME_METADATA_H_
