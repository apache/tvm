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
#include <tvm/ffi/function.h>
#include <tvm/runtime/module.h>
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
constexpr const char* kUseDynamicSharedMemoryTag = "tir.use_dyn_shared_memory";
/*! \brief A tag to specify whether or not use programatic dependent launch */
constexpr const char* kUseProgramaticDependentLaunch = "tir.use_programtic_dependent_launch";
/*! \brief A tag to specify whether or not use cooperative launch */
constexpr const char* kUseCooperativeLaunch = "tir.use_cooperative_launch";

}  // namespace launch_param

/*! \brief Extra tags for function arguments */
enum class ArgExtraTags : int { kNone = 0, kTensorMap = 1 };

/*! \brief function information needed by device */
class FunctionInfoObj : public Object {
 public:
  ffi::String name;
  ffi::Array<DLDataType> arg_types;
  ffi::Array<ffi::String> launch_param_tags;
  ffi::Array<ArgExtraTags> arg_extra_tags;

  ffi::json::Value SaveToJSON() const;
  void LoadFromJSON(ffi::json::Object src);

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.FunctionInfo", FunctionInfoObj, Object);
};

class FunctionInfo : public ObjectRef {
 public:
  FunctionInfo(ffi::String name, ffi::Array<DLDataType> arg_types,
               ffi::Array<ffi::String> launch_param_tags, ffi::Array<ArgExtraTags> arg_extra_tags);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(FunctionInfo, ObjectRef, FunctionInfoObj);
};

}  // namespace runtime

namespace support {

// ---- Serializer for runtime::FunctionInfo ----
template <>
struct Serializer<runtime::FunctionInfo> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const runtime::FunctionInfo& info) {
    Serializer<ffi::String>::Write(strm, info->name);
    Serializer<ffi::Array<DLDataType>>::Write(strm, info->arg_types);
    Serializer<ffi::Array<ffi::String>>::Write(strm, info->launch_param_tags);
    Serializer<ffi::Array<runtime::ArgExtraTags>>::Write(strm, info->arg_extra_tags);
  }

  static bool Read(Stream* strm, runtime::FunctionInfo* info) {
    auto n = ffi::make_object<runtime::FunctionInfoObj>();
    if (!Serializer<ffi::String>::Read(strm, &(n->name))) return false;
    if (!Serializer<ffi::Array<DLDataType>>::Read(strm, &(n->arg_types))) return false;
    if (!Serializer<ffi::Array<ffi::String>>::Read(strm, &(n->launch_param_tags))) return false;
    if (!Serializer<ffi::Array<runtime::ArgExtraTags>>::Read(strm, &(n->arg_extra_tags)))
      return false;
    *info = runtime::FunctionInfo(std::move(n));
    return true;
  }
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_RUNTIME_METADATA_H_
