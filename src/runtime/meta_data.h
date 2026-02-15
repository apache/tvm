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
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef TVM_RUNTIME_META_DATA_H_
#define TVM_RUNTIME_META_DATA_H_

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <string>
#include <unordered_map>
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

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<DLDataType> arg_types;
  std::vector<std::string> launch_param_tags;

  enum class ArgExtraTags : int { kNone = 0, kTensorMap = 1 };
  std::vector<ArgExtraTags> arg_extra_tags;

  ffi::json::Value SaveToJSON() const;
  void LoadFromJSON(ffi::json::Object src);
  void Save(support::Stream* writer) const;
  bool Load(support::Stream* reader);
};
}  // namespace runtime
}  // namespace tvm

namespace tvm {
namespace support {
template <>
struct Serializer<::tvm::runtime::FunctionInfo> {
  static constexpr bool enabled = true;
  static void Write(Stream* strm, const ::tvm::runtime::FunctionInfo& data) { data.Save(strm); }
  static bool Read(Stream* strm, ::tvm::runtime::FunctionInfo* data) { return data->Load(strm); }
};
}  // namespace support
}  // namespace tvm

#endif  // TVM_RUNTIME_META_DATA_H_
