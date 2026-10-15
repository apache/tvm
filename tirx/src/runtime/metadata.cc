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
 * \file metadata.cc
 * \brief Runtime function metadata reflection and JSON conversion.
 */
#include "metadata.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <utility>

namespace tvm {
namespace runtime {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<FunctionInfoObj>()
      .def_ro("name", &FunctionInfoObj::name)
      .def_ro("arg_types", &FunctionInfoObj::arg_types)
      .def_ro("launch_param_tags", &FunctionInfoObj::launch_param_tags)
      .def_ro("arg_extra_tags", &FunctionInfoObj::arg_extra_tags)
      .def_ro("cuda_launch_fields", &FunctionInfoObj::cuda_launch_fields,
              refl::DefaultValue(ffi::Array<ffi::String>()))
      .def_ro("cuda_kernel_attrs", &FunctionInfoObj::cuda_kernel_attrs,
              refl::DefaultValue(ffi::Map<ffi::String, int64_t>()));
  refl::GlobalDef()
      // Registry: "runtime.LoadMetaDataFromJSON" — parse a tvm_meta.json
      // string into Map<String, FunctionInfo>.  Used by Python callers that
      // build FunctionInfo maps in-memory (e.g. tirx external_kernel) without
      // a disk round-trip.
      .def("runtime.LoadMetaDataFromJSON", [](const ffi::String& json_str) {
        namespace json = ::tvm::ffi::json;
        ffi::Map<ffi::String, FunctionInfo> fmap;
        auto root = json::Parse(std::string(json_str)).cast<json::Object>();
        auto func_info_obj = root.at("func_info").cast<json::Object>();
        for (const auto& kv : func_info_obj) {
          auto info_node = ffi::make_object<FunctionInfoObj>();
          info_node->LoadFromJSON(kv.second.cast<json::Object>());
          fmap.Set(kv.first.cast<ffi::String>(),
                   FunctionInfo(ffi::UnsafeInit{}, std::move(info_node)));
        }
        return fmap;
      });
}

}  // namespace runtime
}  // namespace tvm
