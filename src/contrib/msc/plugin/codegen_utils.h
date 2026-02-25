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
 * \file src/contrib/msc/plugin/codegen_utils.h
 * \brief Common utilities for print.
 */
#ifndef TVM_CONTRIB_MSC_PLUGIN_CODEGEN_UTILS_H_
#define TVM_CONTRIB_MSC_PLUGIN_CODEGEN_UTILS_H_

#include <tvm/ffi/extra/json.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace contrib {
namespace msc {

#define PLUGIN_CODEGEN_CONFIG_MEMBERS                 \
  bool need_convert{false};                           \
  bool with_runtime{false};                           \
  std::string project_name{"msc_plugin"};             \
  std::string cmake_version{"3.5"};                   \
  std::string install_dir;                            \
  std::vector<size_t> version{0, 0, 0};               \
  std::vector<std::string> includes;                  \
  std::vector<std::string> libs;                      \
  std::unordered_map<std::string, std::string> flags; \
  std::unordered_map<std::string, std::string> ops_info;

#define PLUGIN_CODEGEN_CONFIG_PARSE                                        \
  namespace json = ::tvm::ffi::json;                                       \
  if (auto it = obj.find(ffi::String("need_convert")); it != obj.end()) {  \
    need_convert = (*it).second.cast<bool>();                              \
  }                                                                        \
  if (auto it = obj.find(ffi::String("with_runtime")); it != obj.end()) {  \
    with_runtime = (*it).second.cast<bool>();                              \
  }                                                                        \
  if (auto it = obj.find(ffi::String("cmake_version")); it != obj.end()) { \
    cmake_version = std::string((*it).second.cast<ffi::String>());         \
  }                                                                        \
  if (auto it = obj.find(ffi::String("project_name")); it != obj.end()) {  \
    project_name = std::string((*it).second.cast<ffi::String>());          \
  }                                                                        \
  if (auto it = obj.find(ffi::String("install_dir")); it != obj.end()) {   \
    install_dir = std::string((*it).second.cast<ffi::String>());           \
  }                                                                        \
  if (auto it = obj.find(ffi::String("version")); it != obj.end()) {       \
    auto arr = (*it).second.cast<json::Array>();                           \
    version.clear();                                                       \
    version.reserve(arr.size());                                           \
    for (const auto& elem : arr) {                                         \
      version.push_back(static_cast<size_t>(elem.cast<int64_t>()));        \
    }                                                                      \
  }                                                                        \
  if (auto it = obj.find(ffi::String("includes")); it != obj.end()) {      \
    auto arr = (*it).second.cast<json::Array>();                           \
    includes.clear();                                                      \
    includes.reserve(arr.size());                                          \
    for (const auto& elem : arr) {                                         \
      includes.push_back(std::string(elem.cast<ffi::String>()));           \
    }                                                                      \
  }                                                                        \
  if (auto it = obj.find(ffi::String("libs")); it != obj.end()) {          \
    auto arr = (*it).second.cast<json::Array>();                           \
    libs.clear();                                                          \
    libs.reserve(arr.size());                                              \
    for (const auto& elem : arr) {                                         \
      libs.push_back(std::string(elem.cast<ffi::String>()));               \
    }                                                                      \
  }                                                                        \
  if (auto it = obj.find(ffi::String("flags")); it != obj.end()) {         \
    auto inner = (*it).second.cast<json::Object>();                        \
    flags.clear();                                                         \
    for (const auto& kv : inner) {                                         \
      flags[std::string(kv.first.cast<ffi::String>())] =                   \
          std::string(kv.second.cast<ffi::String>());                      \
    }                                                                      \
  }                                                                        \
  if (auto it = obj.find(ffi::String("ops_info")); it != obj.end()) {      \
    auto inner = (*it).second.cast<json::Object>();                        \
    ops_info.clear();                                                      \
    for (const auto& kv : inner) {                                         \
      ops_info[std::string(kv.first.cast<ffi::String>())] =                \
          std::string(kv.second.cast<ffi::String>());                      \
    }                                                                      \
  }

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_CODEGEN_UTILS_H_
