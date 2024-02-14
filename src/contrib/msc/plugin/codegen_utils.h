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

#define PLUGIN_CODEGEN_CONFIG_PARSE             \
  if (key == "need_convert") {                  \
    reader->Read(&need_convert);                \
  } else if (key == "with_runtime") {           \
    reader->Read(&with_runtime);                \
  } else if (key == "cmake_version") {          \
    reader->Read(&cmake_version);               \
  } else if (key == "project_name") {           \
    reader->Read(&project_name);                \
  } else if (key == "install_dir") {            \
    reader->Read(&install_dir);                 \
  } else if (key == "version") {                \
    reader->Read(&version);                     \
  } else if (key == "includes") {               \
    reader->Read(&includes);                    \
  } else if (key == "libs") {                   \
    reader->Read(&libs);                        \
  } else if (key == "flags") {                  \
    reader->Read(&flags);                       \
  } else if (key == "ops_info") {               \
    reader->Read(&ops_info);                    \
  } else {                                      \
    LOG(FATAL) << "Do not support key " << key; \
  }

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_CODEGEN_UTILS_H_
