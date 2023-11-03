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
 * \file sdaccel_module.cc
 */
#include "sdaccel_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "sdaccel_common.h"

namespace tvm {
namespace runtime {

class SDAccelModuleNode : public OpenCLModuleNode {
 public:
  explicit SDAccelModuleNode(std::string data, std::string fmt,
                             std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : OpenCLModuleNode(data, fmt, fmap, source) {}
  cl::OpenCLWorkspace* GetGlobalWorkspace() final;
};

cl::OpenCLWorkspace* SDAccelModuleNode::GetGlobalWorkspace() {
  return cl::SDAccelWorkspace::Global();
}

Module SDAccelModuleCreate(std::string data, std::string fmt,
                           std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<SDAccelModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

Module SDAccelModuleLoadFile(const std::string& file_name, const String& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return SDAccelModuleCreate(data, fmt, fmap, std::string());
}

Module SDAccelModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return SDAccelModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_xclbin").set_body_typed(SDAccelModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_awsxclbin").set_body_typed(SDAccelModuleLoadFile);
}  // namespace runtime
}  // namespace tvm
