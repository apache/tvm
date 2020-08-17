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
 * \file aocl_module.cc
 */
#include "aocl_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "aocl_common.h"

namespace tvm {
namespace runtime {

class AOCLModuleNode : public OpenCLModuleNode {
 public:
  explicit AOCLModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : OpenCLModuleNode(data, fmt, fmap, source) {}
  cl::OpenCLWorkspace* GetGlobalWorkspace() final;
};

cl::OpenCLWorkspace* AOCLModuleNode::GetGlobalWorkspace() { return cl::AOCLWorkspace::Global(); }

Module AOCLModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<AOCLModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

Module AOCLModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return AOCLModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_aocx").set_body_typed(AOCLModuleLoadFile);

}  // namespace runtime
}  // namespace tvm
