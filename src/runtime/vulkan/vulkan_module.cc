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

#include "vulkan_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include "../file_utils.h"
#include "vulkan_wrapped_func.h"

namespace tvm {
namespace runtime {
namespace vulkan {

Module VulkanModuleCreate(std::unordered_map<std::string, SPIRVShader> smap,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<VulkanModuleNode>(smap, fmap, source);
  return Module(n);
}

Module VulkanModuleLoadFile(const std::string& file_name, const String& format) {
  std::string data;
  std::unordered_map<std::string, SPIRVShader> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  dmlc::MemoryStringStream fs(&data);
  dmlc::Stream* stream = &fs;
  uint32_t magic;
  stream->Read(&magic);
  ICHECK_EQ(magic, kVulkanModuleMagic) << "VulkanModule Magic mismatch";
  stream->Read(&smap);
  return VulkanModuleCreate(smap, fmap, "");
}

Module VulkanModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::unordered_map<std::string, SPIRVShader> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;

  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&smap);
  return VulkanModuleCreate(smap, fmap, "");
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_vulkan").set_body_typed(VulkanModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_vulkan").set_body_typed(VulkanModuleLoadBinary);

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
