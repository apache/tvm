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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include "../file_utils.h"
#include "vulkan_wrapped_func.h"

namespace tvm {
namespace runtime {
namespace vulkan {

ffi::Module VulkanModuleCreate(std::unordered_map<std::string, SPIRVShader> smap,
                               std::unordered_map<std::string, FunctionInfo> fmap,
                               std::string source) {
  auto n = ffi::make_object<VulkanModuleNode>(smap, fmap, source);
  return ffi::Module(n);
}

ffi::Module VulkanModuleLoadFile(const std::string& file_name, const ffi::String& format) {
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

ffi::Module VulkanModuleLoadFromBytes(const ffi::Bytes& bytes) {
  dmlc::MemoryFixedSizeStream ms(const_cast<char*>(bytes.data()), bytes.size());
  dmlc::Stream* stream = &ms;
  std::unordered_map<std::string, SPIRVShader> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;

  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&smap);
  return VulkanModuleCreate(smap, fmap, "");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.Module.load_from_file.vulkan", VulkanModuleLoadFile)
      .def("ffi.Module.load_from_bytes.vulkan", VulkanModuleLoadFromBytes);
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
