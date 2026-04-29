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
 * \file vulkan_fallback_module.cc
 * \brief VulkanFallbackModuleNode — codegen-time placeholder used when the
 *        Vulkan runtime is not linked.  Mirrors `VulkanModuleNode`'s
 *        save/load format byte-for-byte; see one-way comment in
 *        `SaveToBytes` below.  Always compiled (independent of
 *        USE_VULKAN); never registered as an FFI factory or loader.
 *        Plain C++ — no Vulkan API types, no SPIR-V tooling deps.
 */
#include "vulkan_fallback_module.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <string>
#include <utility>

#include "../../support/bytes_io.h"

namespace tvm {
namespace target {

class VulkanFallbackModuleNode : public ffi::ModuleObj {
 public:
  VulkanFallbackModuleNode(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                           ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                           ffi::Map<ffi::String, ffi::String> source)
      : smap_(std::move(smap)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  // Mirror the real module's kind so consumers cannot distinguish at the
  // kind/api layer.  Saved bytes load back as a real VulkanModuleNode on
  // a Vulkan-equipped (USE_VULKAN=ON) receiver.
  const char* kind() const final { return "vulkan"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(RuntimeError)
        << "Vulkan runtime is not linked into this build; cannot launch kernels. "
        << "Re-link with USE_VULKAN=ON or load this module in a Vulkan-equipped "
        << "environment via tvm.runtime.load_module.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::Bytes SaveToBytes() const final {
    // NOTE: serialization format MUST remain byte-identical to
    // VulkanModuleNode::SaveToBytes in src/runtime/vulkan/vulkan_module.cc
    // (the source of truth).  Both produce a kind="vulkan" artifact that
    // the loader (ffi.Module.load_from_bytes.vulkan, registered only when
    // USE_VULKAN=ON) deserializes.  If the real impl's format changes,
    // mirror the change here.  The dependency is one-way: this file
    // follows; vulkan_module.cc does not reference this file.
    //
    // 3 fields only — the source map is in-memory inspection material and
    // is NEVER serialized (matches upstream behavior for all backends).
    // Each value in `smap_` is a self-packed SPIRVShader (flag + data
    // vector); see src/runtime/vulkan/spirv_shader.h.
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(smap_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      // Default: aggregated SPIR-V text dump (key "spv").
      if (auto it = source_.find("spv"); it != source_.end()) {
        return (*it).second;
      }
    }
    return ffi::String();
  }

 private:
  // Per-kernel payload: kernel-name -> bytes.  Each value is a
  // serialized SPIRVShader (flag + uint32_t data segment); the runtime
  // (USE_VULKAN=ON) deserializes via the inverse helper in
  // src/runtime/vulkan/vulkan_module.cc.  Multi-shader uniform
  // Map<String, Bytes> across all multi-shader backends.
  ffi::Map<ffi::String, ffi::Bytes> smap_;
  // Format identifier — always "vulkan" today.
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
};

ffi::Module VulkanFallbackModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                       ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                       ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<VulkanFallbackModuleNode>(std::move(smap), std::move(fmt),
                                                      std::move(fmap), std::move(source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
