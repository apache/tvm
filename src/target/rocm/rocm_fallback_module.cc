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
 * \file rocm_fallback_module.cc
 * \brief ROCmFallbackModuleNode — codegen-time placeholder used when the ROCm
 *        runtime is not linked.  Mirrors `ROCMModuleNode`'s save/load format
 *        byte-for-byte; see one-way comment in `SaveToBytes` below.
 *        Always compiled (independent of USE_ROCM); never registered as an
 *        FFI factory or loader.
 */
#include "rocm_fallback_module.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <string>
#include <utility>

#include "../../support/bytes_io.h"

namespace tvm {
namespace target {

class ROCmFallbackModuleNode : public ffi::ModuleObj {
 public:
  ROCmFallbackModuleNode(ffi::Bytes code, ffi::String fmt,
                         ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                         ffi::Map<ffi::String, ffi::String> source)
      : code_(std::move(code)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  // Mirror the real module's kind so consumers cannot distinguish at the
  // kind/api layer.  Saved bytes load back as a real ROCMModuleNode on a
  // ROCm-equipped receiver.
  const char* kind() const final { return "hip"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(RuntimeError)
        << "ROCm runtime is not linked into this build; cannot launch kernels. "
        << "Re-link with USE_ROCM=ON or load this module in a ROCm-equipped "
        << "environment via tvm.runtime.load_module.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::Bytes SaveToBytes() const final {
    // NOTE: serialization format MUST remain byte-identical to
    // ROCMModuleNode::SaveToBytes in src/runtime/rocm/rocm_module.cc (the
    // source of truth).  Both produce a kind="hip" artifact that the loader
    // (ffi.Module.load_from_bytes.hsaco, registered only when USE_ROCM=ON)
    // deserializes.  If the real impl's format changes, mirror the change
    // here.  The dependency is one-way: this file follows; rocm_module.cc
    // does not reference this file.
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty() || format == "llvm") {
      // Backward-compat: legacy ROCMModuleNode returned `hip_source_` for both
      // empty-format and "llvm".  Map both to source["hip"] (the LLVM IR text
      // emitted by the AMDGPU backend; populated by codegen_amdgpu.cc).
      if (auto it = source_.find("hip"); it != source_.end()) {
        return (*it).second;
      }
    }
    if (format == "asm") {
      if (auto it = source_.find("asm"); it != source_.end()) {
        return (*it).second;
      }
    }
    return ffi::String();
  }

 private:
  // Whatever the codegen produced (compiled hsaco bytes — ROCm has no
  // source-JIT path, so fmt is always "hsaco").
  ffi::Bytes code_;
  // Format identifier ("hsaco").
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
};

ffi::Module ROCmFallbackModuleCreate(ffi::Bytes code, ffi::String fmt,
                                     ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                     ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<ROCmFallbackModuleNode>(std::move(code), std::move(fmt),
                                                    std::move(fmap), std::move(source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
