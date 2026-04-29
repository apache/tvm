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
 * \file opencl_fallback_module.cc
 * \brief OpenCLFallbackModuleNode — codegen-time placeholder used when
 *        the OpenCL runtime is not linked.  Mirrors `OpenCLModuleNode`'s
 *        save/load format byte-for-byte; see one-way comment in
 *        `SaveToBytes` below.  Always compiled (independent of
 *        USE_OPENCL); never registered as an FFI factory or loader.
 *        Plain C++ — no OpenCL-API types.
 */
#include "opencl_fallback_module.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <string>
#include <utility>

#include "../../support/bytes_io.h"

namespace tvm {
namespace target {

class OpenCLFallbackModuleNode : public ffi::ModuleObj {
 public:
  OpenCLFallbackModuleNode(ffi::Bytes code, ffi::String fmt,
                           ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                           ffi::Map<ffi::String, ffi::String> source)
      : code_(std::move(code)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  // Mirror the real module's kind ("opencl") so consumers cannot
  // distinguish at the kind/api layer.  Saved bytes load back as a real
  // OpenCLModuleNode on a USE_OPENCL=ON receiver.
  const char* kind() const final { return "opencl"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(RuntimeError)
        << "OpenCL runtime is not linked into this build; cannot launch kernels. "
        << "Re-link with USE_OPENCL=ON or load this module in an OpenCL-equipped "
        << "environment via tvm.runtime.load_module.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::Bytes SaveToBytes() const final {
    // NOTE: serialization format MUST remain byte-identical to
    // OpenCLModuleNode::SaveToBytes in src/runtime/opencl/opencl_module.cc
    // (the source of truth).  Both produce a kind="opencl" artifact
    // that the loader (ffi.Module.load_from_bytes.opencl, registered
    // only when USE_OPENCL=ON) deserializes.  If the real impl's
    // format changes, mirror the change here.  The dependency is
    // one-way: this file follows; opencl_module.cc does not reference
    // this file.
    //
    // 3 fields only — the source map is in-memory inspection material
    // and is NEVER serialized (matches upstream behavior for all
    // backends).
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      // Default: aggregated OpenCL C source dump (key "cl").
      if (auto it = source_.find("cl"); it != source_.end()) {
        return (*it).second;
      }
      // Fallback: if fmt=="cl" the code is itself the OpenCL source.
      if (fmt_ == "cl") {
        return ffi::String(code_.data(), code_.size());
      }
    }
    return ffi::String();
  }

 private:
  // Single-binary payload: for fmt=="cl" the bytes are the OpenCL C
  // source; for fmt=="xclbin"/"awsxclbin"/"aocx" the bytes are a
  // pre-compiled OpenCL binary.  Single-path uniform Bytes (vs
  // multi-shader Map<String, Bytes> on Vulkan/Metal/WebGPU).
  ffi::Bytes code_;
  // Format identifier ("cl" / "xclbin" / "awsxclbin" / "aocx").
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
};

ffi::Module OpenCLFallbackModuleCreate(ffi::Bytes code, ffi::String fmt,
                                       ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                       ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<OpenCLFallbackModuleNode>(std::move(code), std::move(fmt),
                                                      std::move(fmap), std::move(source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
