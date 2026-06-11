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
 * \file metal_fallback_module.cc
 * \brief MetalFallbackModuleNode — codegen-time placeholder used when the
 *        Metal runtime is not linked.  Mirrors `MetalModuleNode`'s save/load
 *        format byte-for-byte; see one-way comment in `SaveToBytes` below.
 *        Always compiled (independent of USE_METAL); never registered as an
 *        FFI factory or loader.  Plain C++ — no Metal/Cocoa types.
 */
#include "metal_fallback_module.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <string>
#include <utility>

#include "../../support/bytes_io.h"

namespace tvm {
namespace target {

class MetalFallbackModuleNode : public ffi::ModuleObj {
 public:
  MetalFallbackModuleNode(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                          ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                          ffi::Map<ffi::String, ffi::String> source)
      : smap_(std::move(smap)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  // Mirror the real module's kind so consumers cannot distinguish at the
  // kind/api layer.  Saved bytes load back as a real MetalModuleNode on a
  // Metal-equipped (macOS, USE_METAL=ON) receiver.
  const char* kind() const final { return "metal"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(RuntimeError)
        << "Metal runtime is not linked into this build; cannot launch kernels. "
        << "Re-link with USE_METAL=ON (macOS) or load this module in a "
        << "Metal-equipped environment via tvm.runtime.load_module.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::Bytes SaveToBytes() const final {
    // NOTE: serialization format MUST remain byte-identical to
    // MetalModuleNode::SaveToBytes in src/runtime/metal/metal_module.mm
    // (the source of truth).  Both produce a kind="metal" artifact that
    // the loader (ffi.Module.load_from_bytes.metal, registered only when
    // USE_METAL=ON) deserializes.  If the real impl's format changes,
    // mirror the change here.  The dependency is one-way: this file
    // follows; metal_module.mm does not reference this file.
    //
    // 3 fields only — the source map is in-memory inspection material and
    // is NEVER serialized (matches upstream behavior for all backends).
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
      // Default: aggregated MSL source dump (key "metal").
      if (auto it = source_.find("metal"); it != source_.end()) {
        return (*it).second;
      }
      // Fallback: concatenate all kernel sources from the smap (handle
      // the fmt=="metal" text case).
      if (fmt_ == "metal") {
        std::string out;
        for (const auto& kv : smap_) {
          out.append(kv.second.data(), kv.second.size());
          out.push_back('\n');
        }
        return ffi::String(std::move(out));
      }
    }
    return ffi::String();
  }

 private:
  // Per-kernel payload: kernel-name -> bytes.  For fmt=="metal" each value
  // is MSL source bytes; for fmt=="metallib" each value is a compiled
  // metallib blob.  Multi-shader uniform Map<String, Bytes> (see
  // tasks/...-tvm-unify-device-module... "Per-kernel smap shape").
  ffi::Map<ffi::String, ffi::Bytes> smap_;
  // Format identifier ("metal" source / "metallib" compiled).
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
};

ffi::Module MetalFallbackModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                      ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                      ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<MetalFallbackModuleNode>(std::move(smap), std::move(fmt),
                                                     std::move(fmap), std::move(source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
