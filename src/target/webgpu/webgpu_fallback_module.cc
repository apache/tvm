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
 * \file webgpu_fallback_module.cc
 * \brief WebGPUFallbackModuleNode — the canonical C++-side module for
 *        WebGPU.  There is no native WebGPU runtime in the C++ tree; the
 *        real receiver is the wasm runtime in
 *        `web/emcc/webgpu_runtime.cc`, which registers only the
 *        load-from-bytes side of the FFI registry.  The fallback's
 *        `SaveToBytes` produces the bytes that the wasm-side
 *        `WebGPUModuleLoadFromBytes` consumes.
 *
 *        Always compiled (independent of any USE_X flag); never registers
 *        as an FFI factory or loader.
 */
#include "webgpu_fallback_module.h"

#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>

#include <sstream>
#include <string>
#include <utility>

#include "../../support/bytes_io.h"

namespace tvm {
namespace target {

class WebGPUFallbackModuleNode : public ffi::ModuleObj {
 public:
  WebGPUFallbackModuleNode(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                           ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                           ffi::Map<ffi::String, ffi::String> source)
      : smap_(std::move(smap)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  // The canonical C++-side WebGPU module — kind matches what the wasm
  // runtime registers (`web/emcc/webgpu_runtime.cc`).  Saved bytes load
  // back as a real `WebGPUModuleNode` on the wasm side.
  const char* kind() const final { return "webgpu"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(InternalError)
        << "WebGPUFallbackModule is not directly runnable on the C++ side; "
        << "export and run through tvmjs / the wasm WebGPU runtime.";
    return std::nullopt;
  }

  ffi::Bytes SaveToBytes() const final {
    // NOTE: serialization format MUST remain byte-identical to the
    // wasm-side reader `WebGPUModuleLoadFromBytes` in
    // web/emcc/webgpu_runtime.cc — which IS the receiver for WebGPU
    // (there is no native C++ WebGPU runtime).  The wasm reader expects
    // 2 fields [fmap][smap] (no fmt — WebGPU is single-format "wgsl"
    // today).  If this format changes, the wasm-side reader MUST be
    // updated in the same commit.
    //
    // Source map is in-memory inspection only and is NEVER serialized
    // (matches cross-backend rule).
    std::string result;
    support::BytesOutStream stream(&result);
    stream.Write(fmap_);
    stream.Write(smap_);
    return ffi::Bytes(std::move(result));
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == "func_info") {
      namespace json = ::tvm::ffi::json;
      json::Object obj;
      for (const auto& kv : fmap_) {
        obj.Set(kv.first, kv.second->SaveToJSON());
      }
      return std::string(json::Stringify(obj));
    }
    if (auto it = source_.find(format); it != source_.end()) {
      return (*it).second;
    }
    if (format.empty()) {
      if (auto it = source_.find("wgsl"); it != source_.end()) {
        return (*it).second;
      }
      // Fall back to concatenated WGSL kernel sources (legacy upstream
      // InspectSource behavior).
      std::ostringstream os;
      for (auto kv : smap_) {
        os.write(kv.second.data(), kv.second.size());
      }
      return os.str();
    }
    return ffi::String();
  }

 private:
  // Per-kernel payload: kernel-name -> WGSL source bytes.  Multi-shader
  // uniform Map<String, Bytes> (see plan "Per-kernel smap shape").
  ffi::Map<ffi::String, ffi::Bytes> smap_;
  // Format identifier ("wgsl" today — WebGPU is single-format).  Not
  // serialized in the on-disk bytes; the wasm receiver assumes "wgsl".
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  ffi::Map<ffi::String, ffi::String> source_;
};

ffi::Module WebGPUFallbackModuleCreate(ffi::Map<ffi::String, ffi::Bytes> smap, ffi::String fmt,
                                       ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                       ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<WebGPUFallbackModuleNode>(std::move(smap), std::move(fmt),
                                                      std::move(fmap), std::move(source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
