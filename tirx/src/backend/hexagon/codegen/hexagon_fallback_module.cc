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
 * \file hexagon_fallback_module.cc
 * \brief HexagonFallbackModuleObj — codegen-time placeholder used when the
 *        Hexagon runtime is not linked.  Preserves the serialized module
 *        format for external runtimes; never registers an FFI factory or loader.
 */
#include "hexagon_fallback_module.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/support/bytes_io.h>

#include <string>
#include <utility>

namespace tvm {
namespace target {

class HexagonFallbackModuleObj : public ffi::ModuleObj {
 public:
  HexagonFallbackModuleObj(ffi::Bytes code, ffi::String fmt,
                           ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                           HexagonSourceMap source)
      : code_(std::move(code)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  // Mirror the real module's kind so consumers cannot distinguish at the
  // kind/api layer.  Saved bytes load back as a real HexagonModuleObj on
  // a Hexagon-equipped receiver.
  const char* kind() const final { return "hexagon"; }

  int GetPropertyMask() const final { return ffi::Module::kBinarySerializable; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(RuntimeError)
        << "Hexagon runtime is not linked into this build; cannot launch "
        << "kernels.  Load this module with an external Hexagon runtime in a "
        << "Hexagon-equipped environment via tvm.runtime.load_module.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::Bytes SaveToBytes() const final {
    // Preserve the Hexagon module wire format for external runtime loaders:
    // format, function metadata, then shared-library bytes.
    // 3 fields only — the source map is in-memory inspection material and
    // is NEVER serialized (matches upstream behavior for all backends).
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
      const auto& v = (*it).second;
      if (auto s = v.template as<ffi::String>()) {
        return *s;
      }
      if (auto b = v.template as<ffi::Bytes>()) {
        return ffi::String(b->data(), b->size());
      }
    }
    // Backward compat: legacy HexagonModuleObj treated "asm" as alias of
    // "s".  Keep both keys discoverable.
    if (format == "asm") {
      if (auto it = source_.find("s"); it != source_.end()) {
        const auto& v = (*it).second;
        if (auto s = v.template as<ffi::String>()) return *s;
      }
    }
    return ffi::String();
  }

 private:
  // Linked Hexagon shared-library bytes (fmt="so").
  ffi::Bytes code_;
  // Format identifier ("so").
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  // Variant lets each upstream call site pass its blob in its natural
  // form: text ("asm"/"s"/"ll") or binary ("obj"/"bc").
  HexagonSourceMap source_;
};

ffi::Module HexagonFallbackModuleCreate(ffi::Bytes code, ffi::String fmt,
                                        ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                                        HexagonSourceMap source) {
  auto n = ffi::make_object<HexagonFallbackModuleObj>(std::move(code), std::move(fmt),
                                                      std::move(fmap), std::move(source));
  return ffi::Module(n);
}

}  // namespace target
}  // namespace tvm
