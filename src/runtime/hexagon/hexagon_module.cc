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
 * \file hexagon_module.cc
 * \brief HexagonModuleNode — runtime-side, plugin-only.  Reachable from C++
 *        only through the FFI registry keys "ffi.Module.create.hexagon" and
 *        "ffi.Module.load_from_bytes.hexagon".  No exported header —
 *        codegen-side construction goes through
 *        src/target/hexagon/hexagon_fallback_module.h.
 *
 *        This carrier holds the linked Hexagon `.so` blob in memory.  The
 *        existing HexagonModuleNode does not perform `dlopen` (Hexagon
 *        execution happens via a separate RPC-based runtime path); kernel
 *        launches via `GetFunction` are unimplemented.  The module exists
 *        primarily for cross-compile shipment (SaveToBytes) and source
 *        inspection.
 */
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <utility>
#include <vector>

#include "../../support/bytes_io.h"
#include "../metadata.h"

namespace tvm {
namespace runtime {

using HexagonSourceMap = ffi::Map<ffi::String, ffi::Variant<ffi::String, ffi::Bytes>>;

class HexagonModuleNode : public ffi::ModuleObj {
 public:
  HexagonModuleNode(ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
                    HexagonSourceMap source)
      : code_(std::move(code)),
        fmt_(std::move(fmt)),
        fmap_(std::move(fmap)),
        source_(std::move(source)) {}

  const char* kind() const final { return "hexagon"; }

  int GetPropertyMask() const final {
    return ffi::Module::kBinarySerializable | ffi::Module::kRunnable;
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String& name) final {
    TVM_FFI_THROW(InternalError) << "HexagonModuleNode::GetFunction is not implemented.";
    TVM_FFI_UNREACHABLE();
  }

  ffi::String InspectSource(const ffi::String& format) const final {
    if (format == fmt_) {
      return ffi::String(code_.data(), code_.size());
    }
    if (auto it = source_.find(format); it != source_.end()) {
      const auto& v = (*it).second;
      if (auto s = v.template as<ffi::String>()) return *s;
      if (auto b = v.template as<ffi::Bytes>()) return ffi::String(b->data(), b->size());
    }
    if (format == "asm") {
      if (auto it = source_.find("s"); it != source_.end()) {
        const auto& v = (*it).second;
        if (auto s = v.template as<ffi::String>()) return *s;
      }
    }
    return ffi::String();
  }

  ffi::Bytes SaveToBytes() const final {
    // Format: [fmt][fmap][code].  Source map is in-memory inspection only
    // and is NEVER serialized — it is lost on save/load round-trip
    // (matches upstream behavior; the receiver rebuilds source from code
    // bytes if possible).  HexagonFallbackModuleNode::SaveToBytes (in
    // src/target/hexagon/hexagon_fallback_module.cc) MUST mirror this
    // format byte-for-byte; see one-way comment there.
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(fmt_);
    stream.Write(fmap_);
    stream.Write(code_);
    return ffi::Bytes(std::move(buffer));
  }

 private:
  // Linked Hexagon shared-library bytes (fmt="so").
  ffi::Bytes code_;
  // Format identifier ("so").
  ffi::String fmt_;
  // function information table.
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  // In-memory source map for InspectSource — never serialized.
  // Variant lets each blob (text "asm"/"s"/"ll" or binary "obj"/"bc") be
  // passed in its natural form.
  HexagonSourceMap source_;
};

static ffi::Module HexagonModuleCreateImpl(ffi::Bytes code, ffi::String fmt,
                                           ffi::Map<ffi::String, FunctionInfo> fmap,
                                           HexagonSourceMap source) {
  auto n = ffi::make_object<HexagonModuleNode>(std::move(code), std::move(fmt), std::move(fmap),
                                               std::move(source));
  return ffi::Module(n);
}

static ffi::Module HexagonModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  // Source map is not serialized — it is lost on save/load round-trip.
  return HexagonModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                 HexagonSourceMap());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.hexagon" — codegen-time Hexagon module factory.
  // Used by src/target/hexagon/hexagon_fallback_module.h:HexagonModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.hexagon" — disk loader.  Only
  // this (real) module registers a loader; the fallback is codegen-only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.hexagon", HexagonModuleLoadFromBytes)
      .def("ffi.Module.create.hexagon",
           [](ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              HexagonSourceMap source) {
             return HexagonModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                            std::move(source));
           });
}

}  // namespace runtime
}  // namespace tvm
