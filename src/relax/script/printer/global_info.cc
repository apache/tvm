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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tvm/target/target.h>

#include <optional>
#include <string>
#include <unordered_map>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

using GlobalInfoMap = ffi::Dict<ffi::String, ffi::List<GlobalInfo>>;

// Selectors are derived from the module's forward registry, never cached as
// object-to-string state. Definitions and standalone objects keep constructors.
ffi::Optional<ffi::String> GlobalInfoSelector(DocTranslatorObj* d, const GlobalInfo& info) {
  auto infos = d->GetOrCreateExtraState<GlobalInfoMap>("ir.global_info_map");
  if (auto devices = infos.Get("vdevice")) {
    std::unordered_map<std::string, size_t> indices;
    for (const GlobalInfo& entry : devices.value()) {
      if (auto device = entry.as<relax::VDevice>()) {
        std::string kind = device.value()->target->kind->name;
        size_t index = indices[kind]++;
        if (entry.same_as(info)) {
          return ffi::String(kind + ":" + std::to_string(index) + ":" +
                             std::string(device.value()->memory_scope));
        }
      }
    }
  }
  if (auto meshes = infos.Get("mesh")) {
    size_t index = 0;
    for (const GlobalInfo& entry : meshes.value()) {
      if (entry.same_as(info)) return ffi::String("mesh[" + std::to_string(index) + "]");
      ++index;
    }
  }
  return std::nullopt;
}

namespace {

ffi::Optional<ExprDoc> TranslateDummyGlobalInfo(DocTranslatorObj*, ffi::AnyView,
                                                const ffi::Object*) {
  return NamespaceDoc("relax")->Attr("dummy_global_info")->Call({});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::DummyGlobalInfoNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TranslateDummyGlobalInfo>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
