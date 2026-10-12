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

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/memory_scope.h>

#include <string_view>

namespace tvm {
namespace runtime {
namespace {
namespace refl = ffi::reflection;

void RegisterBuiltinScopes() {
  // Lookups and defaults can run in other translation units' static initializers.
  // Initialize the complete set once, before any lookup, without relying on order.
  static const bool initialized = [] {
    refl::ObjectDef<MemoryScopeNode>(refl::init(false));
    for (const char* name : {"global",
                             "nvshmem",
                             "ipc_memory",
                             "tmem",
                             "shared",
                             "shared.dyn",
                             "warp",
                             "local",
                             "wmma.matrix_a",
                             "wmma.matrix_b",
                             "wmma.accumulator",
                             "m16n8k8.matrixA",
                             "m16n8k8.matrixB",
                             "m16n8k8.matrixC",
                             "texture",
                             "amx.tmm",
                             "metal.simdgroup",
                             "metal.cooperative_tensor",
                             "trn.sbuf",
                             "trn.psum",
                             "global.vtcm",
                             "global.workspace"}) {
      refl::EnumDef<MemoryScopeNode>{name};
    }
    for (const char* name :
         {"global.ddr", "global.texture", "global.texture-1d", "global.texture-nchw",
          "global.texture-nhwc", "global.texture-weight"}) {
      refl::EnumDef<MemoryScopeNode>{name}.set_attr("tirx.storage_rewrite_merge", true);
    }
    return true;
  }();
  (void)initialized;
}
}  // namespace

MemoryScope MemoryScope::Get(const ffi::String& name) {
  RegisterBuiltinScopes();
  return ffi::EnumObj::_GetByStrIndex<MemoryScopeNode>(name).as_or_throw<MemoryScope>();
}

MemoryScope MemoryScope::Register(const ffi::String& name) {
  RegisterBuiltinScopes();
  TVM_FFI_CHECK(!name.empty(), ValueError) << "Memory scope name must not be empty";
  TVM_FFI_CHECK(std::string_view(name.data(), name.size()).find('\0') == std::string_view::npos,
                ValueError)
      << "Memory scope name must not contain a null character";
  refl::EnumDef<MemoryScopeNode>(name.c_str());
  return Get(name);
}

MemoryScope MemoryScope::Global() {
  static const MemoryScope scope = Get("global");
  return scope;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterBuiltinScopes();
  refl::TypeAttrDef<MemoryScopeNode>()
      .def(refl::type_attr::kDataToJson,
           [](const MemoryScopeNode* node) -> ffi::String { return node->_str_index; })
      .def(refl::type_attr::kDataFromJson,
           [](const ffi::String& name) { return MemoryScope::Get(name); });
  refl::GlobalDef().def("runtime.MemoryScopeRegister", &MemoryScope::Register);
}
}  // namespace runtime
}  // namespace tvm
