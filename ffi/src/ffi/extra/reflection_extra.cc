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
/*
 * \file src/ffi/extra/reflection_extra.cc
 *
 * \brief Extra reflection registrations. *
 */
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace ffi {
namespace reflection {

void MakeObjectFromPackedArgs(ffi::PackedArgs args, Any* ret) {
  int32_t type_index;
  if (auto opt_type_index = args[0].try_cast<int32_t>()) {
    type_index = *opt_type_index;
  } else {
    String type_key = args[0].cast<String>();
    TVMFFIByteArray type_key_array = TVMFFIByteArray{type_key.data(), type_key.size()};
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  }

  TVM_FFI_ICHECK(args.size() % 2 == 1);
  const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);

  if (type_info->metadata == nullptr || type_info->metadata->creator == nullptr) {
    TVM_FFI_THROW(RuntimeError) << "Type `" << TypeIndexToTypeKey(type_index)
                                << "` does not support reflection creation";
  }
  TVMFFIObjectHandle handle;
  TVM_FFI_CHECK_SAFE_CALL(type_info->metadata->creator(&handle));
  ObjectPtr<Object> ptr =
      details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<TVMFFIObject*>(handle));

  std::vector<String> keys;
  std::vector<bool> keys_found;

  for (int i = 1; i < args.size(); i += 2) {
    keys.push_back(args[i].cast<String>());
  }
  keys_found.resize(keys.size(), false);

  auto search_field = [&](const TVMFFIByteArray& field_name) {
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys_found[i]) continue;
      if (keys[i].compare(field_name) == 0) {
        return i;
      }
    }
    return keys.size();
  };

  auto update_fields = [&](const TVMFFITypeInfo* tinfo) {
    for (int i = 0; i < tinfo->num_fields; ++i) {
      const TVMFFIFieldInfo* field_info = tinfo->fields + i;
      size_t arg_index = search_field(field_info->name);
      void* field_addr = reinterpret_cast<char*>(ptr.get()) + field_info->offset;
      if (arg_index < keys.size()) {
        AnyView field_value = args[arg_index * 2 + 2];
        field_info->setter(field_addr, reinterpret_cast<const TVMFFIAny*>(&field_value));
        keys_found[arg_index] = true;
      } else if (field_info->flags & kTVMFFIFieldFlagBitMaskHasDefault) {
        field_info->setter(field_addr, &(field_info->default_value));
      } else {
        TVM_FFI_THROW(TypeError) << "Required field `"
                                 << String(field_info->name.data, field_info->name.size)
                                 << "` not set in type `" << TypeIndexToTypeKey(type_index) << "`";
      }
    }
  };

  // iterate through acenstors in parent to child order
  // skip the first one since it is always the root object
  for (int i = 1; i < type_info->type_depth; ++i) {
    update_fields(type_info->type_acenstors[i]);
  }
  update_fields(type_info);

  for (size_t i = 0; i < keys.size(); ++i) {
    if (!keys_found[i]) {
      TVM_FFI_THROW(TypeError) << "Type `" << TypeIndexToTypeKey(type_index)
                               << "` does not have field `" << keys[i] << "`";
    }
  }
  *ret = ObjectRef(ptr);
}

inline void AccessStepRegisterReflection() {
  // register access step reflection here since it is only needed for bindings
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AccessStepObj>()
      .def_ro("kind", &AccessStepObj::kind)
      .def_ro("key", &AccessStepObj::key);
}

inline void AccessPathRegisterReflection() {
  // register access path reflection here since it is only needed for bindings
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AccessPathObj>()
      .def_ro("parent", &AccessPathObj::parent)
      .def_ro("step", &AccessPathObj::step)
      .def_ro("depth", &AccessPathObj::depth)
      .def_static("_root", &AccessPath::Root)
      .def("_extend", &AccessPathObj::Extend)
      .def("_attr", &AccessPathObj::Attr)
      .def("_array_item", &AccessPathObj::ArrayItem)
      .def("_map_item", &AccessPathObj::MapItem)
      .def("_attr_missing", &AccessPathObj::AttrMissing)
      .def("_array_item_missing", &AccessPathObj::ArrayItemMissing)
      .def("_map_item_missing", &AccessPathObj::MapItemMissing)
      .def("_is_prefix_of", &AccessPathObj::IsPrefixOf)
      .def("_to_steps", &AccessPathObj::ToSteps)
      .def("_path_equal",
           [](const AccessPath& self, const AccessPath& other) { return self->PathEqual(other); });
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  AccessStepRegisterReflection();
  AccessPathRegisterReflection();
  refl::GlobalDef().def_packed("ffi.MakeObjectFromPackedArgs", MakeObjectFromPackedArgs);
});

}  // namespace reflection
}  // namespace ffi
}  // namespace tvm
