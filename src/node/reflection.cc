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
 * Reflection utilities.
 * \file node/reflection.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/node.h>

namespace tvm {

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

// Expose to FFI APIs.
void NodeGetAttr(ffi::PackedArgs args, ffi::Any* ret) {
  Object* self = const_cast<Object*>(args[0].cast<const Object*>());
  String field_name = args[1].cast<String>();

  bool success;
  if (field_name == "type_key") {
    *ret = self->GetTypeKey();
    success = true;
  } else if (!self->IsInstance<DictAttrsNode>()) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(self->type_index());
    success = false;
    // use new reflection mechanism
    if (type_info->metadata != nullptr) {
      ffi::reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
        if (field_name.compare(field_info->name) == 0) {
          ffi::reflection::FieldGetter field_getter(field_info);
          *ret = field_getter(self);
          success = true;
        }
      });
    }
  } else {
    // specially handle dict attr
    DictAttrsNode* dnode = static_cast<DictAttrsNode*>(self);
    auto it = dnode->dict.find(field_name);
    if (it != dnode->dict.end()) {
      success = true;
      *ret = (*it).second;
    } else {
      success = false;
    }
  }
  if (!success) {
    TVM_FFI_THROW(AttributeError) << self->GetTypeKey() << " object has no attribute `"
                                  << field_name << "`";
  }
}

void NodeListAttrNames(ffi::PackedArgs args, ffi::Any* ret) {
  Object* self = const_cast<Object*>(args[0].cast<const Object*>());

  std::vector<String> names;
  if (!self->IsInstance<DictAttrsNode>()) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(self->type_index());
    if (type_info->metadata != nullptr) {
      // use new reflection mechanism
      ffi::reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
        names.push_back(std::string(field_info->name.data, field_info->name.size));
      });
    }
  } else {
    // specially handle dict attr
    DictAttrsNode* dnode = static_cast<DictAttrsNode*>(self);
    for (const auto& kv : dnode->dict) {
      names.push_back(kv.first);
    }
  }

  *ret = ffi::Function::FromPacked([names](ffi::PackedArgs args, ffi::Any* rv) {
    int64_t i = args[0].cast<int64_t>();
    if (i == -1) {
      *rv = static_cast<int64_t>(names.size());
    } else {
      *rv = names[i];
    }
  });
}

// API function to make node.
// args format:
//   key1, value1, ..., key_n, value_n
void MakeNode(const ffi::PackedArgs& args, ffi::Any* rv) {
  // TODO(tvm-team): consider further simplify by removing DictAttrsNode special handling
  String type_key = args[0].cast<String>();
  int32_t type_index;
  TVMFFIByteArray type_key_array = TVMFFIByteArray{type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  if (type_index == DictAttrsNode::RuntimeTypeIndex()) {
    ObjectPtr<DictAttrsNode> attrs = make_object<DictAttrsNode>();
    attrs->InitByPackedArgs(args.Slice(1), false);
    *rv = ObjectRef(attrs);
  } else {
    auto fcreate_object = ffi::Function::GetGlobalRequired("ffi.MakeObjectFromPackedArgs");
    fcreate_object.CallPacked(args, rv);
  }
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("node.NodeGetAttr", NodeGetAttr)
      .def_packed("node.NodeListAttrNames", NodeListAttrNames)
      .def_packed("node.MakeNode", MakeNode);
});

}  // namespace tvm
