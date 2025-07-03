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
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/node.h>
#include <tvm/node/reflection.h>

namespace tvm {

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

ffi::Any ReflectionVTable::GetAttr(Object* self, const String& field_name) const {
  ffi::Any ret;
  bool success;
  if (field_name == "type_key") {
    ret = self->GetTypeKey();
    success = true;
  } else if (!self->IsInstance<DictAttrsNode>()) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(self->type_index());
    success = false;
    // use new reflection mechanism
    if (type_info->extra_info != nullptr) {
      ffi::reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
        if (field_name.compare(field_info->name) == 0) {
          ffi::reflection::FieldGetter field_getter(field_info);
          ret = field_getter(self);
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
      ret = (*it).second;
    } else {
      success = false;
    }
  }
  if (!success) {
    LOG(FATAL) << "AttributeError: " << self->GetTypeKey() << " object has no attributed "
               << field_name;
  }
  return ret;
}

std::vector<std::string> ReflectionVTable::ListAttrNames(Object* self) const {
  std::vector<std::string> names;

  if (!self->IsInstance<DictAttrsNode>()) {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(self->type_index());
    if (type_info->extra_info != nullptr) {
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
  return names;
}

ReflectionVTable* ReflectionVTable::Global() {
  static ReflectionVTable inst;
  return &inst;
}

ObjectPtr<Object> ReflectionVTable::CreateInitObject(const std::string& type_key,
                                                     const std::string& repr_bytes) const {
  int32_t tindex;
  TVMFFIByteArray type_key_arr{type_key.data(), type_key.length()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_arr, &tindex));
  if (static_cast<size_t>(tindex) >= fcreate_.size() || fcreate_[tindex] == nullptr) {
    LOG(FATAL) << "TypeError: " << type_key << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  return fcreate_[tindex](repr_bytes);
}

ObjectRef ReflectionVTable::CreateObject(const std::string& type_key,
                                         const ffi::PackedArgs& kwargs) {
  int32_t type_index;
  TVMFFIByteArray type_key_array = TVMFFIByteArray{type_key.data(), type_key.size()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_array, &type_index));
  if (type_index == DictAttrsNode::RuntimeTypeIndex()) {
    ObjectPtr<Object> n = this->CreateInitObject(type_key);
    static_cast<BaseAttrsNode*>(n.get())->InitByPackedArgs(kwargs);
    return ObjectRef(n);
  }
  // TODO(tvm-team): remove this once all objects are transitioned to the new reflection
  auto fcreate_object = ffi::Function::GetGlobalRequired("ffi.MakeObjectFromPackedArgs");
  std::vector<AnyView> packed_args(kwargs.size() + 1);
  packed_args[0] = type_index;
  for (int i = 0; i < kwargs.size(); i++) {
    packed_args[i + 1] = kwargs[i];
  }
  ffi::Any rv;
  fcreate_object.CallPacked(ffi::PackedArgs(packed_args.data(), packed_args.size()), &rv);
  return rv.cast<ObjectRef>();
}

ObjectRef ReflectionVTable::CreateObject(const std::string& type_key,
                                         const Map<String, Any>& kwargs) {
  // Redirect to the ffi::PackedArgs version
  // It is not the most efficient way, but CreateObject is not meant to be used
  // in a fast code-path and is mainly reserved as a flexible API for frontends.
  std::vector<AnyView> packed_args(kwargs.size() * 2);
  int index = 0;

  for (const auto& kv : *static_cast<const ffi::MapObj*>(kwargs.get())) {
    packed_args[index] = kv.first.cast<String>().c_str();
    packed_args[index + 1] = kv.second;
    index += 2;
  }

  return CreateObject(type_key, ffi::PackedArgs(packed_args.data(), packed_args.size()));
}

// Expose to FFI APIs.
void NodeGetAttr(ffi::PackedArgs args, ffi::Any* ret) {
  Object* self = const_cast<Object*>(args[0].cast<const Object*>());
  *ret = ReflectionVTable::Global()->GetAttr(self, args[1].cast<std::string>());
}

void NodeListAttrNames(ffi::PackedArgs args, ffi::Any* ret) {
  Object* self = const_cast<Object*>(args[0].cast<const Object*>());

  auto names =
      std::make_shared<std::vector<std::string>>(ReflectionVTable::Global()->ListAttrNames(self));

  *ret = ffi::Function([names](ffi::PackedArgs args, ffi::Any* rv) {
    int64_t i = args[0].cast<int64_t>();
    if (i == -1) {
      *rv = static_cast<int64_t>(names->size());
    } else {
      *rv = (*names)[i];
    }
  });
}

// API function to make node.
// args format:
//   key1, value1, ..., key_n, value_n
void MakeNode(const ffi::PackedArgs& args, ffi::Any* rv) {
  *rv = ReflectionVTable::Global()->CreateObject(args[0].cast<std::string>(), args.Slice(1));
}

TVM_FFI_REGISTER_GLOBAL("node.NodeGetAttr").set_body_packed(NodeGetAttr);

TVM_FFI_REGISTER_GLOBAL("node.NodeListAttrNames").set_body_packed(NodeListAttrNames);

TVM_FFI_REGISTER_GLOBAL("node.MakeNode").set_body_packed(MakeNode);

Optional<String> GetAttrKeyByAddress(const Object* object, const void* attr_address) {
  const TVMFFITypeInfo* tinfo = TVMFFIGetTypeInfo(object->type_index());
  if (tinfo->extra_info != nullptr) {
    Optional<String> result;
    // visit fields with the new reflection
    ffi::reflection::ForEachFieldInfoWithEarlyStop(tinfo, [&](const TVMFFIFieldInfo* field_info) {
      Any field_value = ffi::reflection::FieldGetter(field_info)(object);
      const void* field_addr = reinterpret_cast<const char*>(object) + field_info->offset;
      if (field_addr == attr_address) {
        result = String(field_info->name);
        return true;
      }
      return false;
    });
    return result;
  }
  return std::nullopt;
}

}  // namespace tvm
