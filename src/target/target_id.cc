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
 * \file src/target/target_id.cc
 * \brief Target id registry
 */
#include <tvm/target/target_id.h>

#include "../node/attr_registry.h"
#include "../runtime/object_internal.h"

namespace tvm {

using TargetIdRegistry = AttrRegistry<TargetIdRegEntry, TargetId>;

TVM_DLL TargetIdRegEntry& TargetIdRegEntry::RegisterOrGet(const String& target_id_name) {
  return TargetIdRegistry::Global()->RegisterOrGet(target_id_name);
}

TVM_DLL void TargetIdRegEntry::UpdateAttr(const String& key, TVMRetValue value, int plevel) {
  TargetIdRegistry::Global()->UpdateAttr(key, id_, value, plevel);
}

TVM_DLL const AttrRegistryMapContainerMap<TargetId>& TargetId::GetAttrMapContainer(
    const String& attr_name) {
  return TargetIdRegistry::Global()->GetAttrMap(attr_name);
}

TVM_DLL const TargetId& TargetId::Get(const String& target_id_name) {
  const TargetIdRegEntry* reg = TargetIdRegistry::Global()->Get(target_id_name);
  CHECK(reg != nullptr) << "TargetId " << target_id_name << " is not registered";
  return reg->id_;
}

void VerifyTypeInfo(const ObjectRef& obj, const TargetIdNode::ValueTypeInfo& info) {
  CHECK(obj.defined()) << "Object is None";
  if (!runtime::ObjectInternal::DerivedFrom(obj.get(), info.type_index)) {
    LOG(FATAL) << "AttributeError: expect type " << info.type_key << " but get "
               << obj->GetTypeKey();
    throw;
  }
  if (info.type_index == ArrayNode::_type_index) {
    int i = 0;
    for (const auto& e : *obj.as<ArrayNode>()) {
      try {
        VerifyTypeInfo(e, *info.key);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "The i-th element of array failed type checking, where i = " << i
                   << ", and the error is:\n"
                   << e.what();
        throw;
      }
      ++i;
    }
  } else if (info.type_index == MapNode::_type_index) {
    for (const auto& kv : *obj.as<MapNode>()) {
      try {
        VerifyTypeInfo(kv.first, *info.key);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "The key of map failed type checking, where key = " << kv.first
                   << ", value = " << kv.second << ", and the error is:\n"
                   << e.what();
        throw;
      }
      try {
        VerifyTypeInfo(kv.second, *info.val);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "The value of map failed type checking, where key = " << kv.first
                   << ", value = " << kv.second << ", and the error is:\n"
                   << e.what();
        throw;
      }
    }
  }
}

TVM_DLL void TargetIdNode::ValidateSchema(const Map<String, ObjectRef>& config) const {
  for (const auto& kv : config) {
    auto it = key2vtype_.find(kv.first);
    if (it == key2vtype_.end()) {
      std::ostringstream os;
      os << "AttributeError: Invalid config option, cannot recognize \'" << kv.first
         << "\' candidates are:";
      bool is_first = true;
      for (const auto& kv : key2vtype_) {
        if (is_first) {
          is_first = false;
        } else {
          os << ',';
        }
        os << ' ' << kv.first;
      }
      LOG(FATAL) << os.str();
      throw;
    }
    const auto& obj = kv.second;
    const auto& info = it->second;
    try {
      VerifyTypeInfo(obj, info);
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "AttributeError: Schema validation failed for TargetId " << name
                 << ", details:\n"
                 << e.what() << "\n"
                 << "The given config is:\n"
                 << config;
      throw;
    }
  }
}

}  // namespace tvm
