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

TargetIdRegEntry& TargetIdRegEntry::RegisterOrGet(const String& target_id_name) {
  return TargetIdRegistry::Global()->RegisterOrGet(target_id_name);
}

void TargetIdRegEntry::UpdateAttr(const String& key, TVMRetValue value, int plevel) {
  TargetIdRegistry::Global()->UpdateAttr(key, id_, value, plevel);
}

const AttrRegistryMapContainerMap<TargetId>& TargetId::GetAttrMapContainer(
    const String& attr_name) {
  return TargetIdRegistry::Global()->GetAttrMap(attr_name);
}

const TargetId& TargetId::Get(const String& target_id_name) {
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

void TargetIdNode::ValidateSchema(const Map<String, ObjectRef>& config) const {
  const String kTargetId = "id";
  for (const auto& kv : config) {
    const String& name = kv.first;
    const ObjectRef& obj = kv.second;
    if (name == kTargetId) {
      CHECK(obj->IsInstance<StringObj>())
          << "AttributeError: \"id\" is not a string, but its type is " << obj->GetTypeKey();
      CHECK(Downcast<String>(obj) == this->name)
          << "AttributeError: \"id\" = " << obj << " is inconsistent with TargetId " << this->name;
      continue;
    }
    auto it = key2vtype_.find(name);
    if (it == key2vtype_.end()) {
      std::ostringstream os;
      os << "AttributeError: Invalid config option, cannot recognize \'" << name
         << "\'. Candidates are:";
      for (const auto& kv : key2vtype_) {
        os << "\n  " << kv.first;
      }
      LOG(FATAL) << os.str();
      throw;
    }
    const auto& info = it->second;
    try {
      VerifyTypeInfo(obj, info);
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "AttributeError: Schema validation failed for TargetId " << this->name
                 << ", details:\n"
                 << e.what() << "\n"
                 << "The config is:\n"
                 << config;
      throw;
    }
  }
}

inline String GetId(const Map<String, ObjectRef>& target, const char* name) {
  const String kTargetId = "id";
  CHECK(target.count(kTargetId)) << "AttributeError: \"id\" does not exist in " << name << "\n"
                                 << name << " = " << target;
  const ObjectRef& obj = target[kTargetId];
  CHECK(obj->IsInstance<StringObj>()) << "AttributeError: \"id\" is not a string in " << name
                                      << ", but its type is " << obj->GetTypeKey() << "\n"
                                      << name << " = " << target;
  return Downcast<String>(obj);
}

void TargetValidateSchema(const Map<String, ObjectRef>& config) {
  try {
    const String kTargetHost = "target_host";
    Map<String, ObjectRef> target = config;
    Map<String, ObjectRef> target_host;
    String target_id = GetId(target, "target");
    String target_host_id;
    if (config.count(kTargetHost)) {
      target.erase(kTargetHost);
      target_host = Downcast<Map<String, ObjectRef>>(config[kTargetHost]);
      target_host_id = GetId(target_host, "target_host");
    }
    TargetId::Get(target_id)->ValidateSchema(target);
    if (!target_host.empty()) {
      TargetId::Get(target_host_id)->ValidateSchema(target_host);
    }
  } catch (const tvm::Error& e) {
    LOG(INFO) << e.what();
    throw e;
  }
}

}  // namespace tvm
