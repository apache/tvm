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
 * \file src/runtime/object.cc
 * \brief Object type management system.
 */
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "object_internal.h"
#include "runtime_base.h"

namespace tvm {
namespace runtime {

TVM_REGISTER_GLOBAL("runtime.ObjectPtrHash").set_body_typed([](ObjectRef obj) {
  return static_cast<int64_t>(ObjectPtrHash()(obj));
});

}  // namespace runtime
}  // namespace tvm

int TVMObjectGetTypeIndex(TVMObjectHandle obj, unsigned* out_tindex) {
  API_BEGIN();
  ICHECK(obj != nullptr);
  out_tindex[0] = static_cast<tvm::runtime::Object*>(obj)->type_index();
  API_END();
}

int TVMObjectRetain(TVMObjectHandle obj) {
  API_BEGIN();
  tvm::runtime::ObjectInternal::ObjectRetain(obj);
  API_END();
}

int TVMObjectFree(TVMObjectHandle obj) {
  API_BEGIN();
  tvm::runtime::ObjectInternal::ObjectFree(obj);
  API_END();
}

int TVMObjectDerivedFrom(uint32_t child_type_index, uint32_t parent_type_index, int* is_derived) {
  API_BEGIN();
  *is_derived = [&]() {
    if (child_type_index == parent_type_index) return true;
    if (child_type_index < parent_type_index) return false;
    const TVMFFITypeInfo* child_type_info = TVMFFIGetTypeInfo(child_type_index);
    const TVMFFITypeInfo* parent_type_info = TVMFFIGetTypeInfo(parent_type_index);
    return (child_type_info->type_depth > parent_type_info->type_depth &&
      child_type_info->type_acenstors[parent_type_info->type_depth] == static_cast<int32_t>(parent_type_index));
  }();
  API_END();
}

int TVMObjectTypeKey2Index(const char* type_key, unsigned* out_tindex) {
  API_BEGIN();
  out_tindex[0] = tvm::runtime::ObjectInternal::ObjectTypeKey2Index(type_key);
  API_END();
}

int TVMObjectTypeIndex2Key(unsigned tindex, char** out_type_key) {
  API_BEGIN();
  auto key = tvm::runtime::Object::TypeIndex2Key(tindex);
  *out_type_key = static_cast<char*>(malloc(key.size() + 1));
  strncpy(*out_type_key, key.c_str(), key.size() + 1);
  API_END();
}
