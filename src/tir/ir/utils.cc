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
 * \file src/tir/ir/utils.cc
 * \brief Utilities for manipulating TIR
 */
#include "utils.h"

#include <tvm/ir/attrs.h>

namespace tvm {
namespace tir {

ObjectRef NormalizeAttributeObject(ObjectRef obj) {
  if (const auto* runtime_int = obj.as<runtime::Int::ContainerType>()) {
    return Integer(runtime_int->value);
  } else if (const auto* runtime_bool = obj.as<runtime::Bool::ContainerType>()) {
    return Bool(runtime_bool->value);
  } else if (const auto* runtime_float = obj.as<runtime::Float::ContainerType>()) {
    return FloatImm(DataType::Float(32), runtime_float->value);
  } else if (auto opt_array = obj.as<Array<ObjectRef>>()) {
    return opt_array.value().Map(NormalizeAttributeObject);
  } else if (auto opt_map = obj.as<Map<ObjectRef, ObjectRef>>()) {
    Map<ObjectRef, ObjectRef> new_map;
    bool is_same = true;

    for (const auto& [key, obj] : opt_map.value()) {
      ObjectRef new_obj = NormalizeAttributeObject(obj);
      is_same = is_same && obj.same_as(new_obj);
      new_map.Set(key, new_obj);
    }

    if (is_same) {
      return obj;
    } else {
      return new_map;
    }
  } else if (auto dict_attrs = obj.as<DictAttrs::ContainerType>()) {
    auto new_attrs = Downcast<Map<String, ObjectRef>>(NormalizeAttributeObject(dict_attrs->dict));
    if (new_attrs.same_as(dict_attrs->dict)) {
      return GetRef<DictAttrs>(dict_attrs);
    } else {
      return DictAttrs(new_attrs);
    }
  } else {
    return obj;
  }
}

}  // namespace tir
}  // namespace tvm
