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

ffi::Any NormalizeAttributeObject(ffi::Any obj) {
  if (obj.type_index() == ffi::TypeIndex::kTVMFFIBool) {
    return Bool(obj.operator bool());
  } else if (auto opt_int = obj.as<int>()) {
    return Integer(opt_int.value());
  } else if (auto opt_float = obj.as<double>()) {
    return FloatImm(DataType::Float(32), opt_float.value());
  } else if (auto opt_array = obj.as<Array<ffi::Any>>()) {
    return opt_array.value().Map(NormalizeAttributeObject);
  } else if (auto opt_map = obj.as<Map<ffi::Any, ffi::Any>>()) {
    Map<ffi::Any, ffi::Any> new_map;
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
    auto new_attrs = Downcast<Map<String, ffi::Any>>(NormalizeAttributeObject(dict_attrs->dict));
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
