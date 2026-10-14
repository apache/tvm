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

/*! \brief Backend registration of attributes on layout-axis singletons. */
#ifndef TVM_TIRX_IR_LAYOUT_AXIS_REGISTRY_H_
#define TVM_TIRX_IR_LAYOUT_AXIS_REGISTRY_H_

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/tirx/layout.h>

#include <utility>

namespace tvm::tirx {
class AxisDef {
 public:
  explicit AxisDef(const char* name) : axis_(Axis::Get(name)) {}

  template <typename T>
  AxisDef& set_attr(const char* name, T value) {
    ffi::reflection::TypeAttrColumn column(ffi::reflection::type_attr::kEnumState);
    auto state =
        ffi::Any(column[AxisNode::_GetOrAllocRuntimeTypeIndex()]).as_or_throw<ffi::EnumState>();
    ffi::String key(name);
    auto attrs = state->attrs;
    auto values = attrs.Get(key).value_or(ffi::Dict<ffi::ObjectRef, ffi::Any>());
    attrs.Set(key, values);
    values.Set(axis_, ffi::Any(std::move(value)));
    return *this;
  }

 private:
  Axis axis_;
};
}  // namespace tvm::tirx
#endif  // TVM_TIRX_IR_LAYOUT_AXIS_REGISTRY_H_
