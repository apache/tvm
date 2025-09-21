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
 * \file attrs.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>

#include "attr_functor.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() {
  AttrFieldInfoNode::RegisterReflection();
  DictAttrsNode::RegisterReflection();
}

DictAttrs WithAttrs(DictAttrs attrs, ffi::Map<ffi::String, ffi::Any> new_attrs) {
  if (new_attrs.empty()) {
    return attrs;
  }

  auto* write_ptr = attrs.CopyOnWrite();
  for (const auto& [key, value] : new_attrs) {
    write_ptr->dict.Set(key, value);
  }
  return attrs;
}

DictAttrs WithAttr(DictAttrs attrs, ffi::String key, ffi::Any value) {
  attrs.CopyOnWrite()->dict.Set(key, value);
  return attrs;
}

DictAttrs WithoutAttr(DictAttrs attrs, const std::string& key) {
  attrs.CopyOnWrite()->dict.erase(key);
  return attrs;
}

void DictAttrsNode::InitByPackedArgs(const ffi::PackedArgs& args, bool allow_unknown) {
  for (int i = 0; i < args.size(); i += 2) {
    ffi::String key = args[i].cast<ffi::String>();
    ffi::AnyView val = args[i + 1];
    dict.Set(key, val);
  }
}

DictAttrs::DictAttrs(ffi::Map<ffi::String, Any> dict) {
  ObjectPtr<DictAttrsNode> n = ffi::make_object<DictAttrsNode>();
  n->dict = std::move(dict);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { tvm::ffi::reflection::ObjectDef<BaseAttrsNode>(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ir.DictAttrsGetDict", [](DictAttrs attrs) { return attrs->dict; });
}

}  // namespace tvm
