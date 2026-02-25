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
 * \file src/target/target_tag.cc
 * \brief Target tag registry
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>

#include "../node/attr_registry.h"

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() { TargetTagNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.TargetTagListTags", TargetTag::ListTags)
      .def("target.TargetTagAddTag", TargetTag::AddTag);
}

/**********  Registry-related code  **********/

using TargetTagRegistry = AttrRegistry<TargetTagRegEntry, TargetTag>;

TargetTagRegEntry& TargetTagRegEntry::RegisterOrGet(const ffi::String& target_tag_name) {
  return TargetTagRegistry::Global()->RegisterOrGet(target_tag_name);
}

ffi::Optional<Target> TargetTag::Get(const ffi::String& target_tag_name) {
  const TargetTagRegEntry* reg = TargetTagRegistry::Global()->Get(target_tag_name);
  if (reg == nullptr) {
    return std::nullopt;
  }
  return Target(reg->tag_->config);
}

ffi::Optional<ffi::Map<ffi::String, ffi::Any>> TargetTag::GetConfig(
    const ffi::String& target_tag_name) {
  const TargetTagRegEntry* reg = TargetTagRegistry::Global()->Get(target_tag_name);
  if (reg == nullptr) {
    return std::nullopt;
  }
  return reg->tag_->config;
}

ffi::Map<ffi::String, Target> TargetTag::ListTags() {
  ffi::Map<ffi::String, Target> result;
  for (const ffi::String& tag : TargetTagRegistry::Global()->ListAllNames()) {
    result.Set(tag, TargetTag::Get(tag).value());
  }
  return result;
}

Target TargetTag::AddTag(ffi::String name, ffi::Map<ffi::String, ffi::Any> config, bool override) {
  TargetTagRegEntry& tag = TargetTagRegEntry::RegisterOrGet(name).set_name();
  TVM_FFI_ICHECK(override || tag.tag_->config.empty())
      << "Tag \"" << name << "\" has been previously defined as: " << tag.tag_->config;
  tag.set_config(config);
  return Target(config);
}

}  // namespace tvm
