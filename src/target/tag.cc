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
#include <tvm/runtime/registry.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>

#include "../node/attr_registry.h"

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetTagNode);

TVM_REGISTER_GLOBAL("target.TargetTagListTags").set_body_typed(TargetTag::ListTags);
TVM_REGISTER_GLOBAL("target.TargetTagAddTag").set_body_typed(TargetTag::AddTag);

/**********  Registry-related code  **********/

using TargetTagRegistry = AttrRegistry<TargetTagRegEntry, TargetTag>;

TargetTagRegEntry& TargetTagRegEntry::RegisterOrGet(const String& target_tag_name) {
  return TargetTagRegistry::Global()->RegisterOrGet(target_tag_name);
}

Optional<Target> TargetTag::Get(const String& target_tag_name) {
  const TargetTagRegEntry* reg = TargetTagRegistry::Global()->Get(target_tag_name);
  if (reg == nullptr) {
    return NullOpt;
  }
  return Target(reg->tag_->config);
}

Map<String, Target> TargetTag::ListTags() {
  Map<String, Target> result;
  for (const String& tag : TargetTagRegistry::Global()->ListAllNames()) {
    result.Set(tag, TargetTag::Get(tag).value());
  }
  return result;
}

Target TargetTag::AddTag(String name, Map<String, ObjectRef> config, bool override) {
  TargetTagRegEntry& tag = TargetTagRegEntry::RegisterOrGet(name).set_name();
  CHECK(override || tag.tag_->config.empty())
      << "Tag \"" << name << "\" has been previously defined as: " << tag.tag_->config;
  tag.set_config(config);
  return Target(config);
}

/**********  Register Target tags  **********/

TVM_REGISTER_TARGET_TAG("nvidia/rtx2080ti")
    .set_config({
        {"kind", String("cuda")},
        {"arch", String("sm_75")},
    });

}  // namespace tvm
