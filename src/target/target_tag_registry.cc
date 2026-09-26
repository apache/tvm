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
 * \file src/target/target_tag_registry.cc
 * \brief Process-wide target tag registry.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target_tag_registry.h>

#include <utility>

namespace tvm {

TargetTagRegistry* TargetTagRegistry::Global() {
  static TargetTagRegistry registry;
  return &registry;
}

ffi::Optional<Target> TargetTagRegistry::Get(const ffi::String& name) const {
  if (auto config = GetConfig(name)) {
    return Target(config.value());
  }
  return std::nullopt;
}

ffi::Optional<TargetTagRegistry::Config> TargetTagRegistry::GetConfig(
    const ffi::String& name) const {
  if (auto config = configs_.Get(name)) {
    return config.value();
  }
  return std::nullopt;
}

ffi::Map<ffi::String, Target> TargetTagRegistry::ListTags() const {
  ffi::Map<ffi::String, Target> result;
  for (const auto& kv : configs_) {
    result.Set(kv.first, Target(kv.second));
  }
  return result;
}

Target TargetTagRegistry::AddTag(ffi::String name, Config config, bool override) {
  auto previous = configs_.Get(name);
  TVM_FFI_ICHECK(override || !previous.has_value())
      << "Tag \"" << name << "\" has been previously defined as: " << previous.value();
  Target target(config);
  configs_.Set(name, std::move(config));
  return target;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.TargetTagListTags", []() { return TargetTagRegistry::Global()->ListTags(); })
      .def("target.TargetTagAddTag", [](ffi::String name, TargetTagRegistry::Config config,
                                        bool override) {
        return TargetTagRegistry::Global()->AddTag(std::move(name), std::move(config), override);
      });
}

}  // namespace tvm
