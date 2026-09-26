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
 * \file tvm/target/target_tag_registry.h
 * \brief Process-wide target tag registry.
 */
#ifndef TVM_TARGET_TARGET_TAG_REGISTRY_H_
#define TVM_TARGET_TARGET_TAG_REGISTRY_H_

#include <tvm/ffi/container/dict.h>
#include <tvm/target/target.h>

namespace tvm {

/*!
 * \brief Registry of target configurations keyed by tag name.
 *
 * Register a tag during static initialization, then resolve it through Target:
 * \code
 * TVM_FFI_STATIC_INIT_BLOCK() {
 *   TargetTagRegistry::Global()->AddTag(
 *       "my_cpu", {{"kind", ffi::String("llvm")}}, false);
 * }
 * \endcode
 */
class TargetTagRegistry {
 public:
  using Config = ffi::Map<ffi::String, ffi::Any>;

  /*! \brief Return the process-wide registry. */
  TVM_DLL static TargetTagRegistry* Global();

  /*!
   * \brief Construct the target registered under a name.
   * \param name The tag name to look up.
   * \return The constructed target, or nullopt if the name is unknown.
   */
  TVM_DLL ffi::Optional<Target> Get(const ffi::String& name) const;

  /*!
   * \brief Return a tag's stored configuration without constructing a target.
   * \param name The tag name to look up.
   * \return The configuration, or nullopt if the name is unknown.
   */
  TVM_DLL ffi::Optional<Config> GetConfig(const ffi::String& name) const;

  /*!
   * \brief List all registered names and their constructed targets.
   * \return A map from tag names to targets; empty if no tags are registered.
   */
  TVM_DLL ffi::Map<ffi::String, Target> ListTags() const;

  /*!
   * \brief Register a target configuration under a tag name.
   * \param name The tag name to register.
   * \param config The target configuration, including its kind.
   * \param override Whether to replace an existing registration; otherwise a duplicate is rejected.
   * \return The target constructed from config.
   */
  TVM_DLL Target AddTag(ffi::String name, Config config, bool override);

 private:
  TargetTagRegistry() = default;
  TargetTagRegistry(const TargetTagRegistry&) = delete;
  TargetTagRegistry& operator=(const TargetTagRegistry&) = delete;

  ffi::Dict<ffi::String, Config> configs_;
};

}  // namespace tvm

#endif  // TVM_TARGET_TARGET_TAG_REGISTRY_H_
