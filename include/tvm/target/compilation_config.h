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
 * \file tvm/target/compilation_config.h
 * \brief A helper class to collect all the targets in canonical form necessary for compilation.
 * CAUTION: Preliminary, currently only used to support device planning, very likely to change.
 */

#ifndef TVM_TARGET_COMPILATION_CONFIG_H_
#define TVM_TARGET_COMPILATION_CONFIG_H_

#include <tvm/target/se_scope.h>

namespace tvm {

/*!
 * \brief Gathers the \p Targets and distinguished \p SEScopes in canonical form needed to
 * compile a Relay module. All centralizes any setup and validation logic needed to transition
 * from configuration options conveyed implicitly (eg in \p PassContexts) or explicitly
 * (eg a a list of \p Targets) to the configuration.
 *
 * CAUTION: This is subject to change as we rework compilation options in general. See
 * https://github.com/apache/tvm-rfcs/blob/main/rfcs/0028-command-line-registry-composition.md.
 * So far this class is only focussed on carrying just the configuration needed by PlanDevices,
 * and removing target-munging code duplication and inconsistencies between the three major build
 * flows for the VM (relay/backend/vm/compile.cc), Graph/AOT (relay/backend/build_module.cc) and
 * Interpreter (relay/backend/interpreter.cc). Over time we expect more global compiler
 * configuration (eg for executor and runtime config, for system memory pool configuration, etc)
 * to migrate into this class, and instances thereof to be attached to \p IRModules using a
 * well-known attribute.
 */
class CompilationConfigNode : public Object {
 public:
  /*!
   * \brief The legacy targets map, mapping device type to \p Targets. Does not include any
   * entry for the host target. Intended to give a unique \p Target for every \p DLDeviceType,
   * though we want to get rid of that limitation.
   *
   * CAUTION: Since keys are \p Integers they are compared by object equality not integer
   * value.
   *
   * TODO(mbs): Remove once codegen updated for new target conventions.
   */
  TargetMap legacy_target_map;

  /*!
   * \brief The host target. Used for 'scalar' data and code (such as shapes and shape
   * functions) and residual Relay expressions and data (such as conditionals and ADTs).
   */
  Target host_target;

  /*!
   * \brief Vector of all available targets for primitive operators. May contain a \p Target
   * for the same device type as for the \p host_target, however the \p host_target should
   * be preferred for all host computations and data.
   */
  Array<Target> primitive_targets;

  /*!
   * \brief \p SEScope for primitive operators which are not otherwise constrained to a particular
   * device.
   */
  SEScope default_primitive_se_scope = SEScope::FullyUnconstrained();

  /*! \brief SEScope for the host. */
  SEScope host_se_scope = SEScope::FullyUnconstrained();

  /*!
   * \brief If defined then compile and/or run in 'homogenous execution mode'. In this mode all
   * primitives are compiled for this target only.
   *
   * This is to support legacy passes which have not been adapted to hetrogeneous execution and
   * rely on an implicit global \p Target to be in scope.
   *
   * TODO(mbs): Remove once all passes are 'hetrogeneous aware'.
   */
  Target optional_homogeneous_target;

  void VisitAttrs(AttrVisitor* v);

  /*!
   * \brief Returns a \p SEScope agreeing with \p se_scope on all its constrained fields, however:
   * - If the target is null then it is filled in from the known available primitive targets by
   *   matching on device type. Fails if no such target is known.
   * - The returned object is unique for the field values w.r.t. all other \p SEScopes returned
   *   by this method.
   *
   * We call the result the 'canonical' \p SEScope. Two canonical \p SEScopes are structurally
   * equal if and only if they are pointer equal.
   */
  SEScope CanonicalSEScope(const SEScope& se_scope) const;

  static constexpr const char* _type_key = "CompilationConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompilationConfigNode, Object)

 private:
  /*!
   * \brief Establishes the default \p SEScope for primitives and the \p SEScope for the host
   * given:
   *  - the vector of available primitive \p Targets.
   *  - any host \p Target.
   *  - any "relay.fallback_device_type" attribute on \p pass_ctx.
   *  - whether the LLVM backend is available.
   * If necessary, creates new default \p Targets to match the required devices.
   *
   * NOTE: The implementation is a bit convoluted since it tries to maintain backwards
   * compatibility with legacy methods for conveying \p Targets.
   */
  void EstablishDefaultSEScopes(const transform::PassContext& pass_ctx);

  /*!
   * \brief Returns a freshly constructed \p Target to represent \p device_type.
   */
  static Target MakeDefaultTarget(DLDeviceType device_type);

  /*!
   * \brief Return the \p Target to use for \p device_type. Fail if no such target exists.
   */
  Target FindPrimitiveTargetOrFail(DLDeviceType device_type) const;

  /*!
   * \brief A cache of constructed SEScopes.
   */
  mutable SEScopeCache se_scope_cache_;

  friend class CompilationConfig;
};

/*!
 * \brief Managed reference class to \p CompilationConfig
 *
 * \sa CompilationConfig
 */
class CompilationConfig : public ObjectRef {
 public:
  /*!
   * \brief Constructs the compilation config given the available \p Targets in the
   * \p legacy_target_map and an optional \p optional_host_target. May use
   * 'relay.fallback_device_type' and the availability of the LLVM compilation module
   * to decide on appropriate default devices.
   */
  TVM_DLL CompilationConfig(const transform::PassContext& pass_ctx, TargetMap legacy_target_map,
                            Target optional_host_target);

  TVM_DEFINE_OBJECT_REF_METHODS(CompilationConfig, ObjectRef, CompilationConfigNode);
};

}  // namespace tvm

#endif  // TVM_TARGET_COMPILATION_CONFIG_H_
