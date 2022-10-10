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
 */

#ifndef TVM_TARGET_COMPILATION_CONFIG_H_
#define TVM_TARGET_COMPILATION_CONFIG_H_

#include <tvm/target/virtual_device.h>

#include <string>

namespace tvm {

/*!
 * \brief Gathers the \p Targets and distinguished \p VirtualDevices in canonical form needed to
 * compile a Relay module for execution over possibly heterogeneous devices. Centralizes the
 * validation and canonicalization logic needed to transition from targets supplied by the Python
 * APIs to a single internal representation. Also holds a cache of canonical \p VirtualDevices
 * so that structural equal virtual devices have pointer equal canonical virtual devices.
 *
 * The construction of \p CompilationConfig is idempotent, in that given the same \p PassContext
 * \p ctx and an arbitrary \p Array<Target> \p raw_targets:
 *
 * \code
 *   CompilationConfig(ctxt, raw_targets)
 *      is structurally equal to
 *   CompilationConfig(ctxt, CompilationConfig(ctxt, raw_targets)->primitive_targets)
 * \endcode
 *
 * TODO(mbs): This is subject to change as we rework compilation options in general. This class
 * is probably better called a 'CompositeTarget', and may be better made a sub-class of Target or
 * some other common-target-root class.
 */
class CompilationConfigNode : public Object {
 public:
  /*!
   * \brief The host target. Used for 'scalar' data and code (such as shapes and shape
   * functions) and residual Relay expressions and data (such as conditionals and ADTs).
   * Each \p primitive_target below will have this exact target object as its 'host'.
   *
   * Note that it is possible for a \p Target used for primitive operations to be structurally
   * equal to the host \p Target (up to the \p host field.) However the \p Target objects will
   * be distinct, and can be used as keys within a \p Map without collision.
   */
  Target host_target;

  /*!
   * \brief Vector of all available \p Targets for partitioning or compiling primitive tensor
   * operators (kernels). May contain a \p Target for the same device type as for the
   * \p host_target, however the \p host_target should be used for all host computations and data.
   * Each \p Target will have \p host_target as its 'host'.
   *
   * Primitive targets must be unique by their kind name. In this way the
   * \p FindPrimitiveTargetForKind method will find the unique target for the given kind name.
   * This method is used when transitioning from an external codegen "Compiler" attribute value
   * to the external codegen target representing that compiler.
   *
   * It is possible to have multiple primitive targets for the same device type. However given
   * primitive targets left and right where:
   *  - left appears before right in the array
   *  - left->GetTargetDeviceType() == right->GetTargetDeviceType()
   * then:
   *  - right.IsExternalCodegenFor(left) must be true
   * In this way the \p FindPrimitiveTargetForDeviceOrFail method will find the 'most general'
   * target for the requested device type. This method is used when transitioning from a device
   * constraint to the target needed to compile for that device.
   *
   * In the homogeneous case primitive_targets will have just one entry, which will be pointer equal
   * to optional_homogeneous_target.
   *
   * In the homogenous case where the 'host' is the same device as used for compiling kernels it
   * is *not* the case that optional_homogenous_target == host_target. This is because all
   * primitive always have their host field set to the host_target. Ie, it is valid to have:
   * \code
   *   host_target=Target("llvm")
   *   optional_homogenous_target=Target("llvm", host=host_target)
   * \endcode
   */
  Array<Target> primitive_targets;

  /*!
   * \brief \p VirtualDevice for primitive operators which are not otherwise constrained to a
   * particular device. Used by the PlanDevices pass to determine a virtual device for every
   * sub-expression.
   */
  VirtualDevice default_primitive_virtual_device = VirtualDevice::FullyUnconstrained();

  /*! \brief VirtualDevice for the host. */
  VirtualDevice host_virtual_device = VirtualDevice::FullyUnconstrained();

  /*!
   * \brief If defined then compile and/or run in 'homogenous execution mode'. In this mode all
   * primitives are compiled for this target only.
   *
   * This is to support legacy passes which have not been adapted to heterogeneous execution and
   * rely on an implicit global \p Target to be in scope.
   *
   * TODO(mbs): Remove once all passes are 'heterogeneous aware'.
   */
  Target optional_homogeneous_target;

  void VisitAttrs(AttrVisitor* v);

  /*!
   * \brief Returns the unique \p Target to use for \p device_type. Fail if no such target exists.
   *
   * This will be the first primitive target with matching device type.
   */
  Target FindPrimitiveTargetForDeviceOrFail(DLDeviceType device_type) const;

  /*!
   * \brief Returns the unique \p Target to use for \p kind_name. Returns null if none such.
   */
  Optional<Target> FindPrimitiveTargetForKind(const std::string& kind_name) const;

  /*!
   * \brief Returns a \p Target structurally equal to \p target, however prefer a structually equal
   * known host or primitive target if the configuration has one.
   */
  Target CanonicalTarget(const Target& target) const;

  /*!
   * \brief Returns a \p VirtualDevice which is structurally equal to \p virtual_device on all its
   * constrained fields, however:
   * - If \p virtual_device has a device type but not a target, fill in a target using
   *   \p FindPrimitiveTargetOrFail. This is the one place we allow targets to be defaulted
   *   from device types alone.
   * - If \p virtual_device has a target, also canonicalize it using \p CanonicalTarget.
   * The returned object will be unique for the adjusted virtual device w.r.t. all other
   * \p VirtualDevices returned by this method.
   *
   * We call the result the 'canonical' \p VirtualDevice. Two canonical \p VirtualDevices are
   * structurally equal if and only if they are pointer equal. In this way we can build maps
   * from virtual devices using just pointer equality.
   */
  VirtualDevice CanonicalVirtualDevice(const VirtualDevice& virtual_device) const;

  static constexpr const char* _type_key = "CompilationConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompilationConfigNode, Object)

 private:
  /*!
   * \brief Sets the primitive targets, the host target, the default primitive virtual device, and
   * the host virtual device given:
   *  - the vector of 'raw' targets (in any order) supplied by one of the TVM entry points.
   *  - any "relay.fallback_device_type" attribute on \p pass_ctx.
   *  - whether the LLVM backend is available.
   * Will look for a suitable host target in the given primitive targets, but if none found may
   * reuse a raw target or create a default CPU target.
   */
  void Init(const transform::PassContext& pass_ctx, const Array<Target>& raw_targets);

  /*!
   * \brief Returns a freshly constructed CPU \p Target.
   */
  static Target MakeDefaultCPUTarget();

  /*!
   * \brief A cache of constructed virtual devices.
   */
  mutable VirtualDeviceCache virtual_device_cache_;

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
   * \brief Constructs the compilation config given the settings in \p pass_ctx and supplied
   * \p raw_targets. See \p CompilationConfigNode::Init for details.
   */
  TVM_DLL CompilationConfig(const transform::PassContext& pass_ctx,
                            const Array<Target>& raw_targets);

  TVM_DEFINE_OBJECT_REF_METHODS(CompilationConfig, ObjectRef, CompilationConfigNode);
};

}  // namespace tvm

#endif  // TVM_TARGET_COMPILATION_CONFIG_H_
