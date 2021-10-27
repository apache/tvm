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
 * \file tvm/target/se_scope.h
 * \brief A compile time representation for a Storage or Execution Scope.
 */

#ifndef TVM_TARGET_SE_SCOPE_H_
#define TVM_TARGET_SE_SCOPE_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {

/*!
 * \brief Describes at compile time where data is to be stored down to the device and memory
 * scope level, or where execution is to take place, down to the device level. It is a quadruple of:
 * - A \p device_type (\p DLDeviceType).
 * - An uninterpreted \p virtual_device_id (\p int) distinguishing the intended device from all
 *   other devices (either of the same \p device_type, or across all available devices in the
 *   system). The virtual device id need not correspond to any physical device id, see
 *   "Virtual Devices" below.
 * - A \p target (\p Target) describing how to compile code for the intended device.
 * - A \p memory_scope (currently just \p String) describing which memory area is to be used to
 *   hold data. The area should be reachable from the device but need not be 'on' the device,
 *   see "Memory Scopes and Devices" below. (We're using a \p String for now but would prefer a
 *   more structured representation once it is available.)
 *
 * All of these fields may be 'unconstrained' (ie null, -1 or ""), signaling that device planning
 * is free to choose a value consistent with the whole program. However if a \p target is given
 * then the \p device_type must equal \p target->kind->device_type.
 *
 * Note that currently we assume if a function returns its result on a particular device
 * then the function body is also executed on that device. See the overview comment in
 * src/relay/transforms/device_planner.cc for more details.
 *
 * By 'data' we include both tensors and additional supporting datastructures such as shapes,
 * Relay AST items, Relay tuples, and Relay references. Typically non-tensor data must reside
 * on a 'CPU'-like device with good support for scalars.
 *
 * By 'execution' we include both (fused) primitive operators, and all the Relay expressions
 * surrounding them which coordinates data and control flow. Again, typically non-primitive
 * operators must be executed on a 'CPU'-like device with good support for control flow.
 *
 * Targets vs Devices
 * ------------------
 * Generally \p Targets (a compile-time only datastructue) describe compiler options for a specific
 * microarchitecture and toolchain, while \p Devices (a runtime datastructure also available at
 * compile time) describe a physical device on the target system. Obviously the target must agree
 * with the device's microarchitecture, but we otherwise don't impose any constraints between them:
 *  - It's ok to use different \p Targets for the same \p Device, eg to squeeze some extra perf
 *    out of a particular primitive.
 *  - It's ok to use the same \p Target for multiple \p Devices, eg if we have multiple CPUs.
 *
 * Traditionally TVM assumes at most one \p Target per \p DLDeviceType. We are moving away from that
 * assumption.
 *
 * Virtual vs Physical Devices
 * ---------------------------
 * The \p virtual_device_id may be left as 0 if not significant. It is up to downstream
 * compilation passes and/or the runtime to map a \p virtual_device_id to an actual physical
 * device id if required. For example, some runtimes may support passing in an array of actual
 * `device` specifications, and the \p virtual_device_id is simply an index known at compile time
 * into that array.
 *
 * Memory Scopes and Devices
 * -------------------------
 * Multi-device systems can have complex memory hierarchies. For example
 * \code
 * (kDLCPU, 0, "llvm", "global")
 * \endcode
 * and
 * \code
 * (kDLCPU, 1, "llvm", "global")
 * \endcode
 * could denote:
 * - The same memory area accessible from two separate CPUs without any CPU affinity;
 * - Distinct memory areas in a NUMA architecture for which cross-device access is handled
 *   by the memory system;
 * - Outright distinct memory areas, where one device cannot directly address the memory of
 *   another.
 *
 * Similarly:
 * \code
 * (kDLCPU, 0, "llvm", "global")
 * \endcode
 * and
 * \code
 * (kDLCUDA, 0, "cuda", "host")
 * \endcode
 * could denote the same memory area, but with very different access costs.
 *
 * We don't currently try to build any of this system-level understanding into \p SEScope. Device
 * planning will simply insert "device_copy" operators wherever \p SEScopes are not exactly
 * pointwise equal, and we leave it to downstream compilation to elide unnecessary copies. We
 * may revisit this in the future.
 *
 * Joining and Defaulting
 * ----------------------
 * It is possible to 'join' two \p SEScopes to yield the most constrained \p SEScope which agrees
 * with both join arguments. Eg:
 * \code
 * Join((kDLCPU, -1, "llvm", ""), (kInvalidDeviceType, 3, null, "global))
 *   => (kDLCPU, 3, "llvm", "global")
 * Join((kDLCPU, -1, "llvm", ""), (kInvalidDeviceType, 3, null, "local))
 *   => null (no join possible)
 * \endcode
 *
 * Related to 'join' is 'default', which only takes constrained fields from the rhs when the
 * lhs is unconstrained:
 * \code
 * Default(kDLCPU, -1, "llvm", "local"), (kDLCPU, 3, null, "global"))
 *   => (kDLCPU, 3, "llvm", "local")
 * \endcode
 *
 * These operations are needed during device planning.
 *
 */
class SEScopeNode : public Object {
 public:
  /*!
   * \brief The \p DLDeviceType of the device. If \p target is known then this will be equal to
   * \p target->kind->device_type. If \p target is null then the target is to be determined by
   * a later pass.
   *
   * This is needed to support the legacy "on_device" and "device_copy" calls which only allow
   * a \p DLDeviceTypes (as an integer) to be given.
   *
   * kInvalidDeviceType denotes unconstrained.
   */
  DLDeviceType device_type() const { return device_type_; }

  /*!
   * \brief The 'virtual' device identifier for the device. This must be resolved to a physical
   * device identifier either during compilation or at runtime.
   *
   * -1 denotes unconstrained. May be 0 if not significant.
   */
  int virtual_device_id() const { return virtual_device_id_; }

  /*!
   * \brief The \p Target describing how to compile for the device.
   *
   * Null denotes unconstrained (though if device_type is known then only a target of that
   * type is allowed).
   */
  const Target& target() const { return target_; }

  /*!
   * \brief The scope of memory within the device.
   *
   * Empty denotes unconstrained.
   */
  // TODO(mbs): We are using String as a stand-in pending a more structured representation, such
  // as runtime::StorageScope or a memory pool.
  const String& memory_scope() const { return memory_scope_; }

  /*!
   * \brief Returns true if scope is fully unconstrained, ie no target/device type, virtual device
   * id or memory scope is specified.
   */
  bool is_fully_unconstrained() const {
    return !target_.defined() && device_type_ == kInvalidDeviceType && virtual_device_id_ == -1 &&
           memory_scope_.empty();
  }

  /*!
   * \brief Returns true if scope is fully constrained, ie target, virtual device id and
   * memory scope are all specified.
   */
  bool is_fully_constrained() const {
    return target_.defined() && virtual_device_id_ != -1 && !memory_scope_.empty();
  }

  Device ToDevice() const {
    ICHECK(device_type_ != kInvalidDeviceType);
    ICHECK(virtual_device_id_ != -1);
    Device device;
    device.device_type = device_type_;
    device.device_id = virtual_device_id_;
    return device;
  }

  void VisitAttrs(AttrVisitor* v);

  // We implement structural equality (which reduces Targets to their str representation)
  // so that unit tests can compare actual and expected SEScopes.
  bool SEqualReduce(const SEScopeNode* other, SEqualReducer equal) const;
  void SHashReduce(SHashReducer hash_reduce) const;

  static constexpr const char* _type_key = "SEScope";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SEScopeNode, Object);

 private:
  // We keep the fields private so the constructor memoization can't be upset by mutation.
  DLDeviceType device_type_;
  int virtual_device_id_;
  Target target_;
  String memory_scope_;

  friend class SEScope;
};

/*!
 * \brief Managed reference class to \p SEScopeNode.
 *
 * \sa SEScopeNode.
 */
class SEScope : public ObjectRef {
 public:
  /*!
   * \brief Construct an SEScope.
   * \param device_type The device type for the device, or kInvalidDeviceType if unconstrained.
   * If \p target is defined then must match its \p target->kind->device_type.
   * \param virtual_device_id The virtual device id for the device, or -1 if unconstrained.
   * \param target The target describing how to compile for the device, or null if unconstrained.
   * \param memory_scope The memory scope within the device, or "" if unconstrained.
   * \return The SEScope
   */
  explicit SEScope(DLDeviceType device_type = kInvalidDeviceType, int virtual_device_id = -1,
                   Target target = {}, String memory_scope = {});

  /*! \brief Returns the unique fully unconstrained \p SEScope. */
  static SEScope FullyUnconstrained();

  /*!
   * \brief Returns the \p SEScope for \p device_type and (if not -1) \p virtual_device_id.
   * The target and memory scope will be unconstrained.
   */
  static SEScope ForDeviceType(DLDeviceType device_type, int virtual_device_id = -1) {
    ICHECK_GT(device_type, 0);
    return SEScope(device_type, virtual_device_id);
  }
  static SEScope ForDeviceType(int device_type, int virtual_device_id = -1) {
    return ForDeviceType(static_cast<DLDeviceType>(device_type), virtual_device_id);
  }
  static SEScope ForDeviceType(const Integer& device_type, int virtual_device_id = -1) {
    return ForDeviceType(static_cast<int>(device_type->value), virtual_device_id);
  }

  /*! \brief Returns the \p SEScope for \p device. */
  static SEScope ForDevice(const Device& device) {
    return ForDeviceType(device.device_type, device.device_id);
  }

  /*! \brief Returns the \p SEScope for \p device and \p target. */
  static SEScope ForDeviceAndTarget(const Device& device, Target target) {
    return SEScope(device.device_type, device.device_id, std::move(target));
  }

  /*! \brief Returns the \p SEScope for \p device, \p target and \p memory_scope. */
  TVM_DLL static SEScope ForDeviceTargetAndMemoryScope(const Device& device, Target target,
                                                       String memory_scope) {
    return SEScope(device.device_type, device.device_id, std::move(target),
                   std::move(memory_scope));
  }

  /*!
   * \brief Returns the 'join' of \p lhs and \p rhs. The result will agree pointwise with
   * \p lhs and \p rhs on all their constrained fields. Returns the null optional if no such
   * join exists, ie there's disagreement on at least one constrained field.
   */
  static Optional<SEScope> Join(const SEScope& lhs, const SEScope& rhs);

  /*!
   * \brief Returns the 'default' of \p lhs and \p rhs. The result will be \p lhs, except any
   * unconstrained fields in \p lhs will take their value from \p rhs. Always well-defined.
   */
  static SEScope Default(const SEScope& lhs, const SEScope& rhs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SEScope, ObjectRef, SEScopeNode);

  friend class SEScopeCache;  // Private implementation helper.
};

/*!
 * \brief A cache of \p SEScopes. This can be used:
 *  - To avoid ending up with lots of identical instances, since the space of SEScopes for any
 *    one compilation is very small but the number of points they need to be constructed can
 *    be very large (eg during device planning).
 *  - So we can assume \p SEScopes are pointer equal if and only if they are structurally equal.
 *    This simplifies the unification of 'device domains' which are built on \p SEScopes.
 */
class SEScopeCache {
 public:
  /*! \brief Returns the unique \p SEScope representing given fields. */
  SEScope Make(DLDeviceType device_type = kInvalidDeviceType, int virtual_device_id = -1,
               Target target = {}, String memory_scope = {});

  /*! \brief Returns the unique \p SEScope structurally equal to the given \p se_scope. */
  SEScope Unique(const SEScope& scope);

 private:
  std::unordered_map<std::string, SEScope> cache_;
};

}  // namespace tvm

#endif  //  TVM_TARGET_SE_SCOPE_H_
