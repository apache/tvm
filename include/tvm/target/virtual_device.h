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
 * \file tvm/target/virtual_device.h
 * \brief A compile time representation for where data is to be stored at runtime, and how to
 * compile code to compute it.
 */

#ifndef TVM_TARGET_VIRTUAL_DEVICE_H_
#define TVM_TARGET_VIRTUAL_DEVICE_H_

#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_set>
#include <utility>

namespace tvm {

/*!
 * Abstract label for an area of memory.
 *
 * Currently uninterpreted and arbitrary. Likely to be replaced by a structured representation
 * of a memory pool in the future. Please try to use this alias instead of String to aid future
 * code migration.
 */
using MemoryScope = String;

// NOTE: cannot use enum as they are out of bound of the original enum
// and results in an undefined behavior
// A 'null' device type, does not correspond to any DLDeviceType enum.
// TODO(mbs): This is to help us as we transition away from representing the 'homogenous' case
// as a singleton target map indexed by the invalid DLDeviceType '0'.
constexpr int kNullDeviceType = 0;

// An 'invalid' device type, does not correspond to any DLDeviceType enum.
constexpr int kInvalidDeviceType = -1;

/*!
 * \brief Describes at compile time the constraints on where data is to be stored at runtime
 * down to the (virtual) device and memory scope level, and how to compile code to compute that
 * data. Used by the \p PlanDevices pass to collect and solve (virtual) device constraints for
 * the whole Relay program.
 *
 * Is a quadruple of:
 * - A \p device_type (\p DLDeviceType). May be \p kInvalidDeviceType if unconstrained.
 * - A \p virtual_device_id (\p int). This allows us to distinguish distinct devices
 *   with the same \p Target, for example in a multi-GPU system. May be -1 if unconstrained.
 *   See "Virtual Devices" below.
 * - A \p target (\p Target) describing how to compile code for the intended device. May be null
 *   if unconstrained.
 * - A \p memory_scope (\p MemoryScope, which is currently just \p String) describing which memory
 *   area is to be used to hold data. May be "" if unconstrained. See "Memory Scopes and Devices"
 *   below.
 *
 * Some or all of these fields may be unconstrained, signaling that device planning is free to
 * choose a value consistent with the whole program. However if a \p target is given then the \p
 * device_type must equal \p target->GetTargetDeviceType().
 *
 * Note that currently we assume if a function returns its result on a particular (virtual) device
 * then the function body is also executed on that device. See the overview comment in
 * src/relay/transforms/device_planner.cc for more details.
 *
 * By 'data' we include both tensors and additional supporting datastructures such as shapes,
 * Relay ADT items (including tuples), Relay references, and Relay closures. Typically non-tensor
 * data must reside on a 'CPU'-like host device with good support for scalars.
 *
 * By 'execution' we include both (fused) primitive operators, and all the Relay expressions
 * surrounding them which coordinates data and control flow. Again, typically non-primitive
 * operators must be executed on a 'CPU'-like device with good support for control flow.
 *
 * Since TVM targets such a wide range of systems it is not possible for \p VirtualDevice to impose
 * much semantics on these fields, particularly for \p virtual_device_id and \p memory_scope.
 * Instead we assume downstream passes and codegen will interpret an validate these fields
 * appropriately.
 *
 * Targets vs Devices
 * ------------------
 * Generally \p Targets (a compile-time only datastructue) describe compiler options for a specific
 * microarchitecture and toolchain, while \p Devices (a runtime datastructure also available at
 * compile time) describe a physical device on the target system. Obviously the target must agree
 * with the device's microarchitecture, but we otherwise don't impose any constraints between them:
 *  - It's ok to use different \p Targets for the same \p Device, eg to squeeze some extra perf
 *    out of a particular primitive using particular compiler flags.
 *  - It's ok to use the same \p Target for multiple \p Devices, eg if we have multiple CPUs.
 *
 * Traditionally TVM assumes at most one \p Target per \p DLDeviceType. We are moving away from that
 * assumption.
 *
 * Virtual vs Physical Devices
 * ---------------------------
 * The \p virtual_device_id may be used by downstream passes or the runtime to help decide which
 * \p device_id to use for a particular physical runtime \p Device. For example:
 *  - Some runtimes may support passing in an array of actual `device` specifications, and the
 *    \p virtual_device_id can be used at runtime as an index into that array.
 *  - Some runtimes may support dynamically allocating computations to physical devices. On these
 *    systems a large space of \p virtual_device_ids could be used at compile time, even though
 *    at runtime only a few physical devices will be present.
 *
 * The \p virtual_device_id may also be left unconstrained if not needed.
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
 * Furthermore, not all memory scopes are accessible to all devices, and it is possible for
 * a memory scope to only be accessible to a device when code is compiled with particular
 * \p Target options.
 *
 * \p VirtualDevices themselves have no system-level understanding. Currently the \p PlanDevices
 * pass will simply insert "device_copy" operators wherever \p VirtualDevices are not exactly
 * pointwise equal. We may revisit this in the future as the work on memory pools matures.
 *
 * Joining and Defaulting
 * ----------------------
 * It is possible to 'join' two \p VirtualDevices to yield the most constrained \p VirtualDevice
 * which agrees with both join arguments. Eg:
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
 */

class VirtualDeviceNode : public AttrsNode<VirtualDeviceNode> {
 private:
  /*!
   * \brief The \p DLDeviceType (represented as an int) of the virtual device. If \p target is
   * known then this will be equal to \p target->GetTargetDeviceType(). If \p target is null then
   * the target is to be determined later.
   *
   * This is needed to support the legacy "on_device" and "device_copy" calls which only allow
   * a \p DLDeviceTypes (as an integer) to be given.
   *
   * kInvalidDeviceType denotes unconstrained. An int since the DLDeviceType enum representation
   * is not fixed. Private to discourage further int vs DLDeviceType confusion.
   */
  int /* actually DLDeviceType */ device_type_int;

 public:
  DLDeviceType device_type() const { return static_cast<DLDeviceType>(device_type_int); }

  /*!
   * \brief The device identifier for the virtual device. This must be resolved to a physical
   * device identifier either during compilation or at runtime.
   *
   * -1 denotes unconstrained.
   */
  int virtual_device_id;

  /*!
   * \brief The \p Target describing how to compile for the virtual device.
   *
   * Null denotes unconstrained. Note that if a target later becomes known for this \p VirtualDevice
   * then it must be consistent with the \p device_type if already known. This is enforced by the
   * Join and Default methods.
   */
  Target target;

  /*!
   * \brief The scope of memory w.r.t. the virtual device which holds data.
   *
   * Empty denotes unconstrained.
   */
  MemoryScope memory_scope;

  /*!
   * \brief Returns true if virtual device is 'fully unconstrained', ie no target/device type,
   * device id or memory scope is specified.
   */
  bool IsFullyUnconstrained() const {
    return !target.defined() && device_type() == kInvalidDeviceType && virtual_device_id == -1 &&
           memory_scope.empty();
  }

  /*!
   * \brief Returns true if virtual device is 'fully constrained', ie target, device id and memory
   * scope are all specified.
   */
  bool IsFullyConstrained() const {
    return target.defined() && virtual_device_id != -1 && !memory_scope.empty();
  }

  /*!
   * \brief Returns the (virtual) \p Device implied by this \p VirtualDevice. Both the \p
   * device_type and \p virtual_device_must be constrained. The returned \p Device may not
   * correspond to any physical device available at compile time or even runtime: see "Virtual vs
   * Physical Devices" above.
   */
  Device ToDevice() const {
    ICHECK(device_type_int != kInvalidDeviceType);
    ICHECK(virtual_device_id != -1);
    Device device;
    device.device_type = device_type();
    device.device_id = virtual_device_id;
    return device;
  }

  TVM_DECLARE_ATTRS(VirtualDeviceNode, "VirtualDevice") {
    TVM_ATTR_FIELD(device_type_int)
        .describe("The type of the virtual device.")
        .set_default(kInvalidDeviceType);
    TVM_ATTR_FIELD(virtual_device_id)
        .describe("The device id of the virtual device.")
        .set_default(-1);
    TVM_ATTR_FIELD(target)
        .describe("The target describing how to compile for the virtual device.")
        .set_default(Target());
    TVM_ATTR_FIELD(memory_scope)
        .describe("The area of memory w.r.t. the virtual device where data is stored.")
        .set_default("");
  }

  friend class VirtualDevice;
};

/*!
 * \brief Managed reference class to \p VirtualDeviceNode.
 */
class VirtualDevice : public ObjectRef {
 public:
  /*!
   * \brief Construct a virtual device.
   * \param device_type_int The device type for the virtual device, or \p kInvalidDeviceType if
   * unconstrained.  If \p target is defined then must match its \p target->GetTargetDeviceType().
   * \param virtual_device_id The device id for the virtual device, or -1 if unconstrained.
   * \param target The target describing how to compile for the virtual device, or null if
   * unconstrained.
   * \param memory_scope The memory scope w.r.t. the virtual device which holds data, or "" if
   * unconstrained.
   */
  explicit VirtualDevice(int device_type_int = kInvalidDeviceType, int virtual_device_id = -1,
                         Target target = {}, MemoryScope memory_scope = {});

  /*! \brief Returns the unique fully unconstrained \p VirtualDevice. */
  static VirtualDevice FullyUnconstrained();

  /*!
   * \brief Returns the \p VirtualDevice for \p device_type and (if not -1) \p virtual_device_id.
   * The target and memory scope will be unconstrained.
   */
  static VirtualDevice ForDeviceType(DLDeviceType device_type, int virtual_device_id = -1) {
    ICHECK_GT(device_type, 0);
    return VirtualDevice(device_type, virtual_device_id);
  }
  static VirtualDevice ForDeviceType(int device_type, int virtual_device_id = -1) {
    return ForDeviceType(static_cast<DLDeviceType>(device_type), virtual_device_id);
  }
  static VirtualDevice ForDeviceType(const Integer& device_type, int virtual_device_id = -1) {
    return ForDeviceType(static_cast<int>(device_type->value), virtual_device_id);
  }

  /*! \brief Returns the \p VirtualDevice for \p device. */
  static VirtualDevice ForDevice(const Device& device) {
    return ForDeviceType(device.device_type, device.device_id);
  }

  /*! \brief Returns the \p VirtualDevice for \p device and \p target. */
  static VirtualDevice ForDeviceAndTarget(const Device& device, Target target) {
    return VirtualDevice(device.device_type, device.device_id, std::move(target));
  }

  /*! \brief Returns the \p VirtualDevice for \p target. */
  static VirtualDevice ForTarget(Target target) {
    DLDeviceType device_type = static_cast<DLDeviceType>(target->GetTargetDeviceType());
    return VirtualDevice(device_type, /*virtual_device_id=*/0, std::move(target));
  }

  /*! \brief Returns the \p VirtualDevice for \p memory_scope alone. */
  static VirtualDevice ForMemoryScope(MemoryScope memory_scope) {
    return VirtualDevice(kInvalidDeviceType, -1, {}, std::move(memory_scope));
  }

  /*! \brief Returns the \p VirtualDevice for \p device, \p target and \p memory_scope. */
  TVM_DLL static VirtualDevice ForDeviceTargetAndMemoryScope(const Device& device, Target target,
                                                             MemoryScope memory_scope) {
    return VirtualDevice(device.device_type, device.device_id, std::move(target),
                         std::move(memory_scope));
  }

  /*!
   * \brief Returns the 'join' of \p lhs and \p rhs. The result will agree pointwise with
   * \p lhs and \p rhs on all their constrained fields. Returns the null optional if no such
   * join exists, ie there's disagreement on at least one constrained field.
   */
  static Optional<VirtualDevice> Join(const VirtualDevice& lhs, const VirtualDevice& rhs);

  /*!
   * \brief Returns the 'default' of \p lhs and \p rhs. The result will be \p lhs, except any
   * unconstrained fields in \p lhs will take their value from \p rhs. Always well-defined.
   */
  static VirtualDevice Default(const VirtualDevice& lhs, const VirtualDevice& rhs);

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(VirtualDevice, ObjectRef, VirtualDeviceNode);

  friend class VirtualDeviceCache;  // Private implementation helper.
};

/*!
 * \brief A cache of \p VirtualDevices. This can be used:
 *  - To avoid ending up with lots of identical instances, since the space of VirtualDevices for any
 *    one compilation is very small but the number of points they need to be constructed can
 *    be very large (eg during device planning).
 *  - So we can assume \p VirtualDevices are pointer equal if and only if they are structurally
 * equal. This simplifies the unification of 'device domains' which are built on \p VirtualDevices.
 */
class VirtualDeviceCache {
 public:
  /*! \brief Returns the unique \p VirtualDevice representing given fields. */
  VirtualDevice Make(int device_type = kInvalidDeviceType, int virtual_device_id = -1,
                     Target target = {}, MemoryScope memory_scope = {});

  /*!
   * \brief Returns the unique \p VirtualDevice structurally equal to the given \p virtual_device.
   */
  VirtualDevice Unique(const VirtualDevice& virtual_device);

 private:
  /*! \brief Already constructed VirtualDevices. */
  std::unordered_set<VirtualDevice, StructuralHash, StructuralEqual> cache_;
};

/*! brief The attribute key for the virtual device. This key will be promoted to first class on
 * functions. For use in the parser and printer only.
 *
 * Type: VirtualDevice
 */
constexpr const char* kVirtualDevice = "virtual_device";

}  // namespace tvm

#endif  //  TVM_TARGET_VIRTUAL_DEVICE_H_
