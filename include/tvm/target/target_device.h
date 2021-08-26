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
 * \file tvm/target/target_device.h
 * \brief A compile time representation of a target device.
 *
 * This data structure consists of both the compiler target and a virtual device,
 * a tvm::Device where the the identifier is a virtual identifier and a concrete
 * device type.
 *
 * Executors are required to handle how to map virtual device identifiers to physical
 * device identifiers.
 *
 * The reason to introduce this data structure is that for much of compilation we
 * require understanding both of the target that we plan to compile the code for
 * as well as the concrete device which is used to initiate copies and other
 * device API actions.
 *
 * The idea is that we will carry around TargetDevice structures until device and
 * target planning at which time we can inject explicit virtual devices in the
 * program, and annotate explicit targets on the code to be generated.
 *
 * This will enable us to mix and match multiple devices of the same type with
 * different targets or compilation options, and eventually resolve to a phyical
 * set of devices with code specialized using the correct target.
 *
 * For example consider mobile SoCs which may contain two CPU types, a mobile GPU,
 * as well as NPU accelerator. It is important in these cases for us to be able to
 * correctly partition the code for each device type and apply different compilation
 * strategies.
 *
 * Today the compiler maps each device "type" to a single target, which does not work
 * when you have multiple types of CPUs, GPUs or accelerators attached.
 */
#ifndef TVM_TARGET_TARGET_DEVICE_H_
#define TVM_TARGET_TARGET_DEVICE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/node/node.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace tvm {

class TargetDevice;

/*!
 * \brief A representation of both the compile time and runtime data structure needed to represent a device.
 * \sa TargetDevice
 */
class TargetDeviceNode : public Object {
 public:
  /*! \brief The compilation target to use for the device.  */
  Target target;
  /*! \brief The virtual device, consisting of a virtual id which must be resolved to a physical one, and concrete device type. */
  Device virtual_device;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("target", &target);
    v->Visit("virtual_device", &virtual_device);
  }

  static constexpr const char* _type_key = "TargetDevice";
  TVM_DECLARE_FINAL_OBJECT_INFO(TargetDeviceNode, Object);
};

/*!
 * \brief Managed reference class to TargetDeviceNode.
 * \sa TargetDeviceNode
 *
 * This data structure consists of both the compiler target and a virtual device,
 * a tvm::Device where the the identifier is a virtual identifier and a concrete
 * device type.
 */
class Target : public ObjectRef {
 public:
  /*!
   * \brief Construct a TargetDevice.
   * \param target The target to compile for.
   * \param host The virtual device to execute on.
   * \return The TargetDevice.
   */
  TVM_DLL explicit TargetDevice(Target target, Device virtual_device);
  TVM_DEFINE_OBJECT_REF_METHODS(TargetDevice, ObjectRef, TargetDeviceNode);
};

}  // namespace tvm
#endif  // TVM_TARGET_TARGET_DEVICE_H_
