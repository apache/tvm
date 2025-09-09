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
 * \file tvm/ir/global_info.h
 * \brief GlobalInfo are globally static object that are referred by the IR itself.
 */

#ifndef TVM_IR_GLOBAL_INFO_H_
#define TVM_IR_GLOBAL_INFO_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/target/target.h>

namespace tvm {

/*!
 * \brief Abstract label for an area of memory.
 */
using MemoryScope = ffi::String;

/*!
 * \brief GlobalInfo are globally static object that are referred by the IR itself.
 *        Base node for all global info that can appear in the IR
 */
class GlobalInfoNode : public Object {
 public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO("ir.GlobalInfo", GlobalInfoNode, Object);
};

/*!
 * \brief Managed reference to GlobalInfoNode.
 * \sa GlobalInfoNode
 */
class GlobalInfo : public ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GlobalInfo, ObjectRef, GlobalInfoNode);
};

/*!
 * \brief A global info subclass for virtual devices.
 */
class VDeviceNode : public GlobalInfoNode {
 public:
  /*! \brief The \p Target describing how to compile for the virtual device. */
  Target target;
  /*! \brief The device identifier for the virtual device. This enables us to
   * differentiate between distinct devices with same Target, such as multiple GPUs.
   */
  int vdevice_id;
  MemoryScope memory_scope;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VDeviceNode>()
        .def_ro("target", &VDeviceNode::target)
        .def_ro("vdevice_id", &VDeviceNode::vdevice_id)
        .def_ro("memory_scope", &VDeviceNode::memory_scope);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.VDevice", VDeviceNode, GlobalInfoNode);
};

/*!
 * \brief Managed reference to VDeviceNode.
 * \sa VDeviceNode
 */
class VDevice : public GlobalInfo {
 public:
  TVM_DLL explicit VDevice(Target tgt, int dev_id, MemoryScope mem_scope);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(VDevice, GlobalInfo, VDeviceNode);
};

/*!
 * \brief A dummy global info sub-class for testing purpose.
 */
class DummyGlobalInfoNode : public GlobalInfoNode {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DummyGlobalInfoNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.DummyGlobalInfo", DummyGlobalInfoNode, GlobalInfoNode);
};

/*!
 * \brief Managed reference to DummyGlobalInfoNode.
 * \sa DummyGlobalInfoNode
 */
class DummyGlobalInfo : public GlobalInfo {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DummyGlobalInfo, GlobalInfo, DummyGlobalInfoNode);
};

}  // namespace tvm

#endif  // TVM_IR_GLOBAL_INFO_H_
