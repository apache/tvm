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
 * \brief Base for globally static objects referred to by the IR.
 */
#ifndef TVM_IR_GLOBAL_INFO_H_
#define TVM_IR_GLOBAL_INFO_H_

#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {

/*!
 * \brief GlobalInfo are globally static object that are referred by the IR itself.
 *        Base node for all global info that can appear in the IR
 */
class GlobalInfoNode : public ffi::Object {
 public:
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;

  TVM_FFI_DECLARE_OBJECT_INFO("ir.GlobalInfo", GlobalInfoNode, ffi::Object);
};

/*!
 * \brief Managed reference to GlobalInfoNode.
 * \sa GlobalInfoNode
 */
class GlobalInfo : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(GlobalInfo, ffi::ObjectRef, GlobalInfoNode);
};

}  // namespace tvm

#endif  // TVM_IR_GLOBAL_INFO_H_
