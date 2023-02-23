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

#include "tvm/ir/expr.h"

namespace tvm {

/*!
 * \brief GlobalInfo are globally static object that are referred by the IR itself.
 *        Base node for all global info that can appear in the IR
 */
class GlobalInfoNode : public Object {
 public:
  static constexpr const char* _type_key = "GlobalInfoNode";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(GlobalInfoNode, Object);
};

/*!
 * \brief Managed reference to GlobalInfoNode.
 * \sa GlobalInfoNode
 */
class GlobalInfo : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(GlobalInfo, ObjectRef, GlobalInfoNode);
};

}  // namespace tvm

#endif  // TVM_IR_GLOBAL_INFO_H_