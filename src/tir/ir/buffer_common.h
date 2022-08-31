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
 * \file tir/ir/buffer_common.h
 * \brief Common utils for buffer access
 */
#ifndef TVM_TIR_IR_BUFFER_COMMON_H_
#define TVM_TIR_IR_BUFFER_COMMON_H_

#include <tvm/ir/type.h>
#include <tvm/runtime/data_type.h>

#include <optional>

namespace tvm {
namespace tir {

/*!
 * \brief Returns the type of object pointed to.
 *
 * \param type The type to be checked.
 *
 * \return An std::optional<DataType> object. If the type is a pointer
 * to a primitive type, the object has a value which is the pointed-to
 * type. Otherwise the object is nullopt.
 */
inline std::optional<runtime::DataType> GetPointerType(const Type& type) {
  if (type.defined()) {
    if (auto* ptr_type = type.as<PointerTypeNode>()) {
      if (auto* prim_type = ptr_type->element_type.as<PrimTypeNode>()) {
        return prim_type->dtype;
      }
    }
  }

  return std::nullopt;
}

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_BUFFER_COMMON_H_
