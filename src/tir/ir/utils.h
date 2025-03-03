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
 * \file tir/ir/utils.h
 * \brief Utilities for manipulating TIR
 */
#ifndef TVM_TIR_IR_UTILS_H_
#define TVM_TIR_IR_UTILS_H_

#include <tvm/tir/expr.h>

namespace tvm {
namespace tir {

/* \brief Normalize an ObjectRef held
 *
 * Where possible, the IR should be normalized contain IR types.  For
 * example, holding a `tir::IntImm` instead of a `runtime::Int`.  In
 * attributes, this is not always possible, as attributes may refer to
 * non-IR objects.
 *
 * This function normalizes any `runtime::Int`, `runtime::Bool`,
 * `runtime::Float`, or containers of those types to the corresponding
 * IR type.
 *
 * \param obj The attribute object to be normalized
 *
 * \returns The normalized attribute
 */
ObjectRef NormalizeAttributeObject(ObjectRef obj);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_IR_UTILS_H_
