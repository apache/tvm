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
 * \file tvm/tirx/op/region.h
 * \brief Region operations for TIRx.
 */
#ifndef TVM_TIRX_OP_REGION_H_
#define TVM_TIRX_OP_REGION_H_

#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Thread launch region: operands are a nonempty StringImm tag and a
 * scalar signed or unsigned integer extent wider than one bit.
 * The sole body parameter is a fresh thread-index PrimVar matching the extent type.
 * Tags starting with vthread denote virtual threads. There are no attrs or results.
 */
TVM_DLL const Op& launch_thread_op();

/*!
 * \brief Mark a user-facing device entry containing device scope definitions.
 * Takes no operands, body parameters, attributes or results.
 */
TVM_DLL const Op& device_entry_op();

/*!
 * \brief Supply lexical device context for allocation and packed-call lowering.
 * Operands are integer device type and device ID; there are no body parameters,
 * attributes or results. The region does not change the active runtime device.
 */
TVM_DLL const Op& device_context_op();

/*!
 * \brief Outline the body as a CPU compute helper named by a StringImm operand.
 * There are no body parameters, attributes or results.
 */
TVM_DLL const Op& compute_scope_op();

/*!
 * \brief Launch a CPU worker team around parallel loops and team barriers.
 * Takes no operands, body parameters, attributes or results.
 */
TVM_DLL const Op& parallel_launch_op();

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_OP_REGION_H_
