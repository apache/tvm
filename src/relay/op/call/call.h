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
 * \file src/relay/op/call/call.h
 * \brief Operators for calling lowered functions.
 */
#ifndef TVM_RELAY_OP_CALL_CALL_H_
#define TVM_RELAY_OP_CALL_CALL_H_

#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr.h>

#include <utility>

namespace tvm {
namespace relay {

/*!
 * \brief Returns the Relay call_lowered op. Use this helper to avoid extraneous calls to
 * Registry::Get.
 */
const Op& CallLoweredOp();

/*!
 * \brief Helper to construct a Relay call with the "call_lowered" op.
 *
 * The callee must:
 *  - Be a global bound to a PrimFunc or an externally defined functions.
 *  - Accept only tensor arguments and return tensor results.
 *  - Arguments and results correspond to the flattened form (see FlattenTupleType) of the
 *    Relay Function type.
 *  - Return results by output pointer, ie use DPS.
 * The arguments remain in Relay form (ie not flattened).
 * The result remains in Relay form (ie returned from the call and not flattened).
 *
 * \param lowered_func Lowered function to call with call_lowered.
 * \param args Arguments to be passed to the function.
 * \param call_lowered_attrs Function attributes.
 * \param span TVM span for propagating debugging info.
 * \return
 */
Call CallLowered(GlobalVar lowered_func, Array<Expr> args, CallLoweredAttrs call_lowered_attrs,
                 Span span);

/*!
 * \brief Lowered function and the arguments to call it with.
 */
struct CallLoweredProps {
  /*! \brief Global variable pointing to the lowered function. */
  GlobalVar lowered_func;
  /*! \brief Array of the arguments to call lowered_func with. */
  Array<Expr> arguments;
  /*! \brief Attributes from the call_lowered op. */
  CallLoweredAttrs attrs;
};

/*!
 * \brief Helper to extract the lowered function and its arguments from a Call("call_lowered", ...).
 * Returns the null/empty \p CallLoweredProps if \p call_node is not in that form.
 */
CallLoweredProps GetCallLoweredProps(const CallNode* call_node);

/*!
 * \brief Returns \p call_node in 'standard' Relay form. Ie if \p call_node is a call_lowered
 * then returns it in un-lowered form, otherwise returns \p call_node directly.
 *
 * Useful for passes which can act uniformly on calls irrespective of their form.
 */
Call GetAnyCall(const CallNode* call_node);

/*!
 * \brief Returns true if lowered call described by \p props is to a reshape primitive.
 */
bool IsReshapeOnly(const CallLoweredProps& props);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_CALL_CALL_H_
