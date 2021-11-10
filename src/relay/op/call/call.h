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
 * \brief Helper to construct a Relay call with the call_lowered op.
 * \param func Lowered function to call with call_lowered.
 * \param inputs Arguments to be passed to the function.
 * \param attrs Function attributes, should be TIRCallAttrs.
 * \param type_args Type arguments for the call.
 * \param span TVM span for propogating debugging info.
 * \return
 */
Expr CallLowered(Expr func, Array<Expr> inputs, Attrs attrs, Array<Type> type_args, Span span);

/*!
 * \brief Returns the Relay call_lowered op. Use this helper to avoid extraneous calls to
 * Registry::Get.
 */
const Op& CallLoweredOp();

/*!
 * \brief Lowered function and the arguments to call it with.
 */
struct CallLoweredProps {
  /*! \brief Global variable pointing to the lowered function. */
  GlobalVar lowered_func;
  /*! \brief Array of the arguments to call lowered_func with. */
  Array<Expr> arguments;
  /*! \brief Arguments from the call_lowered op. */
  CallLoweredAttrs attrs;
};

/*!
 * \brief Helper to extract the lowered function and its arguments from Call("call_lowered", ...).
 * Will fail if called on a Call whose op is not "call_lowered" \param call_node CallNode that we
 * want to get the function and its arguments from.
 */
CallLoweredProps GetCallLoweredProps(const CallNode* call_node);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_CALL_CALL_H_
