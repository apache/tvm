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
 * \file relay/op/memory/device_copy.h
 * \brief Helpers for working with "device_copy" attributes.
 */

#ifndef TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_
#define TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_

#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>

#include <utility>

#include "../call/call.h"

namespace tvm {
namespace relay {

/*! \brief Returns the "device_copy" operator. */
const Op& DeviceCopyOp();

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated and
 * stored at \p src_se_scope but then copied to \p dst_se_scope.
 */
Expr DeviceCopy(Expr expr, SEScope src_se_scope, SEScope dst_se_scope);

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated and
 * stored at \p src_se_scope but then copied to \p dst_se_scope.However, return \p expr
 * directly if \p src_se_scope and \p dst_se_scope are (structurally) the same.
 */
Expr MaybeDeviceCopy(Expr expr, SEScope src_se_scope, SEScope dst_se_scope);

/*! \brief Result of \p GetDeviceCopyProps. */
struct DeviceCopyProps {
  Expr body;  // = null
  SEScope src_se_scope = SEScope::FullyUnconstrained();
  SEScope dst_se_scope = SEScope::FullyUnconstrained();

  DeviceCopyProps() = default;

  DeviceCopyProps(Expr body, SEScope src_se_scope, SEScope dst_se_scope)
      : body(std::move(body)),
        src_se_scope(std::move(src_se_scope)),
        dst_se_scope(std::move(dst_se_scope)) {}
};

/*!
 * \brief Returns the body expression, source, and destination \p SEScopes for \p call_node
 * if it is a "device_copy" CallNode. Otherwise returns the null expression and unconstrained
 * device and scopes.
 */
DeviceCopyProps GetDeviceCopyProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, source, and destination \p SEScopes for \p expr if it
 * is a "device_copy" Call. Otherwise returns the null expression and unconstrained device and
 * scopes.
 */
DeviceCopyProps GetDeviceCopyProps(const Expr& expr);

/*!
 * \brief As for GetDeviceCopyProps, but for a lowered call rather than the original
 * "device_copy" operator.
 *
 * See te_compiler.cc for where this rewriting occurs.
 */
DeviceCopyProps GetLoweredDeviceCopyProps(const CallLoweredProps& props);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_
