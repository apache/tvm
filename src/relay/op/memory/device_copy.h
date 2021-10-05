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

namespace tvm {
namespace relay {

/*! \brief Returns the "device_copy" operator. */
const Op& DeviceCopyOp();

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated on
 * a device of type \p src_dev_type but then copied to a device of type \p dst_dev_type.
 */
Expr DeviceCopy(Expr expr, DLDeviceType src_dev_type, DLDeviceType dst_dev_type);

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated on
 * a device of type \p src_dev_type but then copied to a device of type \p dst_dev_type.
 * However, return \p expr directly if \p src_dev_type equals \p dst_dev_type.
 */
Expr MaybeDeviceCopy(Expr expr, DLDeviceType src_dev_type, DLDeviceType dst_dev_type);

/*! \brief Result of \p GetDeviceCopyProps. */
struct DeviceCopyProps {
  Expr body;  // = null
  DLDeviceType src_dev_type = kInvalidDeviceType;
  DLDeviceType dst_dev_type = kInvalidDeviceType;

  DeviceCopyProps() = default;

  DeviceCopyProps(const Expr& body, DLDeviceType srcDevType, DLDeviceType dstDevType)
      : body(body), src_dev_type(srcDevType), dst_dev_type(dstDevType) {}
};

/*!
 * \brief Returns the body expression, source, and destination device types for \p call_node if it
 * is a "device_copy" CallNode. Otherwise returns the null expression and \p kInvalidDeviceType
 * device types.
 */
DeviceCopyProps GetDeviceCopyProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, source, and destination device types for \p expr if it
 * is a "device_copy" CallNode. Otherwise returns the null expression and \p kInvalidDeviceType
 * device types.
 */
DeviceCopyProps GetDeviceCopyProps(const Expr& expr);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_
