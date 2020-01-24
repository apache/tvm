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
 * \file src/relay/backend/contrib/tensorrt/common_utils.h
 * \brief Utility functions used by compilation and runtime.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_TENSORRT_COMMON_UTILS_H_
#define TVM_RELAY_BACKEND_CONTRIB_TENSORRT_COMMON_UTILS_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief Extract the shape from a Relay tensor type.
 *
 * \param type The provided type.
 *
 * \return The extracted shape in a list.
 */
std::vector<int> GetShape(const Type& type);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_TENSORRT_COMMON_UTILS_H_
