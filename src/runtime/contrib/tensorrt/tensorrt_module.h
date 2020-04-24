/* * Licensed to the Apache Software Foundation (ASF) under one
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
 * \file runtime/contrib/tensorrt/tensorrt_module.h
 * \brief TensorRTModule is the runtime module for tensorrt backend.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_MODULE_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_MODULE_H_

#include <tvm/ir/module.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace runtime {

/*!
 * \brief Create a TensorRTModule.
 * \param serialized_subgraphs Function name -> Relay func serialized with
 * SaveJSON.
 * \return TensorRTModule created from subgraphs.
 */
Module TensorRTModuleCreate(
    const std::unordered_map<std::string, std::string>& serialized_subgraphs);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_MODULE_H_
