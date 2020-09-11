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
 * \file ethosn_device.h
 * \brief Ethos-N NPU device integration.
 */
#ifndef TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_DEVICE_H_
#define TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_DEVICE_H_

#include <vector>

#include "ethosn_support_library/Support.hpp"

namespace tvm {
namespace runtime {
namespace ethosn {

namespace sl = ::ethosn::support_library;

bool Inference(tvm::runtime::TVMArgs args, sl::CompiledNetwork* network,
               const std::vector<uint32_t>& input_order, const std::vector<uint32_t>& output_order);

}  // namespace ethosn
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_ETHOSN_ETHOSN_DEVICE_H_
