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
 * \file src/runtime/contrib/mrvl/mrvl_sw_runtime_lib.h
 * \brief Runtime library for Marvell Software Simulator
 */

#ifndef TVM_RUNTIME_CONTRIB_MRVL_MRVL_SW_RUNTIME_LIB_H_
#define TVM_RUNTIME_CONTRIB_MRVL_MRVL_SW_RUNTIME_LIB_H_

#include <tvm/runtime/packed_func.h>

#include <cstddef>
#include <string>

namespace tvm {
namespace runtime {
namespace contrib {
namespace mrvl {

void RunMarvellSimulator(tvm::runtime::TVMArgs args, const std::string& symbol_name,
                         const std::string& bin_code, size_t num_inputs, size_t num_outputs);
}
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MRVL_MRVL_SW_RUNTIME_LIB_H_
