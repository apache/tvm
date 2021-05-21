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
 * \file src/runtime/contrib/verilator/verilator_kernel.h
 * \brief Use external verilator library kernels.
 */

#ifndef TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_KERNEL_H_

#include <tvm/runtime/c_runtime_api.h>

#include "verilator_device.h"

namespace tvm {
namespace runtime {
namespace contrib {

extern "C" TVM_DLL void verilator_add(VerilatorHandle handle, int* left, int* right, int* out,
                                      int p_h_, int p_w_);

extern "C" TVM_DLL void verilator_bias_add(VerilatorHandle handle, int* data, int* bias, int* out,
                                           int p_n_, int p_c_, int p_h_, int p_w_);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_VERILATOR_VERILATOR_KERNEL_H_
