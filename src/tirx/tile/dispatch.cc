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
#include <tvm/tirx/tile/dispatch.h>

namespace tvm::tirx::tile {

Function Dispatch(const Call& call, const DispatchContext& context) {
  static auto dispatcher = ffi::Function::GetGlobal("tirx.f_op_dispatcher");
  TVM_FFI_ICHECK(dispatcher.has_value())
      << "Internal Error: tirx.f_op_dispatcher is not registered";
  Function implementation = dispatcher.value()(call, context).cast<Function>();
  TVM_FFI_ICHECK(implementation.defined()) << "TIRx dispatcher did not return a Function";
  return implementation;
}

}  // namespace tvm::tirx::tile
