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
  Op op = call->op.as_or_throw<Op>();
  TVM_FFI_CHECK(Op::HasAttrMap(op_attr::kTileOpDispatch), ValueError)
      << "No FTileOpDispatch registered for " << op->name;
  static const auto& dispatchers = Op::GetAttrMap(op_attr::kTileOpDispatch);
  TVM_FFI_CHECK(dispatchers.count(op), ValueError)
      << "No FTileOpDispatch registered for " << op->name;
  Function implementation = dispatchers[op](call, context);
  TVM_FFI_ICHECK(implementation.defined()) << "TIRx dispatcher did not return a Function";
  return implementation;
}

}  // namespace tvm::tirx::tile
