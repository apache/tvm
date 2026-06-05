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
 * \file tir/tirx_stmt.cc
 * TIRX statement nodes.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tirx_op.h>

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() { TilePrimitiveCallNode::RegisterReflection(); }

// TilePrimitiveCall
TilePrimitiveCall::TilePrimitiveCall(tvm::Op op, ffi::Array<ffi::Any> args,
                                     ffi::Map<ffi::String, Buffer> workspace,
                                     ffi::Map<ffi::String, ffi::Any> config,
                                     ffi::Optional<ffi::String> dispatch, ExecScope scope) {
  // Check if the op is a TIRX op.
  static const auto& tirx_op_map = Op::GetAttrMap<bool>("TIsTIRxOp");
  TVM_FFI_ICHECK_EQ(tirx_op_map.count(op), 1)
      << "Only TIRX ops can be used in tirx::TilePrimitiveCall";
  // Construct the TilePrimitiveCall.
  ffi::ObjectPtr<TilePrimitiveCallNode> n = ffi::make_object<TilePrimitiveCallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->workspace = std::move(workspace);
  n->config = std::move(config);
  n->dispatch = std::move(dispatch);
  n->scope = std::move(scope);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.TilePrimitiveCall",
      [](tvm::Op op, ffi::Array<ffi::Any> args, ffi::Map<ffi::String, Buffer> workspace,
         ffi::Map<ffi::String, ffi::Any> config, ffi::Optional<ffi::String> dispatch,
         ExecScope scope) {
        return TilePrimitiveCall(op, args, workspace, config, dispatch, scope);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TilePrimitiveCallCopyHandle",
                        [](const TilePrimitiveCall& op) { return TilePrimitiveCall(op); });
}

}  // namespace tirx
}  // namespace tvm
