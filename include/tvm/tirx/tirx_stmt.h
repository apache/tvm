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
 * \file tvm/tirx/tirx_op.h
 * \brief TIRX statements.
 */
#ifndef TVM_TIRX_TIRX_STMT_H_
#define TVM_TIRX_TIRX_STMT_H_

#include <tvm/ir/op.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tirx {

/*!
 * \brief TIRX TilePrimitiveCall stmt.
 */
class TilePrimitiveCallNode : public StmtNode {
 public:
  // tvm::Op which corresponds to the TIRX operator.
  tvm::Op op;

  // Arguments to the operator.
  ffi::Array<ffi::Any> args;

  // Workspace (pre-allocated buffers) for the operator.
  ffi::Map<ffi::String, Buffer> workspace;

  // Config for the operator/scheduler.
  ffi::Map<ffi::String, ffi::Any> config;

  // Optional dispatch variant name registered via @register_dispatch.
  ffi::Optional<ffi::String> dispatch{std::nullopt};

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TilePrimitiveCallNode>()
        .def_ro("op", &TilePrimitiveCallNode::op)
        .def_ro("args", &TilePrimitiveCallNode::args)
        .def_ro("workspace", &TilePrimitiveCallNode::workspace)
        .def_ro("config", &TilePrimitiveCallNode::config)
        .def_ro("dispatch", &TilePrimitiveCallNode::dispatch);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TilePrimitiveCall", TilePrimitiveCallNode, StmtNode);
};

/*!
 * \brief Managed reference to TilePrimitiveCallNode
 * \sa TilePrimitiveCallNode
 */
class TilePrimitiveCall : public Stmt {
 public:
  TVM_DLL TilePrimitiveCall(tvm::Op op, ffi::Array<ffi::Any> args,
                            ffi::Map<ffi::String, Buffer> workspace = {},
                            ffi::Map<ffi::String, ffi::Any> config = {},
                            ffi::Optional<ffi::String> dispatch = std::nullopt);

  static bool IsValidOpCallArgType(const ffi::Any& arg);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TilePrimitiveCall, Stmt, TilePrimitiveCallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TilePrimitiveCallNode);
};

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TIRX_STMT_H_
