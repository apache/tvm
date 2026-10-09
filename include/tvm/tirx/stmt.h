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
/*! \file tvm/tirx/stmt.h
 * \brief TIRx statements and shared statement entry header.
 */
#ifndef TVM_TIRX_STMT_H_
#define TVM_TIRX_STMT_H_

#include <tvm/ir/stmt.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/expr.h>

namespace tvm {
namespace tirx {

/*! \brief Interpret and validate TIRx thread placement on a shared loop. */
inline ffi::Optional<ffi::String> GetThreadBinding(const ForNode* loop) {
  if (auto tag = loop->annotations.Get("thread_binding")) {
    TVM_FFI_CHECK(loop->kind == ForKind::kParallel, ValueError)
        << "thread_binding requires a parallel loop";
    TVM_FFI_CHECK(tag->as<ffi::String>().has_value(), TypeError)
        << "thread_binding annotation must be a string";
    TVM_FFI_CHECK(loop->HasTrivialStep(), ValueError) << "Thread binding loops require a unit step";
    return tag->cast<ffi::String>();
  }
  return std::nullopt;
}
inline ffi::Optional<ffi::String> GetThreadBinding(const For& loop) {
  return GetThreadBinding(loop.get());
}

/*!
 * \brief Standalone statement that declares a scope-id binding (e.g. cta_id,
 * warp_id, lane_id). Carries a ``ScopeIdDef`` value.
 *
 * Each declaration is a flat stmt within the device-region body. The declared
 * ``Var``\ s are visible in subsequent stmts in the same enclosing scope
 * (the ``tirx.device_entry`` region body), analogous to ``BindNode``.
 */
class ScopeIdDefStmtNode : public StmtNode {
 public:
  /*! \brief The scope-id definition (Vars + extents + binding). */
  ScopeIdDef def;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ScopeIdDefStmtNode>().def_ro("def", &ScopeIdDefStmtNode::def);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ScopeIdDefStmt", ScopeIdDefStmtNode, StmtNode);
};

/*! \brief Managed reference to ScopeIdDefStmtNode. */
class ScopeIdDefStmt : public Stmt {
 public:
  TVM_DLL ScopeIdDefStmt(ScopeIdDef def, Span span = Span());

  explicit ScopeIdDefStmt(ffi::ObjectPtr<ScopeIdDefStmtNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ScopeIdDefStmt, Stmt, ScopeIdDefStmtNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ScopeIdDefStmtNode);
};

/*! \brief Statement attribute and loop annotation keys. */
namespace attr {
/*!
 * \brief For annotation: maximum work for automatic unrolling.
 *
 * Integer policy inherited by nested loops unless they override it. Consumed by UnrollLoop.
 */
constexpr const char* auto_unroll_max_step = "auto_unroll_max_step";
/*!
 * \brief For annotation: expand unrolled bodies instead of preserving unrolled loops.
 *
 * Integer policy inherited by nested loops unless they override it. Consumed by UnrollLoop.
 */
constexpr const char* unroll_explicit = "unroll_explicit";
/*! \brief Annotation key on AllocTensor marking the allocation as volatile. */
constexpr const char* kVolatile = "tirx.volatile";
/*! \brief Mark buffer initial addr alignment in bytes */
constexpr const char* buffer_data_alignment = "buffer_data_alignment";
/*! \brief Mark buffer allocated addr in bytes */
constexpr const char* buffer_allocated_addr = "buffer_allocated_addr";

/*!
 * \brief Mark the kernel as persistent.
 */
constexpr const char* kPersistentKernel = "tirx.persistent_kernel";

}  // namespace attr

/*! \brief Whether stmt is undefined, an integer Evaluate, or an empty sequence. */
inline bool IsNoOp(const Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (const auto* op = stmt.as<EvaluateNode>()) return op->value.as<IntImmNode>() != nullptr;
  if (const auto* op = stmt.as<SeqStmtNode>()) return op->seq.empty();
  return false;
}

/*!
 * \brief TIRX TileOpCall stmt.
 */
class TileOpCallNode : public StmtNode {
 public:
  explicit TileOpCallNode(ffi::UnsafeInit tag) : op(tag) {}

  TileOpCallNode(tvm::Op op, ffi::Array<Expr> args, ffi::Map<ffi::String, TensorVar> workspace,
                 ffi::Map<ffi::String, Expr> config, ffi::Optional<ffi::String> dispatch,
                 ExecScope scope)
      : op(std::move(op)),
        args(std::move(args)),
        workspace(std::move(workspace)),
        config(std::move(config)),
        dispatch(std::move(dispatch)),
        scope(std::move(scope)) {}

  // tvm::Op which corresponds to the TIRX operator.
  tvm::Op op;

  // Arguments to the operator.
  ffi::Array<Expr> args;

  // Workspace (pre-allocated buffers) for the operator.
  ffi::Map<ffi::String, TensorVar> workspace;

  // Config for the operator/scheduler.
  ffi::Map<ffi::String, Expr> config;

  // Optional dispatch variant name registered via @register_dispatch.
  ffi::Optional<ffi::String> dispatch{std::nullopt};

  // Cooperation scope of this call. Default thread (an unscoped call).
  ExecScope scope = ExecScope(ScopeKind::kThread);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TileOpCallNode>()
        .def_ro("op", &TileOpCallNode::op)
        .def_ro("args", &TileOpCallNode::args)
        .def_ro("workspace", &TileOpCallNode::workspace)
        .def_ro("config", &TileOpCallNode::config)
        .def_ro("dispatch", &TileOpCallNode::dispatch)
        .def_ro("scope", &TileOpCallNode::scope);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TileOpCall", TileOpCallNode, StmtNode);
};

/*!
 * \brief Managed reference to TileOpCallNode
 * \sa TileOpCallNode
 */
class TileOpCall : public Stmt {
 public:
  TVM_DLL TileOpCall(tvm::Op op, ffi::Array<Expr> args,
                     ffi::Map<ffi::String, TensorVar> workspace = {},
                     ffi::Map<ffi::String, Expr> config = {},
                     ffi::Optional<ffi::String> dispatch = std::nullopt,
                     ExecScope scope = ExecScope(ScopeKind::kThread));

  explicit TileOpCall(ffi::ObjectPtr<TileOpCallNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TileOpCall, Stmt, TileOpCallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TileOpCallNode);
};

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIRX_STMT_H_
