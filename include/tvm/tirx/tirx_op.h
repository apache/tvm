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
 * \brief TIRX built-in operators.
 */
#ifndef TVM_TIRX_TIRX_OP_H_
#define TVM_TIRX_TIRX_OP_H_

#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tirx_stmt.h>

namespace tvm {
namespace tirx {

/*!
 * \brief The type of the function that sanitizes the arguments of a TIRX operator.
 * \param op The operator.
 * \param args The arguments.
 */
using FArgSanitizer = ffi::TypedFunction<void(tvm::Op, ffi::Array<ffi::ObjectRef>)>;

namespace callback {
/*! \brief The buffers allocated by the operator. */
constexpr const char* kPrivateAlloc = "private_alloc";
/*! \brief The initialization statement of the operator.
 *  which will be inserted at the beginning of the kernel
 */
constexpr const char* kDeviceInitStmt = "device_init_stmt";
/*! \brief The initialization statement of the operator.
 *  which will be inserted at the beginning of the kernel
 */
constexpr const char* kHostInitStmt = "host_init_stmt";
/*! \brief Statements to be inserted after a specific buffer's definition (DeclBuffer/AllocBuffer).
 *  Stored as Map<Buffer, ffi::Array<Stmt>>.
 */
constexpr const char* kPostBufferDefStmt = "post_buffer_def_stmt";
}  // namespace callback

/*!
 * \brief The context information of the kernel required by op dispatch.
 */
class DispatchContextNode : public ffi::Object {
 public:
  /*! \brief The target of the kernel. */
  Target target;
  /*! \brief The exec scope of the operator */
  ExecScope exec_scope;
  /*! \brief The kernel launch parameters. */
  ffi::Map<ffi::String, IterVar> launch_params;
  /*! \brief A map from loop variables to their ranges. */
  ffi::Map<Var, Range> var_range_map;
  /*! \brief Whether the dispatch context is only used for buffer allocation. */
  bool alloc_only;
  /*! \brief Callback to be handled when the operator is scheduled. */
  ffi::Map<ffi::String, ffi::ObjectRef> callbacks;
  /*! \brief Shared state that persists across dispatch calls within a single lowering pass. */
  ffi::Map<ffi::String, ffi::ObjectRef> shared_state;
  /*!
   * \brief ExecContext inter-team view at this op site.
   *
   * Maps axis name ("laneid"/"warpid"/"cta_id"/"wid_in_wg"/"wgid") to a
   * 2-element [extent, offset] PrimExpr array. Empty map = no ExecContext
   * tracking available (fallback for unresolved filters, pre-Phase-4 call
   * sites, etc.); dispatchers should fall back to exec_scope.name in that
   * case.
   */
  ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter;
  /*! \brief ExecContext intra-team view. Same encoding as ``inter``. */
  ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra;
  /*! \brief Scope kind string ("kernel"/"cta"/"warpgroup"/"warp"/"thread"/"cluster"). */
  ffi::String scope_kind;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DispatchContextNode>()
        .def_ro("target", &DispatchContextNode::target)
        .def_ro("exec_scope", &DispatchContextNode::exec_scope)
        .def_ro("launch_params", &DispatchContextNode::launch_params)
        .def_ro("var_range_map", &DispatchContextNode::var_range_map)
        .def_ro("alloc_only", &DispatchContextNode::alloc_only)
        .def_ro("callbacks", &DispatchContextNode::callbacks)
        .def_ro("shared_state", &DispatchContextNode::shared_state)
        .def_ro("inter", &DispatchContextNode::inter)
        .def_ro("intra", &DispatchContextNode::intra)
        .def_ro("scope_kind", &DispatchContextNode::scope_kind);
  }

  /*! \brief Add a buffer to be allocated in the kernel. */
  void AddAllocBuffer(Buffer buffer);

  /*! \brief Add an initialization statement to be inserted. */
  void AddInitStmt(Stmt stmt, bool host = false);

  /*! \brief Add a statement to be inserted after a buffer's definition. */
  void AddPostBufferDefStmt(Buffer buffer, Stmt stmt);

  /*! \brief Set a value in the shared state cache. */
  void SharedStateSet(ffi::String key, ffi::ObjectRef value);

  /*! \brief Get a value from the shared state cache. */
  ffi::Optional<ffi::ObjectRef> SharedStateGet(ffi::String key);

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.DispatchContext", DispatchContextNode, ffi::Object);
};

/*!
 * \brief Managed reference to DispatchContextNode.
 */
class DispatchContext : public ffi::ObjectRef {
 public:
  TVM_DLL DispatchContext(Target target, ExecScope exec_scope,
                          ffi::Map<ffi::String, IterVar> launch_params = {},
                          ffi::Map<Var, Range> var_range_map = {}, bool alloc_only = false,
                          ffi::Map<ffi::String, ffi::ObjectRef> callbacks = {},
                          ffi::Map<ffi::String, ffi::ObjectRef> shared_state = {},
                          ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter = {},
                          ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra = {},
                          ffi::String scope_kind = "");

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DispatchContext, ffi::ObjectRef, DispatchContextNode);
};

/*!
 * \brief See pesudo code below:
 *
 * Tx.cast(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& cast();

/*!
 * \brief See pesudo code below:
 *
 * Tx.copy(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& copy();

/*!
 * \brief See pesudo code below:
 *
 * Tx.Async.copy(BufferRegion dst, BufferRegion src)
 */
TVM_DLL const Op& copy_async();

/*!
 * \brief See pesudo code below:
 *
 *  Tx.fill(BufferRegion dst, PrimExpr value)
 */
TVM_DLL const Op& fill();

/*!
 * \brief See pesudo code below:
 *
 * Tx.gemm(Buffer A, Buffer B, Buffer C, Buffer D, PrimExpr alpha, PrimExpr beta)
 */
TVM_DLL const Op& gemm();

/*!
 * \brief See pesudo code below:
 *
 * Tx.gemm_async(BufferRegion C, BufferRegion A, BufferRegion B, bool transA, bool transB,
 * bool accum)
 */
TVM_DLL const Op& gemm_async();

TVM_DLL const Op& zero();

TVM_DLL const Op& sqrt();

TVM_DLL const Op& exp();

TVM_DLL const Op& add();

TVM_DLL const Op& sub();

TVM_DLL const Op& mul();

TVM_DLL const Op& fdiv();

TVM_DLL const Op& minimum();

TVM_DLL const Op& maximum();

TVM_DLL const Op& reciprocal();

TVM_DLL const Op& sum();

TVM_DLL const Op& max();

TVM_DLL const Op& min();

TVM_DLL const Op& memset();

TVM_DLL const Op& reduce_negate();

TVM_DLL const Op& binary_reduce();

TVM_DLL const Op& unary_reduce();

TVM_DLL const Op& binary_chain();

TVM_DLL const Op& select();

/*!
 * \brief See pesudo code below:
 *
 *  tvm_kernel_replace_point()
 */
TVM_DLL const Op& tvm_kernel_replace_point();

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TIRX_OP_H_
