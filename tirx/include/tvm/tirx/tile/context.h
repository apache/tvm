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
 * \file tvm/tirx/tile/context.h
 * \brief Context and callbacks shared by tile operation dispatchers.
 */
#ifndef TVM_TIRX_TILE_CONTEXT_H_
#define TVM_TIRX_TILE_CONTEXT_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/tile/thread_scope.h>

namespace tvm {
namespace tirx {
namespace tile {

namespace callback {
/*! \brief The allocation Bind statements requested by the operator. */
inline constexpr ffi::reflection::AttrKey<ffi::Array<Bind>> kPrivateAlloc{"private_alloc"};
/*! \brief The initialization statement of the operator.
 *  which will be inserted at the beginning of the kernel
 */
inline constexpr ffi::reflection::AttrKey<ffi::Array<Stmt>> kDeviceInitStmt{"device_init_stmt"};
/*! \brief The initialization statement of the operator.
 *  which will be inserted at the beginning of the kernel
 */
inline constexpr ffi::reflection::AttrKey<ffi::Array<Stmt>> kHostInitStmt{"host_init_stmt"};
/*! \brief Statements to be inserted after a specific tensor's definition (DeclTensor/AllocTensor).
 *  Stored as Map<TensorVar, ffi::Array<Stmt>>.
 */
inline constexpr ffi::reflection::AttrKey<ffi::Map<TensorVar, ffi::Array<Stmt>>> kPostTensorDefStmt{
    "post_tensor_def_stmt"};
}  // namespace callback

/*!
 * \brief The context information of the kernel required by op dispatch.
 */
class DispatchContextObj : public ffi::Object {
 public:
  DispatchContextObj(Target target, ThreadScope thread_scope)
      : target(std::move(target)), thread_scope(std::move(thread_scope)) {}
  explicit DispatchContextObj(ffi::UnsafeInit)
      : target(ffi::UnsafeInit{}), thread_scope(ffi::UnsafeInit{}) {}

  /*! \brief The target of the kernel. */
  Target target;
  /*! \brief The thread scope of the operator */
  ThreadScope thread_scope;
  /*! \brief Hardware tag to (index variable, extent) launch parameters. */
  ffi::Map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params;
  /*! \brief A map from loop variables to their ranges. */
  ffi::Map<Var, Range> var_range_map;
  /*! \brief Whether the dispatch context is only used for tensor allocation. */
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
   * sites, etc.); dispatchers should fall back to thread_scope.local_name in that
   * case.
   */
  ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter;
  /*! \brief ExecContext intra-team view. Same encoding as ``inter``. */
  ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra;
  /*! \brief Scope kind string ("kernel"/"cta"/"warpgroup"/"warp"/"thread"/"cluster"). */
  ffi::String scope_kind;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<DispatchContextObj>()
        .def_ro("target", &DispatchContextObj::target)
        .def_ro("thread_scope", &DispatchContextObj::thread_scope)
        .def_ro("launch_params", &DispatchContextObj::launch_params)
        .def_ro("var_range_map", &DispatchContextObj::var_range_map)
        .def_ro("alloc_only", &DispatchContextObj::alloc_only)
        .def_ro("callbacks", &DispatchContextObj::callbacks)
        .def_ro("shared_state", &DispatchContextObj::shared_state)
        .def_ro("inter", &DispatchContextObj::inter)
        .def_ro("intra", &DispatchContextObj::intra)
        .def_ro("scope_kind", &DispatchContextObj::scope_kind);
  }

  /*! \brief Add a tensor to be allocated in the kernel. */
  void AddAllocTensor(TensorVar tensor, ffi::Array<PrimExpr> allocated_addr = {},
                      ffi::Map<ffi::String, ffi::Any> annotations = {});

  /*! \brief Add an initialization statement to be inserted. */
  void AddInitStmt(Stmt stmt, bool host = false);

  /*! \brief Add a statement to be inserted after a tensor's definition. */
  void AddPostTensorDefStmt(TensorVar tensor, Stmt stmt);

  /*! \brief Set a value in the shared state cache. */
  void SharedStateSet(ffi::String key, ffi::ObjectRef value);

  /*! \brief Get a value from the shared state cache. */
  ffi::Optional<ffi::ObjectRef> SharedStateGet(ffi::String key);

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.DispatchContext", DispatchContextObj, ffi::Object);
};

/*!
 * \brief Managed reference to DispatchContextObj.
 */
class DispatchContext : public ffi::ObjectRef {
 public:
  TVM_TIRX_DLL DispatchContext(
      Target target, ThreadScope thread_scope,
      ffi::Map<ffi::String, ffi::Tuple<PrimVar, PrimExpr>> launch_params = {},
      ffi::Map<Var, Range> var_range_map = {}, bool alloc_only = false,
      ffi::Map<ffi::String, ffi::ObjectRef> callbacks = {},
      ffi::Map<ffi::String, ffi::ObjectRef> shared_state = {},
      ffi::Map<ffi::String, ffi::Array<PrimExpr>> inter = {},
      ffi::Map<ffi::String, ffi::Array<PrimExpr>> intra = {}, ffi::String scope_kind = "");

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(DispatchContext, ffi::ObjectRef,
                                                DispatchContextObj);
};

}  // namespace tile
}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TIRX_TILE_CONTEXT_H_
