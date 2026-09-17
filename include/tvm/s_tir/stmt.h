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
 * \file tvm/s_tir/stmt.h
 * \brief S-TIR (Schedulable TIR) statements and attributes.
 *
 * This file contains attribute keys that are specific to the schedulable TIR
 * (S-TIR) layer, including meta_schedule annotations and schedule primitive /
 * SBlock annotations.
 */
#ifndef TVM_S_TIR_STMT_H_
#define TVM_S_TIR_STMT_H_

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {

/*!
 * \brief Match introduces a constraint that the source buffer region can be remapped to the data
 * layout specified by the buffer field. The constraint can be checked in later part of lowering (or
 * optionally during runtime).
 *
 * MatchBufferRegion provides a mechanism to represent data layout and compactness constraints in
 * low-level hardware primitives in the IR and defer the check after the sequence of
 * transformations.
 */
class MatchBufferRegionNode : public ffi::Object {
 public:
  /*! \brief The target buffer. */
  tirx::BufferVar buffer;
  /*! \brief The source buffer region. */
  TensorRegion source;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchBufferRegionNode>()
        .def_ro("buffer", &MatchBufferRegionNode::buffer, refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("source", &MatchBufferRegionNode::source);
  }

  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind = kTVMFFISEqHashKindTreeNode;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.MatchBufferRegion", MatchBufferRegionNode, ffi::Object);
};

/*!
 * \brief Managed reference to MatchBufferRegionNode.
 * \sa MatchBufferRegionNode
 */
class MatchBufferRegion : public ffi::ObjectRef {
 public:
  TVM_DLL explicit MatchBufferRegion(tirx::BufferVar buffer, TensorRegion source);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(MatchBufferRegion, ffi::ObjectRef,
                                             MatchBufferRegionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchBufferRegionNode);
};

/*!
 * \brief A block is a basic schedule unit in TIR.
 * \note SBlock's body is parameterized by iter vars.
 * \code
 *
 *  with T.sblock(name):
 *      v0 = T.axis.S(domain, value0)
 *      v1 = T.axis.R(domain, value1)
 *      ...
 *      T.reads([buffer0[start:end, ...], ...])
 *      T.writes([buffer1[start:end, ...], ...])
 *      T.where(predicate)
 *      buffer2 = T.alloc_buffer(shape, dtype)
 *      buffer3 = T.match_buffer(source_buffer[start:end, ...])
 *      T.attr({attr_key: attr_value, ...})
 *      with T.init():
 *          // init body
 *      // body
 *
 * \endcode
 */
class SBlockNode : public tirx::StmtNode {
 public:
  /*! \brief The variables of the block. */
  ffi::Array<tirx::IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  ffi::Array<TensorRegion> reads;
  /*! \brief The write buffer regions of the block. */
  ffi::Array<TensorRegion> writes;
  /*! \brief The name_hint of the block. */
  ffi::String name_hint;
  /*! \brief The buffer allocated in the block. */
  ffi::Array<tirx::BufferVar> alloc_buffers;
  /*! \brief The match buffer regions. */
  ffi::Array<MatchBufferRegion> match_buffers;
  /*! \brief The annotation of the block. */
  ffi::Map<ffi::String, ffi::Any> annotations;
  /*!
   * \brief The init statement is executed during the first iteration of reduction loops in a
   *  reduction block. The optional init field allows us to represent initialization and
   *  reduction update in a single block and transform them collectively.
   *  We also provide primitives to decompose the init into a separate block during scheduling.
   *  Init field is `std::nullopt` if there is no reduction iter_vars
   */
  ffi::Optional<tirx::Stmt> init;
  /*! \brief The body of the block. */
  tirx::Stmt body;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockNode>()
        .def_ro("iter_vars", &SBlockNode::iter_vars)
        .def_ro("reads", &SBlockNode::reads)
        .def_ro("writes", &SBlockNode::writes)
        .def_ro("name_hint", &SBlockNode::name_hint, refl::AttachFieldFlag::SEqHashIgnore())
        .def_ro("alloc_buffers", &SBlockNode::alloc_buffers,
                refl::AttachFieldFlag::SEqHashDefSimple())
        .def_ro("match_buffers", &SBlockNode::match_buffers)
        .def_ro("annotations", &SBlockNode::annotations)
        .def_ro("init", &SBlockNode::init)
        .def_ro("body", &SBlockNode::body);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlock", SBlockNode, tirx::StmtNode);
};

/*!
 * \brief Managed reference to SBlockNode.
 * \sa SBlockNode
 */
class SBlock : public tirx::Stmt {
 public:
  TVM_DLL explicit SBlock(
      ffi::Array<tirx::IterVar> iter_vars, ffi::Array<TensorRegion> reads,
      ffi::Array<TensorRegion> writes, ffi::String name_hint, tirx::Stmt body,
      ffi::Optional<tirx::Stmt> init = std::nullopt,
      ffi::Array<tirx::BufferVar> alloc_buffers = ffi::Array<tirx::BufferVar>(),
      ffi::Array<MatchBufferRegion> match_buffers = ffi::Array<MatchBufferRegion>(),
      ffi::Map<ffi::String, ffi::Any> annotations = ffi::Map<ffi::String, ffi::Any>(),
      Span span = Span());

  TVM_DLL explicit SBlock(ffi::String name_hint, tirx::Stmt body,
                          ffi::Array<tirx::BufferVar> alloc_buffers = ffi::Array<tirx::BufferVar>(),
                          Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SBlock, tirx::Stmt, SBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockNode);
};

/*!
 * \brief A block realization node represents execution of the block at the binding values.
 */
class SBlockRealizeNode : public tirx::StmtNode {
 public:
  /*! \brief The corresponding values of the iter vars. */
  ffi::Array<PrimExpr> iter_values;
  /*!
   * \brief The predicate of the block realization, the block will only be executed when the
   * predicate is true.
   */
  PrimExpr predicate;
  /*! \brief The block to be realized. */
  SBlock block;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SBlockRealizeNode>()
        .def_ro("iter_values", &SBlockRealizeNode::iter_values)
        .def_ro("predicate", &SBlockRealizeNode::predicate)
        .def_ro("block", &SBlockRealizeNode::block);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlockRealize", SBlockRealizeNode, tirx::StmtNode);
};

/*!
 * \brief Managed reference to BlockRealizeNode
 * \sa BlockRealizeNode
 */
class SBlockRealize : public tirx::Stmt {
 public:
  TVM_DLL explicit SBlockRealize(ffi::Array<PrimExpr> iter_values, PrimExpr predicate, SBlock block,
                                 Span span = Span());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(SBlockRealize, tirx::Stmt, SBlockRealizeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockRealizeNode);
};

namespace attr {

/*!
 * \brief Annotations for invoking and synchronizing asynchronous operations.
 */
constexpr const char* async_commit_queue_scope = "async_commit_queue_scope";
constexpr const char* async_wait_queue_scope = "async_wait_queue_scope";
constexpr const char* async_wait_inflight_count = "async_wait_inflight_count";

/*!
 * \brief Mark that the attached statement runs asynchronously.
 */
constexpr const char* async_scope = "async_scope";

/*! \brief Mark stores/loads with their bounds. */
constexpr const char* buffer_bound = "buffer_bound";

/*!
 * \brief Marks production of double buffer data
 */
constexpr const char* double_buffer_scope = "double_buffer_scope";

/*!
 * \brief Marks region used by double buffer write
 */
constexpr const char* double_buffer_write = "double_buffer_write";

/*!
 * \brief Mark that the shape of TensorCore fragment
 */
constexpr const char* fragment_shape = "fragment_shape";

/*!
 * \brief Mark that the layout of TensorCore fragment
 */
constexpr const char* fragment_layout = "fragment_layout";

/*!
 * \brief Mark that the loop should be partitioned.
 */
constexpr const char* pragma_loop_partition_hint = "pragma_loop_partition_hint";

/*! \brief Mark of reduce scope */
constexpr const char* reduce_scope = "reduce_scope";

/*! \brief Mark launching of a virtual thread. */
constexpr const char* virtual_thread = "virtual_thread";

// -----------------------------------------------------------------------
// meta_schedule annotations
// -----------------------------------------------------------------------

/*! \brief Mark the tiling structure of blocks that are applied by rule Multi-Level-Tiling */
constexpr const char* meta_schedule_tiling_structure = "meta_schedule.tiling_structure";

/*!
 * \brief Mark that the loop should be further skip and bound to environment threads to enable
 * cooperative fetching.
 */
constexpr const char* meta_schedule_cooperative_fetch = "meta_schedule.cooperative_fetch";

/*! \brief The allowed range of thread extent in thread bindings */
constexpr const char* meta_schedule_thread_extent_low_inclusive =
    "meta_schedule.thread_extent_low_inclusive";

/*! \brief The allowed range of thread extent in thread bindings */
constexpr const char* meta_schedule_thread_extent_high_inclusive =
    "meta_schedule.thread_extent_high_inclusive";

/*! \brief Mark the block whose producer needs to be applied by rule Random-Compute-Location */
constexpr const char* meta_schedule_random_compute_producer =
    "meta_schedule.random_compute_producer";

/*! \brief Mark auto-parallel setting on the block. */
constexpr const char* meta_schedule_parallel = "meta_schedule.parallel";

/*! \brief Mark auto-vectorize setting on the block. */
constexpr const char* meta_schedule_vectorize = "meta_schedule.vectorize";

/*! \brief Mark auto-unroll setting on the block. */
constexpr const char* meta_schedule_unroll_explicit = "meta_schedule.unroll_explicit";

/*! \brief Mark auto-unroll setting on the block. */
constexpr const char* meta_schedule_unroll_implicit = "meta_schedule.unroll_implicit";

/*! \brief Mark that a block should be further rewritten using tensorization. */
constexpr const char* meta_schedule_auto_tensorize = "meta_schedule.auto_tensorize";

/*! \brief Mark that a block is a preprocessor block for layout rewrite. */
constexpr const char* meta_schedule_layout_rewrite_preproc = "meta_schedule.layout_rewrite_preproc";

/*!
 * \brief Mark that the init statement of a block should be further rewritten using tensorization.
 */
constexpr const char* meta_schedule_auto_tensorize_init = "meta_schedule.auto_tensorize_init";

/*! \brief Mark that a block is disallowed in auto inline. */
constexpr const char* meta_schedule_inline_rule = "meta_schedule.inline_rule";

// -----------------------------------------------------------------------
// Schedule primitive / SBlock annotations
// -----------------------------------------------------------------------

/*!
 * \brief Mark whether the script-completer need to fill in missing access region
 *        during script parsing.
 * \note The result should be a integer mask with range [0, 4).
 *       if (mask & 1) the read region should be detected,
 *       if (mask & 2) the write region should be detected.
 */
constexpr const char* script_parsing_detect_access = "tirx.script_parsing_detect_access";

/*!
 * \brief Mark that the block need to add predicate for block var bounds during lowering
 */
constexpr const char* require_block_var_bound_predicate = "require_bound_predicate";

/*! \brief Mark the stage of a statement in the software pipeline */
constexpr const char* software_pipeline_stage = "software_pipeline_stage";

/*! \brief Mark the order of a statement in the software pipeline */
constexpr const char* software_pipeline_order = "software_pipeline_order";

/*! \brief List stages in the software pipeline that should run asynchronously
 * \note All statements in the provided stages are assumed to have asynchronous
 *       semantics (e.g. CUDA async global to shared memory copy).
 */
constexpr const char* software_pipeline_async_stages = "software_pipeline_async_stages";

/*! \brief Mark the buffers which is const access and can be transformed layout. */
constexpr const char* layout_free_buffers = "layout_free_buffers";

/*! \brief Mark the local stage for the shared memory access should be added. */
constexpr const char* manifest_shared_memory_local_stage =
    "tirx.manifest_shared_memory_local_stage";

/*!
 * \brief Mark alignment of buffer dimension
 *  stmt.node is Tensor
 *  stmt.value is tvm_tuple(dim, align, offset)
 *  This gives hint to require stride of dim to be k * align + offset.
 */
constexpr const char* buffer_dim_align = "buffer_dim_align";

/*! \brief Mark that a block has an explicitly specified read region.
 * This is used to override the default read region inference in TIR.
 */
constexpr const char* explicit_read_region = "explicit_read_region";

/*! \brief Mark that a block has an explicitly specified write region.
 * This is used to override the default write region inference in TIR.
 */
constexpr const char* explicit_write_region = "explicit_write_region";

/*! \brief ,ark a ForNode represent an irregular loop of non-structural control flow edges. */
constexpr const char* irregular_loop_mark = "irregular_loop_mark";

/*! \brief Mark auto copy for memhammer */
constexpr const char* auto_copy = "auto_copy";

/*! \brief Mark local stage constraint on data copy */
constexpr const char* local_stage = "local_stage";

/*! \brief Mark vectorization length constraint on block */
constexpr const char* vector_bytes = "vector_bytes";

/*!
 * \brief Mark that a block is executed by a warp. This implies the extend of threadIdx.x is
 * warp size.
 */
constexpr const char* warp_execution = "warp_execution";

/*!
 * \brief Marks the layout transforms to be used for a tensor.
 *
 * Only applies to a tensor-like input, as it should be made part of the
 * PrimFunc attributes for TIR.
 */
constexpr const char* layout_transforms = "layout_transforms";

/*!
 * \brief Mark that the kernel is hand threaded and doesn't need syncs inserted
 */
constexpr const char* hand_threaded = "hand_threaded";

}  // namespace attr
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_STMT_H_
