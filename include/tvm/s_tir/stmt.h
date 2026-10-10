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

#include <tvm/ffi/container/tuple.h>
#include <tvm/s_tir/iter_var.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {

/*! \brief Placement entries owned by a block, traversable with their buffer references. */
using BufferAllocatedAddresses = ffi::Array<ffi::Tuple<Var, ffi::Array<PrimExpr>>>;

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
  explicit MatchBufferRegionNode(ffi::UnsafeInit tag) : buffer(tag), source(tag) {}

  MatchBufferRegionNode(tirx::TensorVar buffer, TensorRegion source)
      : buffer(std::move(buffer)), source(std::move(source)) {}

  /*! \brief The target buffer. */
  tirx::TensorVar buffer;
  /*! \brief The source buffer region. */
  TensorRegion source;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<MatchBufferRegionNode>()
        .def_ro("buffer", &MatchBufferRegionNode::buffer,
                refl::AttachFieldFlag::SEqHashDefPattern())
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
  TVM_DLL explicit MatchBufferRegion(tirx::TensorVar buffer, TensorRegion source);

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
 *      buffer2 = T.alloc_tensor(shape, dtype)
 *      buffer3 = Ts.match_buffer(source_buffer[start:end, ...])
 *      Ts.sblock_attr({attr_key: attr_value, ...})
 *      with T.init():
 *          // init body
 *      // body
 *
 * \endcode
 */
class SBlockNode : public StmtNode {
 public:
  explicit SBlockNode(ffi::UnsafeInit tag) : body(tag) {}

  explicit SBlockNode(SeqStmt body) : body(std::move(body)) {}

  /*! \brief The variables of the block. */
  ffi::Array<s_tir::IterVar> iter_vars;
  /*! \brief The read buffer regions of the block. */
  ffi::Array<TensorRegion> reads;
  /*! \brief The write buffer regions of the block. */
  ffi::Array<TensorRegion> writes;
  /*! \brief The name_hint of the block. */
  ffi::String name_hint;
  /*! \brief The buffer allocated in the block. */
  ffi::Array<tirx::TensorVar> alloc_buffers;
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
  ffi::Optional<SeqStmt> init;
  /*! \brief The body of the block. */
  SeqStmt body;

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlock", SBlockNode, StmtNode);
};

/*!
 * \brief Managed reference to SBlockNode.
 * \sa SBlockNode
 */
class SBlock : public Stmt {
 public:
  TVM_DLL explicit SBlock(
      ffi::Array<s_tir::IterVar> iter_vars, ffi::Array<TensorRegion> reads,
      ffi::Array<TensorRegion> writes, ffi::String name_hint, SeqStmt body,
      ffi::Optional<SeqStmt> init = std::nullopt,
      ffi::Array<tirx::TensorVar> alloc_buffers = ffi::Array<tirx::TensorVar>(),
      ffi::Array<MatchBufferRegion> match_buffers = ffi::Array<MatchBufferRegion>(),
      ffi::Map<ffi::String, ffi::Any> annotations = ffi::Map<ffi::String, ffi::Any>(),
      Location loc = UnknownLoc());

  TVM_DLL explicit SBlock(ffi::String name_hint, SeqStmt body,
                          ffi::Array<tirx::TensorVar> alloc_buffers = ffi::Array<tirx::TensorVar>(),
                          Location loc = UnknownLoc());

  explicit SBlock(ffi::ObjectPtr<SBlockNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlock, Stmt, SBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockNode);
};

/*!
 * \brief A block realization node represents execution of the block at the binding values.
 */
class SBlockRealizeNode : public StmtNode {
 public:
  explicit SBlockRealizeNode(ffi::UnsafeInit tag) : predicate(tag), block(tag) {}

  SBlockRealizeNode(PrimExpr predicate, SBlock block)
      : predicate(std::move(predicate)), block(std::move(block)) {}

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
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.SBlockRealize", SBlockRealizeNode, StmtNode);
};

/*!
 * \brief Managed reference to BlockRealizeNode
 * \sa BlockRealizeNode
 */
class SBlockRealize : public Stmt {
 public:
  TVM_DLL explicit SBlockRealize(ffi::Array<PrimExpr> iter_values, PrimExpr predicate, SBlock block,
                                 Location loc = UnknownLoc());

  explicit SBlockRealize(ffi::ObjectPtr<SBlockRealizeNode> node) : Stmt(std::move(node)) {}

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SBlockRealize, Stmt, SBlockRealizeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SBlockRealizeNode);
};

/*! \brief Region marking copies eligible for asynchronous lowering. */
TVM_DLL const Op& async_copy_scope();
/*! \brief Commit the preceding asynchronous copies to a queue. */
TVM_DLL const Op& async_commit();
/*! \brief Wait until at most the given number of committed groups remain in flight. */
TVM_DLL const Op& async_wait();
/*! \brief Region whose author manages cross-thread synchronization explicitly. */
TVM_DLL const Op& manual_sync();

namespace attr {

/*!
 * \brief SBlock annotation selecting a write-buffer index for double buffering in
 * InjectSoftwarePipeline.
 */
constexpr const char* kDoubleBufferScope = "double_buffer_scope";

/*!
 * \brief String-valued allocation Call attribute containing the TensorCore fragment shape
 */
constexpr const char* kFragmentShape = "fragment_shape";

/*!
 * \brief String-valued allocation Call attribute containing the TensorCore fragment layout
 */
constexpr const char* kFragmentLayout = "fragment_layout";

/*!
 * \brief For annotation requesting partitioning when its PrimExpr value is provably true.
 */
constexpr const char* kLoopPartitionHint = "loop_partition_hint";

// -----------------------------------------------------------------------
// meta_schedule annotations
// -----------------------------------------------------------------------

/*! \brief Mark the tiling structure of blocks that are applied by rule Multi-Level-Tiling */
constexpr const char* kMetaScheduleTilingStructure = "meta_schedule.tiling_structure";

/*!
 * \brief Mark that the loop should be further skip and bound to environment threads to enable
 * cooperative fetching.
 */
constexpr const char* kMetaScheduleCooperativeFetch = "meta_schedule.cooperative_fetch";

/*! \brief The allowed range of thread extent in thread bindings */
constexpr const char* kMetaScheduleThreadExtentLowInclusive =
    "meta_schedule.thread_extent_low_inclusive";

/*! \brief The allowed range of thread extent in thread bindings */
constexpr const char* kMetaScheduleThreadExtentHighInclusive =
    "meta_schedule.thread_extent_high_inclusive";

/*! \brief Mark the block whose producer needs to be applied by rule Random-Compute-Location */
constexpr const char* kMetaScheduleRandomComputeProducer = "meta_schedule.random_compute_producer";

/*! \brief Mark auto-parallel setting on the block. */
constexpr const char* kMetaScheduleParallel = "meta_schedule.parallel";

/*! \brief Mark auto-vectorize setting on the block. */
constexpr const char* kMetaScheduleVectorize = "meta_schedule.vectorize";

/*! \brief Mark auto-unroll setting on the block. */
constexpr const char* kMetaScheduleUnrollExplicit = "meta_schedule.unroll_explicit";

/*! \brief Mark auto-unroll setting on the block. */
constexpr const char* kMetaScheduleUnrollImplicit = "meta_schedule.unroll_implicit";

/*! \brief Mark that a block should be further rewritten using tensorization. */
constexpr const char* kMetaScheduleAutoTensorize = "meta_schedule.auto_tensorize";

/*! \brief Mark that a block is a preprocessor block for layout rewrite. */
constexpr const char* kMetaScheduleLayoutRewritePreproc = "meta_schedule.layout_rewrite_preproc";

/*!
 * \brief Mark that the init statement of a block should be further rewritten using tensorization.
 */
constexpr const char* kMetaScheduleAutoTensorizeInit = "meta_schedule.auto_tensorize_init";

/*! \brief Mark that a block is disallowed in auto inline. */
constexpr const char* kMetaScheduleInlineRule = "meta_schedule.inline_rule";

// -----------------------------------------------------------------------
// Schedule primitive / SBlock annotations
// -----------------------------------------------------------------------

/*! \brief BufferAllocatedAddresses for block-owned allocations and match buffers. */
constexpr const char* kBufferAllocatedAddr = "s_tir.buffer_allocated_addr";

/*!
 * \brief Mark whether the script-completer need to fill in missing access region
 *        during script parsing.
 * \note The result should be a integer mask with range [0, 4).
 *       if (mask & 1) the read region should be detected,
 *       if (mask & 2) the write region should be detected.
 */
constexpr const char* kScriptParsingDetectAccess = "tirx.script_parsing_detect_access";

/*!
 * \brief Mark that the block need to add predicate for block var bounds during lowering
 */
constexpr const char* kRequireBlockVarBoundPredicate = "require_bound_predicate";

/*! \brief Mark the stage of a statement in the software pipeline */
constexpr const char* kSoftwarePipelineStage = "software_pipeline_stage";

/*! \brief Mark the order of a statement in the software pipeline */
constexpr const char* kSoftwarePipelineOrder = "software_pipeline_order";

/*! \brief List stages in the software pipeline that should run asynchronously
 * \note All statements in the provided stages are assumed to have asynchronous
 *       semantics (e.g. CUDA async global to shared memory copy).
 */
constexpr const char* kSoftwarePipelineAsyncStages = "software_pipeline_async_stages";

/*! \brief Mark the local stage for the shared memory access should be added. */
constexpr const char* kManifestSharedMemoryLocalStage = "tirx.manifest_shared_memory_local_stage";

/*!
 * \brief Mark alignment of buffer dimension
 *  The annotation value is an array of explicit tuples
 *  (buffer_index, axis, factor, offset).
 *  This requires the stride of an axis to be k * factor + offset.
 */
constexpr const char* kBufferDimAlign = "buffer_dim_align";

/*! \brief Mark that a block has an explicitly specified read region.
 * This is used to override the default read region inference in TIR.
 */
constexpr const char* kExplicitReadRegion = "explicit_read_region";

/*! \brief Mark that a block has an explicitly specified write region.
 * This is used to override the default write region inference in TIR.
 */
constexpr const char* kExplicitWriteRegion = "explicit_write_region";

/*! \brief ,ark a ForNode represent an irregular loop of non-structural control flow edges. */
constexpr const char* kIrregularLoopMark = "irregular_loop_mark";

/*! \brief Mark auto copy for memhammer */
constexpr const char* kAutoCopy = "auto_copy";

/*! \brief Mark local stage constraint on data copy */
constexpr const char* kLocalStage = "local_stage";

/*! \brief Mark vectorization length constraint on block */
constexpr const char* kVectorBytes = "vector_bytes";

/*!
 * \brief Mark that a block is executed by a warp. This implies the extend of threadIdx.x is
 * warp size.
 */
constexpr const char* kWarpExecution = "warp_execution";

constexpr const char* kPermutedLayout = "permuted_layout";

constexpr const char* kScheduleRule = "schedule_rule";

constexpr const char* kMetaScheduleWriteCacheLevel = "s_tir.meta_schedule.write_cache_level";

}  // namespace attr
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_STMT_H_
