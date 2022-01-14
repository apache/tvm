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
 * \file tir/usmp/utils.h
 * \brief Utilities for Unified Static Memory Planner
 */

#ifndef TVM_TIR_USMP_UTILS_H_
#define TVM_TIR_USMP_UTILS_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/target.h>
#include <tvm/tir/stmt.h>

namespace tvm {

/*!
 * \brief PassContext option to enable the USMP
 */
constexpr const char* kUSMPEnableOption = "tir.usmp.enable";
/*!
 * \brief PassContext option to select the memory planning algorithm in USMP
 */
constexpr const char* kUSMPAlgorithmOption = "tir.usmp.algorithm";

namespace tir {
namespace usmp {

/*!
 * \brief The string parameter to indicate read and write access to a pool
 * This needs to be kept in sync with PoolInfo.READ_WRITE_ACCESS in
 * python/tvm/tir/usmp/utils.py
 */
static constexpr const char* kTargetPoolReadWriteAccess = "rw";
/*!
 * \brief The string parameter to indicate read only access to a pool
 * This needs to be kept in sync with PoolInfo.READ_ONLY_ACCESS in
 * python/tvm/tir/usmp/utils.py
 */
static constexpr const char* kTargetPoolReadOnlyAccess = "ro";

/*!
 * \brief Describes a pool of memory accessible by one or more targets.
 */
struct PoolInfoNode : public Object {
  /*! \brief The name of the memory pool */
  String pool_name;
  /*! \brief The expected size hint to be used by the allocator.
   * The size_hint_bytes is defaulted to kUnrestrictedPoolSizeHint
   * to indicate the pool is not size restricted.
   */
  Integer size_hint_bytes;
  /*! \brief The accessibility from each Target*/
  Map<Target, String> target_access;  // 'rw' or 'ro'
  /*! \brief Whether pool is internally generated.
   * The internal pools will be generated as part of
   * the entry point code generation of the executor*/
  bool is_internal = false;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pool_name", &pool_name);
    v->Visit("size_hint_bytes", &size_hint_bytes);
    v->Visit("target_access", &target_access);
    v->Visit("is_internal", &is_internal);
  }

  bool SEqualReduce(const PoolInfoNode* other, SEqualReducer equal) const {
    return equal(pool_name, other->pool_name) && equal(size_hint_bytes, other->size_hint_bytes) &&
           equal(target_access, other->target_access) && equal(is_internal, other->is_internal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(pool_name);
    hash_reduce(size_hint_bytes);
    hash_reduce(target_access);
    hash_reduce(is_internal);
  }

  static constexpr const char* _type_key = "tir.usmp.PoolInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(PoolInfoNode, Object);
};

/*!
 * \brief The PoolSize is unrestricted for the memory planner
 */
static const int kUnrestrictedPoolSizeHint = -1;

class PoolInfo : public ObjectRef {
 public:
  TVM_DLL PoolInfo(String pool_name, Map<Target, String> target_access,
                   Integer size_hint_bytes = kUnrestrictedPoolSizeHint,
                   Bool is_internal = Bool(false));
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PoolInfo, ObjectRef, PoolInfoNode);
};

/*!
 * \brief Describes an abstract memory buffer that will get allocated inside a pool.
 * The actual memory buffer in represented by PoolAllocationNode after static memory planning.
 *
 * See also for relay-level counterparts:
 * relay::StorageToken (graph_plan_memory.cc)
 * relay::backend::StorageInfoNode (relay/backend/utils.h)
 * Region (python/tvm/relay/transform/memory_plan.py)
 */
struct BufferInfoNode : public Object {
  /*! \brief The name of the buffer var */
  String name_hint;
  /*! \brief The size in terms of bytes */
  Integer size_bytes;
  /*! \brief The pool candidates that this buffer can get pooled to*/
  Array<PoolInfo> pool_candidates;
  /*! \brief The byte alignment required for buffers that will placed within the pool */
  Integer alignment;
  /*! \brief The liveness conflicting other buffer info objects */
  Array<ObjectRef> conflicts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name_hint", &name_hint);
    v->Visit("size_bytes", &size_bytes);
    v->Visit("pool_candidates", &pool_candidates);
    v->Visit("alignment", &alignment);
    v->Visit("conflicts", &conflicts);
  }

  bool SEqualReduce(const BufferInfoNode* other, SEqualReducer equal) const {
    return equal(name_hint, other->name_hint) && equal(size_bytes, other->size_bytes) &&
           equal(pool_candidates, other->pool_candidates) && equal(alignment, other->alignment) &&
           equal(conflicts, other->conflicts);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name_hint);
    hash_reduce(size_bytes);
    hash_reduce(alignment);
    hash_reduce(conflicts);
    hash_reduce(pool_candidates);
  }
  /*!
   * \brief Set the liveness conflicts of this BufferInfo
   *
   * \param conflicting_buffer_info_objs An array of BufferInfo that conflicts in liveness
   */
  TVM_DLL void SetConflicts(Array<ObjectRef> conflicting_buffer_info_objs);

  static constexpr const char* _type_key = "tir.usmp.BufferInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(BufferInfoNode, Object);
};

class BufferInfo : public ObjectRef {
 public:
  TVM_DLL BufferInfo(String name_hint, Integer size_bytes, Array<PoolInfo> pool_candidates,
                     Integer alignment = runtime::kDefaultWorkspaceAlignment);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferInfo, ObjectRef, BufferInfoNode);
};

/*!
 * \brief This is a composite node that is produced by extract_buffer_info
 * analysis pass that contains useful global information that could be useful
 * for memory planning algorithms.
 */
struct BufferInfoAnalysisNode : public Object {
  /*! \brief The BufferInfo object and its associated TIR statement */
  Map<BufferInfo, tir::Stmt> buffer_info_stmts;
  /*! \brief This represent maximum amount of memory being used at
   * any point of time in the inference. This value is largely the
   * best allocation an algorithm could achieve. Due to
   * the complexities of conflict graphs, it would not be feasible
   * to achieve this value, practically. However, it can be useful
   * for iterative algorithms to know this value to define termination
   * criteria.*/
  Integer memory_pressure;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("buffer_info_stmts", &buffer_info_stmts);
    v->Visit("memory_pressure", &memory_pressure);
  }

  bool SEqualReduce(const BufferInfoAnalysisNode* other, SEqualReducer equal) const {
    return equal(buffer_info_stmts, other->buffer_info_stmts) &&
           equal(memory_pressure, other->memory_pressure);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(buffer_info_stmts);
    hash_reduce(memory_pressure);
  }
};

class BufferInfoAnalysis : public ObjectRef {
 public:
  TVM_DLL BufferInfoAnalysis(Map<BufferInfo, tir::Stmt> buffer_info_stmts, Integer memory_pressure);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(BufferInfoAnalysis, ObjectRef, BufferInfoAnalysisNode);
};

/*!
 * \brief The pool allocation produced after the USMP algorithm
 */
struct PoolAllocationNode : public Object {
  /*! \brief The assigned PoolInfo object */
  PoolInfo pool_info;
  /*! \brief The byte offset where the tensor is supposed to be placed within the pool*/
  Integer byte_offset;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pool_info", &pool_info);
    v->Visit("byte_offset", &byte_offset);
  }

  bool SEqualReduce(const PoolAllocationNode* other, SEqualReducer equal) const {
    return equal(pool_info, other->pool_info) && equal(byte_offset, other->byte_offset);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(pool_info);
    hash_reduce(byte_offset);
  }

  static constexpr const char* _type_key = "tir.usmp.PoolAllocation";
  TVM_DECLARE_FINAL_OBJECT_INFO(PoolAllocationNode, Object);
};

class PoolAllocation : public ObjectRef {
 public:
  TVM_DLL PoolAllocation(PoolInfo pool_info, Integer byte_offset);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PoolAllocation, ObjectRef, PoolAllocationNode);
};

/*!
 * \brief This object contains information post-allocation for PoolInfo objects
 */
struct AllocatedPoolInfoNode : public Object {
  /*! \brief The assigned PoolInfo object */
  PoolInfo pool_info;
  /*! \brief The allocated size into this pool */
  Integer allocated_size;
  /*! \brief An optional associated pool Var*/
  Optional<Var> pool_var;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pool_info", &pool_info);
    v->Visit("allocated_size", &allocated_size);
    v->Visit("pool_var", &pool_var);
  }

  bool SEqualReduce(const AllocatedPoolInfoNode* other, SEqualReducer equal) const {
    return equal(pool_info, other->pool_info) && equal(allocated_size, other->allocated_size) &&
           equal(pool_var, other->pool_var);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(pool_info);
    hash_reduce(allocated_size);
    hash_reduce(pool_var);
  }

  static constexpr const char* _type_key = "tir.usmp.AllocatedPoolInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocatedPoolInfoNode, Object);
};

class AllocatedPoolInfo : public ObjectRef {
 public:
  TVM_DLL AllocatedPoolInfo(PoolInfo pool_info, Integer allocated_size, Var pool_var = Var());
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AllocatedPoolInfo, ObjectRef, AllocatedPoolInfoNode);
};

/*!
 * \brief Convert the IR-bound BufferInfo map to an array of BufferInfo
 *
 * \param buffer_info_map IR-bound BufferInfo map
 */
Array<BufferInfo> CreateArrayBufferInfo(const Map<BufferInfo, Stmt>& buffer_info_map);

/*!
 * \brief Calculate workspace required to execute a IRModule with main expressed in TIR
 *
 * \param mod the IRModule with TIR-based main function
 */
Integer CalculateModuleWorkspaceSize(const IRModule& mod);

/*!
 * \brief The allocate node attribute to indicate candidate memory pools.
 * This needs to be kept in sync with CANDIDATE_MEMORY_POOL_ATTR in
 * python/tvm/tir/usmp/utils.py.
 */
static constexpr const char* kPoolCandidatesAllocateAttr = "candidate_memory_pools";

/*!
 * \brief Calculate the size of the extents in bytes
 *
 * \param op the allocate node
 */
Integer CalculateExtentsSize(const AllocateNode* op);

/*!
 * \brief Joins the Stmt nodes with PoolAllocation objects
 *
 * \param buffer_info_to_stmt the map of BufferInfo objects to Stmt nodes
 * \param buffer_info_to_pool_allocation the map of BufferInfo objects to PoolAllocation objects
 */
Map<Stmt, PoolAllocation> AssignStmtPoolAllocations(
    const Map<BufferInfo, Stmt>& buffer_info_to_stmt,
    const Map<BufferInfo, PoolAllocation>& buffer_info_to_pool_allocation);

}  // namespace usmp
}  // namespace tir

namespace attr {
/*!
 * \brief This is a BaseFunc attribute to indicate which input var represent
 * a PoolInfo Object in the form of a Map<Var, PoolInfo>.
 */
static constexpr const char* kPoolArgs = "pool_args";

/*!
 * \brief This is a IRModule attribute that contains all the PoolInfo objects
 * as an Array.
 */
static constexpr const char* kPoolInfoIRModuleAttr = "pool_infos";

}  // namespace attr

}  // namespace tvm

#endif  // TVM_TIR_USMP_UTILS_H_
