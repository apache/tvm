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
#include <tvm/target/target.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {
namespace usmp {

static constexpr const char* kTargetPoolReadWriteAccess = "rw";
static constexpr const char* kTargetPoolReadOnlyAccess = "ro";

/*!
 * \brief The pool information to be used by USMP
 */
struct PoolInfoNode : public Object {
  /*! \brief The name of the memory pool */
  String pool_name;
  /*! \brief The expected size hint to be used by the allocator.
   * The size_hint is defaulted to -1 to indicate the pool is not
   * size restricted.
   */
  Integer size_hint_bytes;
  /*! \brief The accessibility from each Target*/
  Map<Target, String> target_access;  // 'rw' or 'ro'

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pool_name", &pool_name);
    v->Visit("size_hint_bytes", &size_hint_bytes);
    v->Visit("target_access", &target_access);
  }

  bool SEqualReduce(const PoolInfoNode* other, SEqualReducer equal) const {
    return equal(pool_name, other->pool_name) && equal(size_hint_bytes, other->size_hint_bytes) &&
           equal(target_access, other->target_access);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(pool_name);
    hash_reduce(size_hint_bytes);
    hash_reduce(target_access);
  }

  static constexpr const char* _type_key = "tir.usmp.PoolInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(PoolInfoNode, Object);
};

class PoolInfo : public ObjectRef {
 public:
  TVM_DLL PoolInfo(String pool_name, Map<Target, String> target_access,
                   Integer size_hint_bytes = -1);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(PoolInfo, ObjectRef, PoolInfoNode);
};

/*!
 * \brief The buffer information to be used by USMP
 */
struct BufferInfoNode : public Object {
  /*! \brief The name of the buffer var */
  String name_hint;
  /*! \brief The size in terms of bytes */
  Integer size_bytes;
  /*! \brief The pool candidates that this buffer can get pooled to*/
  Array<PoolInfo> pool_candidates;
  /*! \brief The byte alignment required within the pool */
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
 * \brief Convert the IR-bound BufferInfo map to an array of BufferInfo
 *
 * \param buffer_info_map IR-bound BufferInfo map
 */
Array<BufferInfo> CreateArrayBufferInfo(const Map<Stmt, BufferInfo>& buffer_info_map);

static constexpr const char* kPoolCandidatesIRModAttr = "candidate_memory_pools";

}  // namespace usmp
}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_USMP_UTILS_H_
