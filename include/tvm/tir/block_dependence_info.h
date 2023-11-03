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
 * \file tvm/tir/block_dependence_info.h
 * \brief Define BlockDependenceInfoNode that uses the BlockScope and StmtSRef objects to
 * store the block level dependences
 * \sa BlockDependenceInfoNode
 */

/**
 * @brief An object that builds and maintains block scope and StmtSref mapping for Dependence
 * analysis
 */

#ifndef TVM_TIR_BLOCK_DEPENDENCE_INFO_H_
#define TVM_TIR_BLOCK_DEPENDENCE_INFO_H_

#include <tvm/tir/block_scope.h>

#include <unordered_map>

namespace tvm {
namespace tir {

/**
 * @brief An object that helps build and query block level dependences using the 2 core objects
 * BlockScope and StmtSRef
 *
 * The data structures exposed are:
 * 1) sref2scope: Mapping from the srefs to its corresponding BlockScope
 * 2) stmt2ref: Mapping from blocks to corresponding StmtSRefs
 *
 * Note that this object does not store SRefs to loops as the purpose is only to expose block level
 * dependences. This provides the advantage that the scope block (parent block) for a given block
 * sref can be directly accessed using the sref->parent member
 */
class BlockDependenceInfoNode : public Object {
 public:
  /*!
   * \brief Mapping from a block sref to its correpsonding BlockScope,
   * tracking the dependency inside the block scope,
   */
  std::unordered_map<StmtSRef, BlockScope, ObjectPtrHash, ObjectPtrEqual> sref2scope;
  /*! \brief The reverse mapping from block/for-loop to their corresponding srefs */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "tir.BlockDependenceInfo";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockDependenceInfoNode, Object);

  /*!
   * \brief Get the BlockScope correpsonding to the sref of scope root block
   * \param scope_root The block sref to be retrieved
   * \return The corresponding BlockScope
   */
  BlockScope GetBlockScope(const StmtSRef& scope_root) const {
    auto it = sref2scope.find(scope_root);
    CHECK(it != sref2scope.end())
        << "IndexError: Cannot find the corresponding BlockScope to the block sref:\n"
        << GetRef<Stmt>(scope_root->stmt);
    return it->second;
  }
};

/*!
 * \brief Managed reference to BlockDependenceInfoNode
 * \sa BlockDependenceInfo
 */
class BlockDependenceInfo : public ObjectRef {
  /*! \brief Construct an empty BlockDependenceInfo
   */
  TVM_DLL BlockDependenceInfo();

 public:
  /*! \brief Construct a BlockDependenceInfo from IRModule
   */
  TVM_DLL BlockDependenceInfo(IRModule mod);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockDependenceInfo, ObjectRef,
                                                    BlockDependenceInfoNode);
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_BLOCK_DEPENDENCE_INFO_H_
