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
 * \file tvm/tir/schedule/block_scope.h
 * \brief Definition of two pillar data structure for TensorIR scheduling: StmtSRef, BlockScope.
 * \sa StmtSRefNode
 * \sa BlockScopeNode
 */
#ifndef TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
#define TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_

#include <tvm/tir/stmt.h>

#include <unordered_map>

namespace tvm {
namespace tir {

/*!
 * \brief An object that refers to schedulable elements (block/for-loop) in TensorIR, aka "sref".
 *
 * Glossary
 * - Block sref: A StmtSRef that points to a TensorIR block.
 * - Loop sref: A StmtSRef that points to a TensorIR for loop.
 * - Parent sref: The parent reference of an sref is the block or loop reference to the closest
 schedulable statement. We define closest to be the nearest schedulable statement of an ancestor in
 the AST.
 * schedulable statement of its ancestors on the TensorIR AST.
 * - Root sref: Sref to the root block. Every sref has exactly one parent sref except for root sref.
 * - Sref tree: The parent-children-relationship of srefs that forms a tree, uniquely determined by
 * the TensorIR AST.
 */
class StmtSRefNode : public Object {
 public:
  /*!
   * \brief The block or `for` stmt the object refers to
   * \note Non-owned reference (raw pointer) is used here, so that we can perform copy-on-write
   * optimization on statements when possible. The strong reference is held in the ScheduleState.
   */
  const StmtNode* stmt;
  /*! \brief The parent sref. */
  StmtSRefNode* parent;
  /*!
   * \brief If the statement the sref points to is an element of a SeqStmt in the AST,
   * then `seq_index` is set to its index; otherwise `seq_index` is -1
   */
  int64_t seq_index;

  void VisitAttrs(AttrVisitor* v) {
    // `stmt` is not visited
    // `parent` is not visited
    v->Visit("seq_index", &seq_index);
  }

  static constexpr const char* _type_key = "tir.StmtSRef";
  TVM_DECLARE_FINAL_OBJECT_INFO(StmtSRefNode, Object);

  /*! \brief Reset the object inplace to the invalid state */
  void Reset() {
    this->stmt = nullptr;
    this->parent = nullptr;
    this->seq_index = -1;
  }

  /*!
   * \brief Get the referenced statement with proper type checking.
   * It serves the same purpose as `ObjectRef::as`, but does not acquire strong reference to `stmt`
   * \tparam StmtType The type that `this->stmt` to be downcasted to. Presumably
   * tvm::tir::BlockNode or tvm::tir::ForNode
   * \return nullptr if type check fails, otherwise the casted result for `this->stmt`
   */
  template <typename StmtType>
  const StmtType* StmtAs() const {
    if (stmt != nullptr && stmt->IsInstance<StmtType>()) {
      return static_cast<const StmtType*>(stmt);
    } else {
      return nullptr;
    }
  }
};

/*!
 * \brief Managed reference to StmtSRefNode
 * \sa StmtSRefNode
 */
class StmtSRef : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param stmt The corresponding stmt node, can be either block or for loop.
   * \param parent The parent sref.
   * \param seq_index The location in an array if the parent of the stmt contains multiple children.
   * -1 if the parent does not contain multiple children.
   */
  TVM_DLL explicit StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index);

  /*! \return The mutable pointer to the StmtSRefNode */
  StmtSRefNode* get() const { return static_cast<StmtSRefNode*>(data_.get()); }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StmtSRef, ObjectRef, StmtSRefNode);

 public:
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do the work of compute-inline
   * \note This is only as a faked loop sref for compute-at and reverse-compute-at,
   * i.e.
   *
   * compute-at(block, loop_sref):
   *   compute-inline(block)                if loop_sref.same_as(InlineMark())
   *   no-op                                if loop_sref.same_as(RootMark())
   *   compute-at-impl(block, loop_sref)    otherwise
   */
  TVM_DLL static StmtSRef InlineMark();
  /*!
   * \return A special StmtSRef, which doesn't point to any stmt in the AST,
   * only serving as a "mark" to hint compute-at to do nothing
   * \note This is only as a faked loop sref for compute-at and reverse-compute-at,
   * i.e.
   *
   * compute-at(block, loop_sref):
   *   compute-inline(block)                if loop_sref.same_as(InlineMark())
   *   no-op                                if loop_sref.same_as(RootMark())
   *   compute-at-impl(block, loop_sref)    otherwise
   */
  TVM_DLL static StmtSRef RootMark();
};

/*!
 * \brief Type of dependency. Right now we have 4 types of dependencies
 * 1) Read-after-write (kRAW)
 * 2) Write-after-write (kWAW)
 * 3) Write-after-read (kWAR)
 * 4) Opaque dependency (kOpaque)
 */
enum class DepKind : int32_t {
  kRAW = 0,
  kWAW = 1,
  kWAR = 2,
  kOpaque = 3,
};

/*!
 * \brief A tuple (src, dst, kind) representing certain types of dependency.
 * For example, (A, B, kRAW) means block B depends on block A, and the dependency kind is
 * read-after-write, which means block B reads the result written by block A.
 */
class DependencyNode : public Object {
 public:
  /*! \brief The source of the dependency relation */
  StmtSRef src;
  /*! \brief The destination of the dependency relation */
  StmtSRef dst;
  /*! \brief The dependency kind */
  DepKind kind;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("src", &src);
    v->Visit("dst", &dst);
    v->Visit("kind", &kind);
  }

  static constexpr const char* _type_key = "tir.Dependency";
  TVM_DECLARE_FINAL_OBJECT_INFO(DependencyNode, Object);
};

/*!
 * \brief Managed reference to DependencyNode
 * \sa DependencyNode
 */
class Dependency : public ObjectRef {
 public:
  /*! \brief Constructor */
  TVM_DLL explicit Dependency(StmtSRef src, StmtSRef dst, DepKind kind);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Dependency, ObjectRef, DependencyNode);
};

/*!
 * \brief An object with 1-to-1 correspondence with each block reference in the sref tree.
 * This data structure is used to track the producer-consumer dependencies between blocks.
 * For example even leaf nodes have a scope node, even though they have no dependencies.
 *
 * Glossary:
 * - Block scope: A contiguous subtree of the sref tree, rooted at each block sref,
 * whose components are:
 *   - scope root: a block sref
 *   - internal srefs: loop srefs
 *   - scope leaves: block srefs
 * - Child block: The scope leaf blocks under the scope root or a specific internal sref
 */
class BlockScopeNode : public Object {
 public:
  /*!
   * \brief Lookup table for the `src` of dependencies
   * \note We intentionally didn't use tvm::Map as the data structure, because we need the values
   * inside to be mutable so that they could be further maintained properly during transformations.
   */
  std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual> src2deps;
  /*! \brief Lookup table for the `dst` of dependencies */
  std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual> dst2deps;
  /*! \brief The mapping from the buffer to the blocks who write it */
  std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual> buffer_writers;
  /*!
   * \brief This property indicates that the block scope (rooted at its corresponding block) is
   * equivalent to of a stage pipeline. Under the following conditions:
   *
   * 1) The region cover property holds for every of its child blocks
   * 2) No write-after-read dependency or opaque dependency, only read-after-write and
   * write-after-write are allowed
   * 3) All the statements in the scope are schedulable statements, i.e. Block and For
   */
  bool stage_pipeline{false};

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "tir.BlockScope";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockScopeNode, Object);

 public:
  /******** Dependency ********/
  /*!
   * \brief Get all dependencies whose `src` equals `src`
   * \param src The queried block
   * \return The dependencies
   */
  TVM_DLL Array<Dependency> GetDepsBySrc(const StmtSRef& src) const;
  /*!
   * \brief Get all dependencies whose `dst` equals `dst`
   * \param dst The queried block
   * \return The dependencies
   */
  TVM_DLL Array<Dependency> GetDepsByDst(const StmtSRef& dst) const;
};

/*!
 * \brief Managed reference to BlockScopeNode
 * \sa BlockScopeNode
 */
class BlockScope : public ObjectRef {
 public:
  /*! \brief The constructor creating an empty block scope with on dependency information */
  TVM_DLL BlockScope();
  /*!
   * \brief Create the object with the specific leaf blocks, and compute the dependency information
   * between the leaf blocks.
   * \param child_block_srefs The srefs to the leaf blocks
   * \note We assume the leaf blocks are given in pre-DFS order
   */
  TVM_DLL explicit BlockScope(const Array<StmtSRef>& child_block_srefs);

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockScope, ObjectRef, BlockScopeNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_BLOCK_SCOPE_H_
