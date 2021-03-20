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
 * \file tir/analysis/block_region_detector.cc
 * \brief Detect block read/write regions by visiting its body
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

/*!
 * \brief Detect which regions of tensors in this block are read or written to. Regions are sorted
 * by order of appearance in the AST. \note This detector can only visit blocks and will not visit
 * child blocks recursively
 */
class BlockReadWriteDetector : public StmtExprVisitor {
 public:
  explicit BlockReadWriteDetector(const Map<Var, Buffer>& buffer_var_map)
      : buffer_var_map_(buffer_var_map) {}

  /*! \brief Return read regions of the block */
  Array<BufferRegion> CollectReads();
  /*! \brief Return write regions of the block */
  Array<BufferRegion> CollectWrites();
  /*!
   * \brief Return opaque buffer regions of the block
   * \note The buffer accessed by load/store or call with buffer.data will
   *       be marked as opaque.
   */
  Array<BufferRegion> CollectOpaques();
  /*! \brief overload operator() to make sure it accepts a block node */
  void operator()(const Stmt& stmt);

 private:
  /*! \brief Iteration range for loop_vars */
  std::unordered_map<const VarNode*, arith::IntSet> dom_map_;
  /*! \brief The buffers that the current block reads */
  std::vector<Buffer> read_buffers_;
  /*! \brief The buffers that the current block writes */
  std::vector<Buffer> writes_buffers_;
  /*! \brief The opaque buffer which is access by buffer.data */
  std::vector<Buffer> opaque_buffers_;
  /*! \brief The read regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> read_regions_;
  /*! \brief The write regions of the current block */
  std::vector<std::vector<tvm::arith::IntSet>> write_regions_;
  /*! \brief The outside buffer data mapping to its buffer */
  Map<Var, Buffer> buffer_var_map_;
  /*! \brief The analyzer for simplifying*/
  arith::Analyzer analyzer_;

  /*!
   * \brief Update read/write buffers and regions with provided buffer and region
   * \param buffers The buffers should be updated
   * \param regions The access regions should be updated
   * \param buffer The provided buffer
   * \param region The provided region
   */
  void Update(std::vector<Buffer>* buffers, std::vector<std::vector<arith::IntSet>>* regions,
              const Buffer& buffer, const std::vector<arith::IntSet>& region);

  /*! \brief Helper function to collect access regions. */
  Array<BufferRegion> CollectRegions(const std::vector<Buffer>& buffers,
                                     const std::vector<std::vector<tvm::arith::IntSet>>& regions);

  /*! \brief Helper function to add a opaque buffer. */
  void AddOpaque(const Var& buffer_var);

  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const BlockRealizeNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const StoreNode* op) override;
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitExpr_(const LoadNode* op) override;
  void VisitExpr_(const VarNode* op) override;
};

void BlockReadWriteDetector::operator()(const Stmt& stmt) {
  ICHECK(stmt.as<BlockNode>() != nullptr)
      << "Only visiting Blocks is allowed, but got " << stmt->GetTypeKey();
  StmtExprVisitor::operator()(stmt);
}

Array<BufferRegion> BlockReadWriteDetector::CollectReads() {
  return CollectRegions(read_buffers_, read_regions_);
}

Array<BufferRegion> BlockReadWriteDetector::CollectWrites() {
  return CollectRegions(writes_buffers_, write_regions_);
}

Array<BufferRegion> BlockReadWriteDetector::CollectOpaques() {
  Array<BufferRegion> res;
  res.reserve(opaque_buffers_.size());
  for (const Buffer& buffer : opaque_buffers_) {
    res.push_back(BufferRegion::FullRegion(buffer));
  }
  return res;
}

void BlockReadWriteDetector::VisitExpr_(const VarNode* op) { AddOpaque(GetRef<Var>(op)); }

void BlockReadWriteDetector::VisitExpr_(const LoadNode* op) {
  AddOpaque(op->buffer_var);
  ExprVisitor::VisitExpr_(op);
}

void BlockReadWriteDetector::VisitExpr_(const BufferLoadNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (const PrimExpr& index : op->indices) {
    relaxed_region.push_back(arith::EvalSet(index, dom_map_));
  }
  Update(&read_buffers_, &read_regions_, op->buffer, relaxed_region);
  ExprVisitor::VisitExpr_(op);
}

void BlockReadWriteDetector::VisitStmt_(const ForNode* op) {
  Range range = Range::FromMinExtent(op->min, op->extent);
  dom_map_[op->loop_var.get()] = arith::IntSet::FromRange(range);
  StmtVisitor::VisitStmt_(op);
  dom_map_.erase(op->loop_var.get());
}

void BlockReadWriteDetector::VisitStmt_(const StoreNode* op) {
  AddOpaque(op->buffer_var);
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteDetector::VisitStmt_(const BufferStoreNode* op) {
  std::vector<arith::IntSet> relaxed_region;
  for (const PrimExpr& index : op->indices) {
    relaxed_region.push_back(arith::EvalSet(index, dom_map_));
  }
  Update(&writes_buffers_, &write_regions_, op->buffer, relaxed_region);
  StmtVisitor::VisitStmt_(op);
}

void BlockReadWriteDetector::VisitStmt_(const BlockRealizeNode* op) {
  /*! \note detector will not visit child block recursively, so it will stop here */
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  for (size_t i = 0; i < op->block->iter_vars.size(); ++i) {
    vmap[op->block->iter_vars[i]->var.get()] = op->iter_values[i];
  }
  for (const auto& read : op->block->reads) {
    std::vector<arith::IntSet> relaxed_region;
    for (const auto& range : read->region) {
      relaxed_region.push_back(
          arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                             Substitute(range->min, vmap), Substitute(range->extent, vmap))),
                         dom_map_));
    }
    Update(&read_buffers_, &read_regions_, read->buffer, relaxed_region);
  }
  for (const auto& write : op->block->writes) {
    std::vector<arith::IntSet> relaxed_region;
    for (const auto& range : write->region) {
      relaxed_region.push_back(
          arith::EvalSet(arith::IntSet::FromRange(Range::FromMinExtent(
                             Substitute(range->min, vmap), Substitute(range->extent, vmap))),
                         dom_map_));
    }
    Update(&writes_buffers_, &write_regions_, write->buffer, relaxed_region);
  }
}

void BlockReadWriteDetector::Update(std::vector<Buffer>* buffers,
                                    std::vector<std::vector<arith::IntSet>>* regions,
                                    const Buffer& buffer,
                                    const std::vector<arith::IntSet>& region) {
  if (buffer_var_map_.find(buffer->data) == buffer_var_map_.end()) return;
  ICHECK_EQ(buffers->size(), regions->size())
      << " Expected the buffer and regions to have the same size ";
  for (size_t i = 0; i < regions->size(); ++i) {
    if ((*buffers)[i].same_as(buffer)) {
      ICHECK_EQ((*regions)[i].size(), region.size()) << "Inconsistent buffer dimension";
      for (size_t j = 0; j < region.size(); ++j) {
        (*regions)[i][j] = arith::Union({(*regions)[i][j], region[j]});
      }
      return;
    }
  }
  buffers->push_back(buffer);
  regions->push_back(region);
}

Array<BufferRegion> BlockReadWriteDetector::CollectRegions(
    const std::vector<Buffer>& buffers,
    const std::vector<std::vector<tvm::arith::IntSet>>& regions) {
  ICHECK_EQ(buffers.size(), regions.size());
  Array<BufferRegion> res;
  res.reserve(buffers.size());
  for (size_t i = 0; i < regions.size(); ++i) {
    Array<Range> region;
    region.reserve(regions[i].size());
    for (size_t j = 0; j < regions[i].size(); j++) {
      tvm::arith::IntSet range = regions[i][j];
      region.push_back(range.CoverRange(Range::FromMinExtent(0, buffers[i]->shape[j])));
    }
    res.push_back(BufferRegion(buffers[i], region));
  }
  return res;
}

void BlockReadWriteDetector::AddOpaque(const Var& buffer_var) {
  auto it = buffer_var_map_.find(buffer_var);
  if (it != buffer_var_map_.end()) {
    const Buffer& buffer = (*it).second;
    for (const Buffer& opaque_buffer : opaque_buffers_) {
      if (buffer.same_as(opaque_buffer)) return;
    }
    opaque_buffers_.push_back(buffer);
  }
}

Array<Array<BufferRegion>> GetBlockAccessRegion(const Block& block,
                                                const Map<Var, Buffer>& buffer_var_map) {
  BlockReadWriteDetector detector(buffer_var_map);
  detector(block);
  return {detector.CollectReads(), detector.CollectWrites(), detector.CollectOpaques()};
}

TVM_REGISTER_GLOBAL("tir.analysis.get_block_access_region").set_body_typed(GetBlockAccessRegion);

}  // namespace tir
}  // namespace tvm
