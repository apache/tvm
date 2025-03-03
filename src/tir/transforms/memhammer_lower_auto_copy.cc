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

#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <array>
#include <stack>

#include "../../runtime/thread_storage_scope.h"
#include "../schedule/utils.h"
#include "./ir_utils.h"
#include "./memhammer_rewrite_rule.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tir {

using support::NDIntSet;

// rewrite rules
static InverseMapping inverse_mapping;
static CoalescedAccess coalesced_access;
static CreateLocalStage create_local_stage;
static SharedToWmma shared_to_wmma;
static WmmaToGlobal wmma_to_global;
static WmmaToShared wmma_to_shared;
static MmaToGlobal mma_to_global;

/*!
 * \brief A class to perform auto padding.
 *
 * One simple way to perform auto padding is to fix each padding size for each dimension at the
 * same time, calculate the precise access index and the bank conflict,
 * and choose the one with minimal conflict. However, this algorithm has exponential complexity.
 * Suppose we have d dimensions and the padding size is 0-31, we need to calculate bank
 * conflict for 32^{d-1} times.
 * We propose a fast incremental algorithm that works for affine inputs, and it only calculate
 * bank conflict for 32*{d-1} times. To be specific, we first decide the optimal padding size for
 * dimension d-2, then for dimension d-3, ..., finally for dimension 0. It involves 2 steps.
 *
 * First, we analyze how a typical warp accesses the shared memory banks.
 * A typical warp means setting all irrelevant loop vars to 0, and only keeps the threads in a warp.
 * For each dimension, the access index is represented by
 * x_1 * scale_1 + ... + x_n * scale_n (x_i is loop var)
 * Note: The affine property guarantees that {x_i} must be independent,
 * otherwise the algorithm is wrong.
 * We will use this information to keep a list for each dimension called "iteration space" that
 * records the resulting index as x_i takes each possible value.
 *
 * For example, the index is [outer*2+ty, tx*4+vec], where tx is threadIdx.x, and ty is threadIdx.y.
 * tx is in [0, 16), and ty is in [0, 2).
 * We will first get a warp access [ty, tx*4] because outer and vec are irrelevant loop vars.
 * It's obvious that ty, tx*4 are both in the form of x_1 * scale_1 + ... + x_n * scale_n.
 * In this case, we will keep lists {{0, 1}, {0, 4, ..., 60}}
 *
 * Next, we choose a padding size that has minimal conflict from the last dimension to first one.
 * To calculate the conflict, we calculate the Cartesian product of the iteration space of all
 * dimensions not higher than this. Each single point of product space represents access index
 * of a particular thread, by which we can calculate the accessed memory bank. The conflict is
 * the highest access frequency among the banks.
 *
 */
class AutoPadder {
 public:
  /**
   * \brief Do padding to the given buffers in shard memory
   * \param buffers the given buffers
   * \return the list of new padded buffers
   */
  Array<Buffer> PadSharedMemory(const Array<Buffer>& buffers) {
    Array<Buffer> result;

    for (const Buffer& buffer : buffers) {
      runtime::StorageScope scope = runtime::StorageScope::Create(buffer.scope());
      if (scope.rank == runtime::StorageRank::kShared) {
        auto iter_spaces = iter_spaces_[buffer.get()];
        if (iter_spaces.empty()) {
          result.push_back(buffer);
          continue;
        }
        // The access index represented by points in the cartesian product of lower dimension
        // iteration spaces
        std::vector<std::vector<int>> low_dim_iter_space(iter_spaces.size(), std::vector<int>());

        int n = buffer->shape.size();
        int data_bits = buffer->dtype.bits();
        // Step 1. initialize `low_dim_iter_space` with the iteration space of the last dim
        for (int i = 0; i < static_cast<int>(iter_spaces.size()); i++) {
          auto last_dim_iter_space = iter_spaces[i][n - 1];
          low_dim_iter_space[i] = last_dim_iter_space;
        }
        PrimExpr stride = 1;
        Array<PrimExpr> reverse_strides;
        int pad_min = padding_min_.Get(buffer).value_or(Integer(1)).IntValue();
        // Step 2. For each dimension, select a padding that has minimal bank conflict
        for (int k = n - 2; k >= 0; k--) {  // dims
          int max_pad_size =
              std::min(static_cast<int>(max_pad_factor_ *
                                        (stride * buffer->shape[k + 1]).as<IntImmNode>()->value),
                       32 * 32 / data_bits);
          int min_conflict = INT32_MAX;
          int min_conflict_pad = -1;
          for (int pad = 0; pad <= max_pad_size; pad += pad_min) {  // select padding
            int padded_stride = ((stride * buffer->shape[k + 1]).as<IntImmNode>()->value + pad) %
                                (32 * 32 / data_bits);
            int conflict = 0;
            for (int i = 0; i < static_cast<int>(iter_spaces.size()); i++) {  // accesses
              auto iter_space = iter_spaces[i][k];
              int bank[32]{0};
              for (int v1 : iter_space) {
                for (int v2 : low_dim_iter_space[i]) {
                  int comb = (v1 * padded_stride + v2) * data_bits / 32 % 32;
                  bank[comb]++;
                }
              }
              for (int j = 0; j < 32; j++) {
                conflict = std::max(conflict, bank[j]);
              }
            }
            if (conflict < min_conflict) {
              min_conflict = conflict;
              min_conflict_pad = pad;
            }
          }
          // update low_dim_iter_space with
          for (int i = 0; i < static_cast<int>(iter_spaces.size()); i++) {  // accesses
            auto iter_space = iter_spaces[i][k];
            if (!iter_space.empty()) {
              int padded_stride =
                  ((stride * buffer->shape[k + 1]).as<IntImmNode>()->value + min_conflict_pad) %
                  (32 * 32 / data_bits);
              std::vector<int> span;
              for (int v1 : iter_space) {
                for (int v2 : low_dim_iter_space[i]) {
                  span.push_back(((v1 * padded_stride + v2) * data_bits) % (32 * 32 / data_bits));
                }
              }
              low_dim_iter_space[i] = span;
            }
          }
          stride = stride * buffer->shape[k + 1] + min_conflict_pad;
          reverse_strides.push_back(stride);
        }
        // Step 3. create the new padded buffer
        ObjectPtr<BufferNode> b = make_object<BufferNode>(*buffer.get());
        Array<PrimExpr> strides;
        for (int i = static_cast<int>(reverse_strides.size()) - 1; i >= 0; i--) {
          strides.push_back(reverse_strides[i]);
        }
        strides.push_back(1);
        b->strides = strides;
        Buffer new_buffer(b);
        result.push_back(new_buffer);
        padded_buffer_map_.Set(buffer, new_buffer);
      } else {
        result.push_back(buffer);
      }
    }
    return result;
  }

  /**
   * \brief Replace all occurrence of the old buffer with the new buffer in the stmt
   * \param stmt the stmt to do replacement
   * \return the stmt after replacement
   */
  Stmt RewriteBufferAccess(const Stmt& stmt) {
    class Rewriter : public StmtExprMutator {
     public:
      explicit Rewriter(const Map<Buffer, Buffer>& buffer_map) : buffer_map_(buffer_map) {}

     private:
      PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
        BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
        BufferLoadNode* op = load.CopyOnWrite();
        if (buffer_map_.count(op->buffer)) {
          op->buffer = buffer_map_[op->buffer];
        }
        return std::move(load);
      }

      Stmt VisitStmt_(const BufferStoreNode* _op) final {
        BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
        BufferStoreNode* op = store.CopyOnWrite();
        if (buffer_map_.count(op->buffer)) {
          op->buffer = buffer_map_[op->buffer];
        }
        return std::move(store);
      }

      Stmt VisitStmt_(const BlockNode* op) final {
        // To reduce the number of blocks in block sref reuse map, we check whether the block is
        // really mutated (i.e., the old buffer appears in the block). If so, we return the block
        // after mutation. Otherwise we just return the original block.
        bool changed = false;
        // Step 1. Mutate the read region.
        Array<BufferRegion> reads;
        for (const BufferRegion& read : op->reads) {
          if (buffer_map_.count(read->buffer)) {
            changed = true;
            reads.push_back(BufferRegion(buffer_map_[read->buffer], read->region));
          } else {
            reads.push_back(read);
          }
        }
        // Step 2. Mutate the write region.
        Array<BufferRegion> writes;
        for (const BufferRegion& write : op->writes) {
          if (buffer_map_.count(write->buffer)) {
            changed = true;
            writes.push_back(BufferRegion(buffer_map_[write->buffer], write->region));
          } else {
            writes.push_back(write);
          }
        }
        // Step 4. Mutate `match_buffers`. If an old buffer appears as a source of
        // MatchBufferRegion, the storage scope of the target buffer also needs to be set.
        Array<MatchBufferRegion> match_buffers;
        for (const MatchBufferRegion& match_buffer : op->match_buffers) {
          if (buffer_map_.count(match_buffer->source->buffer)) {
            changed = true;
            Buffer new_buffer = buffer_map_[match_buffer->source->buffer];
            match_buffers.push_back(MatchBufferRegion(
                match_buffer->buffer, BufferRegion(new_buffer, match_buffer->source->region)));
          } else {
            match_buffers.push_back(match_buffer);
          }
        }
        // Step 5. Recursively mutate the block.
        Stmt res = StmtMutator::VisitStmt_(op);
        if (res.get() != op) {
          changed = true;
        }

        if (changed) {
          ObjectPtr<BlockNode> block = CopyOnWrite(res.as<BlockNode>());
          block->reads = std::move(reads);
          block->writes = std::move(writes);
          block->match_buffers = std::move(match_buffers);
          return Stmt(block);
        } else {
          return GetRef<Block>(op);
        }
      }
      const Map<Buffer, Buffer>& buffer_map_;
    };
    Rewriter rewriter(padded_buffer_map_);
    return rewriter(stmt);
  }

  /**
   * \brief an equivalent of scale * loop_var with loop_var: {min=0, extent=extent}
   */
  struct Pattern {
    int extent;
    int scale;
  };

  /**
   * \brief Collect pattern from indices
   */
  class PatternCollector : public StmtExprVisitor {
    void VisitExpr_(const VarNode* op) final {
      if (!success_) {
        return;
      }
      int extent = var_range_[GetRef<Var>(op)]->extent.as<IntImmNode>()->value;
      if (extent > 1) {
        stack_.push({{extent, 1}});
      } else {
        stack_.push({});
      }
    }

    void VisitExpr_(const AddNode* op) final {
      ExprVisitor::VisitExpr_(op);
      if (!success_) {
        return;
      }
      std::vector<Pattern> merged_patterns;
      std::vector<Pattern> r = stack_.top();
      stack_.pop();
      std::vector<Pattern> l = stack_.top();
      stack_.pop();
      for (const Pattern& pattern : l) {
        merged_patterns.push_back(pattern);
      }
      for (const Pattern& pattern : r) {
        merged_patterns.push_back(pattern);
      }
      if (merged_patterns.empty()) {
        stack_.push({});
        return;
      }
      std::vector<Pattern> ret;
      ret.push_back(merged_patterns[0]);
      for (int i = 0; i < static_cast<int>(merged_patterns.size()); i++) {
        Pattern prev_pattern = ret.back();
        if (merged_patterns[i].extent * merged_patterns[i].scale == prev_pattern.scale) {
          ret.pop_back();
          ret.push_back(
              {prev_pattern.extent * merged_patterns[i].extent, merged_patterns[i].scale});
        }
      }
      stack_.push(ret);
    }

    void VisitExpr_(const FloorDivNode* op) final {
      ExprVisitor::VisitExpr_(op);
      if (!success_) {
        return;
      }
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int lower_factor = op->b.as<IntImmNode>()->value;
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        if (pattern.scale >= lower_factor) {
          if (pattern.scale % lower_factor == 0) {
            ret.push_back({pattern.extent, pattern.scale / lower_factor});
          } else {
            success_ = false;
          }
        } else if (pattern.scale * pattern.extent > lower_factor) {
          if ((pattern.scale * pattern.extent) % lower_factor == 0) {
            ret.push_back({pattern.extent * pattern.scale / lower_factor, 1});
          } else {
            success_ = false;
          }
        }
      }
      stack_.push(ret);
    }

    void VisitExpr_(const FloorModNode* op) final {
      ExprVisitor::VisitExpr_(op);
      if (!success_) {
        return;
      }
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int extent = op->b.as<IntImmNode>()->value;
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        if (pattern.scale < extent) {
          if (extent % pattern.scale == 0) {
            if (extent / pattern.scale < pattern.extent) {
              ret.push_back({extent / pattern.scale, pattern.scale});
            } else {
              ret.push_back({pattern.extent, pattern.scale});
            }
          } else {
            success_ = false;
          }
        }
      }
      stack_.push(ret);
    }

    void VisitExpr_(const MulNode* op) final {
      ExprVisitor::VisitExpr_(op);
      if (!success_) {
        return;
      }
      std::vector<Pattern> inner = stack_.top();
      stack_.pop();
      int scale = op->b.as<IntImmNode>()->value;
      std::vector<Pattern> ret;
      for (const Pattern& pattern : inner) {
        ret.push_back({pattern.extent, pattern.scale * scale});
      }
      stack_.push(ret);
    }

   public:
    explicit PatternCollector(const Map<Var, Range>& var_range) : var_range_(var_range) {}

    /*!
     * \brief Collect the iteration space for given indices. The iteration space is the possible
     * values that an index can take (do not remove duplicate).
     * For example, the input is [ty, tx*4], where tx is in [0, 16), and ty is in [0, 2).
     * The output would be {{0, 1}, {0, 4, ..., 60}}
     * \param indices The indices to analyze
     * \param var_range The range of loop variables
     * \param data_bits The size of dtype in bits
     * \return The iteration space. The first array represents dimensions, and the second array
     * represents the iteration space of one dimension
     */
    static std::vector<std::vector<int>> CollectIterationSpace(const Array<PrimExpr>& indices,
                                                               const Map<Var, Range>& var_range,
                                                               int data_bits) {
      PatternCollector collector(var_range);
      std::vector<std::vector<int>> ret;
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        collector(indices[i]);
        if (collector.success_ && collector.stack_.size() == 1) {
          auto patterns = collector.stack_.top();
          int extent_prod = 1;
          for (const Pattern& p : patterns) {
            extent_prod *= p.extent;
          }
          std::vector<int> iter_space;
          for (int thread_id = 0; thread_id < extent_prod; thread_id++) {
            int index = 0;
            int n = thread_id;
            for (int j = static_cast<int>(patterns.size()) - 1; j >= 0; j--) {
              int val = n % patterns[j].extent;
              index += val * patterns[j].scale;
              n /= patterns[j].extent;
            }
            iter_space.push_back(index);
          }

          ret.push_back(iter_space);
          collector.stack_.pop();
        } else {
          ret.push_back({});
        }
      }
      return ret;
    }

    std::stack<std::vector<Pattern>> stack_;
    const Map<Var, Range>& var_range_;
    bool success_ = true;
  };

  /*! A utility class for calling CollectIterationSpace to each buffer access*/
  class IterSpaceAnalyzer : public StmtExprVisitor {
   public:
    IterSpaceAnalyzer(const Map<Var, PrimExpr>& substitute_map, AutoPadder* self, int data_bits,
                      const Map<String, Integer> warp_thread_extent)
        : substitute_map_(substitute_map),
          self(self),
          data_bits_(data_bits),
          warp_thread_extent_(warp_thread_extent) {}

   private:
    bool CheckVarContiguous(PrimExpr e, Var var, const Map<Var, PrimExpr>& subst_map) {
      PrimExpr e1 = Substitute(e, [var](const Var& v) -> Optional<PrimExpr> {
        if (v.same_as(var)) {
          return Integer(0);
        } else {
          return v;
        }
      });
      PrimExpr e2 = Substitute(e, [var](const Var& v) -> Optional<PrimExpr> {
        if (v.same_as(var)) {
          return Integer(1);
        } else {
          return v;
        }
      });
      arith::Analyzer analyzer;
      return !analyzer.CanProve(Substitute(e2 - e1, subst_map) != 1);
    }

    void VisitStmt_(const ForNode* op) final {
      if (op->kind != ForKind::kThreadBinding) {
        substitute_map_.Set(op->loop_var, op->min);
      } else {
        Integer extent =
            warp_thread_extent_.Get(op->thread_binding.value()->thread_tag).value_or(1);
        var_range_.Set(op->loop_var, Range::FromMinExtent(op->min, extent));
      }
      if (op->kind == ForKind::kVectorized) {
        vector_var = op->loop_var;
        vector_length_ = op->extent.as<IntImmNode>()->value;
      }
      StmtExprVisitor::VisitStmt_(op);
      if (op->kind == ForKind::kVectorized) {
        vector_length_ = -1;
      }
      if (op->kind != ForKind::kThreadBinding) {
        substitute_map_.erase(op->loop_var);
      }
    }
    /*!
     * \brief Take a typical warp and collect the iteration space for buffer store
     * For example, the access is A[outer*2+ty, tx*4+vec] = xxx, where tx is threadIdx.x, and ty is
     * threadIdx.y. tx is in [0, 16), and ty is in [0, 2).
     * The iteration space would be {{0, 1}, {0, 4, ..., 60}}.
     * \param op the buffer store
     */
    void VisitStmt_(const BufferStoreNode* op) final {
      runtime::StorageScope scope = runtime::StorageScope::Create(op->buffer.scope());
      if (scope.rank == runtime::StorageRank::kShared) {
        Array<PrimExpr> substitued_indices;
        arith::Analyzer analyzer;
        for (const PrimExpr& e : op->indices) {
          substitued_indices.push_back(analyzer.Simplify(Substitute(e, substitute_map_)));
        }
        std::vector<std::vector<int>> iter_space =
            PatternCollector::CollectIterationSpace(substitued_indices, var_range_, data_bits_);
        if (!iter_space.empty()) {
          self->iter_spaces_[op->buffer.get()].push_back(iter_space);
        }
        if (vector_length_ != -1 &&
            CheckVarContiguous(op->indices.back(), vector_var, substitute_map_)) {
          Integer m = self->padding_min_.Get(op->buffer).value_or(1);
          self->padding_min_.Set(op->buffer, Downcast<Integer>(max(vector_length_, m)));
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }
    /*!
     * \brief Take a typical warp and collect the iteration space for buffer load
     * For example, the access is xxx = A[outer*2+ty, tx*4+vec], where tx is threadIdx.x, and ty is
     * threadIdx.y. tx is in [0, 16), and ty is in [0, 2).
     * The iteration space would be {{0, 1}, {0, 4, ..., 60}}.
     * \param op the buffer load
     */
    void VisitExpr_(const BufferLoadNode* op) final {
      runtime::StorageScope scope = runtime::StorageScope::Create(op->buffer.scope());
      if (scope.rank == runtime::StorageRank::kShared) {
        Array<PrimExpr> substitued_indices;
        arith::Analyzer analyzer;
        for (const PrimExpr& e : op->indices) {
          substitued_indices.push_back(analyzer.Simplify(Substitute(e, substitute_map_)));
        }
        std::vector<std::vector<int>> iter_space =
            PatternCollector::CollectIterationSpace(substitued_indices, var_range_, data_bits_);
        if (!iter_space.empty()) {
          self->iter_spaces_[op->buffer.get()].push_back(iter_space);
        }
        if (vector_length_ != -1 &&
            CheckVarContiguous(substitued_indices.back(), vector_var, substitute_map_)) {
          Integer m = self->padding_min_.Get(op->buffer).value_or(1);
          self->padding_min_.Set(op->buffer, Downcast<Integer>(max(vector_length_, m)));
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }

    /*!
     * \brief Take a typical warp and collect the iteration space for load_matrix_sync and
     * store_matrix_sync
     * For example, the access region is A[y*16+16, x*16+16], where y and x are not bound to
     * threadIdx. The iteration space would be {{0, 1, ..., 15}, {0, 1, ..., 15}}.
     * \param op the call node
     */
    void VisitStmt_(const BlockNode* op) final {
      if (const auto* eval = op->body.as<EvaluateNode>()) {
        if (const auto* call = eval->value.as<CallNode>()) {
          if (call->op == builtin::tvm_load_matrix_sync() ||
              call->op == builtin::tvm_store_matrix_sync()) {
            for (const MatchBufferRegion& r : op->match_buffers) {
              Buffer src_buffer = r->source->buffer;
              runtime::StorageScope scope = runtime::StorageScope::Create(src_buffer.scope());
              if (scope.rank == runtime::StorageRank::kShared) {
                Region region = r->source->region;
                Array<PrimExpr> indices;
                for (int i = 0; i < static_cast<int>(region.size()); i++) {
                  Var var("region" + std::to_string(i));
                  indices.push_back(region[i]->min + var);
                  var_range_.Set(var, Range::FromMinExtent(0, region[i]->extent));
                }
                Array<PrimExpr> substitued_indices;
                arith::Analyzer analyzer;
                for (const PrimExpr& e : indices) {
                  substitued_indices.push_back(analyzer.Simplify(Substitute(e, substitute_map_)));
                }
                std::vector<std::vector<int>> iter_space = PatternCollector::CollectIterationSpace(
                    substitued_indices, var_range_, data_bits_);
                if (!iter_space.empty()) {
                  self->iter_spaces_[src_buffer.get()].push_back(iter_space);
                }
              }
            }
          }
        }
      }
    }

    Map<Var, PrimExpr> substitute_map_;
    AutoPadder* self;
    int data_bits_;
    Map<String, Integer> warp_thread_extent_;
    Map<Var, Range> var_range_;
    int vector_length_ = -1;
    Var vector_var;
  };

  /*!
   * \brief Analyze the shared memory access
   * \param stmt The data copy
   * \param outer_loops The outer loops of the stmt
   * \param data_bits The length of dtype in bits
   * \param thread_extent The extents of all thread binding loops
   */
  void AnalyzeSharedMemoryAccess(const Stmt& stmt, const Array<For>& outer_loops, int data_bits,
                                 const Map<String, Integer>& thread_extent) {
    Map<String, Integer> warp_thread_extent;
    Integer prod = 1;
    Array<String> thread_tags{"threadIdx.x", "threadIdx.y", "threadIdx.z"};
    arith::Analyzer analyzer;
    for (int i = 0; i < 3; i++) {
      Integer extent = thread_extent.Get(thread_tags[i]).value_or(1);
      if (analyzer.CanProve(prod * extent >= 32)) {
        warp_thread_extent.Set(thread_tags[i], Downcast<Integer>(floordiv(32, prod)));
        prod *= floordiv(32, prod);
        break;
      } else {
        warp_thread_extent.Set(thread_tags[i], Downcast<Integer>(extent));
        prod *= extent;
      }
    }
    Map<Var, PrimExpr> substitute_map;
    for (const For& loop : outer_loops) {
      substitute_map.Set(loop->loop_var, loop->min);
    }
    IterSpaceAnalyzer iter_space_analyzer(substitute_map, this, data_bits, warp_thread_extent);
    iter_space_analyzer(stmt);
  }

 private:
  /*! \brief A map from the old buffers to the new padded buffers */
  Map<Buffer, Buffer> padded_buffer_map_;
  /*! \brief A map from each buffer to the iteration spaces of the accesses*/
  std::unordered_map<const BufferNode*, std::vector<std::vector<std::vector<int>>>> iter_spaces_;
  /*! \brief A map from each buffer to their minimal padding size */
  Map<Buffer, Integer> padding_min_;
  /*! \brief max padding size in relative to the original shape*/
  const double max_pad_factor_ = 0.25;

  friend class AutoCopyMutator;
};

class AutoCopyMutator : public StmtExprMutator {
 public:
  explicit AutoCopyMutator(Map<String, Integer> thread_extent) : thread_extent_(thread_extent) {}
  /**
   * \brief Replace old buffers with padded buffers in the stmt
   * \param stmt The stmt to rewrite
   * \return The stmt after rewrite
   */
  Stmt RewritePaddingBody(const Stmt& stmt) { return padder.RewriteBufferAccess(stmt); }

 private:
  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    // only rewrite the block annotated with "auto_copy"
    if (GetAnn<Integer>(op, tir::attr::auto_copy).value_or(0)->value == 0) {
      BlockNode* n = block.CopyOnWrite();
      n->alloc_buffers = padder.PadSharedMemory(std::move(n->alloc_buffers));
      return std::move(block);
    }
    ICHECK_EQ(block->writes.size(), 1);
    ICHECK_GE(block->reads.size(), 1);

    BufferRegion target_read = block->reads[0];
    if (block->reads.size() > 1) {
      bool found = false;
      for (size_t i = 0; i < block->reads.size(); i++) {
        if (block->reads[i]->buffer.scope() == "wmma.accumulator") {
          found = true;
          target_read = block->reads[i];
        }
      }
      ICHECK(found) << "Multiple buffer read";
    }

    int data_bits = target_read->buffer->dtype.bits();
    ConstraintSet constraints(this->thread_extent_,  //
                              this->outer_loops_,    //
                              target_read,           //
                              block->writes[0],      //
                              data_bits,             //
                              block->annotations);
    BlockNode* n = block.CopyOnWrite();
    OutputSet outputs;
    for (RewriteRule* rule : rules) {
      n->body = rule->Apply(std::move(n->body), constraints, &outputs);
    }
    for (const Buffer& buffer : outputs.alloc_buffer) {
      n->alloc_buffers.push_back(buffer);
    }
    for (const auto& p : outputs.padding_min) {
      Integer m = padder.padding_min_.Get(p.first).value_or(1);
      padder.padding_min_.Set(p.first, Downcast<Integer>(max(p.second, m)));
    }
    padder.AnalyzeSharedMemoryAccess(block->body, outer_loops_, data_bits, thread_extent_);
    n->alloc_buffers = padder.PadSharedMemory(std::move(n->alloc_buffers));
    return std::move(block);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    outer_loops_.push_back(GetRef<For>(op));
    Stmt stmt = StmtMutator::VisitStmt_(op);
    outer_loops_.pop_back();
    return stmt;
  }

  /*! \brief Thread extents collected. */
  Map<String, Integer> thread_extent_;
  /*! \brief The outer loops during recursive visit */
  Array<For> outer_loops_;
  /*! \brief Calculating optimal padding size */
  AutoPadder padder;

  /*! \brief All rewrite rules. */
  const std::array<RewriteRule*, 7> rules = {&inverse_mapping,     //
                                             &coalesced_access,    //
                                             &create_local_stage,  //
                                             &shared_to_wmma,      //
                                             &wmma_to_global,      //
                                             &wmma_to_shared,      //
                                             &mma_to_global};
};

/*!
 * \brief Collect the extent for all thread binding loops.
 */
class ThreadExtentCollector : public StmtVisitor {
 public:
  static Map<String, Integer> CollectThreadExtent(const Stmt& stmt) {
    ThreadExtentCollector collector;
    collector(stmt);
    return collector.thread_extent_;
  }

 private:
  void VisitStmt_(const BlockNode* op) final {
    if (Optional<Integer> warp_execution = GetAnn<Integer>(op, "warp_execution")) {
      if (warp_execution.value()->value != 0) {
        thread_extent_.Set("threadIdx.x", Integer(32));
      }
    }
    StmtVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const ForNode* op) final {
    if (op->thread_binding.defined() && op->thread_binding.value()->iter_type == kThreadIndex) {
      if (const auto* extent = op->extent.as<IntImmNode>()) {
        thread_extent_.Set(op->thread_binding.value()->thread_tag, GetRef<Integer>(extent));
      }
    }
    StmtVisitor::VisitStmt_(op);
  }

  /*! \brief the map from thread tag to its extent */
  Map<String, Integer> thread_extent_;
};

namespace transform {

Pass LowerAutoCopy() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    AutoCopyMutator mutator(ThreadExtentCollector::CollectThreadExtent(n->body));
    n->body = mutator(std::move(n->body));
    n->body = mutator.RewritePaddingBody(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerAutoCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerAutoCopy").set_body_typed(LowerAutoCopy);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
