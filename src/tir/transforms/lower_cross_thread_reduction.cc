/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_cross_thread_reduction.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../schedule/analysis.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output buffers
 * \param scope_block The scope block of the block to be checked
 * \param block The block whose dominant property is to be checked
 * \return A boolean indicating if the block is a dominant block
 */
bool IsDominantBlock(const Block& scope_block, const Block& block) {
  // Step 1. Count the number of writers for each buffer written by the scope block.
  std::unordered_map<const BufferNode*, int> buffer_writer_cnt;
  PreOrderVisit(scope_block->body, [&buffer_writer_cnt](const ObjectRef& obj) {
    if (const auto* block = obj.as<BlockNode>()) {
      for (const BufferRegion& buffer_region : block->writes) {
        ++buffer_writer_cnt[buffer_region->buffer.get()];
      }
      return false;
    }
    return true;
  });
  // Step 2. Check whether `block` is the only writer of its outputs.
  for (const BufferRegion& buffer_region : block->writes) {
    ICHECK(buffer_writer_cnt.count(buffer_region->buffer.get()));
    if (buffer_writer_cnt[buffer_region->buffer.get()] != 1) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Check whether the input block is a reduction block.
 * \param block_realize The block to be checked
 * \param loop_range_map The mapping from the loop variables outside the input block to their ranges
 * \param scope_block The scope block of the input block
 * \param analyzer The analyzer
 * \return A boolean indicating whether the input block is a reduction block.
 * \note A similar check has been implemented in "src/tir/schedule/analysis.h", but that check is
 * based on `tir.Schedule`. Here we have no schedule information, and thus we must implement the
 * check again.
 */
bool IsReductionBlock(const BlockRealize& block_realize, const Map<Var, Range>& loop_range_map,
                      const Block& scope_block, arith::Analyzer* analyzer) {
  const auto* block = block_realize->block.as<BlockNode>();
  // Cond 1. The block has the `init` statement.
  if (!block->init.defined()) {
    return false;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!IsAffineBinding(block_realize, loop_range_map, analyzer)) {
    return false;
  }
  // Cond 3. All block vars are either data parallel block vars or reduction block vars. Meanwhile,
  // we collect all the reduction block vars.
  if (!ContainsOnlyDataParAndReductionBlockIter(block->iter_vars)) {
    return false;
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output buffers.
  if (!IsDominantBlock(scope_block, GetRef<Block>(block))) {
    return false;
  }
  // Cond 5. The reduction block vars are not used to index the output buffers.
  return ReductionIterNotIndexOutputBuffer(GetRef<Block>(block));
}

/*!
 * \brief Create an intermediate buffer with specified name and data type
 * \param name The specified name
 * \param dtype The specified data type
 * \return The created buffer
 */
Buffer CreateReductionBuffer(String name, const DataType& dtype) {
  Var var(name, PointerType(PrimType(dtype), "local"));
  return Buffer(var, dtype, {1}, {1}, PrimExpr(), std::move(name), 0, 0, kDefault);
}

/*!
 * \brief Remove the BufferRegions whose buffer is the input buffer
 * \param buffer_regions The array of BufferRegions to be
 * \param buffer_to_remove The specified buffer
 * \return The mutated array of BufferRegions, no longer containing BufferRegion of the input buffer
 */
Array<BufferRegion> RemoveBufferFromBufferRegions(const Array<BufferRegion>& buffer_regions,
                                                  const Buffer& buffer_to_remove) {
  Array<BufferRegion> res;
  res.reserve(buffer_regions.size());
  for (const BufferRegion& buffer_region : buffer_regions) {
    if (!buffer_region->buffer.same_as(buffer_to_remove)) {
      res.push_back(buffer_region);
    }
  }
  return res;
}

/*!
 * \brief Substitute a given source buffer with a given target buffer in statements or expressions
 */
class BufferAccessReplacer : public StmtExprMutator {
 public:
  explicit BufferAccessReplacer(Buffer src_buffer, Buffer tgt_buffer)
      : src_buffer_(std::move(src_buffer)), tgt_buffer_(std::move(tgt_buffer)) {}

 private:
  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    return load->buffer.same_as(src_buffer_) ? BufferLoad(tgt_buffer_, {0})
                                             : GetRef<BufferLoad>(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    if (store->buffer.same_as(src_buffer_)) {
      PrimExpr value = StmtExprMutator::VisitExpr(store->value);
      return BufferStore(tgt_buffer_, value, {0});
    } else {
      return StmtMutator::VisitStmt_(store);
    }
  }

  Buffer src_buffer_;
  Buffer tgt_buffer_;
};

/*!
 * \brief Substitute a given source block with a given target block, or remove the source block
 * branch from the AST if the target block is undefined
 */
class ReductionBlockReplacer : public StmtMutator {
 public:
  explicit ReductionBlockReplacer(const BlockRealizeNode* src_block, BlockRealize tgt_block)
      : src_block_(src_block), tgt_block_(std::move(tgt_block)) {}

 private:
  Stmt VisitStmt_(const BlockRealizeNode* block_realize) final {
    return block_realize == src_block_ ? tgt_block_ : GetRef<BlockRealize>(block_realize);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    For res = Downcast<For>(StmtMutator::VisitStmt_(loop));
    return !res.defined() ? Stmt{nullptr} : (res->thread_binding.defined() ? res->body : res);
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> results;
    results.reserve(seq->size());
    for (const Stmt& stmt : seq->seq) {
      Stmt res = StmtMutator::VisitStmt(stmt);
      if (res.defined()) {
        results.push_back(res);
      }
    }
    return results.empty() ? Stmt{nullptr} : SeqStmt(results);
  }

  const BlockRealizeNode* src_block_;
  BlockRealize tgt_block_;
};

/*!
 * \brief Detect cross-thread reduction pattern and then transform
 */
class CrossThreadReductionTransformer : public StmtMutator {
 private:
  // Check if the input block needs cross-thread reduction.
  bool NeedCrossThreadReduction(const BlockRealizeNode* block_realize) {
    // Step 0. If the block is the root block, just return.
    if (block_stack_.empty()) {
      return false;
    }

    // Step 1. If the block is not a reduction block, cross-thread reduction is not needed.
    if (!IsReductionBlock(GetRef<BlockRealize>(block_realize), loop_range_map_,
                          GetRef<Block>(block_stack_.back()), &analyzer_)) {
      return false;
    }

    // Step 2. Collect all the vars that appear in the bindings of reduction block iters.
    std::unordered_set<const VarNode*> reduction_vars;
    GetVarsTouchedByBlockIters(GetRef<BlockRealize>(block_realize), nullptr, &reduction_vars);

    // Step 3. Collect the loops whose loop vars appear in the bindings of reduction block iters.
    // We call these loops "reduction-related".
    // Step 4. See whether at least one reduction-related loop is bound to thread axis in GPU - if
    // so, cross-thread reduction is needed. If none of the reduction-related loops is bound to
    // thread axis, cross-thread reduction is not needed for the input block.
    bool need = false;
    reduction_loops_.clear();
    for (const ForNode* loop : loop_stack_) {
      if (reduction_vars.count(loop->loop_var.get())) {
        // Step 3. Collect the loop.
        reduction_loops_.push_back(loop);
        // Step 4. See whether the loop is bound to some thread axis.
        if (loop->thread_binding.defined()) {
          need = true;
        }
      }
    }

    return need;
  }

  // Given that the input block needs cross-thread reduction, check if cross-thread reduction can
  // be applied to the block (i.e., the block satisfies all necessary conditions of cross-thread
  // reduction).
  void CheckCanApplyCrossThreadReduction(const BlockNode* block) {
    const String& block_name = block->name_hint;

    // Condition 1. The block being applied cross-thread reduction should write to single buffer.
    int n_write_buffer = static_cast<int>(block->writes.size());
    CHECK_EQ(n_write_buffer, 1) << "ValueError: Cross-thread reduction requires the block to only "
                                   "write to single buffer. However, the block "
                                << block_name << " writes to " << n_write_buffer << " buffer(s).";

    // Condition 2. All the reduction-related loops should be the deepest among all statements
    // outside the block (ignoring SeqStmt here).
    int n_deepest_reduction_loops = 0;
    for (auto rit = statement_stack_.rbegin() + 1; rit != statement_stack_.rend(); ++rit) {
      if ((*rit)->IsInstance<SeqStmtNode>()) {
        // Skip SeqStmt.
        continue;
      }
      if (std::find(reduction_loops_.begin(), reduction_loops_.end(),
                    reinterpret_cast<const ForNode*>(*rit)) == reduction_loops_.end()) {
        break;
      }
      ++n_deepest_reduction_loops;
    }
    CHECK_EQ(n_deepest_reduction_loops, reduction_loops_.size())
        << "ValueError: Cross-thread reduction requires all the reduction-related loops to be the "
           "deepest among all statements outside the desired block. However, block "
        << block_name
        << " needs cross-thread reduction, while the reduction-related loops outside of it are not "
           "the deepest statements, which violates the condition.";

    // Condition 3. All the reduction-related loops that are bound to thread axes should only be
    // bound to `threadIdx.x/y/z`.
    n_bound_reduction_loops_ = 0;
    for (const ForNode* reduction_loop : reduction_loops_) {
      if (reduction_loop->thread_binding.defined()) {
        ++n_bound_reduction_loops_;
        const String& thread_tag = reduction_loop->thread_binding.value()->thread_tag;
        CHECK(thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
              thread_tag == "threadIdx.z")
            << "ValueError: Cross-thread reduction requires all the reduction-related loops that "
               "are bound to GPU thread axes to only be bound `threadIdx.x/y/z`. However, loop "
            << reduction_loop->loop_var->name_hint << " is bound to " << thread_tag
            << ", which violates the condition.";
      }
    }

    // Condition 4. Get the `init` identity and the `update` combiner of the reduction. They should
    // both be BufferStores with the same buffer and indices.
    BufferStore init;
    BufferStore update;
    std::tie(init, update) =
        GetBufferStoresFromReductionBlock<false>(ScheduleState{nullptr}, GetRef<Block>(block));

    // Condition 5. Extract the commutative reducer, combiner lhs and combiner rhs from the
    // reduction identity and the reduction combiner.
    PrimExpr combiner_lhs;
    std::tie(reducer_, combiner_lhs, combiner_rhs_) =
        GetReducerAndCombinerLhsRhs<false>(ScheduleState{nullptr}, init->value, update);

    // Condition 6. The block should be the last block under the first reduction-related loop.
    bool visit = false;
    PreOrderVisit(GetRef<For>(reduction_loops_[0]), [block, &visit](const ObjectRef& obj) {
      if (const auto* realize = obj.as<BlockRealizeNode>()) {
        CHECK(!visit) << "ValueError: Cross-thread reduction cannot be applied when the reduction "
                         "block isn't the last block under its first reduction-related loop";
        if (realize->block.get() == block) {
          visit = true;
        }
        return false;
      }
      return true;
    });
  }

  void TransformReductionBlock(const BlockRealizeNode* block_realize, bool with_normal_reduction) {
    const BlockNode* block = block_realize->block.get();
    Buffer result_buffer = block->writes[0]->buffer;

    BufferRegion ct_reduction_buf_region(cross_thread_reduction_buf_, {Range::FromMinExtent(0, 1)});
    BufferRegion normal_reduction_buf_region{nullptr};
    if (with_normal_reduction) {
      normal_reduction_buf_region =
          BufferRegion(normal_reduction_buf_, {Range::FromMinExtent(0, 1)});
    }

    Array<Stmt> seq_stmt;
    seq_stmt.reserve(4);

    if (with_normal_reduction) {
      // Step 1. Create the BufferStore which initializes `normal_reduction_buf_`.
      seq_stmt.push_back(BufferStore(/*buffer=*/normal_reduction_buf_,
                                     /*value=*/block->init.value().as<BufferStoreNode>()->value,
                                     /*indices=*/{0}));

      // Step 2. Create the block and loops which do the normal reduction.
      //  - 2.1. Create the block.
      ObjectPtr<BlockNode> p_new_block = make_object<BlockNode>(*block);
      {
        p_new_block->reads = RemoveBufferFromBufferRegions(block->reads, result_buffer);
        p_new_block->reads.push_back(normal_reduction_buf_region);
        p_new_block->writes = {normal_reduction_buf_region};
        p_new_block->name_hint = block->name_hint + "_normal_reduction";
        p_new_block->body = BufferAccessReplacer(result_buffer, normal_reduction_buf_)(block->body);
        p_new_block->init = NullOpt;
      }
      //  - 2.2. Create the block-realize.
      ObjectPtr<BlockRealizeNode> p_new_block_realize =
          make_object<BlockRealizeNode>(*block_realize);
      p_new_block_realize->block = Block(p_new_block);
      //  - 2.3. Replace the original reduction block with the normal reduction block.
      Stmt replace_result = ReductionBlockReplacer(
          block_realize, BlockRealize(p_new_block_realize))(GetRef<For>(reduction_loops_[0]));
      ICHECK(replace_result.defined());
      seq_stmt.push_back(replace_result);
    } else {
      // Remove the original reduction block.
      Stmt replace_result = ReductionBlockReplacer(
          block_realize, BlockRealize{nullptr})(GetRef<For>(reduction_loops_[0]));
      if (replace_result.defined()) {
        seq_stmt.push_back(replace_result);
      }
    }

    // Step 3. Create the statement which calls the intrinsic and does the cross-thread reduction.
    //  - 3.1. Create the intrinsic parameters.
    PrimExpr reduction_value =
        with_normal_reduction ? BufferLoad(normal_reduction_buf_, {0}) : combiner_rhs_;
    Array<PrimExpr> parameters{make_const(DataType::UInt(32), static_cast<uint32_t>(1)),
                               std::move(reduction_value), const_true(),
                               cross_thread_reduction_buf_->data};
    parameters.reserve(reduction_loops_.size() + 4);
    for (const ForNode* reduction_loop : reduction_loops_) {
      if (reduction_loop->thread_binding.defined()) {
        parameters.push_back(reduction_loop->loop_var);
      }
    }
    //  - 3.2. Create the intrinsic call and the block body.
    AttrStmt ct_reduction_body(/*node=*/reducer_,
                               /*attr_key=*/tir::attr::reduce_scope,
                               /*value=*/make_zero(DataType::Handle()),
                               /*body=*/
                               Evaluate(Call(/*dtype=*/DataType::Handle(),
                                             /*op=*/tir::builtin::tvm_thread_allreduce(),
                                             /*args=*/std::move(parameters))));
    //  - 3.3. Create the block and the block-realize.
    {
      Array<IterVar> iters;
      Array<PrimExpr> bindings;
      Array<BufferRegion> reads{nullptr};
      if (with_normal_reduction) {
        reads = {std::move(normal_reduction_buf_region)};
      } else {
        iters = block->iter_vars;
        bindings = block_realize->iter_values;
        reads = {RemoveBufferFromBufferRegions(block->reads, result_buffer)};
      }
      Block ct_reduction_block(/*iter_vars=*/std::move(iters),
                               /*reads=*/std::move(reads),
                               /*writes=*/{ct_reduction_buf_region},
                               /*name_hint=*/block->name_hint + "_cross_thread_reduction",
                               /*body=*/std::move(ct_reduction_body));
      seq_stmt.push_back(BlockRealize(/*iter_values=*/std::move(bindings),
                                      /*predicate=*/const_true(),
                                      /*block=*/std::move(ct_reduction_block)));
    }

    // Step 4. Create the block which writes the cross-thread reduction result back to the original
    // result buffer.
    //  - 4.1. Create the block iters and their corresponding iter bindings.
    ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
    int n_iter = static_cast<int>(block->iter_vars.size());
    Array<IterVar> write_back_block_iters;
    Array<PrimExpr> write_back_block_bindings;
    std::unordered_map<const VarNode*, PrimExpr> write_back_block_var_map;
    write_back_block_iters.reserve(n_iter);
    write_back_block_bindings.reserve(n_iter);
    write_back_block_var_map.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
      IterVar iter = block->iter_vars[i];
      PrimExpr binding = block_realize->iter_values[i];
      if (iter->iter_type == kCommReduce) {
        continue;
      }
      ObjectPtr<IterVarNode> p_new_iter = make_object<IterVarNode>(*iter.get());
      p_new_iter->var = Var(make_object<VarNode>(*iter->var.get()));
      IterVar new_iter(p_new_iter);
      write_back_block_iters.push_back(new_iter);
      write_back_block_bindings.push_back(binding);
      write_back_block_var_map[iter->var.get()] = std::move(new_iter);
    }
    //  - 4.2. Create the body of the write-back block.
    const auto* old_reduction_body = block->body.as<BufferStoreNode>();
    BufferStore write_back_body(/*buffer=*/std::move(result_buffer),
                                /*value=*/BufferLoad(cross_thread_reduction_buf_, {0}),
                                /*indices=*/old_reduction_body->indices);
    //  - 4.3. Create the block and block-realize.
    Block write_back_block(/*iter_vars=*/std::move(write_back_block_iters),
                           /*reads=*/{std::move(ct_reduction_buf_region)},
                           /*writes=*/block->writes,
                           /*name_hint=*/block->name_hint + "_write_back",
                           /*body=*/std::move(write_back_body));
    write_back_block =
        Downcast<Block>(Substitute(Stmt(write_back_block), write_back_block_var_map));
    seq_stmt.push_back(BlockRealize(/*iter_values=*/std::move(write_back_block_bindings),
                                    /*predicate=*/const_true(),
                                    /*block=*/std::move(write_back_block)));

    // Step 5. Wrap all the above four statements with the reduction loops were bound.
    Stmt new_stmt = SeqStmt::Flatten(seq_stmt);
    for (auto rit = reduction_loops_.rbegin(); rit != reduction_loops_.rend(); ++rit) {
      if ((*rit)->thread_binding.defined()) {
        ObjectPtr<ForNode> p_new_loop = make_object<ForNode>(*(*rit));
        p_new_loop->body = std::move(new_stmt);
        new_stmt = For(p_new_loop);
      }
    }

    // Step 6. Replace the first reduction-related loop the new statement.
    loop2new_stmt_[reduction_loops_[0]] = std::move(new_stmt);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    statement_stack_.push_back(stmt.get());
    Stmt result = StmtMutator::VisitStmt(stmt);
    statement_stack_.pop_back();
    return result;
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    loop_stack_.push_back(loop);
    loop_range_map_.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    Stmt result = StmtMutator::VisitStmt_(loop);
    loop_stack_.pop_back();
    loop_range_map_.erase(loop->loop_var);

    // Replace `result` with the pre-stored result if `loop` appears as a key in `loop2new_stmt_`.
    auto it = loop2new_stmt_.find(loop);
    if (it != loop2new_stmt_.end()) {
      return it->second;
    } else {
      return result;
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    Map<Var, Range> old_loop_range_map;

    block_stack_.push_back(block);
    std::swap(old_loop_range_map, loop_range_map_);
    Block new_block = Downcast<Block>(StmtMutator::VisitStmt_(block));
    block_stack_.pop_back();
    std::swap(old_loop_range_map, loop_range_map_);

    // Insert the new allocated buffers into the block's `alloc_buffers` field.
    auto it = block2new_buffers_.find(block);
    if (it != block2new_buffers_.end()) {
      BlockNode* p_new_block = new_block.CopyOnWrite();
      for (const Buffer& new_buffer : it->second) {
        if (new_buffer.defined()) {
          p_new_block->alloc_buffers.push_back(new_buffer);
        }
      }
    }
    return new_block;
  }

  Stmt VisitStmt_(const BlockRealizeNode* block_realize) final {
    const BlockNode* block = block_realize->block.get();

    // Step 1. Check whether cross-thread reduction is needed. If no, skip this block.
    if (!NeedCrossThreadReduction(block_realize)) {
      return StmtMutator::VisitStmt_(block_realize);
    }
    ++reduction_id_;

    // Step 2. Check whether cross-thread reduction can be applied. If no, throw an exception on
    // which condition the block violates.
    CheckCanApplyCrossThreadReduction(block);

    // Step 3. When not all the reduction-related loops are bound to thread axes, normal reduction
    // is needed in this cross-thread reduction.
    bool need_normal_reduction =
        n_bound_reduction_loops_ < static_cast<int>(reduction_loops_.size());

    // Step 4. Create intermediate buffers, storing them in `cross_thread_reduction_buf_` and
    // `normal_reduction_buf_`. Let the scope block allocate these new buffers.
    std::vector<Buffer>& new_buffers = block2new_buffers_[block_stack_.back()];
    DataType dtype = block->writes[0]->buffer->dtype;
    cross_thread_reduction_buf_ =
        CreateReductionBuffer("reduce_temp" + std::to_string(reduction_id_), dtype);
    new_buffers.push_back(cross_thread_reduction_buf_);
    if (need_normal_reduction) {
      normal_reduction_buf_ =
          CreateReductionBuffer("normal_reduce_temp" + std::to_string(reduction_id_), dtype);
      new_buffers.push_back(normal_reduction_buf_);
    }

    // Step 5. Transform.
    TransformReductionBlock(block_realize, need_normal_reduction);

    // Step 6. Return an empty statement, because the transformation result will be inserted when
    // returning to the first reduction-related loop.
    return Stmt{nullptr};
  }

 private:
  int reduction_id_ = -1;

  std::vector<const StmtNode*> statement_stack_;
  std::vector<const ForNode*> loop_stack_;
  std::vector<const BlockNode*> block_stack_;
  Map<Var, Range> loop_range_map_;

  int n_bound_reduction_loops_ = 0;
  std::vector<const ForNode*> reduction_loops_;

  CommReducer reducer_;
  PrimExpr combiner_rhs_;

  Buffer cross_thread_reduction_buf_;
  Buffer normal_reduction_buf_;

  std::unordered_map<const BlockNode*, std::vector<Buffer>> block2new_buffers_;
  std::unordered_map<const ForNode*, Stmt> loop2new_stmt_;

  arith::Analyzer analyzer_;
};

PrimFunc LowerCrossThreadReduction(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = CrossThreadReductionTransformer()(f->body);
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass LowerCrossThreadReduction() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerCrossThreadReduction(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerCrossThreadReduction", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerCrossThreadReduction")
    .set_body_typed(LowerCrossThreadReduction);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
