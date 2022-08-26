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
#include "./ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Checks if a loop is bound to threadIdx.x/y/z
 * \brief loop The loop to be checked
 * \return True if the loop is bound to threadIdx.x/y/z
 */
bool IsBoundToThreadIdx(const ForNode* loop) {
  if (!loop->thread_binding.defined()) {
    return false;
  }
  runtime::ThreadScope scope =
      runtime::ThreadScope::Create(loop->thread_binding.value()->thread_tag);
  return scope.rank == 1 && scope.dim_index >= 0;
}

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
 * \param realize The block to be checked
 * \param loop_range_map The mapping from the loop variables outside the input block to their ranges
 * \param scope_block The scope block of the input block
 * \param analyzer The analyzer
 * \return A boolean indicating whether the input block is a reduction block.
 * \note A similar check has been implemented in "src/tir/schedule/analysis.h", but that check is
 * based on `tir.Schedule`. Here we have no schedule information, and thus we must implement the
 * check again.
 */
bool IsReductionBlock(const BlockRealize& realize, const Map<Var, Range>& loop_range_map,
                      const Block& scope_block, arith::Analyzer* analyzer) {
  const auto* block = realize->block.as<BlockNode>();
  // Cond 1. The block has the `init` statement.
  if (!block->init.defined()) {
    return false;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!IsAffineBinding(realize, loop_range_map, analyzer)) {
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
Buffer MakeScratchpad(String name, const DataType& dtype) {
  return Buffer(/*ptr=*/Var(name, PointerType(PrimType(dtype), "local")),
                /*dtype=*/dtype,
                /*shape=*/{Integer(1)},
                /*strides=*/{Integer(1)},
                /*elem_offset=*/PrimExpr{nullptr},
                /*name=*/name,
                /*data_alignment=*/0,
                /*offset_factor=*/0,
                /*buffer_type=*/kDefault);
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
class BufferReplacer : private StmtExprMutator {
 public:
  static Stmt Run(Buffer src_buffer, Buffer tgt_buffer, Stmt stmt) {
    return BufferReplacer(src_buffer, tgt_buffer)(std::move(stmt));
  }

 private:
  explicit BufferReplacer(Buffer src_buffer, Buffer tgt_buffer)
      : src_buffer_(std::move(src_buffer)), tgt_buffer_(std::move(tgt_buffer)) {}

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
class InThreadReducerMaker : private StmtMutator {
 public:
  static Optional<Stmt> Make(const BlockRealizeNode* src_realize,
                             Optional<BlockRealize> tgt_realize, Stmt stmt) {
    return InThreadReducerMaker(src_realize, std::move(tgt_realize))(std::move(stmt));
  }

 private:
  explicit InThreadReducerMaker(const BlockRealizeNode* src_realize,
                                Optional<BlockRealize> tgt_realize)
      : src_realize_(src_realize), tgt_realize_(tgt_realize) {}
  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    if (realize == src_realize_) {
      return tgt_realize_.defined()  //
                 ? tgt_realize_.value()
                 : Stmt{nullptr};
    }
    return GetRef<BlockRealize>(realize);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (Optional<For> opt_res = Downcast<Optional<For>>(StmtMutator::VisitStmt_(loop))) {
      For res = opt_res.value();
      if (res->thread_binding.defined()) {
        return res->body;
      } else {
        return std::move(res);
      }
    } else {
      return Stmt{nullptr};
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> stmts;
    stmts.reserve(seq->size());
    for (const Stmt& stmt : seq->seq) {
      if (Optional<Stmt> opt_res = VisitStmt(stmt)) {
        stmts.push_back(opt_res.value());
      }
    }
    return stmts.empty() ? Stmt{nullptr} : SeqStmt::Flatten(stmts);
  }

  const BlockRealizeNode* src_realize_;
  Optional<BlockRealize> tgt_realize_;
};

/*!
 * \brief Create the lowered allreduce block transformed from the input reduction block
 * \param reduction_block The input reduction block
 * \param it_buffer The buffer to store in-thread reduction results
 * \param ct_buffer The buffer to store cross-thread reduction results
 * \param reducer The reduction function
 * \param combiner_rhs The RHS of the combiner
 * \param reduction_loops The reduction loops
 */
Stmt TransformReductionBlock(const BlockRealizeNode* realize, const Optional<Buffer>& it_buffer,
                             const Buffer& ct_buffer, const CommReducer& reducer,
                             const PrimExpr& combiner_rhs,
                             const std::vector<const ForNode*>& reduction_loops) {
  const BlockNode* block = realize->block.get();
  Buffer wb_buffer = block->writes[0]->buffer;
  Array<Range> wb_region = block->writes[0]->region;

  BufferRegion ct_buffer_region(ct_buffer, {Range::FromMinExtent(0, 1)});
  Optional<BufferRegion> it_buffer_region = NullOpt;
  if (it_buffer.defined()) {
    it_buffer_region = BufferRegion(it_buffer.value(), {Range::FromMinExtent(0, 1)});
  }
  // In total, the block is transformed into at most 4 statements
  // - Stmt 1: initialize the buffer for in-thread reduction
  // - Stmt 2: do in-thread reduction
  // - Stmt 3: do cross-thread reduction
  // - Stmt 4: write cross-thread reduction result to the original buffer
  Array<Stmt> stmts;
  stmts.reserve(4);
  // Stmt 1: initialize the buffer for in-thread reduction
  if (it_buffer.defined()) {
    BufferStore init = Downcast<BufferStore>(block->init);
    stmts.push_back(BlockRealize(
        /*iter_values=*/{},
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{},
              /*reads=*/{},
              /*writes=*/{it_buffer_region.value()},
              /*name_hint=*/block->name_hint + "_in_thread_init",
              /*body=*/
              BufferStore(/*buffer=*/it_buffer.value(),
                          /*value=*/init->value,
                          /*indices=*/{Integer(0)}))));
  }
  // Stmt 2: do in-thread reduction
  {
    Optional<BlockRealize> new_realize = NullOpt;
    // If need to generate in-thread reduction,
    // then replace `wb_buffer` with `it_buffer` accordingly in given BlockRealize
    // otherwise, directly remove given BlockRealize
    if (it_buffer.defined()) {
      ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
      new_block->reads = RemoveBufferFromBufferRegions(std::move(new_block->reads), wb_buffer);
      new_block->reads.push_back(it_buffer_region.value());
      new_block->writes = {it_buffer_region.value()};
      new_block->name_hint = new_block->name_hint + "_in_thread";
      new_block->body =
          BufferReplacer::Run(wb_buffer, it_buffer.value(), std::move(new_block->body));
      new_block->init = NullOpt;
      ObjectPtr<BlockRealizeNode> n = make_object<BlockRealizeNode>(*realize);
      n->block = Block(new_block);
      new_realize = BlockRealize(n);
    }
    For loop = GetRef<For>(reduction_loops[0]);
    if (Optional<Stmt> stmt = InThreadReducerMaker::Make(realize, new_realize, std::move(loop))) {
      stmts.push_back(stmt.value());
    }
  }
  // Stmt 3: do cross-thread reduction
  {
    // Step 3.1. Create the parameters to the intrinsic
    Array<PrimExpr> parameters;
    parameters.reserve(reduction_loops.size() + 4);
    // 1-st argument: size
    parameters.push_back(make_const(DataType::UInt(32), 1));
    // 2-nd argument: source
    if (it_buffer.defined()) {
      parameters.push_back(BufferLoad(it_buffer.value(), {Integer(0)}));
    } else {
      parameters.push_back(combiner_rhs);
    }
    // 3-rd argument: predicate
    parameters.push_back(const_true());
    // 4-th argument: destination
    parameters.push_back(BufferLoad(ct_buffer, {0}));
    // next arguments: all the reduction threads
    for (const ForNode* reduction_loop : reduction_loops) {
      if (reduction_loop->thread_binding.defined()) {
        parameters.push_back(reduction_loop->loop_var);
      }
    }
    // Step 3.2. Create the block and the block-realize.
    Array<IterVar> iter_vars{nullptr};
    Array<PrimExpr> bindings{nullptr};
    Array<BufferRegion> reads{nullptr};
    if (it_buffer.defined()) {
      iter_vars = Array<IterVar>{};
      bindings = Array<PrimExpr>{};
      reads = {it_buffer_region.value()};
    } else {
      iter_vars = block->iter_vars;
      bindings = realize->iter_values;
      reads = {RemoveBufferFromBufferRegions(block->reads, wb_buffer)};
    }
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/std::move(iter_vars),
              /*reads=*/std::move(reads),
              /*writes=*/{ct_buffer_region},
              /*name_hint=*/block->name_hint + "_cross_thread",
              /*body=*/
              AttrStmt(/*node=*/reducer,
                       /*attr_key=*/tir::attr::reduce_scope,
                       /*value=*/make_zero(DataType::Handle()),
                       /*body=*/
                       Evaluate(Call(/*dtype=*/DataType::Handle(),
                                     /*op=*/tir::builtin::tvm_thread_allreduce(),
                                     /*args=*/std::move(parameters)))))));
  }
  // Stmt 4: write cross-thread reduction result to the original buffer
  {
    ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    int n_iter = static_cast<int>(block->iter_vars.size());
    Array<IterVar> iter_vars;
    Array<PrimExpr> bindings;
    Map<Var, PrimExpr> var_map;
    iter_vars.reserve(n_iter);
    bindings.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != kCommReduce) {
        IterVar new_iter_var{nullptr};
        {
          ObjectPtr<IterVarNode> n = make_object<IterVarNode>(*iter_var.get());
          ObjectPtr<VarNode> v = make_object<VarNode>(*iter_var->var.get());
          n->var = Var(v);
          new_iter_var = IterVar(n);
        }
        iter_vars.push_back(new_iter_var);
        bindings.push_back(binding);
        var_map.Set(iter_var->var, new_iter_var->var);
      }
    }
    BufferStore update = Downcast<BufferStore>(block->body);
    update = Downcast<BufferStore>(Substitute(std::move(update), var_map));
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings),
        /*predicate=*/const_true(),
        /*block=*/
        Block(
            /*iter_vars=*/std::move(iter_vars),
            /*reads=*/{std::move(ct_buffer_region)},
            /*writes=*/{BufferRegion(wb_buffer, Substitute(wb_region, var_map))},
            /*name_hint=*/block->name_hint + "_write_back",
            /*body=*/
            BufferStore(/*buffer=*/wb_buffer,
                        /*value=*/BufferLoad(ct_buffer, {Integer(0)}),
                        /*indices=*/update->indices))));
  }
  // Final step: Wrap all the above four statements with the reduction loops bound to threadIdx
  Stmt new_stmt = SeqStmt::Flatten(std::move(stmts));
  for (auto rit = reduction_loops.rbegin(); rit != reduction_loops.rend(); ++rit) {
    const ForNode* loop = *rit;
    if (loop->thread_binding.defined()) {
      ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
      n->body = std::move(new_stmt);
      new_stmt = For(n);
    }
  }
  return new_stmt;
}

/*!
 * \brief Detect cross-thread reduction pattern and then transform
 */
class CrossThreadReductionTransformer : public StmtMutator {
 private:
  // Check if the input block needs cross-thread reduction.
  std::vector<const ForNode*> NeedCrossThreadReduction(const BlockRealizeNode* realize) {
    // Step 0. If the block is the root block, just return.
    if (block_stack_.empty()) {
      return {};
    }

    // Step 1. If the block is not a reduction block, cross-thread reduction is not needed.
    if (!IsReductionBlock(GetRef<BlockRealize>(realize), loop_range_map_,
                          GetRef<Block>(block_stack_.back()), &analyzer_)) {
      return {};
    }

    // Step 2. Collect all the vars that appear in the bindings of reduction block iters.
    std::unordered_set<const VarNode*> reduction_vars;
    GetVarsTouchedByBlockIters(GetRef<BlockRealize>(realize), nullptr, &reduction_vars);

    // Step 3. Collect the loops whose loop vars appear in the bindings of reduction block iters.
    // We call these loops "reduction-related".
    // Step 4. See whether at least one reduction-related loop is bound to thread axis in GPU - if
    // so, cross-thread reduction is needed. If none of the reduction-related loops is bound to
    // thread axis, cross-thread reduction is not needed for the input block.
    bool need = false;
    std::vector<const ForNode*> reduction_loops;
    for (const ForNode* loop : loop_stack_) {
      if (reduction_vars.count(loop->loop_var.get())) {
        // Step 3. Collect the loop.
        reduction_loops.push_back(loop);
        // Step 4. See whether the loop is bound to some thread axis.
        if (loop->thread_binding.defined()) {
          need = true;
        }
      }
    }
    return need ? reduction_loops : std::vector<const ForNode*>{};
  }

  // Given that the input block needs cross-thread reduction, check if cross-thread reduction can
  // be applied to the block (i.e., the block satisfies all necessary conditions of cross-thread
  // reduction).
  std::tuple<int, CommReducer, PrimExpr> CheckCanApplyCrossThreadReduction(
      const BlockNode* block, const std::vector<const ForNode*>& reduction_loops) const {
    // Condition 1. The block being applied cross-thread reduction should write to single buffer.
    CHECK_EQ(block->writes.size(), 1)
        << "ValueError: Cross-thread reduction requires the block to only "
           "write to single buffer. However, the block "
        << block->name_hint << " writes to " << block->writes.size() << " buffer(s).";

    // Condition 2. All the reduction-related loops should be the deepest among all statements
    // outside the block (ignoring SeqStmt here).
    int n_deepest_reduction_loops = 0;
    for (auto rit = statement_stack_.rbegin() + 1; rit != statement_stack_.rend(); ++rit) {
      const StmtNode* stmt = *rit;
      if ((*rit)->IsInstance<SeqStmtNode>()) {
        // Skip SeqStmt.
        continue;
      }
      if (std::find(reduction_loops.begin(), reduction_loops.end(),
                    reinterpret_cast<const ForNode*>(stmt)) == reduction_loops.end()) {
        break;
      }
      ++n_deepest_reduction_loops;
    }
    CHECK_EQ(n_deepest_reduction_loops, reduction_loops.size())
        << "ValueError: Cross-thread reduction requires all the reduction-related loops to be the "
           "deepest among all statements outside the desired block. However, block "
        << block->name_hint
        << " needs cross-thread reduction, while the reduction-related loops outside of it are not "
           "the deepest statements, which violates the condition.";

    // Condition 3. All the reduction-related loops that are bound to thread axes should only be
    // bound to `threadIdx.x/y/z`.
    int n_bound_reduction_loops = 0;
    for (const ForNode* reduction_loop : reduction_loops) {
      if (reduction_loop->thread_binding.defined()) {
        ++n_bound_reduction_loops;
        CHECK(IsBoundToThreadIdx(reduction_loop))
            << "ValueError: Cross-thread reduction requires all the reduction-related loops that "
               "are bound to GPU thread axes to only be bound `threadIdx.x/y/z`. However, loop "
            << reduction_loop->loop_var->name_hint << " violates the condition.";
      }
    }

    // Condition 4. Get the `init` identity and the `update` combiner of the reduction. They should
    // both be BufferStores with the same buffer and indices;
    // Extract the commutative reducer, combiner lhs and combiner rhs from the reduction identity
    // and the reduction combiner.
    auto [init, update] = GetBufferStoresFromReductionBlock(NullOpt, GetRef<Block>(block));
    auto [reducer, combiner_lhs, combiner_rhs] =
        GetReducerAndCombinerLhsRhs(NullOpt, init->value, update);
    (void)combiner_lhs;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767

    // Condition 5. The block should be the last block under the first reduction-related loop.
    bool visit = false;
    PreOrderVisit(GetRef<For>(reduction_loops[0]), [block, &visit](const ObjectRef& obj) {
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
    return std::make_tuple(n_bound_reduction_loops, reducer, combiner_rhs);
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
    return std::move(new_block);
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    const BlockNode* block = realize->block.get();
    // Step 1. Check whether cross-thread reduction is needed. If no, skip this block.
    std::vector<const ForNode*> reduction_loops = NeedCrossThreadReduction(realize);
    if (reduction_loops.empty()) {
      return StmtMutator::VisitStmt_(realize);
    }
    ++reduction_id_;
    // Step 2. Check whether cross-thread reduction can be applied. If no, throw an exception on
    // which condition the block violates.
    auto [n_bound_reduction_loops, reducer, combiner_rhs] =
        CheckCanApplyCrossThreadReduction(block, reduction_loops);
    // Step 3. Before doing the cross-thread reduction, in-thread reduction is needed when
    //  - not all the reduction-related loops are bound to thread axes, or
    //  - the block-realize has a non-constant-true predicate.
    bool need_in_thread_reduction =
        n_bound_reduction_loops < static_cast<int>(reduction_loops.size()) ||
        !is_one(realize->predicate);
    // Step 4. Create intermediate buffers, storing them in `ct_buffer` and
    // `it_buffer`. Let the scope block allocate these new buffers.
    std::vector<Buffer>& new_buffers = block2new_buffers_[block_stack_.back()];
    DataType dtype = block->writes[0]->buffer->dtype;
    Buffer ct_buffer = MakeScratchpad("cross_thread_" + std::to_string(reduction_id_), dtype);
    new_buffers.push_back(ct_buffer);
    Optional<Buffer> it_buffer = NullOpt;
    if (need_in_thread_reduction) {
      it_buffer = MakeScratchpad("in_thread_" + std::to_string(reduction_id_), dtype);
      new_buffers.push_back(it_buffer.value());
    }
    // Step 5. Transform.
    loop2new_stmt_[reduction_loops[0]] = TransformReductionBlock(
        realize, it_buffer, ct_buffer, reducer, combiner_rhs, reduction_loops);
    // Step 6. Return an empty statement, because the transformation result will be inserted when
    // returning to the first reduction-related loop.
    return Stmt{nullptr};
  }

 private:
  int reduction_id_ = -1;
  std::vector<const StmtNode*> statement_stack_;
  std::vector<const ForNode*> loop_stack_;
  std::vector<const BlockNode*> block_stack_;
  std::unordered_map<const BlockNode*, std::vector<Buffer>> block2new_buffers_;
  std::unordered_map<const ForNode*, Stmt> loop2new_stmt_;
  Map<Var, Range> loop_range_map_;
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
