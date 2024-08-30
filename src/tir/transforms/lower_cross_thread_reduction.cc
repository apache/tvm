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

#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "../schedule/analysis.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

using runtime::ThreadScope;
using support::StartsWith;

constexpr const char* kIsCrossThreadReductionApplied = "tir.cross_thread_reduction_applied";

// Implement a hash and equality function for ThreadScope so that
// ThreadScope can serve as map key class
struct ThreadScopeHash {
  size_t operator()(const ThreadScope& scope) const {
    return static_cast<size_t>(scope.rank * 30 + scope.dim_index);
  }
};

struct ThreadScopeEqual {
  bool operator()(const ThreadScope& a, const ThreadScope& b) const {
    return a.rank == b.rank && a.dim_index == b.dim_index;
  }
};

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

  // Cond 1. as some reduction block didn't have init statement.
  // Thus, we detect the reduction block by checking the iter vars.
  // If the block has no reduce iter vars, it is not a reduction block, then return false.
  bool has_reduce_iter_vars =
      std::any_of(block->iter_vars.begin(), block->iter_vars.end(),
                  [](const IterVar& iter_var) { return iter_var->iter_type == kCommReduce; });
  // should have init or has reduce iter vars and not cross thread reduction block
  Optional<ObjectRef> cross_thread_reduction_applied =
      block->annotations.Get(kIsCrossThreadReductionApplied);
  if (!has_reduce_iter_vars || cross_thread_reduction_applied) {
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
  // Cond 4. The reduction block vars are not used to index the output buffers.
  return ReductionIterNotIndexOutputBuffer(GetRef<Block>(block));
}

/*!
 * \brief Create intermediate buffers according to the input buffers and buffer kind
 * \param reduction_buffers The old reduction buffers which provide the buffer names and data types
 * \param is_cross_thread_buffer A boolean indicating whether to create buffers for the cross-thread
 * computation results or not, which is used for determine the buffer name prefix
 * \return The created buffers
 */
Array<Buffer> MakeScratchpads(const Array<Buffer>& reduction_buffers, bool is_cross_thread_buffer) {
  Array<Buffer> new_buffers;
  new_buffers.reserve(reduction_buffers.size());
  for (const Buffer& buffer : reduction_buffers) {
    String name = is_cross_thread_buffer ? "cross" : "in";
    name = name + "_thread_" + buffer->name;
    new_buffers.push_back(Buffer(/*ptr=*/Var(name, PointerType(PrimType(buffer->dtype), "local")),
                                 /*dtype=*/buffer->dtype,
                                 /*shape=*/{Integer(1)},
                                 /*strides=*/{Integer(1)},
                                 /*elem_offset=*/PrimExpr{nullptr},
                                 /*name=*/name,
                                 /*data_alignment=*/0,
                                 /*offset_factor=*/0,
                                 /*buffer_type=*/kDefault));
  }
  return new_buffers;
}

/*!
 * \brief Get init value from BufferStore Node
 * \param block The block to be checked
 * \return The init value
 */
class InitUpdateValueFinder : public StmtExprVisitor {
 public:
  /*!
   * \brief Find the init value of the given block
   * \param block The block to be checked
   * \return The init value of the given block
   */
  static PrimExpr FindInit(const Block& block) {
    InitUpdateValueFinder finder;
    finder(block->body);
    CHECK(finder.init_value_.defined()) << "The init value of the block is not found";
    return finder.init_value_;
  }

  /*!
   * \brief Find the update value of the given block
   * \param block The block to be checked
   * \return The update value of the given block
   */
  static BufferStore FindUpdate(const Block& block) {
    InitUpdateValueFinder finder;
    finder(block->body);
    CHECK(finder.update_value_.defined()) << "The update value of the block is not found";
    return finder.update_value_;
  }

  /*!
   * \brief Check whether the input block has MMA operation
   * \param realize The block to be checked
   * \return A boolean indicating whether the input block has MMA operation.
   */
  static bool CheckHasMMA(const Block& block) {
    InitUpdateValueFinder checker;
    checker(block->body);
    return checker.has_mma_;
  }

 private:
  void VisitStmt_(const BufferStoreNode* node) final {
    BufferStore store = GetRef<BufferStore>(node);
    init_value_ = store->value;
    update_value_ = store;
    return StmtVisitor::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode* op) {
    // TODO: Should append more test case for wmma
    if (op->op.same_as(tir::builtin::ptx_mma())) {
      has_mma_ = true;
    } else if (op->op.same_as(tir::builtin::mma_fill())) {
      has_mma_ = true;
      init_value_ = make_const(DataType::Float(16), 0);
    }
    return StmtExprVisitor::VisitExpr_(op);
  }

  bool has_mma_{false};
  PrimExpr init_value_{nullptr};
  BufferStore update_value_{nullptr};
};

/*!
 * \brief Check whether the input block has child blocks.
 * \param realize The block to be checked
 * \return A boolean indicating whether the input block has child blocks.
 * \note A similar check has been implemented in "src/tir/schedule/analysis.h", but that check is
 * based on `tir.Schedule`. Here we have no schedule information, and thus we must implement the
 * check again.
 */
class HasChildBlocksChecker : public StmtVisitor {
 public:
  /*!
   * \brief Check if the given block has child blocks.
   * \param realize The block to be checked
   * \return True if the block has child blocks, false otherwise.
   */
  static bool Check(const Block& block) {
    HasChildBlocksChecker checker;
    checker(block->body);
    return checker.has_child_blocks_;
  }

  /*!
   * \brief Get child blocks of the given block realize
   * \param realize The block to be checked
   * \return The child blocks of the given block realize
   */
  static Array<BlockRealize> GetChildBlockRealizes(const Block& block) {
    HasChildBlocksChecker checker;
    checker(block->body);
    return checker.child_blocks_;
  }

 private:
  void VisitStmt_(const BlockNode* block) final {
    has_child_blocks_ = true;
    return StmtVisitor::VisitStmt_(block);
  }

  void VisitStmt_(const BlockRealizeNode* block_realize) final {
    child_blocks_.push_back(GetRef<BlockRealize>(block_realize));
    return StmtVisitor::VisitStmt_(block_realize);
  }

  bool has_child_blocks_{false};
  Array<BlockRealize> child_blocks_;
};
/*!
 * \brief Visit Block Stmt and Find blocks that write the specific buffer
 */

class BufferInitBlockFinder : public StmtVisitor {
 public:
  /*!
   * \brief Find the blocks that write the specific buffer
   * \param stmt The statement to be visited
   * \param buffer The buffer to be found
   * \return The blocks that write the specific buffer
   */
  static Array<Block> Find(const Stmt& stmt, const Buffer& buffer) {
    const Block scope_block = Downcast<Block>(stmt);
    BufferInitBlockFinder finder(scope_block, buffer);
    finder(stmt);
    return finder.blocks_;
  }

 private:
  explicit BufferInitBlockFinder(const Block& scope_block, const Buffer& buffer)
      : scope_block_(scope_block), buffer_(buffer) {}

  void VisitStmt_(const BlockNode* block) final {
    // We assume that init block doesn't have reads.
    if (block->reads.size() != 0) {
      return StmtVisitor::VisitStmt_(block);
    }
    // Skip the block that has no writes.
    if (block->writes.size() == 0) {
      return StmtVisitor::VisitStmt_(block);
    }
    // Skip the block that is not the dominant block.
    if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
      return StmtVisitor::VisitStmt_(block);
    }
    for (const BufferRegion& buffer_region : block->writes) {
      if (buffer_region->buffer == buffer_) {
        blocks_.push_back(GetRef<Block>(block));
      }
    }
    return StmtVisitor::VisitStmt_(block);
  }

  const Block& scope_block_;
  const Buffer& buffer_;
  Array<Block> blocks_;
};

/* !
 * \brief LoopVar Class to store the loop variables related information
 * \brief loop_var The loop variable
 * \brief min The minimum value of iteration
 * \brief extent The extent of the iteration
 * \brief kind The kind of the for loop
 * \return The loop variables between stmt1 and stmt2
 */
class LoopVar {
 public:
  LoopVar() = default;
  LoopVar(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind)
      : loop_var(loop_var), min(min), extent(extent), kind(kind) {}
  /*! \brief The loop variable. */
  Var loop_var;
  /*! \brief The minimum value of iteration. */
  PrimExpr min;
  /*! \brief The extent of the iteration. */
  PrimExpr extent;
  /*! \brief The kind of the for loop. */
  ForKind kind;
};

/*!
 * \brief Collect the loop variables between stmt1 and stmt2
 * \param stmt1 The first statement
 * \param stmt2 The second statement
 * \return The loop variables between stmt1 and stmt2
 */
class LoopVarCollector : public StmtVisitor {
 public:
  static std::vector<LoopVar> Collect(const Stmt& stmt1, const Block& stmt2) {
    LoopVarCollector collector(stmt2);
    collector(stmt1);
    return collector.loop_vars_;
  }

 private:
  explicit LoopVarCollector(const Block& target_block) : target_block(target_block) {}
  void VisitStmt_(const ForNode* loop) final {
    // If loop dominant stmt2_, put loop_var into loop_vars_
    tir::PostOrderVisit(loop->body, [&](const ObjectRef& obj) {
      if (const auto* block = obj.as<tir::BlockNode>()) {
        if (block == target_block.get()) {
          loop_vars_.push_back(LoopVar(loop->loop_var, loop->min, loop->extent, loop->kind));
        }
      }
    });
    StmtVisitor::VisitStmt_(loop);
  }

  std::vector<LoopVar> loop_vars_;
  const Block& target_block;
};

/*!
 * \brief Collect Used IterVars in the given Stmt
 * \param stmt The statement to be visited
 * \param var_set The set of variables to be checked
 * \return A boolean indicating whether the given statement uses the given variables
 */
class UsedIterVarCollector : public StmtExprVisitor {
 public:
  explicit UsedIterVarCollector(std::function<bool(const VarNode*)> var_set)
      : var_set_(std::move(var_set)) {}

  static bool Collect(const Stmt& stmt, std::function<bool(const VarNode*)> var_set) {
    UsedIterVarCollector visitor(std::move(var_set));
    visitor(stmt);
    return visitor.find_var_;
  }

  static bool Collect(const PrimExpr& expr, std::function<bool(const VarNode*)> var_set) {
    UsedIterVarCollector visitor(std::move(var_set));
    visitor(expr);
    return visitor.find_var_;
  }

  void VisitStmt(const Stmt& stmt) final { StmtExprVisitor::VisitStmt(stmt); }

  void VisitExpr(const PrimExpr& e) final { StmtExprVisitor::VisitExpr(e); }

  void VisitExpr_(const VarNode* op) final { Handle(op); }

  void VisitStmt_(const BufferStoreNode* op) final {
    Handle(op->buffer->data.get());
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    Handle(op->buffer->data.get());
    ExprVisitor::VisitExpr_(op);
  }

  void Handle(const VarNode* var) {
    if (var_set_(var)) find_var_ = true;
  }

  bool find_var_{false};

 private:
  std::function<bool(const VarNode*)> var_set_;
};

/*!
 * \brief Substitute given source buffers with given target buffers respectively in the input
 * statement
 */
class BufferReplacer : private StmtExprMutator {
 public:
  static Stmt Run(Array<Buffer> src_buffers, Array<Buffer> tgt_buffers, Stmt stmt) {
    Map<Buffer, Buffer> buffer_map;
    ICHECK_EQ(src_buffers.size(), tgt_buffers.size());
    int n_buffers = src_buffers.size();
    for (int i = 0; i < n_buffers; ++i) {
      buffer_map.Set(src_buffers[i], tgt_buffers[i]);
    }
    return BufferReplacer(buffer_map)(std::move(stmt));
  }

 private:
  explicit BufferReplacer(Map<Buffer, Buffer> buffer_map) : buffer_map_(std::move(buffer_map)) {}

  PrimExpr VisitExpr_(const BufferLoadNode* load) final {
    auto it = buffer_map_.find(load->buffer);
    return it != buffer_map_.end() ? BufferLoad((*it).second, {0}) : GetRef<BufferLoad>(load);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    auto it = buffer_map_.find(store->buffer);
    if (it != buffer_map_.end()) {
      PrimExpr value = StmtExprMutator::VisitExpr(store->value);
      return BufferStore((*it).second, std::move(value), {0});
    } else {
      return StmtMutator::VisitStmt_(store);
    }
  }

  Map<Buffer, Buffer> buffer_map_;
};

/*!
 * \brief Substitute a given source block with a given target block, or remove the source block
 * branch from the AST if the target block is undefined
 */
class InThreadReducerMaker : private StmtMutator {
 public:
  /*!
   * \brief Visitor class to collect all reduction block variables under a loop.
   */
  class UnderLoopReductionBlockVarCollector : public StmtVisitor {
   public:
    /*!
     * \brief Check if the given statement has any reduction blocks.
     * \param stmt The statement to check.
     * \return True if the statement has reduction blocks, false otherwise.
     */
    static bool CheckHasReductionBlocks(const Stmt& stmt) {
      UnderLoopReductionBlockVarCollector collector;
      collector(stmt);
      return collector.reduction_block_vars_.size() > 0;
    }

   private:
    void VisitStmt_(const BlockNode* block) final {
      Array<IterVar> iter_vars = block->iter_vars;
      for (const IterVar& iter_var : block->iter_vars) {
        if (iter_var->iter_type == kCommReduce) {
          reduction_block_vars_.push_back(iter_var);
        }
      }
      StmtVisitor::VisitStmt_(block);
    }

    /*! \brief the map from thread tag to its extent */
    Array<IterVar> reduction_block_vars_;
  };

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
        UnderLoopReductionBlockVarCollector collector;
        if (!res->body.defined() || collector.CheckHasReductionBlocks(res)) {
          return res->body;
        }
        return std::move(res);
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
 * \param realize The block-realize which contains the old reduction block
 * \param it_buffers The buffers to store in-thread reduction results
 * \param ct_buffers The buffers to store cross-thread reduction results
 * \param wb_buffers The buffers to store the final reduction results
 * \param old_wb_indices The indices used to access the write-back buffers when storing the final
 * reduction results into the write-back buffers
 * \param reducer The reduction function
 * \param combiner_rhs The RHS values of the combiner
 * \param reduction_loops The reduction loops
 */
Stmt TransformReductionBlock(const BlockRealizeNode* realize,            //
                             const Optional<Array<Buffer>>& it_buffers,  //
                             const Array<Buffer>& ct_buffers,            //
                             const Array<Buffer>& wb_buffers,            //
                             const Array<PrimExpr>& old_wb_indices,      //
                             const CommReducer& reducer,                 //
                             const Array<PrimExpr>& combiner_rhs,        //
                             const std::vector<const ForNode*>& reduction_loops) {
  int n_buffers = wb_buffers.size();
  const BlockNode* block = realize->block.get();

  auto f_create_buffer_regions = [](Array<Buffer> buffers) {
    Array<BufferRegion> regions;
    regions.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      regions.push_back(BufferRegion(buffer, {Range::FromMinExtent(0, 1)}));
    }
    return regions;
  };

  Array<BufferRegion> ct_buffer_regions = f_create_buffer_regions(ct_buffers);
  Optional<Array<BufferRegion>> it_buffer_regions = NullOpt;
  if (it_buffers.defined()) {
    it_buffer_regions = f_create_buffer_regions(it_buffers.value());
  }
  // In total, the block is transformed into at most 4 statements
  // - Stmt 1: initialize the buffer for in-thread reduction
  // - Stmt 2: do in-thread reduction
  // - Stmt 3: do cross-thread reduction
  // - Stmt 4: write cross-thread reduction result to the original buffer
  Array<Stmt> stmts;
  stmts.reserve(4);
  // Stmt 1: initialize the buffer for in-thread reduction
  if (it_buffers.defined()) {
    Array<Stmt> inits;
    inits.reserve(n_buffers);
    for (int i = 0; i < n_buffers; ++i) {
      inits.push_back(
          BufferStore(it_buffers.value()[i], reducer->identity_element[i], {Integer(0)}));
    }
    stmts.push_back(BlockRealize(/*iter_values=*/{},
                                 /*predicate=*/const_true(),
                                 /*block=*/
                                 Block(/*iter_vars=*/{},
                                       /*reads=*/{},
                                       /*writes=*/it_buffer_regions.value(),
                                       /*name_hint=*/block->name_hint + "_in_thread_init",
                                       /*body=*/n_buffers > 1 ? SeqStmt(inits) : inits[0])));
  }
  // Stmt 2: do in-thread reduction
  {
    Optional<BlockRealize> new_realize = NullOpt;
    // If need to generate in-thread reduction,
    // then replace `wb_buffers` with `it_buffers` accordingly in given BlockRealize
    // otherwise, directly remove given BlockRealize
    if (it_buffers.defined()) {
      ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
      new_block->reads = std::move(new_block->reads);
      new_block->writes = it_buffer_regions.value();
      new_block->name_hint = new_block->name_hint + "_in_thread";
      new_block->body =
          BufferReplacer::Run(wb_buffers, it_buffers.value(), std::move(new_block->body));
      new_block->init = NullOpt;
      new_block->annotations.Set(kIsCrossThreadReductionApplied, Bool(true));
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
    // 1-st argument: number of buffers
    parameters.push_back(make_const(DataType::UInt(32), n_buffers));
    // Next `n_buffers` arguments: sources
    if (it_buffers.defined()) {
      for (int i = 0; i < n_buffers; ++i) {
        parameters.push_back(BufferLoad(it_buffers.value()[i], {Integer(0)}));
      }
    } else {
      parameters.insert(parameters.end(), combiner_rhs.begin(), combiner_rhs.end());
    }
    // Next argument: predicate
    parameters.push_back(const_true());
    // Next `n_buffers` arguments: destinations
    for (int i = 0; i < n_buffers; ++i) {
      parameters.push_back(BufferLoad(ct_buffers[i], {0}));
    }
    // Next arguments: all the reduction threads
    for (const ForNode* reduction_loop : reduction_loops) {
      if (reduction_loop->thread_binding.defined()) {
        parameters.push_back(reduction_loop->loop_var);
      }
    }
    // Step 3.2. Create the block and the block-realize.
    Array<IterVar> iter_vars{nullptr};
    Array<PrimExpr> bindings{nullptr};
    Array<BufferRegion> reads{nullptr};
    if (it_buffers.defined()) {
      iter_vars = Array<IterVar>{};
      bindings = Array<PrimExpr>{};
      reads = it_buffer_regions.value();
    } else {
      iter_vars = block->iter_vars;
      bindings = realize->iter_values;
      reads = block->reads;
    }
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/std::move(iter_vars),
              /*reads=*/std::move(reads),
              /*writes=*/ct_buffer_regions,
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
    Map<Var, Var> var_map;
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
    Array<Stmt> wb_updates;
    Array<BufferRegion> wb_regions;
    wb_updates.reserve(n_buffers);
    wb_regions.reserve(n_buffers);
    int n_dim = static_cast<int>(old_wb_indices.size());
    Array<Range> region = Substitute(block->writes[0]->region, var_map);
    Array<PrimExpr> wb_indices;
    wb_indices.reserve(n_dim);
    for (int d = 0; d < n_dim; ++d) {
      wb_indices.push_back(Substitute(old_wb_indices[d], var_map));
    }
    for (int i = 0; i < n_buffers; ++i) {
      wb_updates.push_back(
          BufferStore(wb_buffers[i], BufferLoad(ct_buffers[i], {Integer(0)}), wb_indices));
      wb_regions.push_back(BufferRegion(wb_buffers[i], region));
    }

    // Construct the predicate of the write-back block. It is the conjunction of
    // - each predicate clause of the original block which contains spatial loop var, and
    // - `t == 0` for each reduction thread dim when the write-back buffer is not local.
    PrimExpr wb_predicate = const_true();
    std::unordered_set<const VarNode*> reduction_loop_vars;
    reduction_loop_vars.reserve(reduction_loops.size());
    for (const ForNode* reduction_loop : reduction_loops) {
      reduction_loop_vars.insert(reduction_loop->loop_var.get());
    }
    PostOrderVisit(realize->predicate, [&wb_predicate, &reduction_loop_vars](const ObjectRef& obj) {
      if (const auto* and_node = obj.as<AndNode>()) {
        Array<PrimExpr> sub_exprs = {and_node->a, and_node->b};
        for (PrimExpr sub_expr : sub_exprs) {
          if (sub_expr->IsInstance<AndNode>()) {
            continue;
          }
          bool is_reduction = [sub_expr, &reduction_loop_vars]() {
            Array<Var> vars = UndefinedVars(sub_expr);
            for (Var var : vars) {
              if (reduction_loop_vars.find(var.get()) != reduction_loop_vars.end()) {
                return true;
              }
            }
            return false;
          }();
          if (!is_reduction) {
            wb_predicate = wb_predicate && sub_expr;
          }
        }
        return true;
      }
      return false;
    });
    if (wb_buffers[0].scope() != "local") {
      for (const ForNode* loop : reduction_loops) {
        if (loop->thread_binding.defined()) {
          wb_predicate = wb_predicate && (loop->loop_var == IntImm(loop->loop_var->dtype, 0));
        }
      }
    }

    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings),
        /*predicate=*/wb_predicate,
        /*block=*/
        Block(/*iter_vars=*/std::move(iter_vars),
              /*reads=*/std::move(ct_buffer_regions),
              /*writes=*/std::move(wb_regions),
              /*name_hint=*/block->name_hint + "_write_back",
              /*body=*/n_buffers > 1 ? SeqStmt(wb_updates) : wb_updates[0])));
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
 * \brief Inject the lowered allreduce block transformed from the input reduction block
 * \param realize The block-realize which contains the old reduction block
 * \param ct_buffers The buffers to store cross-thread reduction results
 * \param wb_buffers The buffers to store the final reduction results
 * \param old_wb_indices The indices used to access the write-back buffers when storing the final
 * reduction results into the write-back buffers
 * \param reducer The reduction function
 * \param combiner_lhs The LHS values of the combiner
 * \param reduction_loops The reduction loops
 */
Stmt InjectReductionBlock(const BlockRealizeNode* realize,                    //
                          const Array<Buffer>& ct_buffers,                    //
                          const Array<Buffer>& wb_buffers,                    //
                          const Array<PrimExpr>& old_wb_indices,              //
                          const CommReducer& reducer,                         //
                          const Array<PrimExpr>& combiner_lhs,                //
                          const std::vector<const ForNode*>& reduction_loops  //
) {
  int n_buffers = wb_buffers.size();
  const BlockNode* block = realize->block.get();

  auto f_create_buffer_regions = [](Array<Buffer> buffers) {
    Array<BufferRegion> regions;
    regions.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      regions.push_back(BufferRegion(buffer, {Range::FromMinExtent(0, 1)}));
    }
    return regions;
  };

  Array<BufferRegion> ct_buffer_regions = f_create_buffer_regions(ct_buffers);
  Optional<Array<BufferRegion>> it_buffer_regions = NullOpt;
  // In total, the block is transformed into at most 4 statements
  // - Stmt 1: do cross-thread reduction
  // - Stmt 2: write cross-thread reduction result to the original buffer
  Array<Stmt> stmts;
  stmts.reserve(4);

  // Stmt 1: do cross-thread reduction
  {
    // Step 1.1. Create the parameters to the intrinsic
    Array<PrimExpr> parameters;
    parameters.reserve(reduction_loops.size() + 4);
    // 1-st argument: number of buffers
    parameters.push_back(make_const(DataType::UInt(32), n_buffers));
    // Next `n_buffers` arguments: sources
    parameters.insert(parameters.end(), combiner_lhs.begin(), combiner_lhs.end());
    // Next argument: predicate
    parameters.push_back(const_true());
    // Next `n_buffers` arguments: destinations
    for (int i = 0; i < n_buffers; ++i) {
      parameters.push_back(BufferLoad(ct_buffers[i], {0}));
    }
    // Next arguments: all the reduction threads
    for (const ForNode* reduction_loop : reduction_loops) {
      if (reduction_loop->thread_binding.defined()) {
        parameters.push_back(reduction_loop->loop_var);
      }
    }
    // Step 1.2. Create the block and the block-realize.
    Array<IterVar> iter_vars = block->iter_vars;
    Array<PrimExpr> bindings = realize->iter_values;
    Array<BufferRegion> reads = block->writes;

    // Blockized block should also be considered
    if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
      Array<BlockRealize> child_blocks =
          HasChildBlocksChecker::GetChildBlockRealizes(GetRef<Block>(block));

      ICHECK_GT(child_blocks.size(), 0) << "Child blocks should be more than 0";

      // If has ChildBlocks, the reads should be analyzed from the child blocks
      reads.clear();
      for (BlockRealize child_block : child_blocks) {
        Array<IterVar> child_iter_vars = child_block->block->iter_vars;
        Array<PrimExpr> child_bindings = child_block->iter_values;
        iter_vars.insert(iter_vars.end(), child_iter_vars.begin(), child_iter_vars.end());
        bindings.insert(bindings.end(), child_bindings.begin(), child_bindings.end());
        reads.insert(reads.end(), child_block->block->writes.begin(),
                     child_block->block->writes.end());
      }
    }

    // Remove unused iter vars which introduced by blockize
    // otherwise may generate duplicated for loops
    Array<IterVar> iter_vars_used;
    Array<PrimExpr> bindings_used;
    auto f_inject = [&iter_vars, &bindings, &iter_vars_used,
                     &bindings_used](const VarNode* var) -> bool {
      for (size_t i = 0; i < iter_vars.size(); ++i) {
        const IterVar& iter_var = iter_vars[i];
        if (iter_var->var.get() == var) {
          if (std::find(iter_vars_used.begin(), iter_vars_used.end(), iter_var) ==
              iter_vars_used.end()) {
            iter_vars_used.push_back(iter_var);
            bindings_used.push_back(bindings[i]);
          }
          return true;
        }
      }
      return false;
    };
    for (const BufferRegion& read : reads) {
      for (const Range& range : read->region) {
        UsedIterVarCollector::Collect(range->min, f_inject);
        UsedIterVarCollector::Collect(range->extent, f_inject);
      }
    }

    Block cross_thread_block =
        Block(/*iter_vars=*/std::move(iter_vars_used),
              /*reads=*/std::move(reads),
              /*writes=*/ct_buffer_regions,
              /*name_hint=*/block->name_hint + "_cross_thread",
              /*body=*/
              AttrStmt(/*node=*/reducer,
                       /*attr_key=*/tir::attr::reduce_scope,
                       /*value=*/make_zero(DataType::Handle()),
                       /*body=*/
                       Evaluate(Call(/*dtype=*/DataType::Handle(),
                                     /*op=*/tir::builtin::tvm_thread_allreduce(),
                                     /*args=*/std::move(parameters)))));
    ObjectPtr<BlockNode> cross_thread_block_node =
        make_object<BlockNode>(*cross_thread_block.operator->());
    cross_thread_block_node->annotations.Set(kIsCrossThreadReductionApplied, tir::const_true());
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings_used),
        /*predicate=*/const_true(),
        /*block=*/cross_thread_block));
  }
  // Stmt 2: write cross-thread reduction result to the original buffer
  {
    ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    int n_iter = static_cast<int>(block->iter_vars.size());
    Array<IterVar> iter_vars;
    Array<PrimExpr> bindings;
    Map<Var, Var> var_map;
    Array<Range> write_region = block->writes[0]->region;
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

    if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
      Array<BlockRealize> child_blocks =
          HasChildBlocksChecker::GetChildBlockRealizes(GetRef<Block>(block));
      ICHECK_GT(child_blocks.size(), 0) << "Child blocks should be more than 0";

      for (BlockRealize child_block : child_blocks) {
        Array<IterVar> child_iter_vars = child_block->block->iter_vars;
        Array<PrimExpr> child_bindings = child_block->iter_values;
        iter_vars.insert(iter_vars.end(), child_iter_vars.begin(), child_iter_vars.end());
        bindings.insert(bindings.end(), child_bindings.begin(), child_bindings.end());
        write_region = child_block->block->writes[0]->region;
      }
    }
    Array<Stmt> wb_updates;
    Array<BufferRegion> wb_regions;
    wb_updates.reserve(n_buffers);
    wb_regions.reserve(n_buffers);
    int n_dim = static_cast<int>(old_wb_indices.size());
    Array<Range> region = Substitute(write_region, var_map);
    Array<PrimExpr> wb_indices;
    wb_indices.reserve(n_dim);
    for (int d = 0; d < n_dim; ++d) {
      wb_indices.push_back(Substitute(old_wb_indices[d], var_map));
    }
    for (int i = 0; i < n_buffers; ++i) {
      wb_updates.push_back(
          BufferStore(wb_buffers[i], BufferLoad(ct_buffers[i], {Integer(0)}), wb_indices));
      wb_regions.push_back(BufferRegion(wb_buffers[i], region));
    }

    // Construct the predicate of the write-back block. It is the conjunction of
    // - each predicate clause of the original block which contains spatial loop var, and
    // - `t == 0` for each reduction thread dim when the write-back buffer is not local.
    PrimExpr wb_predicate = const_true();
    std::unordered_set<const VarNode*> reduction_loop_vars;
    reduction_loop_vars.reserve(reduction_loops.size());
    for (const ForNode* reduction_loop : reduction_loops) {
      reduction_loop_vars.insert(reduction_loop->loop_var.get());
    }
    PostOrderVisit(realize->predicate, [&wb_predicate, &reduction_loop_vars](const ObjectRef& obj) {
      if (const auto* and_node = obj.as<AndNode>()) {
        Array<PrimExpr> sub_exprs = {and_node->a, and_node->b};
        for (PrimExpr sub_expr : sub_exprs) {
          if (sub_expr->IsInstance<AndNode>()) {
            continue;
          }
          bool is_reduction = [sub_expr, &reduction_loop_vars]() {
            Array<Var> vars = UndefinedVars(sub_expr);
            for (Var var : vars) {
              if (reduction_loop_vars.find(var.get()) != reduction_loop_vars.end()) {
                return true;
              }
            }
            return false;
          }();
          if (!is_reduction) {
            wb_predicate = wb_predicate && sub_expr;
          }
        }
        return true;
      }
      return false;
    });
    if (wb_buffers[0].scope() != "local") {
      for (const ForNode* loop : reduction_loops) {
        if (loop->thread_binding.defined()) {
          wb_predicate = wb_predicate && (loop->loop_var == IntImm(loop->loop_var->dtype, 0));
        }
      }
    }
    // Remove unused iter vars which introduced by blockize
    // otherwise may generate duplicated for loops
    Array<IterVar> iter_vars_used;
    Array<PrimExpr> bindings_used;
    auto f_inject = [&iter_vars, &bindings, &iter_vars_used,
                     &bindings_used](const VarNode* var) -> bool {
      for (size_t i = 0; i < iter_vars.size(); ++i) {
        const IterVar& iter_var = iter_vars[i];
        if (iter_var->var.get() == var) {
          if (std::find(iter_vars_used.begin(), iter_vars_used.end(), iter_var) ==
              iter_vars_used.end()) {
            iter_vars_used.push_back(iter_var);
            bindings_used.push_back(bindings[i]);
          }
          return true;
        }
      }
      return false;
    };
    for (const BufferRegion& write : wb_regions) {
      for (const Range& range : write->region) {
        UsedIterVarCollector::Collect(range->min, f_inject);
        UsedIterVarCollector::Collect(range->extent, f_inject);
      }
    }
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings_used),
        /*predicate=*/wb_predicate,
        /*block=*/
        Block(/*iter_vars=*/std::move(iter_vars_used),
              /*reads=*/std::move(ct_buffer_regions),
              /*writes=*/std::move(wb_regions),
              /*name_hint=*/block->name_hint + "_write_back",
              /*body=*/n_buffers > 1 ? SeqStmt(wb_updates) : wb_updates[0])));
  }
  // Final step: Wrap all the above four statements with the reduction loops bound to threadIdx
  Stmt new_stmt = Stmt();
  for (auto rit = reduction_loops.rbegin(); rit != reduction_loops.rend(); ++rit) {
    const ForNode* loop = *rit;
    if (loop->thread_binding.defined()) {
      // Colelct Loop vars between the reduction loops
      std::vector<LoopVar> chain_loop_vars =
          LoopVarCollector::Collect(loop->body, GetRef<Block>(block));
      std::vector<LoopVar> used_chain_loop_vars_array;
      if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
        chain_loop_vars.clear();
        Array<BlockRealize> child_blocks =
            HasChildBlocksChecker::GetChildBlockRealizes(GetRef<Block>(block));
        for (BlockRealize child_block : child_blocks) {
          std::vector<LoopVar> child_loop_vars =
              LoopVarCollector::Collect(loop->body, child_block->block);
          chain_loop_vars.insert(chain_loop_vars.end(), child_loop_vars.begin(),
                                 child_loop_vars.end());
        }
      }

      // Remove Unused Loop from the chain loops, otherwise may generate duplicated for loops
      for (auto it = chain_loop_vars.begin(); it != chain_loop_vars.end(); ++it) {
        Var target_var = (*it).loop_var;
        auto f_find = [&target_var](const VarNode* var) -> bool {
          if (target_var.get() == var) {
            return true;
          }
          return false;
        };
        for (const Stmt& stmt : stmts) {
          if (UsesVar(stmt, f_find)) {
            used_chain_loop_vars_array.push_back(*it);
            break;
          }
        }
      }
      chain_loop_vars = used_chain_loop_vars_array;

      ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
      if (chain_loop_vars.size() == 0) {
        stmts.insert(stmts.begin(), n->body);
        new_stmt = SeqStmt::Flatten(std::move(stmts));
        n->body = std::move(new_stmt);
        new_stmt = For(n);
        break;
      } else {
        new_stmt = SeqStmt::Flatten(std::move(stmts));
        For new_for = For(chain_loop_vars.back().loop_var, chain_loop_vars.back().min,
                          chain_loop_vars.back().extent, chain_loop_vars.back().kind, new_stmt);

        ObjectPtr<ForNode> current_loop = make_object<ForNode>(*new_for.get());
        for (int i = chain_loop_vars.size() - 2; i >= 0; i--) {
          new_for = For(chain_loop_vars[i].loop_var, chain_loop_vars[i].min,
                        chain_loop_vars[i].extent, chain_loop_vars[i].kind, new_for);
        }
        new_stmt = SeqStmt::Flatten(std::move((SeqStmt({std::move(n->body), std::move(new_for)}))));
        n->body = std::move(new_stmt);
        new_stmt = For(n);
        break;
      }
    }
  }
  return new_stmt;
}

/*!
 * \brief Inject the lowered warp evaluate allreduce block transformed from the input reduction
 * block
 * \param realize The block-realize which contains the old reduction block
 * \param ct_buffers The buffers to store cross-thread reduction results
 * \param wb_buffers The buffers to store the final reduction results
 * \param old_wb_indices The indices used to access the write-back buffers when storing the final
 * reduction results into the write-back buffers
 * \param reducer The reduction function
 * \param combiner_lhs The LHS values of the combiner
 * \param reduction_loops The reduction loops
 * \param used_block The block used in the warp evaluate (init block)
 */
Stmt InjectWarpEvaluateReductionBlock(const BlockRealizeNode* realize,                    //
                                      const Array<Buffer>& ct_buffers,                    //
                                      const Array<Buffer>& wb_buffers,                    //
                                      const Array<PrimExpr>& old_wb_indices,              //
                                      const CommReducer& reducer,                         //
                                      const Array<PrimExpr>& combiner_evaluate,           //
                                      const std::vector<const ForNode*>& reduction_loops  //
) {
  int n_buffers = wb_buffers.size();
  const BlockNode* block = realize->block.get();
  Buffer write_buffer = (*block->writes.begin())->buffer;
  PrimExpr warp_size = write_buffer->shape[write_buffer->shape.size() - 2];
  PrimExpr local_size = write_buffer->shape[write_buffer->shape.size() - 1];
  // Create IterVars
  Var v_lane_id = Var("v_lane_id");
  Var v_local_id = Var("v_local_id");
  // Create Bindings
  Var ax_lane_id = Var("ax_lane_id");
  Var ax_local_id = Var("ax_local_id");
  // Create IterVar
  IterVar lane_id = IterVar(Range(0, warp_size), v_lane_id, kDataPar, "threadIdx.x");
  IterVar local_id = IterVar(Range(0, local_size), v_local_id, kDataPar);

  auto f_create_buffer_regions = [](Array<Buffer> buffers) {
    Array<BufferRegion> regions;
    regions.reserve(buffers.size());
    for (const Buffer& buffer : buffers) {
      regions.push_back(BufferRegion(buffer, {Range::FromMinExtent(0, 1)}));
    }
    return regions;
  };

  Array<BufferRegion> ct_buffer_regions = f_create_buffer_regions(ct_buffers);
  Optional<Array<BufferRegion>> it_buffer_regions = NullOpt;
  // In total, the block is transformed into at most 4 statements
  // - Stmt 1: do cross-thread reduction
  // - Stmt 2: write cross-thread reduction result to the original buffer
  Array<Stmt> stmts;
  stmts.reserve(4);

  // Stmt 1: do cross-thread reduction
  {
    // Step 1.1. Create the parameters to the intrinsic
    Array<PrimExpr> parameters;

    // Step 1.2. Create the block and the block-realize.
    Array<IterVar> iter_vars = block->iter_vars;
    Array<PrimExpr> bindings = realize->iter_values;
    Array<BufferRegion> reads = block->writes;

    // Blockized block should also be considered
    if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
      Array<BlockRealize> child_blocks =
          HasChildBlocksChecker::GetChildBlockRealizes(GetRef<Block>(block));

      ICHECK_GT(child_blocks.size(), 0) << "Child blocks should be more than 0";

      // If has ChildBlocks, the reads should be analyzed from the child blocks
      reads.clear();
      for (BlockRealize child_block : child_blocks) {
        Array<IterVar> child_iter_vars = child_block->block->iter_vars;
        Array<PrimExpr> child_bindings = child_block->iter_values;
        iter_vars.insert(iter_vars.end(), child_iter_vars.begin(), child_iter_vars.end());
        bindings.insert(bindings.end(), child_bindings.begin(), child_bindings.end());
        reads.insert(reads.end(), child_block->block->writes.begin(),
                     child_block->block->writes.end());
      }
    }

    // Remove unused iter vars which introduced by blockize
    // otherwise may generate duplicated for loops
    Array<IterVar> iter_vars_used;
    Array<PrimExpr> bindings_used;
    auto f_inject = [&iter_vars, &bindings, &iter_vars_used,
                     &bindings_used](const VarNode* var) -> bool {
      for (size_t i = 0; i < iter_vars.size(); ++i) {
        const IterVar& iter_var = iter_vars[i];
        if (iter_var->var.get() == var) {
          if (std::find(iter_vars_used.begin(), iter_vars_used.end(), iter_var) ==
              iter_vars_used.end()) {
            iter_vars_used.push_back(iter_var);
            bindings_used.push_back(bindings[i]);
          }
          return true;
        }
      }
      return false;
    };
    for (const BufferRegion& read : reads) {
      for (const Range& range : read->region) {
        UsedIterVarCollector::Collect(range->min, f_inject);
        UsedIterVarCollector::Collect(range->extent, f_inject);
      }
    }

    // Create the vars for the warp
    // Collect the Warp Information from the Buffer
    // The threadIdx and Warp Information is stored in the last two dimensions of the write buffer

    iter_vars_used.push_back(lane_id);
    iter_vars_used.push_back(local_id);

    bindings_used.push_back(ax_lane_id);
    bindings_used.push_back(ax_local_id);

    Array<PrimExpr> combiner_warp;
    for (size_t i = 0; i < combiner_evaluate.size(); i++) {
      Array<PrimExpr> new_indices;  // indices that maintain the new indices with warp information
      for (size_t j = 0; j < combiner_evaluate[i].as<BufferLoadNode>()->indices.size() - 2; j++) {
        new_indices.push_back(combiner_evaluate[i].as<BufferLoadNode>()->indices[j]);
      }
      new_indices.push_back(v_lane_id);
      new_indices.push_back(v_local_id);
      combiner_warp.push_back(
          BufferLoad(combiner_evaluate[i].as<BufferLoadNode>()->buffer, new_indices));
    }
    // Create the parameters to the intrinsic
    parameters.reserve(reduction_loops.size() + 4);
    // 1-st argument: number of buffers
    parameters.push_back(make_const(DataType::UInt(32), n_buffers));
    // Next `n_buffers` arguments: sources
    parameters.insert(parameters.end(), combiner_warp.begin(), combiner_warp.end());
    // Next argument: predicate
    parameters.push_back(const_true());
    // Next `n_buffers` arguments: destinations
    for (int i = 0; i < n_buffers; ++i) {
      parameters.push_back(BufferLoad(ct_buffers[i], {0}));
    }
    // Next arguments: all the reduction threads
    for (const ForNode* reduction_loop : reduction_loops) {
      if (reduction_loop->thread_binding.defined()) {
        parameters.push_back(reduction_loop->loop_var);
      }
    }
    // update param indices
    Block cross_thread_block =
        Block(/*iter_vars=*/std::move(iter_vars_used),
              /*reads=*/std::move(reads),
              /*writes=*/ct_buffer_regions,
              /*name_hint=*/block->name_hint + "_cross_thread",
              /*body=*/
              AttrStmt(/*node=*/reducer,
                       /*attr_key=*/tir::attr::reduce_scope,
                       /*value=*/make_zero(DataType::Handle()),
                       /*body=*/
                       Evaluate(Call(/*dtype=*/DataType::Handle(),
                                     /*op=*/tir::builtin::tvm_thread_allreduce(),
                                     /*args=*/std::move(parameters)))));

    ObjectPtr<BlockNode> cross_thread_block_node =
        make_object<BlockNode>(*cross_thread_block.operator->());
    cross_thread_block_node->annotations.Set(kIsCrossThreadReductionApplied, tir::const_true());
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings_used),
        /*predicate=*/const_true(),
        /*block=*/cross_thread_block));
  }
  // Stmt 2: write cross-thread reduction result to the original buffer
  {
    ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    int n_iter = static_cast<int>(block->iter_vars.size());
    Array<IterVar> iter_vars;
    Array<PrimExpr> bindings;
    Array<Range> write_region = block->writes[0]->region;
    iter_vars.reserve(n_iter);
    bindings.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != kCommReduce) {
        iter_vars.push_back(iter_var);
        bindings.push_back(binding);
      }
    }

    if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
      Array<BlockRealize> child_blocks =
          HasChildBlocksChecker::GetChildBlockRealizes(GetRef<Block>(block));
      ICHECK_GT(child_blocks.size(), 0) << "Child blocks should be more than 0";

      for (BlockRealize child_block : child_blocks) {
        Array<IterVar> child_iter_vars = child_block->block->iter_vars;
        Array<PrimExpr> child_bindings = child_block->iter_values;
        iter_vars.insert(iter_vars.end(), child_iter_vars.begin(), child_iter_vars.end());
        bindings.insert(bindings.end(), child_bindings.begin(), child_bindings.end());
        write_region = child_block->block->writes[0]->region;
      }
    }
    Array<Stmt> wb_updates;
    Array<BufferRegion> wb_regions;
    wb_updates.reserve(n_buffers);
    wb_regions.reserve(n_buffers);

    Array<Range> region = write_region;
    Array<PrimExpr> wb_indices;
    for (size_t i = 0; i < block->writes[0]->region.size() - 2; i++) {
      Range range = block->writes[0]->region[i];
      wb_indices.push_back(range->min);
    }
    // Append the warp and local id to the indices
    wb_indices.push_back(v_lane_id);
    wb_indices.push_back(v_local_id);
    for (int i = 0; i < n_buffers; ++i) {
      wb_updates.push_back(
          BufferStore(wb_buffers[i], BufferLoad(ct_buffers[i], {Integer(0)}), wb_indices));
      wb_regions.push_back(BufferRegion(wb_buffers[i], region));
    }

    // Construct the predicate of the write-back block. It is the conjunction of
    // - each predicate clause of the original block which contains spatial loop var, and
    // - `t == 0` for each reduction thread dim when the write-back buffer is not local.
    PrimExpr wb_predicate = const_true();
    std::unordered_set<const VarNode*> reduction_loop_vars;
    reduction_loop_vars.reserve(reduction_loops.size());
    for (const ForNode* reduction_loop : reduction_loops) {
      reduction_loop_vars.insert(reduction_loop->loop_var.get());
    }
    PostOrderVisit(realize->predicate, [&wb_predicate, &reduction_loop_vars](const ObjectRef& obj) {
      if (const auto* and_node = obj.as<AndNode>()) {
        Array<PrimExpr> sub_exprs = {and_node->a, and_node->b};
        for (PrimExpr sub_expr : sub_exprs) {
          if (sub_expr->IsInstance<AndNode>()) {
            continue;
          }
          bool is_reduction = [sub_expr, &reduction_loop_vars]() {
            Array<Var> vars = UndefinedVars(sub_expr);
            for (Var var : vars) {
              if (reduction_loop_vars.find(var.get()) != reduction_loop_vars.end()) {
                return true;
              }
            }
            return false;
          }();
          if (!is_reduction) {
            wb_predicate = wb_predicate && sub_expr;
          }
        }
        return true;
      }
      return false;
    });
    if (wb_buffers[0].scope() != "local") {
      for (const ForNode* loop : reduction_loops) {
        if (loop->thread_binding.defined()) {
          wb_predicate = wb_predicate && (loop->loop_var == IntImm(loop->loop_var->dtype, 0));
        }
      }
    }
    // Remove unused iter vars which introduced by blockize
    // otherwise may generate duplicated for loops
    Array<IterVar> iter_vars_used;
    Array<PrimExpr> bindings_used;
    auto f_inject = [&iter_vars, &bindings, &iter_vars_used,
                     &bindings_used](const VarNode* var) -> bool {
      for (size_t i = 0; i < iter_vars.size(); ++i) {
        const IterVar& iter_var = iter_vars[i];
        if (iter_var->var.get() == var) {
          if (std::find(iter_vars_used.begin(), iter_vars_used.end(), iter_var) ==
              iter_vars_used.end()) {
            iter_vars_used.push_back(iter_var);
            bindings_used.push_back(bindings[i]);
          }
          return true;
        }
      }
      return false;
    };
    for (const BufferRegion& write : wb_regions) {
      for (const Range& range : write->region) {
        UsedIterVarCollector::Collect(range->min, f_inject);
        UsedIterVarCollector::Collect(range->extent, f_inject);
      }
    }

    iter_vars_used.push_back(IterVar(Range(0, warp_size), v_lane_id, kDataPar, "threadIdx.x"));
    iter_vars_used.push_back(IterVar(Range(0, local_size), v_local_id, kDataPar));

    bindings_used.push_back(ax_lane_id);
    bindings_used.push_back(ax_local_id);
    stmts.push_back(BlockRealize(
        /*iter_values=*/std::move(bindings_used),
        /*predicate=*/wb_predicate,
        /*block=*/
        Block(/*iter_vars=*/std::move(iter_vars_used),
              /*reads=*/std::move(ct_buffer_regions),
              /*writes=*/std::move(wb_regions),
              /*name_hint=*/block->name_hint + "_write_back",
              /*body=*/n_buffers > 1 ? SeqStmt(wb_updates) : wb_updates[0])));
  }
  // Final step: Wrap all the above four statements with the reduction loops bound to threadIdx
  Stmt new_stmt = Stmt();
  for (auto rit = reduction_loops.rbegin(); rit != reduction_loops.rend(); ++rit) {
    const ForNode* loop = *rit;
    if (loop->thread_binding.defined()) {
      // Colelct Loop vars between the reduction loops
      std::vector<LoopVar> chain_loop_vars =
          LoopVarCollector::Collect(loop->body, GetRef<Block>(block));
      std::vector<LoopVar> used_chain_loop_vars_array;
      if (HasChildBlocksChecker::Check(GetRef<Block>(block))) {
        chain_loop_vars.clear();
        Array<BlockRealize> child_blocks =
            HasChildBlocksChecker::GetChildBlockRealizes(GetRef<Block>(block));
        for (BlockRealize child_block : child_blocks) {
          std::vector<LoopVar> child_loop_vars =
              LoopVarCollector::Collect(loop->body, child_block->block);
          chain_loop_vars.insert(chain_loop_vars.end(), child_loop_vars.begin(),
                                 child_loop_vars.end());
        }
      }

      // Remove Unused Loop from the chain loops, otherwise may generate duplicated for loops
      for (auto it = chain_loop_vars.begin(); it != chain_loop_vars.end(); ++it) {
        Var target_var = (*it).loop_var;
        auto f_find = [&target_var](const VarNode* var) -> bool {
          if (target_var.get() == var) {
            return true;
          }
          return false;
        };
        for (const Stmt& stmt : stmts) {
          if (UsesVar(stmt, f_find)) {
            used_chain_loop_vars_array.push_back(*it);
            break;
          }
        }
      }
      chain_loop_vars = used_chain_loop_vars_array;
      // append warp related loops
      chain_loop_vars.push_back(LoopVar(ax_lane_id, IntImm(loop->loop_var->dtype, 0), warp_size,
                                        ForKind::kThreadBinding));
      chain_loop_vars.push_back(
          LoopVar(ax_local_id, IntImm(loop->loop_var->dtype, 0), local_size, ForKind::kSerial));
      ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
      if (chain_loop_vars.size() == 0) {
        stmts.insert(stmts.begin(), n->body);
        new_stmt = SeqStmt::Flatten(std::move(stmts));
        n->body = std::move(new_stmt);
        new_stmt = For(n);
        break;
      } else {
        new_stmt = SeqStmt::Flatten(std::move(stmts));
        For new_for = For(chain_loop_vars.back().loop_var, chain_loop_vars.back().min,
                          chain_loop_vars.back().extent, chain_loop_vars.back().kind, new_stmt);

        ObjectPtr<ForNode> current_loop = make_object<ForNode>(*new_for.get());
        for (int i = chain_loop_vars.size() - 2; i >= 0; i--) {
          LoopVar loop_var = chain_loop_vars[i];
          if (loop_var.kind == ForKind::kThreadBinding) {
            new_for = For(loop_var.loop_var, loop_var.min, loop_var.extent, loop_var.kind, new_for,
                          lane_id);
          } else {
            new_for = For(loop_var.loop_var, loop_var.min, loop_var.extent, loop_var.kind, new_for);
          }
        }
        new_stmt = SeqStmt::Flatten(std::move((SeqStmt({std::move(n->body), std::move(new_for)}))));
        n->body = std::move(new_stmt);
        new_stmt = For(n);
        break;
      }
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

  // Check if the input block needs thread broadcast rewrite.
  // One block needs broadcast rewrite when
  // 1. it consumes a buffer produced by cross-thread reduction under
  // the same kernel (i.e., same group of blockIdx),
  // 2. it writes to non-local memory,
  // 3. at least one of the reduction thread vars of the cross-thread reduction
  // is free to this block (i.e., not bound to the block).
  std::vector<std::pair<ThreadScope, Range>> NeedCrossThreadBroadcast(
      const BlockRealizeNode* realize) {
    Block block = realize->block;

    // If the block writes to local memory, no rewrite is needed.
    for (BufferRegion write_region : block->writes) {
      if (write_region->buffer.scope() == "local") {
        return {};
      }
    }

    // Find out the reduction threads for the read-buffers which are produced by
    // cross-thread reduction.
    std::unordered_map<ThreadScope, Range, ThreadScopeHash, ThreadScopeEqual> thread2range;
    for (BufferRegion read_region : block->reads) {
      auto buf_it = crt_buf2threads_.find(read_region->buffer.get());
      if (buf_it == crt_buf2threads_.end()) {
        continue;
      }
      for (auto [scope, range] : buf_it->second) {
        thread2range[scope] = range;
      }
    }

    // Erase those threads which are not free to this block.
    for (const ForNode* loop : loop_stack_) {
      if (loop->thread_binding.defined()) {
        ThreadScope scope = ThreadScope::Create(loop->thread_binding.value()->thread_tag);
        thread2range.erase(scope);
      }
    }
    std::vector<std::pair<ThreadScope, Range>> unbound_thread2range_list;
    for (auto [scope, range] : thread2range) {
      unbound_thread2range_list.emplace_back(scope, range);
    }
    return unbound_thread2range_list;
  }

  /*!
   * \brief Given that the input block needs cross-thread reduction, check if cross-thread reduction
   * can be applied to the block (i.e., the block satisfies all necessary conditions of cross-thread
   * reduction)
   * \param block The block to be checked
   * \param reduction_loops The reduction loops above the block
   * \return A tuple consisting of five elements:
   *  - an integer which indicates the number of reduction loops that are bound to thread axes,
   *  - the detected commutative reducer of the reduction,
   *  - the reduction buffers which store the reduction results,
   *  - the RHS values of the reduction updates,
   *  - the indices which is used to access the reduction buffers when storing the reduction results
   */
  std::tuple<int, CommReducer, Array<Buffer>, Array<PrimExpr>, Array<PrimExpr>, Array<PrimExpr>,
             Array<PrimExpr>>
  CheckCanApplyCrossThreadReduction(const BlockNode* block,
                                    const std::vector<const ForNode*>& reduction_loops) const {
    // To implement the block reduction with Tiling, the reduction-related may not be the deepest
    // Condition 1. All the reduction-related loops should be the deepest among all statements
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
        continue;
      }
      ++n_deepest_reduction_loops;
    }

    // Condition 2. All the reduction-related loops that are bound to thread axes should only be
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

    // Condition 3. Get the identity values of the block init and the BufferStore block combiner
    // updates of the reduction. Extract the commutative reducer, combiner lhs and combiner rhs from
    // the reduction identities and the reduction combiner.
    Array<PrimExpr> init_values{nullptr};
    Array<BufferStore> updates{nullptr};
    CommReducer reducer{nullptr};
    Array<PrimExpr> combiner_lhs{nullptr};
    Array<PrimExpr> combiner_rhs{nullptr};
    Array<PrimExpr> combiner_evaluate{nullptr};  // combiner for mma like operations
    std::unordered_set<const BlockNode*> used_blocks;
    bool has_mma = false;

    if (block->init.defined()) {
      std::tie(init_values, updates) =
          GetInitValuesAndUpdatesFromReductionBlock(NullOpt, GetRef<Block>(block));
    } else {
      // check the block is not cross_thread_reduction block
      CHECK(!block->annotations.Get(kIsCrossThreadReductionApplied))
          << "ValueError: Cross-thread reduction requires all reduction blocks to have init "
             "statements. However, here the block "
          << block->name_hint << " is a cross-thread reduction block but has no init statement.";
      // get write buffer
      Array<BufferRegion> writes = block->writes;
      // block_stack usually only have root block
      for (BufferRegion write : writes) {
        used_blocks.clear();
        // find other blocks that used the write buffer
        for (const auto* stack_block : block_stack_) {
          Array<Block> blocks =
              BufferInitBlockFinder::Find(GetRef<Block>(stack_block), write->buffer);
          for (const Block& b : blocks) {
            if (b != GetRef<Block>(block) && b != GetRef<Block>(stack_block)) {
              used_blocks.insert(b.get());
            }
          }
        }

        for (auto used_block : used_blocks) {
          has_mma = InitUpdateValueFinder::CheckHasMMA(GetRef<Block>(used_block));
          init_values.push_back(InitUpdateValueFinder::FindInit(GetRef<Block>(used_block)));
          if (!has_mma) {
            updates.push_back(InitUpdateValueFinder::FindUpdate(GetRef<Block>(block)));
          }
        }
      }
    }

    if (!has_mma) {
      std::tie(reducer, combiner_lhs, combiner_rhs) =
          GetReducerAndCombinerLhsRhs(NullOpt, init_values, updates);
    } else {
      // Create a dummy combiner for MMA
      auto span = block->span;
      Var x("x", init_values.back().dtype(), span), y("y", init_values.back().dtype(), span);
      PrimExpr result = tir::Add(x, y, span);
      PrimExpr identity_element = make_zero(init_values.back().dtype(), span);
      reducer = tir::CommReducer({x}, {y}, {result}, {identity_element}, span);
      // Create BufferLoad Node From Update Buffer
      Buffer init_buffer = block->writes[0]->buffer;
      Array<PrimExpr> indices;
      for (auto range : block->writes[0]->region) {
        indices.push_back(range->min);
      }
      combiner_evaluate.push_back(BufferLoad(init_buffer, indices));
    }

    // Condition 4. All reduction buffers should be all local or all non-local.
    int is_local_buf = -1;
    Array<Buffer> reduction_buffers;
    reduction_buffers.reserve(updates.size());

    Array<PrimExpr> wb_indices{nullptr};
    if (updates.defined()) {
      wb_indices = updates[0]->indices;
      for (const BufferStore& buf_store : updates) {
        reduction_buffers.push_back(buf_store->buffer);
        if (buf_store->buffer.scope() == "local") {
          CHECK_NE(is_local_buf, 0)
              << "ValueError: Cross-thread reduction requires all reduction buffers to be all "
                 "local or all non-local. However, here some buffer is local while some buffer is "
                 "shared or global.";
          is_local_buf = 1;
        } else {
          CHECK_NE(is_local_buf, 1)
              << "ValueError: Cross-thread reduction requires all reduction buffers to be all "
                 "local or all non-local. However, here some buffer is local while some buffer is "
                 "shared or global.";
          is_local_buf = 0;
        }
      }
    } else {
      // otherwise the reduction buffer is the same as the init write buffer
      for (const BufferRegion& write : block->writes) {
        reduction_buffers.push_back(write->buffer);
      }
    }

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

    return std::make_tuple(n_bound_reduction_loops,       //
                           std::move(reducer),            //
                           std::move(reduction_buffers),  //
                           std::move(combiner_lhs),       //
                           std::move(combiner_rhs),       //
                           std::move(combiner_evaluate),  //
                           wb_indices);
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

    // Collect loop-thread information:
    // - when encountering a threadIdx loop, we keep note of its domain and
    // the "loop var -> thread scope" relation, in order to collect all existing
    // threads within a thread block.
    // - we are careful about thread block boundary for safety.
    bool is_block_idx = false;
    bool is_thread_idx = false;
    if (loop->kind == ForKind::kThreadBinding) {
      ThreadScope scope = ThreadScope::Create(loop->thread_binding.value()->thread_tag);
      if (scope.rank == 1 && scope.dim_index >= 0) {
        is_thread_idx = true;
        ++thread_idx_depth;
      } else if (scope.rank == 0) {
        is_block_idx = true;
        ++block_idx_depth;
      }
    }

    Stmt result = StmtMutator::VisitStmt_(loop);
    loop_stack_.pop_back();
    loop_range_map_.erase(loop->loop_var);
    if (is_thread_idx) {
      --thread_idx_depth;
    }
    if (is_block_idx) {
      --block_idx_depth;
    }
    if (is_block_idx || (is_thread_idx && thread_idx_depth == 0 && block_idx_depth == 0)) {
      crt_buf2threads_.clear();
    }

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

  void MakeCrossThreadReduction(const BlockRealizeNode* realize,
                                const std::vector<const ForNode*> reduction_loops) {
    const BlockNode* block = realize->block.get();

    // Step 1. Check whether cross-thread reduction can be applied. If no, throw an exception on
    // which condition the block violates.
    int n_bound_reduction_loops = 0;
    CommReducer reducer{nullptr};
    Array<Buffer> reduction_buffers{nullptr};
    Array<PrimExpr> combiner_lhs{nullptr};
    Array<PrimExpr> combiner_rhs{nullptr};
    Array<PrimExpr> combiner_evaluate{nullptr};
    Array<PrimExpr> wb_indices{nullptr};
    std::tie(n_bound_reduction_loops, reducer, reduction_buffers, combiner_lhs, combiner_rhs,
             combiner_evaluate, wb_indices) =
        CheckCanApplyCrossThreadReduction(block, reduction_loops);
    // Step 2. Before doing the cross-thread reduction, in-thread reduction is needed when
    //  - not all the reduction-related loops are bound to thread axes, or
    //  - the block-realize has a non-constant-true predicate.
    bool need_in_thread_reduction =
        n_bound_reduction_loops < static_cast<int>(reduction_loops.size()) ||
        !is_one(realize->predicate);
    bool need_in_thread_init_block = block->init.defined();
    // Step 3. Create intermediate buffers, storing them in `ct_buffers` and
    // `it_buffers`. Let the scope block allocate these new buffers.
    Array<Buffer>& new_buffers = block2new_buffers_[block_stack_.back()];
    Array<Buffer> ct_buffers = MakeScratchpads(reduction_buffers, /*is_cross_thread_buffer=*/true);
    new_buffers.insert(new_buffers.end(), ct_buffers.begin(), ct_buffers.end());
    Optional<Array<Buffer>> it_buffers = NullOpt;
    if (need_in_thread_reduction) {
      it_buffers = MakeScratchpads(reduction_buffers, /*is_cross_thread_buffer=*/false);
      CHECK(it_buffers.defined())
          << "ValueError: Cross-thread reduction requires the block to have "
             "an init statement. However, here the block "
          << block->name_hint << " is a cross-thread reduction block but has no init statement.";
      new_buffers.insert(new_buffers.end(), it_buffers.value().begin(), it_buffers.value().end());
    }

    // Step 4. Transform.
    if (need_in_thread_init_block) {
      loop2new_stmt_[reduction_loops[0]] =
          TransformReductionBlock(realize, it_buffers, ct_buffers, reduction_buffers, wb_indices,
                                  reducer, combiner_rhs, reduction_loops);
    } else {
      // Otherwise the init block is already be decomposed.
      if (combiner_evaluate.defined()) {
        // Inject ReductionBlock Specially for MMA
        loop2new_stmt_[reduction_loops[0]] =
            InjectWarpEvaluateReductionBlock(realize, ct_buffers, reduction_buffers, wb_indices,
                                             reducer, combiner_evaluate, reduction_loops);
      } else {
        loop2new_stmt_[reduction_loops[0]] =
            InjectReductionBlock(realize, ct_buffers, reduction_buffers, wb_indices, reducer,
                                 combiner_lhs, reduction_loops);
      }

      // Step 4.1. Mark the block as a cross-thread reduction block.
      ObjectPtr<BlockNode> p_block = make_object<BlockNode>(*block);
      p_block->annotations.Set(kIsCrossThreadReductionApplied, tir::const_true());
    }

    // Step 5. Record the reduction thread dims for the write-back buffers.
    // The information is used for consumer block broadcasting detection.
    std::vector<std::pair<ThreadScope, Range>> reduction_threads;
    reduction_threads.reserve(reduction_loops.size());
    for (const ForNode* loop : reduction_loops) {
      if (loop->thread_binding.defined()) {
        reduction_threads.emplace_back(
            ThreadScope::Create(loop->thread_binding.value()->thread_tag),
            Range::FromMinExtent(loop->min, loop->extent));
      }
    }
    for (const Buffer& reduction_buf : reduction_buffers) {
      crt_buf2threads_[reduction_buf.get()] = reduction_threads;
    }
  }

  Stmt MakeCrossThreadBroadcast(
      const BlockRealizeNode* realize,
      const std::vector<std::pair<ThreadScope, Range>>& unbound_thread2range) {
    // Step 1. Generate loop var for each unbound thread.
    // Update the block predicate with clauses of `thread_var == min`.
    PrimExpr predicate = realize->predicate;
    Array<Var> loop_vars;
    loop_vars.reserve(unbound_thread2range.size());
    for (auto [scope, range] : unbound_thread2range) {
      std::string dim_index(1, static_cast<char>(scope.dim_index + 'x'));
      Var loop_var("t" + dim_index, range->min->dtype);
      loop_vars.push_back(loop_var);
      predicate = (loop_var == range->min) && predicate;
    }

    // Step 2. Update the BlockRealize with the new predicate.
    ObjectPtr<BlockRealizeNode> p_realize = make_object<BlockRealizeNode>(*realize);
    p_realize->predicate = std::move(predicate);

    // Step 3. Wrap the updated BlockRealize with the new loops.
    Stmt body(p_realize);
    for (int i = 0; i < static_cast<int>(unbound_thread2range.size()); ++i) {
      std::string dim_index(1, static_cast<char>(unbound_thread2range[i].first.dim_index + 'x'));
      body = For(
          /*loop_var=*/loop_vars[i],                          //
          /*min=*/unbound_thread2range[i].second->min,        //
          /*extent=*/unbound_thread2range[i].second->extent,  //
          /*kind=*/ForKind::kThreadBinding,                   //
          /*body=*/body,                                      //
          /*thread_binding=*/
          IterVar(NullValue<Range>(), Var("", loop_vars[i]->dtype), IterVarType::kThreadIndex,
                  "threadIdx." + dim_index));
    }
    return body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* realize) final {
    // Part 1. Check if the block needs cross-thread reduction rewrite.
    std::vector<const ForNode*> reduction_loops = NeedCrossThreadReduction(realize);
    if (!reduction_loops.empty()) {
      // Return an empty statement, because the transformation result will
      // be inserted when returning to the first reduction-related loop.
      has_cross_thread_reduction_ = true;
      MakeCrossThreadReduction(realize, reduction_loops);
      return Stmt{nullptr};
    }
    if (!has_cross_thread_reduction_) {
      return StmtMutator::VisitStmt_(realize);
    }

    // Part 2. Check if the block needs all-thread broadcasting rewrite.
    // We only check this when cross-thread reduction was detected.
    std::vector<std::pair<ThreadScope, Range>> unbound_thread2range =
        NeedCrossThreadBroadcast(realize);
    if (!unbound_thread2range.empty()) {
      return MakeCrossThreadBroadcast(realize, unbound_thread2range);
    }

    realize = StmtMutator::VisitStmt_(realize).as<BlockRealizeNode>();
    return GetRef<BlockRealize>(realize);
  }

 private:
  bool has_cross_thread_reduction_ = false;
  std::vector<const StmtNode*> statement_stack_;
  std::vector<const ForNode*> loop_stack_;
  std::vector<const BlockNode*> block_stack_;
  std::unordered_map<const BlockNode*, Array<Buffer>> block2new_buffers_;
  std::unordered_map<const ForNode*, Stmt> loop2new_stmt_;
  Map<Var, Range> loop_range_map_;
  arith::Analyzer analyzer_;

  int block_idx_depth = 0;
  int thread_idx_depth = 0;
  std::unordered_map<const BufferNode*, std::vector<std::pair<ThreadScope, Range>>>
      crt_buf2threads_;
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