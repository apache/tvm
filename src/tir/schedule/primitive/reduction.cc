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
#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief A helper class to create a new scope that contains decomposed init body
 * and replaced old reduction block.
 */
class DecomposeReductionBlockReplacer : public StmtMutator {
 public:
  /*!
   * \brief The open interface to users to call the helper class
   * \param old_scope_root The original block scope before decomposition
   * \param target_loop The loop we insert the decomposed init body before
   * \param decompose_body The decomposed init body
   * \param old_reduction_block The reduction block we want to decompose
   * \return The new block scope and the updated reduction block
   */
  static std::pair<Block, Block> Replace(Block old_scope_root, For target_loop,
                                         Stmt decomposed_body, Block old_reduction_block) {
    DecomposeReductionBlockReplacer replacer(std::move(target_loop), std::move(decomposed_body),
                                             std::move(old_reduction_block));
    return std::make_pair(Downcast<Block>(replacer(std::move(old_scope_root))),
                          replacer.new_reduction_block_);
  }

 private:
  explicit DecomposeReductionBlockReplacer(For target_loop, Stmt decomposed_body,
                                           Block old_reduction_block)
      : target_loop_(std::move(target_loop)),
        decomposed_body_(std::move(decomposed_body)),
        old_reduction_block_(std::move(old_reduction_block)) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    Stmt mutated_stmt = StmtMutator::VisitStmt_(loop);
    if (loop == target_loop_.get()) {
      return SeqStmt({decomposed_body_, mutated_stmt});
    } else {
      return mutated_stmt;
    }
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    if (block == old_reduction_block_.get()) {
      ObjectPtr<BlockNode> p_new_block = CopyOnWrite(block);
      p_new_block->name_hint = p_new_block->name_hint + "_update";
      p_new_block->init = NullOpt;
      // Add write regions back to read regions in update block.
      Array<BufferRegion> new_reads;
      std::unordered_set<const BufferNode*> read_bufs;
      for (const BufferRegion& read_access : block->reads) {
        read_bufs.insert(read_access->buffer.get());
      }
      for (const BufferRegion& write_access : block->writes) {
        if (read_bufs.find(write_access->buffer.get()) == read_bufs.end()) {
          new_reads.push_back(write_access);
        }
      }
      for (const BufferRegion& read_access : block->reads) {
        new_reads.push_back(read_access);
      }
      p_new_block->reads = new_reads;
      new_reduction_block_ = Block(p_new_block);
      return new_reduction_block_;
    } else {
      return StmtMutator::VisitStmt_(block);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> new_stmts;
    new_stmts.reserve(seq->seq.size());
    for (const Stmt& old_stmt : seq->seq) {
      new_stmts.push_back(VisitStmt(old_stmt));
    }
    return SeqStmt::Flatten(new_stmts);
  }

 private:
  For target_loop_;
  Stmt decomposed_body_;
  Block old_reduction_block_;
  Block new_reduction_block_;
};

class LoopHeightError : public ScheduleError {
 public:
  static void CheckLoopHigherThanReduceLoops(const IRModule& mod, const BlockNode* block,
                                             const BlockRealizeNode* realize,
                                             const Array<StmtSRef>& loops,
                                             const StmtSRef& loop_sref) {
    for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
      // For each block var of type kCommReduce, check its binding
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != IterVarType::kCommReduce) {
        continue;
      }
      for (const StmtSRef& higher_loop : loops) {
        // Only check loops not lower than the target loop
        if (higher_loop.same_as(loop_sref)) {
          break;
        }
        // loop_var of a higher loop shouldn't contain loop var
        const Var& loop_var = higher_loop->StmtAs<ForNode>()->loop_var;
        if (UsesVar(binding, [v = loop_var.get()](const VarNode* var) { return var == v; })) {
          const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
          throw LoopHeightError(mod, GetRef<For>(loop), GetRef<Block>(block));
        }
      }
    }
  }

  explicit LoopHeightError(IRModule mod, For loop, Block block)
      : mod_(std::move(mod)), loop_(std::move(loop)), block_(std::move(block)) {}

  String FastErrorString() const final {
    return "ScheduleError: decompose_reduction expect the loop to be higher than all the loops "
           "related to reduce block var";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: decompose_reduction expect the loop {0} to be higher than all the loops "
          "related to reduce block var of block {1}";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_, block_}; }

  IRModule mod_;
  For loop_;
  Block block_;
};

PrimExpr RemakePredicate(PrimExpr pred, const std::unordered_set<const VarNode*>& discarded_loops) {
  if (is_one(pred)) return Bool(true);
  PrimExpr new_pred = Bool(true);
  auto f = [&](const VarNode* var) { return discarded_loops.count(var); };
  arith::PVar<PrimExpr> lhs, rhs, rest;
  for (;;) {
    if ((rest && (lhs < rhs)).Match(pred)) {
      if (!UsesVar(lhs.Eval(), f)) new_pred = new_pred && (lhs.Eval() < rhs.Eval());
      pred = rest.Eval();
    } else if ((lhs < rhs).Match(pred)) {
      if (!UsesVar(lhs.Eval(), f)) new_pred = new_pred && (lhs.Eval() < rhs.Eval());
      break;
    } else {
      ICHECK(false) << "Unexpected predicate for reduction block";
    }
  }
  return new_pred;
}

StmtSRef DecomposeReduction(ScheduleState self, const StmtSRef& block_sref,
                            const StmtSRef& loop_sref) {
  /*!
   *  Check
   *    - block is a reduction block
   *    - loop is not lower than all the loops related to reduce block var
   *  Mutate
   *    - generate loops related to data par block vars
   *    - generate corresponding init block and update block
   */
  // Condition Checks and Information Collection
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  // Get the outer loops from high to low
  Array<StmtSRef> loops = GetLoops(block_sref);
  const BlockRealizeNode* realize = GetBlockRealize(self, block_sref).get();
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/false);
  if (self->enable_check) {
    // Cond 0. Check loop_sref is an ancestor of block_sref
    if (std::find(loops.begin(), loops.end(), loop_sref) == loops.end()) {
      throw LoopPositionError(self->mod, GetRef<For>(loop), GetRef<Block>(block),
                              "decompose_reduction");
    }
    // Cond 1. Check block is reduction
    CheckReductionBlock(self, block_sref, scope_root_sref);
    // Cond 2. Check 'loop' is higher than all the loops related to block var of type reduction
    LoopHeightError::CheckLoopHigherThanReduceLoops(self->mod, block, realize, loops, loop_sref);
  }
  // IR Manipulation
  ObjectPtr<BlockNode> init_block = make_object<BlockNode>();
  ObjectPtr<BlockRealizeNode> init_realize = make_object<BlockRealizeNode>();
  init_block->name_hint = block->name_hint + "_init";
  init_block->annotations = block->annotations;
  init_realize->iter_values = {};
  init_realize->block = Block(init_block);
  // Step 1. Create new block vars and their bindings
  // Maps an old block var to the new corresponding block var
  std::unordered_map<Var, Var> block_var_map;
  block_var_map.reserve(block->iter_vars.size());
  for (int i = 0, n = block->iter_vars.size(); i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->iter_values[i];
    // Only process data parallel block vars
    if (iter_var->iter_type != IterVarType::kDataPar) {
      continue;
    }
    // Create a new block var
    IterVar new_iter_var(/*dom=*/iter_var->dom,
                         /*var=*/iter_var->var.copy_with_suffix(""),
                         /*iter_type=*/iter_var->iter_type,
                         /*thread_tag=*/iter_var->thread_tag);
    // Add a block var and its binding
    init_block->iter_vars.push_back(new_iter_var);
    init_realize->iter_values.push_back(binding);
    // Add a mapping from old block vars to new block vars
    block_var_map[iter_var->var] = new_iter_var->var;
  }
  // Step 2. After copying block vars, substitute them in init block
  init_block->body = Substitute(block->init.value(), block_var_map);
  for (const BufferRegion& write : block->writes) {
    init_block->writes.push_back(
        BufferRegion(write->buffer, Substitute(write->region, block_var_map)));
  }
  // Step 3. Scan loops not higher than the specified loop above the reduction block.
  //         If the loop is used in the init block binding, then it is chosen.
  //         Otherwise, it is discarded.
  std::unordered_set<const VarNode*> discarded_loops;
  std::vector<int> chosen_loops;
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    const VarNode* loop_var = loops[i]->StmtAs<ForNode>()->loop_var.get();
    bool discarded = true;
    for (const PrimExpr& expr : init_realize->iter_values) {
      if (!UsesVar(expr, [v = loop_var](const VarNode* var) { return var == v; })) {
        continue;
      }
      // The loop is related to init block bindings;
      chosen_loops.push_back(i);
      discarded = false;
      break;
    }
    if (discarded) discarded_loops.insert(loop_var);
    // Only scan loops not higher than the given loop
    if (loops[i].same_as(loop_sref)) {
      break;
    }
  }
  // Step 4. After scanning loops, make a new predicate in the init block realize
  //         We discard predicate that is related to discarded loops
  init_realize->predicate = RemakePredicate(realize->predicate, discarded_loops);
  // Step 5. Create new loops above init block
  std::unordered_map<Var, Var> loop_var_map;
  Stmt body = BlockRealize(init_realize);
  for (int i : chosen_loops) {
    const ForNode* old_loop = TVM_SREF_TO_FOR(loops[i]);
    // Create a new equivalent to the chosen loop
    Var old_loop_var = old_loop->loop_var;
    Var new_loop_var = old_loop_var.copy_with_suffix("_init");
    loop_var_map[old_loop_var] = new_loop_var;
    Optional<IterVar> opt_thread_binding = old_loop->thread_binding;
    if (opt_thread_binding) {
      auto thread_binding = opt_thread_binding.value();
      auto new_var = thread_binding->var.copy_with_suffix("");
      thread_binding.CopyOnWrite()->var = new_var;
      opt_thread_binding = thread_binding;
    }
    body = For(/*loop_var=*/new_loop_var,
               /*min=*/old_loop->min,
               /*extent=*/old_loop->extent,
               /*kind=*/old_loop->kind,
               /*body=*/body,
               /*thread_binding=*/opt_thread_binding);
  }
  body = Substitute(body, loop_var_map);
  // Step 6. Mutate IR
  const BlockNode* old_scope_root = TVM_SREF_TO_BLOCK(scope_root_sref);
  auto [new_scope_root, new_reduction_block] = DecomposeReductionBlockReplacer::Replace(
      GetRef<Block>(old_scope_root), GetRef<For>(loop), body, GetRef<Block>(block));
  self->Replace(scope_root_sref, new_scope_root,
                {{GetRef<Block>(old_scope_root), new_scope_root},
                 {GetRef<Block>(block), new_reduction_block}});
  self->UpdateScopeBlockInfo(new_scope_root);
  return self->stmt2ref.at(init_block.get());
}

/******** Commutative Reducer ********/

/*!
 * \brief A structure used for registering new commutative reducers, and store all the registered
 * reducers. The reducers are preserved in a list, in the form of "reducer-getter function". When
 * invoking a reducer-getter function with a specific datatype, the reducer-getter will return the
 * CommReducer of the corresponding reduction pattern and the specific datatype
 */
struct ReducerRegistry {
  ReducerRegistry()
      : reducer_getters{
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{x[0] + y[0]};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, 0)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{x[0] * y[0]};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, 1)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{min(x[0], y[0])};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{max_value(values[0]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/1,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{max(x[0], y[0])};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{min_value(values[0]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  return Array<PrimExpr>{x[0] + y[0], x[1] + y[1]};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, 0),
                                         make_const(values[1]->dtype, 0)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx = Select(x[1] >= y[1], x[0], y[0]);
                  PrimExpr val = Select(x[1] >= y[1], x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         min_value(values[1]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx =
                      Select(Or(greater(x[1], y[1]), And(equal(x[1], y[1]), less(x[0], y[0]))),
                             x[0], y[0]);
                  PrimExpr val = Select(greater(x[1], y[1]), x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         min_value(values[1]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx = Select(x[1] <= y[1], x[0], y[0]);
                  PrimExpr val = Select(x[1] <= y[1], x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         max_value(values[1]->dtype)};
                }),
            CreateReducerGetter(
                /*n_buffers=*/2,
                [](const Array<Var>& x, const Array<Var>& y) {
                  PrimExpr idx = Select(
                      Or(less(x[1], y[1]), And(equal(x[1], y[1]), less(x[0], y[0]))), x[0], y[0]);
                  PrimExpr val = Select(less(x[1], y[1]), x[1], y[1]);
                  return Array<PrimExpr>{idx, val};
                },
                [](const Array<PrimExpr>& values) {
                  return Array<PrimExpr>{make_const(values[0]->dtype, -1),
                                         max_value(values[1]->dtype)};
                })} {}

  static void RegisterReducer(
      int n_buffers, TypedPackedFunc<Array<PrimExpr>(Array<Var>, Array<Var>)> combiner_getter,
      TypedPackedFunc<Array<PrimExpr>(Array<PrimExpr>)> identity_getter) {
    ReducerRegistry::Global()->reducer_getters.push_back(ReducerRegistry::CreateReducerGetter(
        n_buffers, std::move(combiner_getter), std::move(identity_getter)));
  }

  static TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)> CreateReducerGetter(
      int n_buffers, TypedPackedFunc<Array<PrimExpr>(Array<Var>, Array<Var>)> combiner_getter,
      TypedPackedFunc<Array<PrimExpr>(Array<PrimExpr>)> identity_getter) {
    return [n_buffers,                                     //
            combiner_getter = std::move(combiner_getter),  //
            identity_getter = std::move(identity_getter)   //
    ](Array<PrimExpr> values) -> Optional<CommReducer> {
      if (static_cast<int>(values.size()) != n_buffers) {
        return NullOpt;
      }
      Array<Var> lhs;
      Array<Var> rhs;
      for (int i = 0; i < n_buffers; ++i) {
        lhs.push_back(Var("x" + std::to_string(i), values[i]->dtype));
        rhs.push_back(Var("y" + std::to_string(i), values[i]->dtype));
      }
      return CommReducer(lhs, rhs, combiner_getter(lhs, rhs), identity_getter(values));
    };
  }

  static ReducerRegistry* Global() {
    static ReducerRegistry instance;
    return &instance;
  }

  std::vector<TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)>> reducer_getters;
};

std::vector<TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)>> GetReducerGetters() {
  return ReducerRegistry::Global()->reducer_getters;
}

class NotSerialLoopKindError : public ScheduleError {
 public:
  explicit NotSerialLoopKindError(IRModule mod, For loop)
      : mod_(std::move(mod)), loop_(std::move(loop)) {}

  String FastErrorString() const final {
    return "ScheduleError: The input loop of rfactor is required to be `kSerial`";
  }

  String DetailRenderTemplate() const final {
    String str_kind = ForKind2String(loop_->kind);
    std::ostringstream os;
    os << "ScheduleError: The input loop {0} of rfactor is required to be `Serial`. However, the "
          "kind of {0} is `"
       << str_kind << "`";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
};

class FactorAxisOutOfRangeError : public ScheduleError {
 public:
  explicit FactorAxisOutOfRangeError(IRModule mod, Buffer buffer, int factor_axis)
      : mod_(std::move(mod)), buffer_(std::move(buffer)), factor_axis_(factor_axis) {}

  String FastErrorString() const final {
    return "ScheduleError: The input `factor_axis` is out of range. It is required to be in range "
           "[-(ndim + 1), ndim] where `ndim` is the number of dimensions of the write buffer";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    int ndim = static_cast<int>(buffer_->shape.size());
    os << "The write buffer " << buffer_->name << " has " << ndim
       << " dimension(s), so `factor_axis` is required to be in [" << -(ndim + 1) << ", " << ndim
       << "] for rfactor. However, the input `factor_axis` is " << factor_axis_
       << ", which is out of the expected range";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  static int CheckAndUpdate(const IRModule& mod, const Buffer& buffer, int factor_axis) {
    int ndim = static_cast<int>(buffer->shape.size());
    if (factor_axis < -(ndim + 1) || factor_axis > ndim) {
      throw FactorAxisOutOfRangeError(mod, buffer, factor_axis);
    }
    // If factor_axis is negative, convert it to a non-negative one.
    if (factor_axis < 0) {
      factor_axis += ndim + 1;
    }
    return factor_axis;
  }

  IRModule mod_;
  Buffer buffer_;
  int factor_axis_;
};

class LoopPropertyError : public ScheduleError {
 public:
  enum ErrorType {
    kDataParIterTouchRFactorLoop = 0,
    kLoopTouchedByBothKindsOfBlockIters = 1,
    kNotFirstChildBlockOfOutermostLoop = 2,
    kUnboundLoopUnderReductionLoop = 3
  };

  explicit LoopPropertyError(IRModule mod, For loop, ErrorType error_type)
      : mod_(std::move(mod)), loop_(std::move(loop)), error_type_(error_type) {}

  String FastErrorString() const final {
    switch (error_type_) {
      case kDataParIterTouchRFactorLoop:
        return "ScheduleError: The loop to be applied rfactor is required not to be touched by any "
               "data parallel block iter of the block";
      case kLoopTouchedByBothKindsOfBlockIters:
        return "ScheduleError: The loops outside of the reduction block are required not to be "
               "touched by both data parallel block iters and reduction block iters";
      case kNotFirstChildBlockOfOutermostLoop:
        return "ScheduleError: The reduction block should be the first child block of the "
               "outermost loop outside of it";
      case kUnboundLoopUnderReductionLoop:
        return "ScheduleError: A loop who has extent greater than one and is not bound to any "
               "block iter should not appear under a reduction loop";
    }
    ICHECK(false) << "Unreachable";
    throw;
  }

  String DetailRenderTemplate() const final {
    switch (error_type_) {
      case kDataParIterTouchRFactorLoop:
        return "The loop to be applied rfactor is {0}, which is required not to be touched by any "
               "data parallel block iter of the block below. However, some of the block's data "
               "parallel block iters touch this loop";
      case kLoopTouchedByBothKindsOfBlockIters:
        return "It is not allowed that the loop {0} is touched by both some data parallel block "
               "iters and some reduction block iters";
      case kNotFirstChildBlockOfOutermostLoop:
        return "The first child block of the outermost loop {0} is not the reduction block.";
      case kUnboundLoopUnderReductionLoop:
        return "The loop {0} has extent greater than one, and is not bound to any block iter. "
               "Therefore it shouldn't appear under a reduction loop";
    }
    ICHECK(false) << "Unreachable";
    throw;
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  static void CheckLoopProperty(const ScheduleState& self, const Array<For>& loops,
                                const ForNode* rf_loop, const Block& block,
                                const std::unordered_set<const VarNode*>& data_par_loop_vars,
                                const std::unordered_set<const VarNode*>& reduce_loop_vars) {
    Array<BlockRealize> children_of_outermost_loop =
        GetChildBlockRealizeOnSRefTree(self->stmt2ref.at(loops[0].get()));
    if (!children_of_outermost_loop[0]->block.same_as(block)) {
      throw LoopPropertyError(self->mod, loops[0], kNotFirstChildBlockOfOutermostLoop);
    }

    bool meet_reduction_loop = false;
    for (const For& loop : loops) {
      bool data_par_touched = data_par_loop_vars.count(loop->loop_var.get());
      bool reduction_touched = reduce_loop_vars.count(loop->loop_var.get());

      if (data_par_touched && reduction_touched) {
        throw LoopPropertyError(self->mod, loop, kLoopTouchedByBothKindsOfBlockIters);
      } else if (data_par_touched) {
        if (loop.get() == rf_loop) {
          throw LoopPropertyError(self->mod, loop, kDataParIterTouchRFactorLoop);
        }
        continue;
      } else if (reduction_touched) {
        if (!meet_reduction_loop) {
          CheckGetSingleChildBlockRealizeOnSRefTree(self, self->stmt2ref.at(loop.get()));
          meet_reduction_loop = true;
        }
        continue;
      } else if (meet_reduction_loop && !is_one(loop->extent)) {
        throw LoopPropertyError(self->mod, loop, kUnboundLoopUnderReductionLoop);
      }
    }
  }

  IRModule mod_;
  For loop_;
  ErrorType error_type_;
};

/*!
 * \brief For each loop in the given array of loop, associate its loop var with the loop itself
 * using a mapping
 * \param loops The loops to be analyzed
 * \return A mapping from loops to their corresponding loop vars
 */
std::unordered_map<const VarNode*, For> GetLoopVar2LoopMap(const Array<For>& loops) {
  std::unordered_map<const VarNode*, For> loop_vars2loop;
  loop_vars2loop.reserve(loops.size());
  for (const For& loop : loops) {
    loop_vars2loop[loop->loop_var.get()] = loop;
  }
  return loop_vars2loop;
}

/*!
 * \brief Create the intermediate rfactor buffers, which the rfactor block writes to and the
 * write-back block reads from
 * \param buf_stores The BufferStores of the original block, where the rfactor buffers will be
 * created from
 * \param factor_axis The `factor_axis` parameter of rfactor
 * \param rf_loop The rfactor loop
 * \return The new created intermediate rfactor buffer
 */
Array<Buffer> CreateRFactorBuffers(const Array<BufferStore>& buf_stores, int factor_axis,
                                   const ForNode* rf_loop) {
  Array<Buffer> rf_buffers;
  rf_buffers.reserve(buf_stores.size());
  for (const BufferStore& buf_store : buf_stores) {
    Buffer buffer = buf_store->buffer;
    Array<PrimExpr> rf_shape = buffer->shape;
    rf_shape.insert(rf_shape.begin() + factor_axis, rf_loop->extent);

    ObjectPtr<BufferNode> n = make_object<BufferNode>(*buffer.get());
    n->shape = rf_shape;
    n->name = buffer->name + ".rf";
    n->data = buffer->data.copy_with_suffix(".rf");
    rf_buffers.push_back(Buffer(n));
  }
  return rf_buffers;
}

/*!
 * \brief The base class of the rfactor/write-back block creator, which creates the blocks in four
 * steps:
 * 1) Create the new block iters and the their iter bindings
 * 2) Create the body and init of the new block
 * 3) Create the read/write regions of the new block
 * 4) Create the new block and the new block-realize
 */
class BaseBlockCreator {
 public:
  explicit BaseBlockCreator(BlockRealize old_block_realize, For rf_loop,
                            Array<BufferStore> old_reduction_updates, CommReducer reducer,
                            Array<Buffer> rf_buffers, bool is_rf_block)
      : old_block_realize_(std::move(old_block_realize)),
        rf_loop_(std::move(rf_loop)),
        old_reduction_updates_(std::move(old_reduction_updates)),
        reducer_(std::move(reducer)),
        rf_buffers_(std::move(rf_buffers)),
        n_buffers_(static_cast<int>(rf_buffers_.size())),
        is_rf_block_(is_rf_block) {
    n_block_iters_ = static_cast<int>(old_block_realize_->iter_values.size());
    update_buffers_.reserve(n_buffers_);
    update_indices_.reserve(n_buffers_);
    update_lhs_.reserve(n_buffers_);
    update_rhs_.reserve(n_buffers_);
  }

  void CreateBlock() {
    CreateAdditionalIter();
    for (int i = 0; i < n_block_iters_; ++i) {
      CreateNormalIters(i);
    }
    bool has_reduce_iter = false;
    for (const IterVar& iter_var : iter_vars_) {
      if (iter_var->iter_type == IterVarType::kCommReduce) {
        has_reduce_iter = true;
        break;
      }
    }

    // The pre-processing finds out the buffers written in the block, the indices of the buffer
    // accesses, and the reduction LHS and RHS of the stored values.
    PreProcess();
    Stmt block_body = Substitute(CreateBlockBody(has_reduce_iter), var_map_);
    Optional<Stmt> block_init = CreateBlockInit(has_reduce_iter);
    if (block_init.defined()) {
      block_init = Substitute(block_init.value(), var_map_);
    }
    CreateReadWriteRegions();

    String new_block_name = old_block_realize_->block->name_hint;
    PrimExpr predicate = const_true();
    if (is_rf_block_) {
      new_block_name = new_block_name + "_rf";
      predicate = old_block_realize_->predicate;
    }
    new_block_ = Block(
        /*iter_vars=*/iter_vars_,
        /*reads=*/read_regions_,
        /*writes=*/write_regions_,
        /*name_hint=*/new_block_name,
        /*body=*/std::move(block_body),
        /*init=*/std::move(block_init),
        /*alloc_buffers=*/{},
        /*match_buffers=*/{},
        /*annotations=*/old_block_realize_->block->annotations);
    new_block_realize_ = BlockRealize(iter_values_, predicate, new_block_);
  }

 private:
  virtual void CreateAdditionalIter() = 0;
  virtual void CreateNormalIters(int idx) = 0;
  virtual void PreProcess() = 0;
  virtual void CreateReadWriteRegions() = 0;

  Stmt CreateBlockBody(bool has_reduce_iter) {
    Array<Stmt> buf_stores;
    buf_stores.reserve(n_buffers_);

    // Case 1. If the block has no reduction iterator, we just store the RHS values into the
    // buffers.
    if (!has_reduce_iter) {
      for (int i = 0; i < n_buffers_; ++i) {
        buf_stores.push_back(BufferStore(update_buffers_[i], update_rhs_[i], update_indices_[i]));
      }
      return n_buffers_ > 1 ? SeqStmt(buf_stores) : buf_stores[0];
    }

    // Case 2. If the reduction is for single buffer, the block body is a single BufferStore.
    Array<PrimExpr> stored_values = (*reducer_.get())(update_lhs_, update_rhs_);
    if (n_buffers_ == 1) {
      return BufferStore(update_buffers_[0], stored_values[0], update_indices_[0]);
    }

    // Case 3. In case the reduction is for multiple buffers, we should create the reduction with
    // LetStmt so that the reduction execution generates correct results.
    Array<Var> let_vars;
    let_vars.reserve(n_buffers_);
    for (int i = 0; i < n_buffers_; ++i) {
      Var var("v_" + update_buffers_[i]->name, PrimType(stored_values[i]->dtype));
      let_vars.push_back(var);
      buf_stores.push_back(BufferStore(update_buffers_[i], var, update_indices_[i]));
    }
    Stmt body = SeqStmt(buf_stores);
    for (int i = n_buffers_ - 1; i >= 0; --i) {
      body = LetStmt(let_vars[i], stored_values[i], std::move(body));
    }
    return body;
  }

  Optional<Stmt> CreateBlockInit(bool has_reduce_iter) {
    if (!has_reduce_iter) {
      return NullOpt;
    }

    Array<Stmt> inits;
    inits.reserve(n_buffers_);
    for (int i = 0; i < n_buffers_; ++i) {
      inits.push_back(
          BufferStore(update_buffers_[i], reducer_->identity_element[i], update_indices_[i]));
    }
    return n_buffers_ > 1 ? SeqStmt(inits) : inits[0];
  }

 public:
  /*! \brief The new created block */
  Block new_block_;
  /*! \brief The new created block-realize */
  BlockRealize new_block_realize_;
  /*! \brief The indices used to access the intermediate rfactor buffer */
  Array<PrimExpr> rf_buf_access_indices_;

 protected:
  /*! \brief The old block-realize */
  BlockRealize old_block_realize_;
  /*! \brief The number of block iters in the old block */
  int n_block_iters_;
  /*! \brief The rfactor loop */
  For rf_loop_;
  /*! \brief The update BufferStores of the old block */
  Array<BufferStore> old_reduction_updates_;
  /*! \brief The matched commutative reducer */
  CommReducer reducer_;
  /*! \brief The intermediate rfactor buffers */
  Array<Buffer> rf_buffers_;
  /*! \brief The number of rfactor buffers. */
  const int n_buffers_;
  /*!
   * \brief A mapping which maps old block iters to new expressions. The old iters will be replaced
   * by the expressions in future substitution for the two blocks
   */
  Map<Var, PrimExpr> var_map_;

  /*! \brief Whether we are creating the rfactor block or the write-back block */
  bool is_rf_block_;
  /*! \brief The new block iters of the new created block */
  std::vector<IterVar> iter_vars_;
  /*! \brief The new block iter bindings of the new created block-realize */
  std::vector<PrimExpr> iter_values_;
  /*! \brief The buffers updated in this block */
  Array<Buffer> update_buffers_;
  /*! \brief The indices of the buffers updated in this block, respectively */
  Array<Array<PrimExpr>> update_indices_;
  /*! \brief The LHS values of the reduction in this block */
  Array<PrimExpr> update_lhs_;
  /*! \brief THe RHS values of the reduction in this block */
  Array<PrimExpr> update_rhs_;
  /*! \brief The read regions of the new created block */
  Array<BufferRegion> read_regions_;
  /*! \brief The write regions of the new created block */
  Array<BufferRegion> write_regions_;
};

/*!
 * \brief The derived class of the rfactor block creator, which implements all virtual methods in
 * the base creator
 * \details Start constructing the rfactor block. The main difficulty to construct the rfactor block
 * is to create its block iters. So here we introduce the algorithm to create the block iters.
 *  1. Create a block iter for the rfactor loop. The block binding of this iter is the loop var, and
 *     the block iter is data parallel.
 *  2. For all the old block's block iters, there are two cases:
 *    (a) If it is data parallel block iter, or a reduction block iter which doesn't touch the
 *        rfactor loop, we keep it and its block binding in the rfactor block.
 *    (b) Otherwise it is a reduction block iter which touches the rfactor loop. In this case, we
 *        "split" the block iter into one or more new block iters and do not keep the old block
 *        var. More specifically, we create a new reduction block iter for each loop var that
 *        appears in the reduction block iter's binding (except for the rfactor loop), and the
 *        binding of the new block iter is exactly the loop var. (Note that for each loop var, we
 *        create at most one block iter, even if there are multiple old block iters which touch
 *        both this loop and the rfactor loop).
 *        Then we substitute the appearances of the old block iter with the new created block
 *        iters by recording two mappings: one maps loops vars to new created block iters which
 *        is used for binding substitution, and another maps old block iters to new expressions
 *        which is used for substitutions of the old block iters.
 */
class RFactorBlockCreator : public BaseBlockCreator {
 public:
  explicit RFactorBlockCreator(BlockRealize old_block_realize, For rf_loop,
                               Array<BufferStore> old_reduction_updates, CommReducer reducer,
                               Array<Buffer> rf_buffers,
                               std::unordered_map<const VarNode*, For> loop_vars2loop,
                               int factor_axis, Array<PrimExpr> combiner_rhs)
      : BaseBlockCreator(std::move(old_block_realize), std::move(rf_loop),
                         std::move(old_reduction_updates), std::move(reducer),
                         std::move(rf_buffers), true),
        loop_vars2loop_(std::move(loop_vars2loop)),
        factor_axis_(factor_axis),
        combiner_rhs_(std::move(combiner_rhs)) {}

 private:
  void CreateAdditionalIter() final {
    // Create a new data parallel block iter for the rfactor loop.
    additional_iter_ =
        IterVarFromLoop(rf_loop_, "v" + rf_loop_->loop_var->name_hint, IterVarType::kDataPar);
    loop_var2block_binding_[rf_loop_->loop_var.get()] = additional_iter_->var;
    iter_vars_.push_back(additional_iter_);
    iter_values_.push_back(rf_loop_->loop_var);
  }

  void CreateNormalIters(int idx) final {
    IterVar old_iter = old_block_realize_->block->iter_vars[idx];
    PrimExpr old_binding = old_block_realize_->iter_values[idx];
    if (old_iter->iter_type == IterVarType::kDataPar ||
        !UsesVar(old_binding,
                 [v = rf_loop_->loop_var.get()](const VarNode* var) { return var == v; })) {
      // The old block iter is either a data parallel block iter, or a reduction block iter that
      // doesn't touch the rfactor loop. In this case reuse the old reduction block iter and its
      // corresponding binding.
      iter_vars_.push_back(old_iter);
      iter_values_.push_back(old_binding);
      return;
    }
    ICHECK(old_iter->iter_type == kCommReduce);
    // This block iter is a reduction block iter that touches the rfactor loop. So next we try to
    // create a new block iter for all loop vars that appear in the old binding.
    Array<Var> vars_in_old_binding = UndefinedVars(old_binding);
    for (const Var& var : vars_in_old_binding) {
      auto it = loop_vars2loop_.find(var.get());
      if (it == loop_vars2loop_.end()) {
        // `var` is not a loop var. So skip.
        continue;
      }
      const For& loop = it->second;
      if (loop_var2block_binding_.find(var.get()) == loop_var2block_binding_.end()) {
        // We haven't created the new block iter for `var`. So here we create it, append it
        // and its binding to `rf_block_iter_vars` and `rf_block_iter_values` respectively.
        IterVar new_iter_var =
            IterVarFromLoop(loop, "v" + loop->loop_var->name_hint, IterVarType::kCommReduce);
        loop_var2block_binding_[var.get()] = new_iter_var->var;
        iter_vars_.push_back(new_iter_var);
        iter_values_.push_back(var);
      }
    }
    // Substitute the original binding with new block iters. Store the result expression
    // in `rf_var_map` for future substitution.
    var_map_.Set(old_iter->var, Substitute(old_binding, loop_var2block_binding_));
  }

  void PreProcess() final {
    // The accessed indices for all reduction buffers are the same.
    rf_buf_access_indices_ = old_reduction_updates_[0]->indices;
    rf_buf_access_indices_.insert(rf_buf_access_indices_.begin() + factor_axis_,
                                  additional_iter_->var);
    for (int i = 0; i < n_buffers_; ++i) {
      update_buffers_.push_back(rf_buffers_[i]);
      update_indices_.push_back(rf_buf_access_indices_);
      update_lhs_.push_back(BufferLoad(update_buffers_[i], rf_buf_access_indices_));
      update_rhs_.push_back(combiner_rhs_[i]);
    }
  }

  void CreateReadWriteRegions() final {
    Map<Buffer, Buffer> buffer_map;
    for (int i = 0; i < n_buffers_; ++i) {
      buffer_map.Set(old_reduction_updates_[i]->buffer, rf_buffers_[i]);
    }
    const Block& old_block = old_block_realize_->block;
    read_regions_.reserve(old_block->reads.size());
    for (const BufferRegion& read_region : old_block->reads) {
      read_regions_.push_back(
          BufferRegion(read_region->buffer, Substitute(read_region->region, var_map_)));
    }
    write_regions_.reserve(old_block->writes.size());
    for (const BufferRegion& write_region : old_block->writes) {
      Array<Range> region = write_region->region;
      region.insert(region.begin() + factor_axis_,
                    Range::FromMinExtent(additional_iter_->var,
                                         make_const(additional_iter_->var.dtype(), 1)));
      Optional<Buffer> rf_buffer = buffer_map.Get(write_region->buffer);
      ICHECK(rf_buffer.defined());
      write_regions_.push_back(BufferRegion(rf_buffer.value(), Substitute(region, var_map_)));
    }
  }

 public:
  /*! \brief The generated additional block iter in rfactor block for the rfactor loop */
  IterVar additional_iter_;

 private:
  /*!
   * \brief A mapping which maps a loop var to its corresponding For loop for all the reduction
   * block's outer loops
   */
  std::unordered_map<const VarNode*, For> loop_vars2loop_;
  /*! \brief The factor_axis specified for rfactor */
  int factor_axis_;
  /*! \brief The RHS values of the reduction in the old block */
  Array<PrimExpr> combiner_rhs_;
  /*!
   * \brief A mapping which maps loop vars to new created block iters. This map is used to
   * substitute the loop vars which appear in the bindings of some old block iters with the new
   * created block iters
   */
  std::unordered_map<const VarNode*, Var> loop_var2block_binding_;
};

/*!
 * \brief The derived class of the write-back block creator, which implements all virtual methods in
 * the base creator
 */
class WriteBackBlockCreator : public BaseBlockCreator {
 public:
  explicit WriteBackBlockCreator(BlockRealize old_block_realize, For rf_loop,
                                 Array<BufferStore> old_reduction_updates, CommReducer reducer,
                                 Array<Buffer> rf_buffers, IterVar rf_additional_iter,
                                 Array<PrimExpr> combiner_lhs,
                                 Array<PrimExpr> rf_buf_access_indices)
      : BaseBlockCreator(std::move(old_block_realize), std::move(rf_loop),
                         std::move(old_reduction_updates), std::move(reducer),
                         std::move(rf_buffers), false),
        rf_additional_iter_(std::move(rf_additional_iter)),
        combiner_lhs_(std::move(combiner_lhs)) {
    iter_vars_.reserve(n_block_iters_);
    iter_values_.reserve(n_block_iters_);
    rf_buf_access_indices_ = std::move(rf_buf_access_indices);
  }

 private:
  void CreateAdditionalIter() final {
    // Create a new reduction block iter for the rfactor loop.
    IterVar wb_new_block_iter =
        IterVarFromLoop(rf_loop_, "v" + rf_loop_->loop_var->name_hint, kCommReduce);
    iter_vars_.push_back(wb_new_block_iter);
    iter_values_.push_back(rf_loop_->loop_var);
    var_map_.Set(rf_additional_iter_->var, wb_new_block_iter->var);
  }

  void CreateNormalIters(int idx) final {
    IterVar old_block_iter = old_block_realize_->block->iter_vars[idx];
    if (old_block_iter->iter_type == IterVarType::kDataPar) {
      iter_vars_.emplace_back(old_block_iter->dom, old_block_iter->var.copy_with_suffix(""),
                              kDataPar);
      iter_values_.push_back(old_block_realize_->iter_values[idx]);
      var_map_.Set(old_block_iter->var, iter_vars_.back());
    }
  }

  void PreProcess() final {
    for (int i = 0; i < n_buffers_; ++i) {
      PrimExpr rhs = BufferLoad(rf_buffers_[i], rf_buf_access_indices_);
      update_buffers_.push_back(old_reduction_updates_[i]->buffer);
      update_indices_.push_back(old_reduction_updates_[i]->indices);
      update_lhs_.push_back(Substitute(combiner_lhs_[i], var_map_));
      update_rhs_.push_back(Substitute(std::move(rhs), var_map_));
    }
  }

  void CreateReadWriteRegions() final {
    CreateRegion(update_rhs_, true);
    CreateRegion(update_lhs_, false);
  }

  void CreateRegion(const Array<PrimExpr>& buf_loads, bool is_read) {
    Array<BufferRegion>& buf_regions = is_read ? read_regions_ : write_regions_;
    for (const PrimExpr& expr : buf_loads) {
      const auto* buf_load = expr.as<BufferLoadNode>();
      ICHECK(buf_load != nullptr);
      Array<Range> region;
      region.reserve(buf_load->indices.size());
      for (const PrimExpr& index : buf_load->indices) {
        region.push_back(Range::FromMinExtent(index, make_const(index.dtype(), 1)));
      }
      buf_regions.push_back(BufferRegion(buf_load->buffer, std::move(region)));
    }
  }

 private:
  /*! \brief The new created additional block iter of the rfactor block */
  IterVar rf_additional_iter_;
  /*! \brief The LHS values of the reduction in the old block */
  Array<PrimExpr> combiner_lhs_;
};

/*!
 * \brief Create new outer loops for the rfactor block, meanwhile update the rfactor block's iter
 * bindings to use the new created loop vars
 * \param rf_block_realize The BlockRealize of the rfactor block
 * \param loops The loops to be wrapped over the rfactor block
 * \return A Stmt which is the wrapping result
 */
Stmt CreateLoopOutsideRfactorBlock(BlockRealize rf_block_realize, const Array<For>& loops) {
  int n_loops = static_cast<int>(loops.size());

  // Step 1. Create new loop vars.
  Array<For> new_loops;
  std::unordered_map<const VarNode*, Var> new_loop_var_map;
  new_loops.reserve(n_loops);
  new_loop_var_map.reserve(n_loops);
  for (const For& old_loop : loops) {
    Var new_loop_var = old_loop->loop_var.copy_with_suffix("");
    new_loop_var_map[old_loop->loop_var.get()] = new_loop_var;
  }

  // Step 2. Update the iter bindings and predicate of the rfactor block.
  Array<PrimExpr> new_bindings;
  new_bindings.reserve(rf_block_realize->iter_values.size());
  for (const PrimExpr& old_binding : rf_block_realize->iter_values) {
    new_bindings.push_back(Substitute(old_binding, new_loop_var_map));
  }
  {
    BlockRealizeNode* p_rf_block_realize = rf_block_realize.CopyOnWrite();
    p_rf_block_realize->iter_values = new_bindings;
    p_rf_block_realize->predicate = Substitute(rf_block_realize->predicate, new_loop_var_map);
  }

  // Step 3. Wrap `rf_block_realize` with outer loops.
  Stmt rf_body = rf_block_realize;
  for (int i = n_loops - 1; i >= 0; --i) {
    ObjectPtr<ForNode> p_loop = make_object<ForNode>(*loops[i].get());
    p_loop->loop_var = Downcast<Var>(new_loop_var_map[loops[i]->loop_var.get()]);
    p_loop->body = rf_body;
    rf_body = For(std::move(p_loop));
  }

  return rf_body;
}

class BlockReplacer : public StmtMutator {
 public:
  /*!
   * \brief The replace takes the old scope root block as input, and does four things:
   *  1) replace the reduction block with the write-back block,
   *  2) remove loops outside the write-back block that are touched by reduction block iters, except
   *  for the rfactor loop
   *  3) combine the rfactor block (wrapped with outer loops) and the transformed outermost loop
   *  into a SeqStmt, and
   *  4) insert the rfactor buffer into the scope root block's `alloc_buffers`
   * After transformation, the function returns the new scope root block
   * \param scope_root_block The old scope root block
   * \param rf_body The rfactor block, which is already wrapped with outer loops
   * \param outermost_loop The loop that is outermost among all loops outside the reduction block
   * \param wb_block_realize The new created BlockRealize of the write-back block
   * \param old_block_realize The BlockRealize of the reduction block
   * \param rf_loop The rfactor loop, which should be kept outside the write-back block
   * \param reduce_loop_vars The loops that are touched by reduction block iters, used to remove
   * loops outside the write-back block
   * \param loop_vars2loop The mapping from loop vars to loops that are outside the reduction block,
   * which is used to reduce redundant recursive visits
   * \param rf_buffer The rfactor buffer to be added into the scope root's `alloc_buffers`
   * \return The transformed new scope root block
   */
  static Block Replace(Block scope_root_block, Stmt rf_body, For outermost_loop,
                       BlockRealize wb_block_realize, BlockRealize old_block_realize, For rf_loop,
                       std::unordered_set<const VarNode*> reduce_loop_vars,
                       std::unordered_map<const VarNode*, For> loop_vars2loop,
                       const Array<Buffer>& rf_buffers) {
    BlockReplacer replacer(std::move(rf_body), std::move(outermost_loop),
                           std::move(wb_block_realize), std::move(old_block_realize),
                           std::move(rf_loop), std::move(reduce_loop_vars),
                           std::move(loop_vars2loop));
    Block new_scope_root = Downcast<Block>(replacer(std::move(scope_root_block)));
    BlockNode* p = new_scope_root.CopyOnWrite();
    for (const Buffer& rf_buffer : rf_buffers) {
      p->alloc_buffers.push_back(rf_buffer);
    }
    return new_scope_root;
  }

 private:
  explicit BlockReplacer(Stmt rf_body, For outermost_loop, BlockRealize wb_block_realize,
                         BlockRealize old_block_realize, For rf_loop,
                         std::unordered_set<const VarNode*> reduce_loop_vars,
                         std::unordered_map<const VarNode*, For> loop_vars2loop)
      : rf_body_(std::move(rf_body)),
        outermost_loop_(std::move(outermost_loop)),
        wb_block_realize_(std::move(wb_block_realize)),
        old_block_realize_(std::move(old_block_realize)),
        rf_loop_(std::move(rf_loop)),
        reduce_loop_vars_(std::move(reduce_loop_vars)),
        loop_vars2loop_(std::move(loop_vars2loop)) {}

  Stmt VisitStmt_(const ForNode* loop) final {
    // Step 1. Check whether this loop is outside the reduction block. Given that we've made sure
    // that the scope root block has stage-pipeline property, if this loop is not outside the
    // reduction block, there's no need to recursively mutate.
    if (!loop_vars2loop_.count(loop->loop_var.get())) {
      return GetRef<For>(loop);
    }

    // Step 2. Recursively mutate.
    Stmt body = StmtMutator::VisitStmt(loop->body);

    // Step 3. If this loop is the rfactor loop and isn't touched by any reduction block iter, it
    // should be kept outside the write-back block. Otherwise it shouldn't.
    if (loop == rf_loop_.get() || !reduce_loop_vars_.count(loop->loop_var.get())) {
      ObjectPtr<ForNode> p_loop = CopyOnWrite(loop);
      p_loop->body = body;
      body = Stmt(p_loop);
    }

    // Step 4. If this loop is the outermost loop of the reduction block, return the combination of
    // `rf_body_` and the mutation result `body`. Otherwise return the mutation result.
    return loop == outermost_loop_.get() ? SeqStmt({rf_body_, body}) : body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* block_realize) final {
    // Due to the visitor's behavior on ForNode, this block-realize must be the reduction block's
    // block-realize. And we directly return the new `wb_block_realize`.
    ICHECK_EQ(block_realize, old_block_realize_.get());
    return wb_block_realize_;
  }

  Stmt VisitStmt_(const SeqStmtNode* seq) final {
    Array<Stmt> new_stmts;
    new_stmts.reserve(static_cast<int>(seq->seq.size()));

    for (const Stmt old_stmt : seq->seq) {
      new_stmts.push_back(VisitStmt(old_stmt));
    }
    return SeqStmt::Flatten(new_stmts);
  }

 private:
  Stmt rf_body_;
  For outermost_loop_;
  BlockRealize wb_block_realize_;
  BlockRealize old_block_realize_;
  For rf_loop_;
  std::unordered_set<const VarNode*> reduce_loop_vars_;
  std::unordered_map<const VarNode*, For> loop_vars2loop_;
};

StmtSRef RFactor(ScheduleState self, const StmtSRef& rf_loop_sref, int factor_axis) {
  // *****************************************************
  // *    Condition Checks and Information Collection    *
  // *****************************************************

  // Step 1. Check some basic conditions for rfactor. Get the block and block-realize.
  BlockRealize block_realize = CheckGetSingleChildBlockRealizeOnSRefTree(self, rf_loop_sref);
  const StmtSRef& block_sref = self->stmt2ref.at(block_realize->block.get());
  const Block& block = block_realize->block;
  StmtSRef scope_root = GetScopeRoot(self, block_sref,  //
                                     /*require_stage_pipeline=*/true);
  if (self->enable_check) {
    CheckReductionBlock(self, block_sref, scope_root);
  }
  const ForNode* rf_loop = TVM_SREF_TO_FOR(rf_loop_sref);
  if (rf_loop->kind != ForKind::kSerial) {
    throw NotSerialLoopKindError(self->mod, GetRef<For>(rf_loop));
  }

  // Step 2. Collect loop vars that are touched by data parallel block iters and reduction block
  // iters, respectively.
  std::unordered_set<const VarNode*> data_par_loop_vars;
  std::unordered_set<const VarNode*> reduce_loop_vars;
  GetVarsTouchedByBlockIters(block_realize, &data_par_loop_vars, &reduce_loop_vars);

  // Step 3. Collect the loops of the reduction block. Construct a mapping from loops to
  // corresponding loop vars.
  Array<For> loops = LoopSRefs2Loops(GetLoops(block_sref));
  std::unordered_map<const VarNode*, For> loop_vars2loop = GetLoopVar2LoopMap(loops);

  // Step 4. Check four properties that the loops should have:
  // - the rfactor loop cannot be touched by any data parallel block iter;
  // - all the loops cannot be touched by both data parallel block iters and reduction block iters;
  // - the outermost loop should have the reduction block as its first child block;
  // - the outermost loop that is touched by some reduction block iters can only have one child
  // block.
  if (self->enable_check) {
    LoopPropertyError::CheckLoopProperty(self, loops, rf_loop, block, data_par_loop_vars,
                                         reduce_loop_vars);
  }

  // Step 5. Get the `init` identity and the `update` combiner of the reduction. Extract the
  // commutative reducer, combiner lhs and combiner rhs from the reduction identity and the
  // reduction combiner. The lhs will be used when constructing the write-back block, and the rhs
  // will be used when constructing the rfactor block.
  Array<PrimExpr> init_values{nullptr};
  Array<BufferStore> updates{nullptr};
  CommReducer reducer{nullptr};
  Array<PrimExpr> combiner_lhs{nullptr};
  Array<PrimExpr> combiner_rhs{nullptr};
  std::tie(init_values, updates) = GetInitValuesAndUpdatesFromReductionBlock(self, block);
  std::tie(reducer, combiner_lhs, combiner_rhs) =
      GetReducerAndCombinerLhsRhs(self, init_values, updates);

  // Step 6. Check whether `factor_axis` is in a correct range, and convert it to non-negative if it
  // is negative.
  factor_axis =
      FactorAxisOutOfRangeError::CheckAndUpdate(self->mod, updates[0]->buffer, factor_axis);

  // *****************************************************
  // *                 IR Manipulation                   *
  // *****************************************************
  // Since rfactor splits the reduction block into two, we call the first one "rfactor block", and
  // the latter one "write-back block", and the intermediate buffer is called "rfactor buffer".

  // Step 1. Create the intermediate buffer (a.k.a. rfactor buffer), which has an additional
  // dimension that specified by `factor_axis` and `rf_loop`.
  Array<Buffer> rf_buffers = CreateRFactorBuffers(updates, factor_axis, rf_loop);

  // Step 2. Create the rfactor block.
  RFactorBlockCreator rf_block_creator(block_realize, GetRef<For>(rf_loop), updates, reducer,
                                       rf_buffers, loop_vars2loop, factor_axis,
                                       std::move(combiner_rhs));
  rf_block_creator.CreateBlock();

  // Step 3. Create the write-back block.
  WriteBackBlockCreator wb_block_creator(block_realize, GetRef<For>(rf_loop), updates, reducer,
                                         rf_buffers, std::move(rf_block_creator.additional_iter_),
                                         std::move(combiner_lhs),
                                         std::move(rf_block_creator.rf_buf_access_indices_));
  wb_block_creator.CreateBlock();

  // Step 4. Wrap the rfactor block with loops.
  Stmt rf_body = CreateLoopOutsideRfactorBlock(rf_block_creator.new_block_realize_, loops);

  // *****************************************************
  // *           Schedule Replacement & Update           *
  // *****************************************************

  // Step 1. Substitute the old scope root block with the new scope root block.
  Block old_scope_root_block = GetRef<Block>(scope_root->StmtAs<BlockNode>());
  Block new_scope_root_block = BlockReplacer::Replace(
      old_scope_root_block, rf_body, loops[0], wb_block_creator.new_block_realize_, block_realize,
      GetRef<For>(rf_loop), reduce_loop_vars, loop_vars2loop, rf_buffers);
  self->Replace(
      scope_root, new_scope_root_block,
      {{old_scope_root_block, new_scope_root_block}, {block, wb_block_creator.new_block_}});

  // Step 2. Update scope information.
  std::vector<StmtSRef> new_block_srefs{self->stmt2ref.at(rf_block_creator.new_block_.get()),
                                        self->stmt2ref.at(wb_block_creator.new_block_.get())};
  for (const StmtSRef& new_block_sref : new_block_srefs) {
    BlockInfo& info = self->block_info[new_block_sref];
    info.affine_binding = true;
    info.region_cover = true;
    info.stage_pipeline = true;
  }
  return new_block_srefs[0];
}

/******** InstructionKind Registration ********/

struct DecomposeReductionTraits : public UnpackedInstTraits<DecomposeReductionTraits> {
  static constexpr const char* kName = "DecomposeReduction";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv) {
    return sch->DecomposeReduction(block_rv, loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv) {
    PythonAPICall py("decompose_reduction");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct RFactorTraits : public UnpackedInstTraits<RFactorTraits> {
  static constexpr const char* kName = "RFactor";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static BlockRV UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, Integer factor_axis) {
    return sch->RFactor(loop_rv, factor_axis->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, Integer factor_axis) {
    PythonAPICall py("rfactor");
    py.Input("loop", loop_rv);
    py.Input("factor_axis", factor_axis->value);
    py.SingleOutput(outputs);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(RFactorTraits);
TVM_REGISTER_INST_KIND_TRAITS(DecomposeReductionTraits);

/******** FFI ********/

TVM_REGISTER_GLOBAL("tir.schedule.RegisterReducer")
    .set_body_typed([](int n_buffers, PackedFunc combiner_getter, PackedFunc identity_getter) {
      ReducerRegistry::RegisterReducer(n_buffers, std::move(combiner_getter),
                                       std::move(identity_getter));
    });

}  // namespace tir
}  // namespace tvm
