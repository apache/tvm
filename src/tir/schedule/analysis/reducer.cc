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

/******** Pattern Matcher ********/

/*!
 * \brief PrimExpr pattern matcher.
 *
 * It is different from the pattern matcher in arith/pattern_match.h, which is dedicated
 * for compile-time constant patterns. This pattern matcher can work on dynamic user-specific
 * patterns.
 *
 * The code below shows how to use the pattern matcher.
 *
 * \code
 *
 * Var x("x"), y("y");
 * // use PrimExpr to declare patterns, x, y are holes that can be filled with
 * PatternMatcher pattern_matcher(x + y);
 * // expr = C[i, j] + A[i, k] * B[k, j], which is the expr we want to match
 * pattern_matcher.Match(expr);
 *
 * if (pattern_matcher.Success()) {
 *   pattern_matcher.Eval(x) // C[i, j]
 *   pattern_matcher.Eval(y) // A[i, k] * B[k, j]
 * }
 *
 * \endcode
 */
class PatternMatcher : public ExprVisitor {
 public:
  explicit PatternMatcher(Array<PrimExpr> pattern) : pattern_(std::move(pattern)) {}

  void VisitExpr_(const VarNode* op) final {
    auto it = filled_map_.find(op);
    if (it == filled_map_.end()) {
      filled_map_[op] = expr_to_match_;
    } else {
      ExprDeepEqual equal;
      if (it->second.same_as(expr_to_match_) || equal(it->second, expr_to_match_)) return;
      match_success_ = false;
    }
  }

  void VisitExpr_(const LetNode* op) final {
    const auto* ptr = expr_to_match_.as<LetNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->var;
      VisitExpr(op->var);
      expr_to_match_ = ptr->value;
      VisitExpr(op->value);
      expr_to_match_ = ptr->body;
      VisitExpr(op->body);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    const auto* ptr = expr_to_match_.as<CallNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->op.same_as(ptr->op)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->args.size(); ++i) {
          expr_to_match_ = ptr->args[i];
          VisitExpr(op->args[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

#define TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OpName) \
  void VisitExpr_(const OpName* op) {              \
    const auto* ptr = expr_to_match_.as<OpName>(); \
    if (ptr == nullptr) {                          \
      match_success_ = false;                      \
    } else {                                       \
      PrimExpr current = expr_to_match_;           \
      expr_to_match_ = ptr->a;                     \
      VisitExpr(op->a);                            \
      expr_to_match_ = ptr->b;                     \
      VisitExpr(op->b);                            \
      std::swap(expr_to_match_, current);          \
    }                                              \
  }

  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AddNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(SubNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MulNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(DivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(ModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorDivNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(FloorModNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MinNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(MaxNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(EQNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(NENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(LENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GTNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(GENode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(AndNode);
  TVM_DECLARE_PATTERN_MATCHER_BIN_OP(OrNode);

  void VisitExpr_(const CastNode* op) final {
    const auto* ptr = expr_to_match_.as<CastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!runtime::TypeEqual(op->dtype, ptr->dtype)) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        expr_to_match_ = ptr->value;
        VisitExpr(op->value);
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const NotNode* op) final {
    const auto* ptr = expr_to_match_.as<NotNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->a;
      VisitExpr(op->a);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const SelectNode* op) final {
    const auto* ptr = expr_to_match_.as<SelectNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->condition;
      VisitExpr(op->condition);
      expr_to_match_ = ptr->true_value;
      VisitExpr(op->true_value);
      expr_to_match_ = ptr->false_value;
      VisitExpr(op->false_value);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const RampNode* op) final {
    const auto* ptr = expr_to_match_.as<RampNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->base;
      VisitExpr(op->base);
      expr_to_match_ = ptr->stride;
      VisitExpr(op->stride);
      expr_to_match_ = ptr->lanes;
      VisitExpr(op->lanes);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const BroadcastNode* op) final {
    const auto* ptr = expr_to_match_.as<BroadcastNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      PrimExpr tmp = expr_to_match_;
      expr_to_match_ = ptr->value;
      VisitExpr(op->value);
      expr_to_match_ = ptr->lanes;
      VisitExpr(op->lanes);
      std::swap(expr_to_match_, tmp);
    }
  }

  void VisitExpr_(const ShuffleNode* op) final {
    const auto* ptr = expr_to_match_.as<ShuffleNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (op->vectors.size() != ptr->vectors.size() || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          VisitExpr(op->indices[i]);
        }
        for (size_t i = 0; i < op->vectors.size(); ++i) {
          expr_to_match_ = ptr->vectors[i];
          VisitExpr(op->vectors[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void VisitExpr_(const IntImmNode* op) final {
    const auto* ptr = expr_to_match_.as<IntImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const FloatImmNode* op) final {
    const auto* ptr = expr_to_match_.as<FloatImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const StringImmNode* op) final {
    const auto* ptr = expr_to_match_.as<StringImmNode>();
    match_success_ = ptr != nullptr && op->value == ptr->value;
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    const auto* ptr = expr_to_match_.as<BufferLoadNode>();
    if (ptr == nullptr) {
      match_success_ = false;
    } else {
      if (!op->buffer.same_as(ptr->buffer) || op->indices.size() != ptr->indices.size()) {
        match_success_ = false;
      } else {
        PrimExpr tmp = expr_to_match_;
        for (size_t i = 0; i < op->indices.size(); ++i) {
          expr_to_match_ = ptr->indices[i];
          VisitExpr(op->indices[i]);
        }
        std::swap(expr_to_match_, tmp);
      }
    }
  }

  void Match(const Array<PrimExpr>& exprs_to_match) {
    this->match_success_ = true;
    this->filled_map_.clear();

    ICHECK_EQ(pattern_.size(), exprs_to_match.size());
    int n_buffers = pattern_.size();
    for (int i = 0; i < n_buffers; ++i) {
      this->expr_to_match_ = exprs_to_match[i];
      this->operator()(pattern_[i]);
    }
  }

  PrimExpr Eval(const Var& var) {
    auto it = filled_map_.find(var.operator->());
    ICHECK(it != filled_map_.end()) << "Unknown pattern variable";
    ICHECK(match_success_) << "Match failed";
    return it->second;
  }

  bool Success() const { return match_success_; }

 private:
  bool match_success_{true};
  Array<PrimExpr> pattern_;
  PrimExpr expr_to_match_;
  std::unordered_map<const VarNode*, PrimExpr> filled_map_;
};

/******** Reduction Block Related ********/

static const char* kRFactorCrossThreadReductionApplicableBlockDef =
    R"(Definition of a reduction block that is applicable by RFactor and Cross-Thread Reduction:
1) The block init should be a single BufferStore or a SeqStmt of BufferStores
2) The buffers initialized in the block init should be all different
3) The number of consecutive LetStmts in the block body (if any) should equal the number of BufferStores in the block init
4) The variables of the LetStmts in the block body should be all different
5) The body of the innermost LetStmt should be a single BufferStore or a SeqStmt of BufferStores
6) The number of BufferStores under the block body should equal the number of BufferStores in the block init, and thereby equal the number of LetStmts above
7) The variables bound by the LetStmts in the block body must all directly serve as values of the BufferStores inside, and the stored values of the BufferStores can only be those variables
8) The variables stored by the BufferStores in the block body should be all different
9) The buffers written by the BufferStores in the block body should be all different
10) The buffers initialized in the block init and written in the block body should match
11) The buffers written by the block should have same shape
12) The indices of all BufferStores in the reduction block should be the same)";

void ErrorRFactorCrossThreadReductionNotApplicable(const Optional<ScheduleState>& self, Block block,
                                                   int violated_cond) {
  class RFactorNotApplicableError : public ScheduleError {
   public:
    explicit RFactorNotApplicableError(IRModule mod, Block block, int violated_cond)
        : mod_(std::move(mod)), block_(std::move(block)), violated_cond_(violated_cond) {}

    String FastErrorString() const final {
      return "ScheduleError: RFactor cannot be applied to the block since the block does not meet "
             "the requirements";
    }

    String DetailRenderTemplate() const final {
      std::ostringstream os;
      os << "RFactor cannot be applied to block {0}, because the block violates condition #"
         << violated_cond_ << ".\n"
         << kRFactorCrossThreadReductionApplicableBlockDef;
      return os.str();
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {block_}; }

    IRModule mod_;
    Block block_;
    int violated_cond_;
  };

  if (self.defined()) {
    throw RFactorNotApplicableError(self.value()->mod, std::move(block), violated_cond);
  } else {
    LOG(FATAL) << "ValueError: Cross-thread reduction cannot be applied to the block "
               << block->name_hint << " because the block violates the condition #" << violated_cond
               << ".\n"
               << kRFactorCrossThreadReductionApplicableBlockDef;
  }
}

/*!
 * \brief Extract the BufferStores, which serve as the reduction updates, from the given LetStmt and
 * the BufferStores inside. And meanwhile set the buffer order of the reduction
 * \param self The schedule state, used for error reporting
 * \param block The reduction block, used for error reporting
 * \param let The LetStmt from which the reduction updates are extracted
 * \param n_buffers The number of buffers participating in the reduction
 * \param updates The extracted reduction updates
 * \param buf2index A mapping from reduction buffers to their indices of the reduction order
 * \throw ScheduleError If rfactor or cross-thread reduction cannot be applied to the block
 */
void ExtractReductionUpdates(const Optional<ScheduleState>& self, Block block,
                             const LetStmtNode* let, int n_buffers, Array<BufferStore>* updates,
                             std::unordered_map<const BufferNode*, int>* buf2index) {
  std::unordered_map<const VarNode*, int> var2index;
  Array<PrimExpr> let_values;
  let_values.reserve(n_buffers);
  updates->resize(n_buffers);

  // Step 1.
  // - Extract the BufferStore values from the LetStmts.
  // - Construct the mapping from let variables to the index.
  for (int i = 0; i < n_buffers; ++i) {
    if (let == nullptr) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/3);
    }

    let_values.push_back(let->value);
    auto insert_result = var2index.insert(std::make_pair(let->var.get(), i));
    if (!insert_result.second) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/4);
    }
    if (i != n_buffers - 1) {
      let = let->body.as<LetStmtNode>();
    }
  }

  // There should be no more LetStmt.
  if (let->body->IsInstance<LetStmtNode>()) {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/3);
  }

  // Now `let` is expected to be the innermost LetStmt, whose body should either be a SeqStmt or
  // a BufferStore
  const auto* p_seq = let->body.as<SeqStmtNode>();
  const auto* p_buf_store = let->body.as<BufferStoreNode>();
  if (p_seq == nullptr && p_buf_store == nullptr) {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/5);
  }
  Array<Stmt> seq = p_seq != nullptr ? p_seq->seq : Array<Stmt>{GetRef<BufferStore>(p_buf_store)};
  if (static_cast<int>(seq.size()) != n_buffers) {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/6);
  }

  // Step 2.
  // - Create BufferStores according to the variables being stored.
  // - Construct the mapping from reduction buffers to the index.
  for (const Stmt& stmt : seq) {
    const auto* buf_store = stmt.as<BufferStoreNode>();
    if (buf_store == nullptr) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/5);
    }
    const auto* var = buf_store->value.as<VarNode>();
    if (var == nullptr) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/7);
    }
    auto it = var2index.find(var);
    if (it == var2index.end()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/7);
    }
    int idx = it->second;
    if ((*updates)[idx].defined()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/8);
    }
    updates->Set(idx, BufferStore(buf_store->buffer, let_values[idx], buf_store->indices));
    auto insert_result = buf2index->insert(std::make_pair(buf_store->buffer.get(), idx));
    if (!insert_result.second) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/9);
    }
  }
  for (int i = 0; i < n_buffers; ++i) {
    ICHECK((*updates)[i].defined());
  }
}

std::pair<Array<PrimExpr>, Array<BufferStore>> GetInitValuesAndUpdatesFromReductionBlock(
    const Optional<ScheduleState>& self, Block block) {
  Array<BufferStore> inits;
  Array<BufferStore> updates;

  // Step 1. Extract the BufferStores serving as block inits.
  if (auto init = block->init.as<BufferStore>()) {
    inits.push_back(init.value());
  } else if (const auto* seq_init = block->init.as<SeqStmtNode>()) {
    std::unordered_set<const BufferNode*> init_buffers;
    for (const Stmt& stmt : seq_init->seq) {
      auto init = stmt.as<BufferStore>();
      if (!init) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/1);
      }
      auto insert_result = init_buffers.insert(init.value()->buffer.get());
      if (!insert_result.second) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/2);
      }
      inits.push_back(init.value());
    }
  } else {
    ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/1);
  }

  // Step 2. Extract the block updates, in the form of BufferStores.
  int n_buffers = inits.size();
  std::unordered_map<const BufferNode*, int> buf2index;
  if (const auto* update = block->body.as<BufferStoreNode>()) {
    updates.push_back(GetRef<BufferStore>(update));
    buf2index[update->buffer.get()] = 0;
  } else {
    const auto* let = block->body.as<LetStmtNode>();
    ExtractReductionUpdates(self, block, let, n_buffers, &updates, &buf2index);
  }
  ICHECK_EQ(updates.size(), n_buffers);

  // Step 3. Set the init values according to the buffer order in `updates`, with the help of the
  // mapping `buf2index`.
  Array<PrimExpr> init_values;
  init_values.resize(n_buffers);

  // - Check all buffers have the same shape
  // - Check all indices of the BufferStores are the same
  // - Check buffers written in the block init and the block body can match
  // - Check buffers do not duplicate
  const Array<PrimExpr>& expected_shape = updates[0]->buffer->shape;
  const Array<PrimExpr>& expected_indices = updates[0]->indices;
  ICHECK_EQ(expected_shape.size(), expected_indices.size());
  int n_dim = expected_indices.size();
  arith::Analyzer ana;
  for (int i = 0; i < n_buffers; ++i) {
    if (static_cast<int>(updates[i]->buffer->shape.size()) != n_dim) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/11);
    }
    if (static_cast<int>(inits[i]->indices.size()) != n_dim ||
        static_cast<int>(updates[i]->indices.size()) != n_dim) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/12);
    }
    for (int d = 0; d < n_dim; ++d) {
      if (!ana.CanProveEqual(updates[i]->buffer->shape[d], expected_shape[d])) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/11);
      }
      if (!ana.CanProveEqual(inits[i]->indices[d], expected_indices[d]) ||
          !ana.CanProveEqual(updates[i]->indices[d], expected_indices[d])) {
        ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/12);
      }
    }

    auto it = buf2index.find(inits[i]->buffer.get());
    if (it == buf2index.end()) {
      ErrorRFactorCrossThreadReductionNotApplicable(self, std::move(block), /*violated_cond=*/10);
    }
    int idx = it->second;
    ICHECK(updates[idx]->buffer.same_as(inits[i]->buffer));
    ICHECK(!init_values[idx].defined());
    init_values.Set(idx, inits[i]->value);
  }
  for (int i = 0; i < n_buffers; ++i) {
    ICHECK(init_values[i].defined());
  }

  return std::make_pair(init_values, updates);
}

bool ContainsOnlyDataParAndReductionBlockIter(const Array<IterVar>& iters) {
  for (const IterVar& iter_var : iters) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  return true;
}

bool ReductionIterNotIndexOutputBuffer(const Block& block) {
  // Step 1. Collect the reduction block iters.
  std::unordered_set<const VarNode*> reduction_block_iters;
  reduction_block_iters.reserve(block->iter_vars.size());
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type == kCommReduce) {
      reduction_block_iters.insert(iter_var->var.get());
    }
  }
  // Step 2. Check if the reduction block iters are used to index the output buffer.
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }

  std::unordered_set<const BufferNode*> buffer_allocated;
  buffer_allocated.reserve(block->alloc_buffers.size());
  for (const Buffer& buffer : block->alloc_buffers) {
    buffer_allocated.insert(buffer.get());
  }

  auto f_uses_reduction_block_var = [&](const PrimExpr& expr) -> bool {
    return UsesVar(expr, [&](const VarNode* var) {  //
      return reduction_block_iters.count(var);
    });
  };

  std::unordered_map<const BufferNode*, const BufferNode*> match_buffer_sources;
  for (const MatchBufferRegion& region : block->match_buffers) {
    match_buffer_sources[region->buffer.get()] = region->source->buffer.get();
  }
  bool affected = false;
  PreOrderVisit(block->body, [&](const ObjectRef& obj) {
    if (affected) {
      return false;
    }
    const auto* block_node = obj.as<BlockNode>();
    if (block_node) {
      for (const MatchBufferRegion& region : block_node->match_buffers) {
        match_buffer_sources[region->buffer.get()] = region->source->buffer.get();
      }
    }
    const auto* store = obj.as<BufferStoreNode>();
    if (!store) {
      return true;
    }

    bool write_is_covered_by_match_buffer =
        match_buffer_sources.count(store->buffer.get()) &&
        buffer_written.count(match_buffer_sources.find(store->buffer.get())->second);
    ICHECK(buffer_written.count(store->buffer.get()) || write_is_covered_by_match_buffer ||
           buffer_allocated.count(store->buffer.get()))
        << "ValueError: The buffer \"" << store->buffer
        << "\" is written in the block but is not in the block's signature nor is it covered by "
           "a match_buffer";
    for (const PrimExpr& index : store->indices) {
      if (f_uses_reduction_block_var(index)) {
        affected = true;
        return false;
      }
    }
    return false;
  });
  return !affected;
}

class NoMatchedReducerError : public ScheduleError {
 public:
  explicit NoMatchedReducerError(IRModule mod, Array<PrimExpr> identities,
                                 Array<BufferStore> combiners)
      : mod_(std::move(mod)),
        identities_(std::move(identities)),
        combiners_(std::move(combiners)) {}

  String FastErrorString() const final {
    return "ScheduleError: No matched reducer for the identity and the combiner of this reduction "
           "block. So rfactor and cross-thread reduction cannot be applied.";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "No matched reducer for identity " << identities_ << " and combiner " << combiners_
       << "In this case rfactor cannot be applied. You can check tvm::tir::ReducerRegistry for "
          "default reducers or registering new reducers.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

  IRModule mod_;
  Array<PrimExpr> identities_;
  Array<BufferStore> combiners_;
};

std::tuple<CommReducer, Array<PrimExpr>, Array<PrimExpr>> GetReducerAndCombinerLhsRhs(
    const Optional<ScheduleState>& self, const Array<PrimExpr>& identities,
    const Array<BufferStore>& combiners) {
  CommReducer reducer{nullptr};
  Array<PrimExpr> combiner_lhs, combiner_rhs;
  bool matched =
      FromIdentityCombiner(identities, combiners, &reducer, &combiner_lhs, &combiner_rhs);
  if (!matched) {
    if (self.defined()) {
      throw NoMatchedReducerError(self.value()->mod, identities, combiners);
    } else {
      LOG(FATAL) << "ValueError: No matched reducer for the identity and the combiner of the "
                    "reduction block. So rfactor and cross-thread reduction cannot be applied.";
    }
  }
  return std::make_tuple(std::move(reducer), std::move(combiner_lhs), std::move(combiner_rhs));
}

/******** Commutative Reducer ********/

bool MatchReducer(const CommReducer& reducer, const Array<PrimExpr>& identities,
                  const Array<PrimExpr>& combined_values, const Array<BufferLoad>& buf_loads,
                  Array<PrimExpr>* lhs, Array<PrimExpr>* rhs) {
  ExprDeepEqual equal;
  ICHECK_EQ(identities.size(), combined_values.size());
  int n_buffers = identities.size();
  for (int i = 0; i < n_buffers; ++i) {
    if (!equal(reducer->identity_element[i], identities[i])) {
      return false;
    }
  }

  PatternMatcher pattern_matcher(reducer->result);
  pattern_matcher.Match(combined_values);
  Array<PrimExpr> lhs_tmp, rhs_tmp;
  lhs_tmp.reserve(n_buffers);
  rhs_tmp.reserve(n_buffers);
  if (!pattern_matcher.Success()) {
    return false;
  }

  for (int i = 0; i < n_buffers; ++i) {
    PrimExpr l = pattern_matcher.Eval(reducer->lhs[i]);
    PrimExpr r = pattern_matcher.Eval(reducer->rhs[i]);
    if (!equal(buf_loads[i], l)) {
      return false;
    }
    lhs_tmp.push_back(l);
    rhs_tmp.push_back(r);
  }
  *lhs = std::move(lhs_tmp);
  *rhs = std::move(rhs_tmp);
  return true;
}

bool FromIdentityCombiner(const Array<PrimExpr>& identities, const Array<BufferStore>& combiners,
                          CommReducer* result_reducer, Array<PrimExpr>* lhs, Array<PrimExpr>* rhs) {
  int n = identities.size();
  Array<BufferLoad> buf_loads;
  Array<PrimExpr> stored_values;
  buf_loads.reserve(n);
  stored_values.reserve(n);

  for (int i = 0; i < n; ++i) {
    buf_loads.push_back(BufferLoad(combiners[i]->buffer, combiners[i]->indices));
    stored_values.push_back(combiners[i]->value);
  }

  // Check reduction patterns.
  for (const TypedPackedFunc<Optional<CommReducer>(Array<PrimExpr>)>& reducer_getter :
       GetReducerGetters()) {
    Optional<CommReducer> reducer = reducer_getter(identities);
    if (!reducer.defined()) {
      continue;
    }
    if (MatchReducer(reducer.value(), identities, stored_values, buf_loads, lhs, rhs)) {
      *result_reducer = reducer.value();
      return true;
    }
  }
  return false;
}

}  // namespace tir
}  // namespace tvm
