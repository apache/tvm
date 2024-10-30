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

using support::NDIntSet;

/******** Error Classes ********/

/*!
 * \brief An error raised when not all required blocks are under the given loop.
 * \tparam is_consumer Indicates if all the required blocks are consumers or producers
 */
template <bool is_consumer>
class NotAllRequiredBlocksAreVisitedError : public ScheduleError {
 public:
  explicit NotAllRequiredBlocksAreVisitedError(IRModule mod, int num_not_visited,
                                               const Array<StmtSRef>& required)
      : mod_(mod), num_not_visited_(num_not_visited) {
    required_.reserve(required.size());
    for (const StmtSRef& block_sref : required) {
      const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
      required_.push_back(GetRef<Block>(block));
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: Not all required blocks are under the loop scope";
  }

  String DetailRenderTemplate() const final {
    String relation = is_consumer ? "consumer(s)" : "producer(s)";
    std::ostringstream os;
    os << "The primitive requires all the " << relation
       << " of the given block to be present under the target loop. However, there are "
       << num_not_visited_ << " " << relation << " not satisfying the constraint. List of the "
       << relation << ":";
    for (int i = 0, n = required_.size(); i < n; ++i) {
      os << "{" << i << "}";
    }
    return os.str();
  }

  IRModule mod() const final { return mod_; }

  Array<ObjectRef> LocationsOfInterest() const final {
    return {required_.begin(), required_.end()};
  }

 private:
  IRModule mod_;
  int num_not_visited_;
  Array<Block> required_;
};

/*!
 * \brief An error raised when the given block is not in the same block scope as the given loop,
 * or the given loop is the ancestor of the given block.
 */
class NotInSameScopeError : public ScheduleError {
 public:
  static void CheckAndBindLoopDomain(const ScheduleState& self, const StmtSRef& block_sref,
                                     const StmtSRef& loop_sref, const StmtSRef& scope_root_sref,
                                     arith::Analyzer* analyzer) {
    for (const StmtSRefNode* p = loop_sref.get();; p = p->parent) {
      if (const ForNode* loop = p->StmtAs<ForNode>()) {
        analyzer->Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      } else if (p != scope_root_sref.get()) {
        throw NotInSameScopeError(self->mod, block_sref, loop_sref);
      } else {
        break;
      }
    }
    for (const StmtSRefNode* p = block_sref->parent; p != scope_root_sref.get(); p = p->parent) {
      if (p == loop_sref.get()) {
        throw NotInSameScopeError(self->mod, block_sref, loop_sref);
      }
    }
  }

  String FastErrorString() const final {
    return "ScheduleError: Expected the block and loop to be under the same block scope, and loop "
           "not to be the ancestor of block";
  }
  String DetailRenderTemplate() const final {
    return "ScheduleError: Expected the block {0} and loop {1} to be under the same block scope, "
           "and loop not to be the ancestor of block";
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {block_, loop_}; }

 private:
  explicit NotInSameScopeError(IRModule mod, const StmtSRef& block_sref, const StmtSRef& loop_sref)
      : mod_(mod),
        block_(GetRef<Block>(block_sref->StmtAs<BlockNode>())),
        loop_(GetRef<For>(loop_sref->StmtAs<ForNode>())) {}

  IRModule mod_;
  Block block_;
  For loop_;
};

/******** Helper Functions/Classes ********/

/*!
 * \brief Find a point where the block can be inserted under the loop
 * \tparam require_all_producers_visited Requires all producer blocks to be present under the loop
 * \tparam require_all_consumers_visited Requires all consumer blocks to be present under the loop
 * \param self The schedule state
 * \param subtrees The subtrees under the loop, among which the insertion points are sought
 * \param producer_srefs The producer blocks
 * \param consumer_srefs The consumer blocks
 * \param block2realize A cache that maps a block to its realize
 * \param index The block index of the loop body subtree blocks:
 * - `index = -1` means inserted into the last possible insertion point;
 * - `index = -2` means inserted into the first possible insertion point;
 * - Otherwise, `index` is a nonnegative number that indicates the insertion point
 * \return The possible position the new block can be inserted into, and the
 * producer-consumer-relationship is still satisfied.
 * \throws ScheduleError if there is no such insertion point found
 */
template <bool require_all_producers_visited, bool require_all_consumers_visited>
int FindInsertionPoint(const ScheduleState& self, const Array<Stmt>& subtrees,
                       const Array<StmtSRef>& producer_srefs, const Array<StmtSRef>& consumer_srefs,
                       std::unordered_map<const BlockNode*, const BlockRealizeNode*>* block2realize,
                       int index) {
  ProducerConsumerSplit split =
      ProducerConsumerSplit::Find(self, subtrees, producer_srefs, consumer_srefs, block2realize);
  // Step 1. Check if all the producers are visited in the subtrees, if required to
  if (require_all_producers_visited) {
    int num_producers = producer_srefs.size();
    if (split.n_producers_visited < num_producers) {
      throw NotAllRequiredBlocksAreVisitedError<false>(
          self->mod, num_producers - split.n_producers_visited, producer_srefs);
    }
  }
  // Step 2. Check if all the consumers are visited in the subtrees, if required to
  if (require_all_consumers_visited) {
    int num_consumers = consumer_srefs.size();
    if (split.n_consumers_visited < num_consumers) {
      throw NotAllRequiredBlocksAreVisitedError<true>(
          self->mod, num_consumers - split.n_consumers_visited, consumer_srefs);
    }
  }
  // Step 3. Check if there is at least one index of the position can be inserted into
  // The valid indices are: (last_producer_position, first_consumer_position]
  ICHECK(split.last_producer_position < split.first_consumer_position);
  // Step 4. Return the possible insertion point according to index
  int insert_position;
  if (index == -1) {
    insert_position = split.first_consumer_position;
  } else if (index == -2) {
    insert_position = split.last_producer_position + 1;
  } else if (index >= 0 && index >= split.last_producer_position + 1 &&
             index <= split.first_consumer_position) {
    insert_position = index;
  } else {
    LOG(FATAL) << "Valid index:(-1, -2, [" << split.last_producer_position + 1 << ", "
               << split.first_consumer_position << "]), "
               << "current index=" << index;
    throw;
  }
  return insert_position;
}

/*!
 * \brief Represent the iteration domain to fully cover the required region of Intersect(dom, bound)
 * The bound region may not get directly intersected with dom region, instead we try to generate
 * extra predicates for non-trivial bound. The domain info class can also union with each other.
 */
struct BlockVarDomainInfo {
  arith::IntSet dom{arith::IntSet::Nothing()};  // dom is ensured to be bounded
  arith::IntSet bound{arith::IntSet::Nothing()};

  /*! \brief Relaxed union operation */
  void Union(const BlockVarDomainInfo& other) {
    // just relax (d0 ^ b0) v (d1 ^ b1) to (d0 v d1) ^ (b0 v b1)
    dom = arith::Union({dom, other.dom});
    bound = arith::Union({bound, other.bound});
  }

  /*! \brief Simplify domain info */
  void Simplify(arith::Analyzer* analyzer) {
    auto to_simplified = [analyzer](const arith::IntSet& set) {
      PrimExpr min = set.HasLowerBound() ? analyzer->Simplify(set.min()) : set.min();
      PrimExpr max = set.HasUpperBound() ? analyzer->Simplify(set.max()) : set.max();
      return arith::IntSet::Interval(min, max);
    };
    // if no dom specified, try use bound as dom
    if (dom.IsNothing()) {
      if (bound.HasLowerBound() && bound.HasUpperBound()) {
        bound = to_simplified(bound);
        std::swap(dom, bound);
      }
      return;
    }
    // simplify intset
    dom = to_simplified(dom);
    bound = to_simplified(bound);
    // if can proof the dom is within bound, remove bound
    auto intersect = to_simplified(arith::Intersect({dom, bound}));
    if (analyzer->CanProveEqual(dom.min(), intersect.min()) &&
        analyzer->CanProveEqual(dom.max(), intersect.max())) {
      bound = arith::IntSet::Nothing();
    } else if (analyzer->CanProveEqual(bound.min(), intersect.min()) &&
               analyzer->CanProveEqual(bound.max(), intersect.max())) {
      dom = bound;
      bound = arith::IntSet::Nothing();
    } else if (is_const_int(intersect.min()) && is_const_int(intersect.max())) {
      // if the bound induce constant iter range, merge bound to loop domain
      dom = intersect;
      bound = arith::IntSet::Nothing();
    }
  }
};

/*!
 * \brief A helper to reconstruct the block scope where the given block is moved under the given
 * loop, and the given block's induced loop nest is regenerated to satisfy the required region.
 */
class ScopeReconstructor : private StmtMutator {
 public:
  explicit ScopeReconstructor(Block scope_root, Block block, For loop)
      : scope_root_(scope_root), block_(block), loop_(loop) {}

  using StmtMutator::operator();

  /*!
   * \brief Create the loop nest on top of the block, induced by the given block var's domain
   * \param insert_position The position among the subtrees where the block and its induced loop
   * nest is inserted
   * \param iter_doms The domain of each block var
   * \param analyzer The arithmetic analyzer
   * \param preserve_unit_loops Whether to generate unit loops where the loop extent is 1
   */
  void MakeNewLoop(int insert_position, std::vector<BlockVarDomainInfo> iter_doms,
                   arith::Analyzer* analyzer, bool preserve_unit_loops) {
    int n_iters = iter_doms.size();
    Array<Var> loop_vars;
    Array<PrimExpr> loop_extents;
    Array<PrimExpr> iter_values;
    loop_vars.reserve(n_iters);
    loop_extents.reserve(n_iters);
    iter_values.reserve(n_iters);
    PrimExpr predicate = const_true();
    for (int i = 0; i < n_iters; ++i) {
      Range iter_dom = iter_doms[i].dom.CoverRange(block_->iter_vars[i]->dom);
      if (preserve_unit_loops || !is_one(iter_dom->extent)) {
        int bits = std::max(iter_dom->min.dtype().bits(), iter_dom->extent.dtype().bits());
        Var var("ax" + std::to_string(loop_vars.size()), DataType::Int(bits));
        loop_vars.push_back(var);
        loop_extents.push_back(analyzer->Simplify(iter_dom->extent));
        iter_values.push_back(iter_dom->min + var);
        analyzer->Bind(var, Range::FromMinExtent(IntImm(var.dtype(), 0), iter_dom->extent));
      } else {
        iter_values.push_back(iter_dom->min);
      }
      const arith::IntSet& pred_bound = iter_doms[i].bound;
      if (!pred_bound.IsNothing()) {
        // NOTE: Apply strong analyzer proofs to get rid of symbolic bound
        if (pred_bound.HasLowerBound()) {
          PrimExpr lower_bound = iter_values[i] >= pred_bound.min();
          if (!analyzer->CanProve(lower_bound, arith::ProofStrength::kSymbolicBound)) {
            predicate = predicate && lower_bound;
          }
        }
        if (pred_bound.HasUpperBound()) {
          PrimExpr upper_bound = iter_values[i] < pred_bound.max() + 1;
          if (!analyzer->CanProve(upper_bound, arith::ProofStrength::kSymbolicBound)) {
            predicate = predicate && upper_bound;
          }
        }
      }
    }
    this->new_block_realize_ =
        BlockRealize(std::move(iter_values), analyzer->Simplify(predicate), std::move(block_));
    Stmt new_subtree = this->new_block_realize_;
    for (int i = static_cast<int>(loop_vars.size()) - 1; i >= 0; --i) {
      const Var& loop_var = loop_vars[i];
      const PrimExpr& loop_extent = loop_extents[i];
      new_subtree = For(/*loop_var=*/loop_var,
                        /*min=*/Integer(0),
                        /*extent=*/loop_extent,
                        /*ForKind=*/ForKind::kSerial,
                        /*body=*/std::move(new_subtree));
    }
    Array<Stmt> subtrees = AsArray(loop_->body);
    subtrees.insert(subtrees.begin() + insert_position, std::move(new_subtree));
    ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop_.get());
    new_loop->body = SeqStmt(std::move(subtrees));
    this->new_loop_ = For(std::move(new_loop));
  }

 private:
  Stmt VisitStmt_(const BlockNode* block) final {
    if (block != scope_root_.get()) {
      return GetRef<Block>(block);
    }
    if (block == rm_src_stmt_.get()) {
      block = TVM_TYPE_AS(rm_tgt_stmt_, BlockNode);
    }
    return StmtMutator::VisitStmt_(block);
  }

  Stmt VisitStmt_(const ForNode* loop) final {
    if (loop == rm_src_stmt_.get()) {
      loop = TVM_TYPE_AS(rm_tgt_stmt_, ForNode);
    }
    if (loop == loop_.get()) {
      return new_loop_;
    }
    return StmtMutator::VisitStmt_(loop);
  }

 public:
  /*! \brief The root block of the block scope */
  Block scope_root_;
  /*! \brief The given block to be moved */
  Block block_;
  /*! \brief The given loop the block and its loop nest to be put under */
  For loop_;
  /*! \brief The new loop to replace the original loop */
  For new_loop_{nullptr};
  /*! \brief The new block realize to the moved block */
  BlockRealize new_block_realize_{nullptr};
  /*! \brief The plan to remove the given block by replacing this loop/block in the AST */
  Stmt rm_src_stmt_{nullptr};
  /*! \brief The plan to remove the given block by replacing to this loop/block in the AST */
  Stmt rm_tgt_stmt_{nullptr};
};

/*!
 * \brief Calculate a list of accessed buffer regions under a path of loops
 * \tparam relax_storage_scope Whether to relax beyond the path according to the storage and
 * execution scope
 * \param binding The block binding, used to unbind the buffer regions
 * \param buffer_regions The buffer regions to be calculated
 * \param relax_path_low_inclusive The lowest point in the loop path, inclusive
 * \param relax_path_high_exclusive The highest point in the loop path, exclusive
 * \param relaxed Where the calculation result is stored
 */
template <bool relax_storage_scope>
void RelaxBufferRegions(const Map<Var, PrimExpr>& binding,
                        const Array<BufferRegion>& buffer_regions,
                        const StmtSRef& relax_path_low_inclusive,
                        const StmtSRef& relax_path_high_exclusive,
                        std::unordered_map<const BufferNode*, std::vector<NDIntSet>>* relaxed) {
  runtime::StorageScope global_scope{runtime::StorageRank::kGlobal, ""};
  // We cache the variable domains
  runtime::StorageRank previous_rank = runtime::StorageRank::kGlobal;
  Optional<Map<Var, arith::IntSet>> var_dom = NullOpt;
  // Enumerate every buffer region
  for (const BufferRegion& buffer_region : buffer_regions) {
    const Buffer& buffer = buffer_region->buffer;
    const Array<Range>& region = buffer_region->region;
    // Skip the buffer regions we are not interested in
    auto it = relaxed->find(buffer.get());
    if (it == relaxed->end()) {
      continue;
    }
    std::vector<NDIntSet>& relaxed_regions = it->second;
    // Check and update the cached `var_dom`
    runtime::StorageScope scope =
        relax_storage_scope ? runtime::StorageScope::Create(buffer.scope()) : global_scope;
    runtime::StorageRank rank = scope.rank;
    if (rank != previous_rank || !var_dom.defined()) {
      previous_rank = rank;
      var_dom = arith::AsIntSet(LoopDomainOfSRefTreePath(
          /*low_inclusive=*/relax_path_low_inclusive,
          /*high_exclusive=*/relax_path_high_exclusive,
          /*extra_relax_scope=*/scope));
    }
    // Relax the region
    Array<arith::IntSet> relaxed_region =
        arith::EvalSet(Substitute(region, binding), var_dom.value());
    relaxed_regions.push_back({relaxed_region.begin(), relaxed_region.end()});
  }
}

/*!
 * \brief Calculate the iteration domain of a provided integer set to fully cover the required
 * domain
 * \param provided The provided integer set to cover the required domain
 * \param required The required domain to be covered
 * \param dim_max The maximum index bound by the buffer shape
 * \param analyzer The arithmetic analyzer
 */
std::pair<Var, BlockVarDomainInfo> SolveBlockVarDomain(const arith::IntSet& provided,
                                                       const arith::IntSet& required,
                                                       PrimExpr dim_max,
                                                       arith::Analyzer* analyzer) {
  PrimExpr provided_min = analyzer->Simplify(provided.min());
  PrimExpr provided_max = analyzer->Simplify(provided.max());
  PrimExpr required_min = analyzer->Simplify(required.min());
  PrimExpr required_max = analyzer->Simplify(required.max());
  arith::IntSet var_dom, var_bound;
  Optional<Var> var;
  arith::PVar<Var> p_v;
  arith::PVar<PrimExpr> p_e;
  if ((p_v * p_e).Match(provided_min) || (p_e * p_v).Match(provided_min)) {
    PrimExpr e = p_e.Eval();
    var = p_v.Eval();
    var_dom = arith::IntSet::Interval(floordiv(required_min, e), floordiv(required_max, e));
    var_bound = arith::IntSet::Interval(0, floordiv(dim_max, e));
  } else if (analyzer->CanProveEqual(provided_min, provided_max)) {
    if (p_v.Match(provided_min)) {
      var = p_v.Eval();
      var_dom = arith::IntSet::Interval(required_min, required_max);
      var_bound = arith::IntSet::Interval(0, dim_max);
    } else {
      arith::PVar<PrimExpr> p_f1, p_f2;
      if ((floordiv(p_f1, p_f2).Match(provided_min))) {
        PrimExpr var_expr = p_f1.Eval();
        PrimExpr fac = p_f2.Eval();
        if (analyzer->CanProveGreaterEqual(fac, 1)) {
          if (var_expr->IsInstance<VarNode>()) {
            // a <= (x // factor) <= b, fac > 0 ==> (a * fac) <= x <= (b * fac + fac - 1)
            var = Downcast<Var>(var_expr);
            var_dom = arith::IntSet::Interval(required_min * fac,
                                              analyzer->Simplify(required_max * fac + fac - 1));
            var_bound = arith::IntSet::Interval(0, analyzer->Simplify(dim_max * fac + fac - 1));
          } else {
            const arith::IntSet new_provided = arith::IntSet::SinglePoint(p_f1.Eval());
            const arith::IntSet new_required = arith::IntSet::Interval(
                required_min * fac, analyzer->Simplify(required_max * fac + fac - 1));
            return SolveBlockVarDomain(new_provided, new_required, dim_max, analyzer);
          }
        }
      } else if ((floormod(p_f1, p_f2).Match(provided_min))) {
        PrimExpr var_expr = p_f1.Eval();
        if (var_expr->IsInstance<VarNode>()) {
          // generally domain of (x % fac) enforce no constraints to domain of x
          Var var_mod = Downcast<Var>(var_expr);
          return {var_mod, BlockVarDomainInfo()};
        } else {
          PrimExpr mod_1 = p_f1.Eval();
          PrimExpr mod_2 = p_f2.Eval();
          if (analyzer->CanProveGreaterEqual(mod_1, 1) &&
              analyzer->CanProveGreaterEqual(mod_2, 1)) {
            const arith::IntSet new_provided = arith::IntSet::SinglePoint(p_f1.Eval());
            if (analyzer->CanProveGreaterEqual(required_min, 0)) {
              const arith::IntSet new_required =
                  arith::IntSet::Interval(required_min, arith::SymbolicLimits::pos_inf_);
              return SolveBlockVarDomain(new_provided, new_required, dim_max, analyzer);
            }
          }
        }
      }
    }
  }
  ICHECK(var.defined()) << "ValueError: BufferRegion pattern match failed: " << provided_min;
  return {var.value(), BlockVarDomainInfo{var_dom, var_bound}};
}

/*!
 * \brief Calculate and update the iteration domain info to fully cover the required domain in
 * dimension-wise fashion. The region relation on each buffer dimension is independently estimated.
 * \param buffer The accessed buffer
 * \param provided_region The provided NDIntSet to cover the required domain
 * \param required_region The required NDIntSet domain to be covered
 * \param analyzer The arithmetic analyzer
 * \param iter_doms The result iteration domains to be updated
 */
void UpdateBlockVarDomainDimwise(
    const BufferNode* buffer, const NDIntSet& provided_region, const NDIntSet& required_region,
    arith::Analyzer* analyzer, std::unordered_map<const VarNode*, BlockVarDomainInfo>* iter_doms) {
  size_t ndim = buffer->shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    arith::IntSet provided = provided_region[i];
    arith::IntSet required = required_region[i];
    PrimExpr dim_max = max(buffer->shape[i] - 1, 0);

    if (provided.CanProveSinglePoint(analyzer) && is_const_int(provided.min())) {
      ICHECK(required.CanProveSinglePoint(analyzer) &&
             analyzer->CanProveEqual(provided.min(), required.min()));
      continue;
    }

    auto [var, dom_info] = SolveBlockVarDomain(provided, required, dim_max, analyzer);
    auto it = iter_doms->find(var.get());
    if (it != iter_doms->end()) {
      it->second.Union(dom_info);
    } else {
      ICHECK(analyzer->CanProveEqual(provided.min(), required.min()));
      ICHECK(analyzer->CanProveEqual(provided.max(), required.max()));
    }
  }
}

/*! \brief Helper function to implement intset version of `InverseAffineIterMap`. */
Map<Var, arith::IntSet> InverseAffineIterMap(const Array<arith::IterSumExpr>& iter_map,
                                             const NDIntSet& outputs, arith::Analyzer* analyzer) {
  Array<PrimExpr> min_point, max_point;
  min_point.reserve(outputs.size());
  max_point.reserve(outputs.size());
  for (const auto& intset : outputs) {
    ICHECK(intset.HasLowerBound() && intset.HasUpperBound());
    min_point.push_back(intset.min());
    max_point.push_back(intset.max());
  }
  auto rev_min = InverseAffineIterMap(iter_map, min_point);
  auto rev_max = InverseAffineIterMap(iter_map, max_point);
  Map<Var, arith::IntSet> dom_map;
  for (const auto& kv : rev_min) {
    const Var& var = kv.first;
    auto it = rev_max.find(var);
    ICHECK(it != rev_max.end());  // InverseAffineIterMap's result vars are assumed stable
    const PrimExpr& rev_min_point = kv.second;
    const PrimExpr& rev_max_point = (*it).second;
    dom_map.Set(var,
                arith::IntSet::Interval(analyzer->Simplify(min(rev_min_point, rev_max_point)),
                                        analyzer->Simplify(max(rev_min_point, rev_max_point))));
  }
  return dom_map;
}

/*!
 * \brief Calculate and update the iteration domain info to fully cover the required domain
 * with affine analysis. It requires bijective mapping of block var to provided region points.
 * \param buffer The accessed buffer
 * \param iter_vars The list of block vars to cover the required region
 * \param provided_region The provided NDIntSet to cover the required domain
 * \param required_region The required NDIntSet domain to be covered
 * \param analyzer The arithmetic analyzer
 * \param iter_doms The result iteration domains to be updated
 * \returns bool. Denotes whether update success
 */
bool UpdateBlockVarDomainAffine(const BufferNode* buffer, const Array<IterVar>& iter_vars,
                                const NDIntSet& provided_region, const NDIntSet& required_region,
                                arith::Analyzer* analyzer,
                                std::unordered_map<const VarNode*, BlockVarDomainInfo>* iter_doms) {
  // we only support single point provided region now, which could cover most cases
  for (const auto& intset : provided_region) {
    if (!intset.CanProveSinglePoint(analyzer)) return false;
  }
  // calculate forward mapping (block vars -> provided region point)
  Map<Var, Range> dom_map;
  for (const IterVar& iter_var : iter_vars) {
    dom_map.Set(iter_var->var, iter_var->dom);
  }
  size_t ndim = buffer->shape.size();
  Array<PrimExpr> provide_indices;
  provide_indices.reserve(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    provide_indices.push_back(provided_region[i].min());
  }
  auto res = arith::DetectIterMap(provide_indices, dom_map, const_true(),
                                  arith::IterMapLevel::Bijective, analyzer, false);
  if (res->indices.empty()) {
    return false;
  }
  // calculate backward mapping (required region point -> block vars)
  NDIntSet required_bound;
  for (size_t i = 0; i < ndim; ++i) {
    required_bound.push_back(
        arith::IntSet::Interval(make_zero(buffer->shape[i]->dtype), max(buffer->shape[i] - 1, 0)));
  }
  Map<Var, arith::IntSet> var_dom = InverseAffineIterMap(res->indices, required_region, analyzer);
  Map<Var, arith::IntSet> var_bound = InverseAffineIterMap(res->indices, required_bound, analyzer);
  for (const auto& kv : var_dom) {
    const Var& var = kv.first;
    auto it = var_bound.find(var);
    ICHECK(it != var_bound.end());  // InverseAffineIterMap's result vars are assumed stable
    (*iter_doms)[var.get()].Union(BlockVarDomainInfo{kv.second, (*it).second});
  }
  return true;
}

/*!
 * \brief Calculate the domain of block vars to cover the required region
 * \param iter_vars The list of block vars to cover the required region
 * \param provided_regions The region provided by one iteration instance of the block vars
 * \param required_regions The region required to be covered
 * \param analyzer The arithmetic analyzer
 * \return A list of iteration domain info corresponding to the given list of block vars
 */
std::vector<BlockVarDomainInfo> CalculateBlockVarDomain(
    const Array<IterVar>& iter_vars,
    std::unordered_map<const BufferNode*, std::vector<NDIntSet>> provided_regions,
    std::unordered_map<const BufferNode*, std::vector<NDIntSet>> required_regions,
    arith::Analyzer* analyzer) {
  int n_iters = iter_vars.size();
  // Step 1. Construct the mapping from block var to their iteration domain (initialized to empty)
  std::unordered_map<const VarNode*, BlockVarDomainInfo> iter_doms;
  iter_doms.reserve(n_iters);
  for (const IterVar& iter_var : iter_vars) {
    iter_doms[iter_var->var.get()] = BlockVarDomainInfo();
  }
  // Step 2. For each buffer, update the domain according to the provided and required regions
  for (const auto& kv : provided_regions) {
    const BufferNode* buffer = kv.first;
    const std::vector<NDIntSet>& many_provided_regions = kv.second;
    // Calculate `provided_region` and `required_region`
    auto it = required_regions.find(buffer);
    if (it == required_regions.end() || it->second.empty()) {
      continue;
    }
    NDIntSet required_region = support::NDIntSetUnion(it->second);
    NDIntSet provided_region = support::NDIntSetUnion(many_provided_regions);
    ICHECK_EQ(provided_region.size(), buffer->shape.size());
    ICHECK_EQ(required_region.size(), buffer->shape.size());
    // Try update iter var domains with current required and provided region pair.
    if (!UpdateBlockVarDomainAffine(buffer, iter_vars, provided_region, required_region, analyzer,
                                    &iter_doms)) {
      UpdateBlockVarDomainDimwise(buffer, provided_region, required_region, analyzer, &iter_doms);
    }
  }
  // Union the iter var domains, put them in the same order of block vars, and return
  std::vector<BlockVarDomainInfo> result;
  result.reserve(n_iters);
  for (const IterVar& iter_var : iter_vars) {
    BlockVarDomainInfo& info = iter_doms.at(iter_var->var.get());
    if (info.bound.IsNothing()) {
      info.bound = arith::IntSet::FromRange(iter_var->dom);
    } else {
      info.bound = arith::Intersect({info.bound, arith::IntSet::FromRange(iter_var->dom)});
    }
    info.Simplify(analyzer);
    ICHECK(!info.dom.IsNothing());
    result.push_back(info);
  }
  return result;
}

/*!
 * \brief Calculate the provided region of the given block by one single of its execution instance,
 * as well as the required buffer regions relaxed to the given loop
 * \tparam is_compute_at Indicates if the operation is compute-at or reverse-compute-at
 * \param block The given block that provides buffer regions
 * \param loop_sref The given loop under which the block is going to be moved to
 * \param block2realize Maps a block to its corresponding BlockRealize
 * \param producer_srefs The producers of the given block
 * \param consumer_srefs The consumers of the given block
 * \param provided_regions The calculated regions provided by the block
 * \param required_regions The calculated regions required by its consumers (in compute-at) or
 * producers (in reverse-compute-at)
 */
template <bool is_compute_at>
void CalculateProvidedRequiredRegions(
    const BlockNode* block, const StmtSRef& loop_sref,
    std::unordered_map<const BlockNode*, const BlockRealizeNode*> block2realize,
    Array<StmtSRef> producer_srefs, Array<StmtSRef> consumer_srefs,
    std::unordered_map<const BufferNode*, std::vector<NDIntSet>>* provided_regions,
    std::unordered_map<const BufferNode*, std::vector<NDIntSet>>* required_regions) {
  // Step 1. Calculate the region provided by a single execution instance of `block`
  const Array<BufferRegion>& provided_buffers = is_compute_at ? block->writes : block->reads;
  provided_regions->reserve(provided_buffers.size());
  required_regions->reserve(provided_buffers.size());
  for (const BufferRegion& provided_buffer_region : provided_buffers) {
    const BufferNode* buffer = provided_buffer_region->buffer.get();
    const Array<Range>& region = provided_buffer_region->region;
    (*provided_regions)[buffer].push_back(support::NDIntSetFromRegion(region));
    (*required_regions)[buffer].clear();
  }
  // Step 2. Calculate the region required by dependent blocks under `loop`
  for (const StmtSRef& required_block_sref : is_compute_at ? consumer_srefs : producer_srefs) {
    const BlockNode* required_block = TVM_SREF_TO_BLOCK(required_block_sref);
    ICHECK(block2realize.count(required_block));
    RelaxBufferRegions</*relax_storage_scope=*/is_compute_at>(
        /*binding=*/GetBindings(GetRef<BlockRealize>(block2realize.at(required_block))),
        /*buffer_regions=*/is_compute_at ? required_block->reads : required_block->writes,
        /*relax_path_low_inclusive=*/GetRef<StmtSRef>(required_block_sref->parent),
        /*relax_path_high_exclusive=*/loop_sref, /*relaxed=*/required_regions);
  }
}

/******** Main Implementation ********/

template <bool is_compute_at>
void ComputeAtOrReverseComputeAtImpl(ScheduleState self, const StmtSRef& block_sref,
                                     const StmtSRef& loop_sref, bool preserve_unit_loops,
                                     arith::Analyzer* analyzer, bool check_only = false,
                                     int index = -1) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  // Step 1. Bunch of checks
  // Check condition 1) : scope stage pipeline
  StmtSRef scope_root_sref = GetScopeRoot(self, block_sref,
                                          /*require_stage_pipeline=*/true);
  Block scope_root = GetRef<Block>(scope_root_sref->StmtAs<BlockNode>());
  AddShapeVarBounds(self, scope_root_sref.get(), analyzer);
  BlockScope scope = self->GetBlockScope(scope_root_sref);
  Array<StmtSRef> producer_srefs = GetProducers(block_sref, scope);
  Array<StmtSRef> consumer_srefs = GetConsumers(block_sref, scope);
  // Check condition 2) : `block` is a complete or reduction block
  CheckCompleteOrReductionBlock(self, block_sref, scope_root_sref);
  // Check condition 3): `block` and `loop` are under the same scope,
  // and `loop` is not the ancestor of `block`
  NotInSameScopeError::CheckAndBindLoopDomain(self, block_sref, loop_sref, scope_root_sref,
                                              analyzer);
  // Check condition 4): `block` is not an output block
  if (is_compute_at) {
    CheckNotOutputBlock(self, block_sref, scope_root_sref);
  }
  // Step 2. Plan for the removal of `block`
  ScopeReconstructor reconstructor(scope_root, GetRef<Block>(block), GetRef<For>(loop));
  LeafBlockRemovalPlan(self, block_sref, &reconstructor.rm_src_stmt_, &reconstructor.rm_tgt_stmt_);
  // Step 3. Find the insertion point under `loop`
  // Check condition 5): all the required block are under the given loop
  std::unordered_map<const BlockNode*, const BlockRealizeNode*> block2realize;
  block2realize.reserve(self->block_info.size());
  int insert_position = FindInsertionPoint<!is_compute_at, is_compute_at>(
      /*self=*/self,
      /*subtrees=*/AsArray(loop->body),
      /*producer_srefs=*/producer_srefs,
      /*consumer_srefs=*/consumer_srefs, /*block2realize=*/&block2realize,
      /*index=*/index);
  // Step 4. Calculate the region provided by a single execution instance of `block`,
  // as well as the region required by dependent blocks under `loop`.
  // Here is the definition of `provide` and `require`:
  // - In compute-at, `provide` means `produce`, and `require` means `consume`
  // - In reverse-compute-at, `provide` means `consume`, and `require` means `produce`
  std::unordered_map<const BufferNode*, std::vector<NDIntSet>> provided_regions;
  std::unordered_map<const BufferNode*, std::vector<NDIntSet>> required_regions;
  CalculateProvidedRequiredRegions<is_compute_at>(
      /*block=*/block, /*loop_sref=*/loop_sref, /*block2realize=*/std::move(block2realize),
      /*producer_srefs=*/std::move(producer_srefs),
      /*consumer_srefs=*/std::move(consumer_srefs),
      /*provided_regions=*/&provided_regions, /*required_regions=*/&required_regions);
  // Step 5. Calculate the iteration domain for each block var
  std::vector<BlockVarDomainInfo> iter_doms =
      CalculateBlockVarDomain(/*iter_vars=*/block->iter_vars,
                              /*provided_regions=*/std::move(provided_regions),
                              /*required_regions=*/std::move(required_regions),
                              /*analyzer=*/analyzer);
  // Step 6. Create the new scope according to the iteration domain
  reconstructor.MakeNewLoop(/*insert_position=*/insert_position, /*iter_doms=*/std::move(iter_doms),
                            /*analyzer=*/analyzer, /*preserve_unit_loops=*/preserve_unit_loops);
  Block new_scope_root = Downcast<Block>(reconstructor(scope_root));

  // Step 7. Do the actual replacement
  if (check_only) {
    return;
  }
  self->Replace(scope_root_sref, new_scope_root, {{scope_root, new_scope_root}});
  // Step 8. Update the cached flags
  BlockInfo& block_info = self->block_info[block_sref];
  block_info.affine_binding = IsAffineBinding(
      /*realize=*/reconstructor.new_block_realize_,
      /*loop_var_ranges=*/LoopDomainOfSRefTreePath(GetRef<StmtSRef>(block_sref->parent)),
      /*analyzer=*/analyzer);
}

void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
               bool preserve_unit_loops, int index) {
  arith::Analyzer analyzer;
  ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops, &analyzer,
                                        false, index);
}

void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                      bool preserve_unit_loops, int index) {
  arith::Analyzer analyzer;
  ComputeAtOrReverseComputeAtImpl<false>(self, block_sref, loop_sref, preserve_unit_loops,
                                         &analyzer, false, index);
}

bool CanComputeAt(const ScheduleState& self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                  bool preserve_unit_loops) {
  arith::Analyzer analyzer;
  try {
    ComputeAtOrReverseComputeAtImpl<true>(self, block_sref, loop_sref, preserve_unit_loops,
                                          &analyzer, true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

bool CanReverseComputeAt(const ScheduleState& self, const StmtSRef& block_sref,
                         const StmtSRef& loop_sref, bool preserve_unit_loops) {
  arith::Analyzer analyzer;
  try {
    ComputeAtOrReverseComputeAtImpl<false>(self, block_sref, loop_sref, preserve_unit_loops,
                                           &analyzer, true);
  } catch (const tvm::runtime::Error& e) {
    return false;
  }
  return true;
}

/******** InstructionKind Registration ********/

struct ComputeAtTraits : public UnpackedInstTraits<ComputeAtTraits> {
  static constexpr const char* kName = "ComputeAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv,
                                      Bool preserve_unit_loops, IntImm index) {
    return sch->ComputeAt(block_rv, loop_rv, preserve_unit_loops.operator bool(), index->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv,
                                 Bool preserve_unit_loops, IntImm index) {
    PythonAPICall py("compute_at");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.Input("preserve_unit_loops", preserve_unit_loops.operator bool());
    py.Input("index", index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct ReverseComputeAtTraits : public UnpackedInstTraits<ReverseComputeAtTraits> {
  static constexpr const char* kName = "ReverseComputeAt";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 2;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block_rv, LoopRV loop_rv,
                                      Bool preserve_unit_loops, IntImm index) {
    return sch->ReverseComputeAt(block_rv, loop_rv, preserve_unit_loops.operator bool(),
                                 index->value);
  }

  static String UnpackedAsPython(Array<String> outputs, String block_rv, String loop_rv,
                                 Bool preserve_unit_loops, IntImm index) {
    PythonAPICall py("reverse_compute_at");
    py.Input("block", block_rv);
    py.Input("loop", loop_rv);
    py.Input("preserve_unit_loops", preserve_unit_loops.operator bool());
    py.Input("index", index);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ComputeAtTraits);
TVM_REGISTER_INST_KIND_TRAITS(ReverseComputeAtTraits);

}  // namespace tir
}  // namespace tvm
