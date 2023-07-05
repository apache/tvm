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
 * \file src/relay/collage/collage_partitioner.cc
 * \brief Search for an optimal partitioning of a Relay model.
 */

#include "./collage_partitioner.h"

#include <math.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/ir/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/transform.h>
#include <tvm/target/target.h>

#include "../ir/dataflow_matcher_impl.h"
#include "../transforms/compiler_function_utils.h"
#include "../transforms/device_aware_visitors.h"
#include "./candidate_partition.h"
#include "./candidate_partition_index.h"
#include "./cost.h"
#include "./cost_estimator.h"
#include "./gather_partition_specs.h"
#include "./name_supply.h"
#include "./partition_rule.h"
#include "./partition_spec.h"
#include "./priority_queue.h"
#include "./sub_graph.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace collage {
namespace {

TVM_REGISTER_PASS_CONFIG_OPTION("relay.collage.tvm_max_depth", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.collage.byoc_max_depth", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.collage.byoc_fusion_style", Array<String>);
/*!
 * \brief Represents the overall expression after some number of non-overlapping candidate
 * partitions have been applied.
 */
class SearchState {
 public:
  explicit SearchState(IndexSet covered) : covered_(std::move(covered)) {}

  /*!
   * \brief Order states by increasing best cost, breaking ties by lexicographic order on
   * the covering sub graph.
   */
  bool operator<(const SearchState& that) const {
    return std::tie(best_cost_, covered_) < std::tie(that.best_cost_, that.covered_);
  }

  const IndexSet& covered() const { return covered_; }

  std::string ToString() const {
    std::ostringstream os;
    os << "State(";
    os << "covered=" << covered_.ToString();
    os << ",best_cost=" << best_cost_.ToString();
    if (best_candidate_.defined()) {
      os << ",best_candidate=" << best_candidate_->ToString();
    }
    os << ")";
    return os.str();
  }

 private:
  /*! \brief Which nodes of overall expression have been placed on all paths to this state. */
  IndexSet covered_;
  /*! \brief Predecessor state for sequence of candidates reaching this state with least
   * cost. Null if initial search state. */
  SearchState* pred_state_ = nullptr;
  /*!
   * \brief Cost of reaching this state using placement implied by path given by pred_state fields.
   * Includes estimated/measured cost of all candidates plus any candidate launch penalty.
   * Initially invalid cost.
   */
  Cost best_cost_ = Cost::Invalid();
  /*! \brief Candidate partition selected in transition from pred_state to this state. */
  CandidatePartition best_candidate_;

  friend class Partitioner;
};

struct CompareSearchStatePtrs {
  bool operator()(const SearchState* left, const SearchState* right) const {
    return *left < *right;
  }
};

struct EqualSearchStatePtrs {
  bool operator()(const SearchState* left, const SearchState* right) const {
    return left->covered() == right->covered();
  }
};

/*!
 * \brief Finds the optimal partitioning of an expression to candidate partitions.
 * Though no candidate partitions overlap, it is possible some sub-expressions end up in
 * no candidate. Those sub-expressions must be evaluated by the host executor (eg VM).
 */
class Partitioner {
 public:
  explicit Partitioner(Array<PartitionSpec> partition_specs,
                       const std::unordered_map<const ExprNode*, VirtualDevice>* virtual_devices,
                       CostEstimator cost_estimator, std::shared_ptr<CandidateFunctionCache> cache,
                       Expr expr)
      : partition_specs_(std::move(partition_specs)),
        virtual_devices_(virtual_devices),
        cost_estimator_(std::move(cost_estimator)),
        cache_(std::move(cache)),
        expr_(std::move(expr)) {}

  Expr Partition() {
    // Establish core data structures.
    dataflow_graph_ = std::make_unique<DataflowGraph>(expr_);
    VLOG(1) << "Created dataflow graph with " << dataflow_graph_->size() << " nodes";

    // Build the candidate index. This is where all the partition rules are invoked .
    index_ = std::make_unique<CandidatePartitionIndex>(virtual_devices_, dataflow_graph_.get());
    index_->Index(partition_specs_);
    VLOG(1) << "All candidates before search:" << std::endl << index_->ToSummary();

    // 'Eagerly' estimate the cost of all candidates.
    //
    // Note if this is not done costs will simply be estimated 'lazily' as the search proceeds.
    // Typically, some candidates are never explored during the search because:
    //  - There are no paths in which the candidate does not intersect candidates already
    //    applied on the path.
    //  - The Dijkstra search terminates early with a least cost path.
    // So eager may result in more estimation overhead. However, eager could be made
    // embarrassingly parallel.
    VLOG(1) << "Beginning eager cost estimation";
    index_->EstimateAllCosts(cost_estimator_, cache_);
    VLOG(1) << "Finished eager cost estimation";

    // Setup initial state.
    SearchState* init_state = GetState(IndexSet(dataflow_graph_->size()));
    init_state->best_cost_ = Cost::Zero();
    pq_.Push(init_state);

    size_t num_transitions = 0;

    VLOG(1) << "#### Commencing Collage search over " << index_->size() << " candidates ####";
    while (!pq_.empty()) {
      SearchState* curr_state = pq_.Pop();
      VLOG(1) << "Looking at state " << curr_state->covered_.ToString();
      PostDfsIndex next_index = curr_state->covered_.FirstOutsideIndex();

      if (next_index >= dataflow_graph_->size()) {
        // The entire expression has been explored. Collect the candidates on the optimal path.
        VLOG(1) << "#### Finished Collage search after exploring " << num_transitions
                << " transitions ####";
        std::vector<CandidatePartition> best_candidates;
        while (curr_state != init_state) {
          ICHECK(curr_state->best_candidate_.defined());
          best_candidates.emplace_back(curr_state->best_candidate_);
          curr_state = curr_state->pred_state_;
          ICHECK(curr_state != nullptr);
        }
        return Finalize(best_candidates);
      }

      size_t num_fires = 0;
      Expr sub_expr = dataflow_graph_->index_to_node(next_index)->ref();
      VLOG(1) << "Looking at index " << next_index << " for sub-expression "
              << SubExprKindAndLabel(sub_expr).second << " out of " << dataflow_graph_->size()
              << " total dataflow nodes";

      // Explore all the outgoing candidates from the current state.
      for (const auto& candidate : index_->candidates_at(next_index)) {
        VLOG(1) << "Considering candidate " << candidate->ToSummary(*dataflow_graph_)
                << " for transition " << ++num_transitions << " over " << index_->size()
                << " total candidates";
        if (!candidate->sub_graph_->inside_.AreDisjoint(curr_state->covered_)) {
          LOG(INFO) << "Candidate overlaps with already partitioned nodes";
          continue;
        }
        IndexSet next_covered = curr_state->covered_ | candidate->sub_graph_->inside_;
        SearchState* next_state = GetState(next_covered);
        Relax(curr_state, next_state, candidate);
        ++num_fires;
      }
      ICHECK_GT(num_fires, 0)
          << "No candidate was found covering sub-expression at index " << next_index
          << ", suggesting the partition rules are incomplete for the given targets.";
    }

    ICHECK(false) << "should have reached end state in which all sub-expressions are covered";
    return {};
  }

  /*! \brief Returns the unique state corresponding to the \p covered sub-graph. */
  SearchState* GetState(const IndexSet& covered) {
    auto itr = covered_to_state_.find(covered);
    if (itr != covered_to_state_.end()) {
      return itr->second.get();
    }
    auto state = std::make_unique<SearchState>(covered);
    SearchState* raw_ptr = state.get();
    covered_to_state_.emplace(covered, std::move(state));
    return raw_ptr;
  }

  /*!
   * \brief Record that it is possible to reach \p next_state by choosing \p candidate
   * in \p curr_state. If the resulting cost is better than the best known so far, update
   * \p next_state's best cost, predecessor and candidate to match.
   */
  void Relax(SearchState* curr_state, SearchState* next_state,
             const CandidatePartition& candidate) {
    // Note this may already be cached if the candidate partition costs were 'eagerly' estimated.
    Cost candidate_cost = candidate->EstimatedCost(*dataflow_graph_, cost_estimator_, cache_);
    VLOG(1) << "Candidate has cost " << candidate_cost.ToString();
    Cost new_state_cost = candidate_cost + curr_state->best_cost_;
    const bool is_new = next_state->best_cost_.is_invalid();
    CandidatePartition previously_best_candidate = next_state->best_candidate_;
    if (is_new || new_state_cost < next_state->best_cost_) {
      next_state->pred_state_ = curr_state;
      Cost previously_best_cost = next_state->best_cost_;
      next_state->best_cost_ = new_state_cost;
      next_state->best_candidate_ = candidate;
      if (is_new) {
        VLOG(1) << "transition " << curr_state->ToString() << " --> " << next_state->ToString()
                << " (New state for spec " << candidate->partition_spec_name() << ")";
        pq_.Push(next_state);
      } else {
        VLOG(1) << "transition " << curr_state->ToString() << " --> " << next_state->ToString()
                << " (Spec " << candidate->partition_spec_name() << " beats previous spec "
                << previously_best_candidate->partition_spec_name() << " by "
                << (previously_best_cost - curr_state->best_cost_).ToString() << ")";
        pq_.Update(next_state);
      }
    } else {
      VLOG(1) << "transition " << curr_state->ToString() << " --> " << next_state->ToString()
              << " (Spec " << candidate->partition_spec_name() << " does not beat existing spec "
              << previously_best_candidate->partition_spec_name() << ")";
    }
  }

  /*!
   * \brief Returns the result of partitioning \p expr according to 'optimal' candidates found
   * by the search.
   */
  Expr Finalize(std::vector<CandidatePartition> best_candidates) {
    best_candidates = CandidatePartition::MaxCoalesce(*dataflow_graph_, best_candidates);

    Cost total_cost = Cost::Zero();
    std::ostringstream os;
    os << "Optimal partitioning:" << std::endl;
    for (const auto& best_candidate : best_candidates) {
      if (best_candidate->partition_spec_name() == kHostSpecName) {
        continue;
      }
      os << best_candidate->ToSummary(*dataflow_graph_);
      os << std::endl;
      total_cost = total_cost + best_candidate->cost_;
    }
    os << "Estimated overall cost is " << total_cost.ToString();
    LOG(INFO) << os.str();

    LOG(INFO) << "All candidates after search:" << std::endl << index_->ToSummary();

    return CandidatePartition::ParallelRewrite(*dataflow_graph_, best_candidates);
  }

 private:
  /*! \brief Available partition specs to use during search. */
  Array<PartitionSpec> partition_specs_;
  /*!
   * \brief The virtual devices for every sub-expression so we can respect any existing target
   * constraints.
   */
  const std::unordered_map<const ExprNode*, VirtualDevice>* virtual_devices_;
  /*! \brief Cost estimator to use for candidates. */
  CostEstimator cost_estimator_;
  /*! \brief Cached names and costs for all partition functions. */
  std::shared_ptr<CandidateFunctionCache> cache_;
  /*! \brief The expression we will be partitioning. */
  Expr expr_;
  /*! \brief Dataflow graph for overall expression. */
  std::unique_ptr<DataflowGraph> dataflow_graph_;
  /*! \brief Index of all avoilable candidates we are searching over. */
  std::unique_ptr<CandidatePartitionIndex> index_;
  /*! \brief Map from covered sub-graphs to the corresponding state. */
  std::unordered_map<IndexSet, std::unique_ptr<SearchState>, IndexSetHash, IndexSetEqual>
      covered_to_state_;
  /*! \brief Priority queue of states, ordered by increasing cost. */
  PriorityQueue<SearchState, CompareSearchStatePtrs, EqualSearchStatePtrs> pq_;
};

}  // namespace

transform::Pass CollagePartition(CompilationConfig config, CostEstimator cost_estimator) {
  runtime::TypedPackedFunc<IRModule(IRModule, transform::PassContext)> pass_func =
      [config = std::move(config), cost_estimator = std::move(cost_estimator)](
          IRModule mod, transform::PassContext ctxt) {
        VLOG(1) << "CollagePartition input:" << std::endl << PrettyPrint(mod);

        Array<PartitionSpec> partition_specs = GatherPartitionSpecs(config);
        VLOG(1) << "Gathered " << partition_specs.size() << " partition specs";

        auto cache =
            std::make_shared<CandidateFunctionCache>(std::make_shared<NameSupply>("collage"));

        IRModule out_mod = mod->ShallowCopy();
        for (const auto& kv : mod->functions) {
          if (const auto* function_node = AsOptimizableFunctionNode(kv.second)) {
            auto function = GetRef<Function>(function_node);
            std::unordered_map<const ExprNode*, VirtualDevice> virtual_devices =
                transform::RecoverVirtualDeviceMap(mod, function);
            Partitioner partitioner(partition_specs, &virtual_devices, cost_estimator, cache,
                                    function);
            Function result = Downcast<Function>(partitioner.Partition());
            out_mod->Add(kv.first, result);
          }
        }

        out_mod = OutlineCompilerFunctions(cache)(std::move(out_mod));
        VLOG(1) << "CollagePartition result:" << std::endl << PrettyPrint(out_mod);
        return out_mod;
      };
  return tvm::transform::CreateModulePass(pass_func, /*opt_level=*/0, "CollagePartition", {});
}

TVM_REGISTER_GLOBAL("relay._transform.CollagePartition").set_body_typed(CollagePartition);

}  // namespace collage
}  // namespace relay
}  // namespace tvm
