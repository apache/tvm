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

#define TVM_META_SCHEDULE_CHECK_PROB_RANGE(p, name)                               \
  CHECK(0.0 <= (p) && (p) <= 1.0) << "ValueError: name should be within [0, 1], " \
                                  << "but get `" << #p << " = " << (p) << '\'';

namespace tvm {
namespace meta_schedule {

using tir::Schedule;

/**************** Data Structure ****************/

/*!
 * \brief A heap with a size up-limit. If overflow happens, it evicted the worst items.
 * \note It maintains a min heap in terms of `Item::score`. Therefore, when
 * overflow happens, the element evicted is the one with the min `Item::score`.
 * As time goes, the elements in the heap are going to be larger.
 */
class SizedHeap {
 public:
  struct Item {
    Schedule sch;
    IRModule mod;
    size_t shash;
    double score;
    bool operator<(const Item& other) const { return score > other.score; }
  };

  struct ItemHash {
    size_t operator()(const Item& hash) const { return hash.shash; }
  };

  struct ItemEqual {
    bool operator()(const Item& lhs, const Item& rhs) const {
      return lhs.shash == rhs.shash && StructuralEqual()(lhs.mod, rhs.mod);
    }
  };
  /*!
   * \brief Constructor
   * \param size_limit The up-limit of the heap size
   */
  explicit SizedHeap(int size_limit) : size_limit(size_limit) { heap.reserve(size_limit); }

  /*!
   * \brief Push the specific item to the heap if its key did not appears in the heap
   * \param item The item to be pushed
   */
  void Push(Schedule sch, IRModule mod, double score) {
    Item item{sch, mod, StructuralHash()(mod), score};
    if (!in_heap.insert(item).second) {
      return;
    }
    int size = heap.size();
    if (size < size_limit) {
      // Heap is not full, just push
      heap.emplace_back(item);
      std::push_heap(heap.begin(), heap.end());
    } else if (item.score > heap.front().score) {
      // if the item is better than the worst one in the heap, we can safely kick it out
      std::pop_heap(heap.begin(), heap.end());
      heap.back() = item;
      std::push_heap(heap.begin(), heap.end());
    }
    // Otherwise, the item is worse than any other element in the heap
  }

  /*! \brief Up-limit of the heap size */
  int size_limit;
  /*! \brief The heap, the worse the topper */
  std::vector<Item> heap;
  /*! \brief The traces that are in the heap */
  std::unordered_set<Item, ItemHash, ItemEqual> in_heap;
};

struct PerThreadData {
  IRModule mod{nullptr};
  TRandState rand_state{-1};
  std::function<int32_t()> trace_sampler = nullptr;
  std::function<Optional<Mutator>()> mutator_sampler = nullptr;

  /*!
   * \brief Set the value for the trace and mutator samplers per thread.
   * \param scores The predicted score for the given samples.
   * \param genetic_mutate_prob The probability of mutation.
   * \param mutator_probs The probability of each mutator as a dict.
   */
  void Set(const std::vector<double>& scores, double genetic_mutate_prob,
           const Map<Mutator, FloatImm>& mutator_probs) {
    trace_sampler = tir::MakeMultinomialSampler(&rand_state, scores);
    mutator_sampler = MakeMutatorSampler(genetic_mutate_prob, mutator_probs, &rand_state);
  }

 private:
  /*!
   * \brief Create a sampler function that picks mutators according to the mass function
   * \param rand_state The random state for sampling
   * \return The sampler created
   */
  static std::function<Optional<Mutator>()> MakeMutatorSampler(
      double genetic_mutate_prob,                   //
      const Map<Mutator, FloatImm>& mutator_probs,  //
      TRandState* rand_state) {
    std::vector<Optional<Mutator>> mutators;
    std::vector<double> masses;
    mutators.push_back(NullOpt);
    masses.push_back(1.0 - genetic_mutate_prob);
    double total_mass_mutator = 0.0;
    if (genetic_mutate_prob > 0) {
      for (const auto& kv : mutator_probs) {
        Mutator mutator = kv.first;
        double mass = kv.second->value;
        total_mass_mutator += mass;
        mutators.push_back(mutator);
        masses.push_back(mass * genetic_mutate_prob);
      }
    }
    // Normalize the sum to 1.0
    if (total_mass_mutator == 0.0) {
      masses[0] = 1.0;
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] = 0.0;
      }
    } else if (total_mass_mutator != 1.0) {
      for (int i = 1, n = masses.size(); i < n; ++i) {
        masses[i] /= total_mass_mutator;
      }
    }
    return [idx_sampler = tir::MakeMultinomialSampler(rand_state, masses),
            mutators = std::move(mutators)]() -> Optional<Mutator> {
      int i = idx_sampler();
      return mutators[i];
    };
  }
};

struct ConcurrentBitmask {
  /*! The bit width. */
  static constexpr const int kBitWidth = 64;
  /*! \brief The size of the concurrent bitmask. */
  int size;
  /*! \brief The bitmasks. */
  std::vector<uint64_t> bitmask;
  /*! \brief The mutexes, one per kBitWidth(64 here) bitmasks. */
  std::vector<std::mutex> mutexes;

  /*!
   * \brief Constructor
   * \param n The total slots managed by the concurrent bitmask.
   */
  explicit ConcurrentBitmask(int n)
      : size((n + kBitWidth - 1) / kBitWidth), bitmask(size, 0), mutexes(size) {}
  /*!
   * \brief Query and mark the given index if not visited before.
   * \param x The index to concurrently check if used. If not, mark as used.
   * \return Whether the index has been used before.
   */
  bool QueryAndMark(int x) {
    constexpr uint64_t one = 1;
    std::unique_lock<std::mutex> lock(mutexes[x / kBitWidth]);
    if (bitmask[x / kBitWidth] & (one << (x % kBitWidth))) {
      return false;
    } else {
      bitmask[x / kBitWidth] |= one << (x % kBitWidth);
      return true;
    }
  }
};

/**************** Util Functions ****************/

/*!
 * \brief Assemble measure candidates from the given candidate traces.
 * \param traces The picked candidate traces.
 * \return The assembled measure candidates.
 */
Array<MeasureCandidate> AssembleCandidates(const std::vector<Schedule>& picks,
                                           const Array<ArgInfo>& args_info) {
  Array<MeasureCandidate> measure_inputs;
  measure_inputs.reserve(picks.size());
  for (const Schedule& sch : picks) {
    measure_inputs.push_back(MeasureCandidate(sch, args_info));
  }
  return measure_inputs;
}

/*!
 * \brief Predict the normalized score of each candidate.
 * \param candidates The candidates for prediction
 * \param task The search task
 * \param space The search space
 * \return The normalized score in the prediction
 */
std::vector<double> PredictNormalizedScore(const std::vector<Schedule>& candidates,
                                           const TuneContext& context, const CostModel& cost_model,
                                           const Array<ArgInfo>& args_info) {
  ICHECK(!candidates.empty()) << "Candidates given for score prediction can not be empty list!";
  std::vector<double> scores =
      cost_model->Predict(context, AssembleCandidates(candidates, args_info));
  for (double& score : scores) {
    score = std::max(0.0, score);
  }
  return scores;
}

/**************** Evolutionary Search ****************/

/*!\brief A search strategy that generates measure candidates using evolutionary search. */
class EvolutionarySearchNode : public SearchStrategyNode {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    EvolutionarySearchNode* self;
    /*! \brief The design spaces. Decisions are not used so traces only. */
    Array<tir::Trace> design_spaces;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(EvolutionarySearchNode* self, Array<tir::Trace> design_spaces)
        : self(self), design_spaces(design_spaces), st(0), ed(self->num_trials_per_iter) {}

    /*!
     * \brief Pick up best candidates from database.
     * \param num The number of traces to produce.
     * \return The picked best candidates.
     */
    inline std::vector<Schedule> PickBestFromDatabase(int num);
    /*!
     * \brief Sample the initial population from previous measured results and randomly generated
     *  traces via trace replaying.
     * \param num The number of traces to produce.
     * \return The initial population of traces sampled.
     */
    inline std::vector<Schedule> SampleInitPopulation(int num);
    /*!
     * \brief Evolve the initial population using mutators and samplers.
     * \param population The initial population of traces sampled.
     * \param num The number of traces to produce.
     * \return The evolved traces from initial population.
     */
    inline std::vector<Schedule> EvolveWithCostModel(std::vector<Schedule> population, int num);
    /*!
     * \brief Pick final candidates from the given initial population and bests of evolved ones.
     * \param inits The initial population of traces sampled.
     * \param bests The best candidates predicted from evolved traces.
     * \param num The number of traces to produce.
     * \return The final picked candidates with a ratio of both.
     */
    inline std::vector<Schedule> PickWithEpsGreedy(const std::vector<Schedule>& inits,
                                                   const std::vector<Schedule>& bests, int num);
    /*! \brief An interface method to be called by it's counterpart in EvolutionarySearchNode */
    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    /*! \brief An interface method to be called by it's counterpart in EvolutionarySearchNode */
    inline void NotifyRunnerResults(const TuneContext& context,
                                    const Array<MeasureCandidate>& measure_candidates,
                                    const Array<RunnerResult>& results);
  };

  /*! \brief The tuning context of the evolutionary search strategy. */
  const TuneContextNode* context_{nullptr};
  /*! \brief The target for the workload. */
  Target target_{nullptr};
  /*! \brief The metadata of the function arguments. */
  Array<ArgInfo> args_info_{nullptr};
  /*! \brief A Database for selecting useful candidates. */
  Database database_{nullptr};
  /*! \brief A cost model helping to explore the search space */
  CostModel cost_model_{nullptr};
  /*! \brief The postprocessors. */
  Array<Postproc> postprocs_{nullptr};
  /*! \brief Mutators and their probability mass */
  Map<Mutator, FloatImm> mutator_probs_{nullptr};
  /*! \brief The number of threads to use. To be initialized with TuneContext. */
  int num_threads_;
  /*! \brief The random state. To be initialized with TuneContext. */
  TRandState rand_state_;
  /*! \brief Pre thread data including module to be tuned and random state. */
  std::vector<PerThreadData> per_thread_data_;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;
  /*! \brief The token registered for the given workload in database. */
  Workload token_{nullptr};

  /*** Configuration: global ***/
  /*! \brief The number of trials per iteration. */
  int num_trials_per_iter;
  /*! \brief The number of total trials. */
  int num_trials_total;
  /*! \brief The population size in the evolutionary search. */
  int population_size;
  /*** Configuration: the initial population ***/
  /*! \brief The ratio of measured states used in the initial population */
  double init_measured_ratio;
  /*! \brief The minimal size of unmeasured population in the initial sampling.*/
  int init_min_unmeasured;
  /*** Configuration: evolution ***/
  /*! \brief The number of iterations performed by generic algorithm. */
  int genetic_num_iters;
  /*! \brief The probability to perform mutation */
  double genetic_mutate_prob;
  /*! \brief The maximum number to try evolving the given trace. */
  int genetic_max_fail_count;
  /*** Configuration: pick states for measurement ***/
  /*! \brief The ratio of measurements to use randomly sampled states. */
  double eps_greedy;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `context_` is not visited
    // `target_` is not visited
    // `args_info_` is not visited
    // `database` is not visited
    // `cost_model` is not visited
    // `postprocs` is not visited
    // `mutator_probs_` is not visited
    // `num_threads` is not visited
    // `rand_state_` is not visited
    // `per_thread_data_` is not visited
    // `state_` is not visited

    /*** Configuration: global ***/
    v->Visit("num_trials_total", &num_trials_total);
    v->Visit("num_trials_per_iter", &num_trials_per_iter);
    v->Visit("population_size", &population_size);
    /*** Configuration: the initial population ***/
    v->Visit("init_measured_ratio", &init_measured_ratio);
    v->Visit("init_min_unmeasured", &init_min_unmeasured);
    /*** Configuration: evolution ***/
    v->Visit("genetic_num_iters", &genetic_num_iters);
    v->Visit("genetic_mutate_prob", &genetic_mutate_prob);
    v->Visit("genetic_max_fail_count", &genetic_max_fail_count);
    /*** Configuration: pick states for measurement ***/
    v->Visit("eps_greedy", &eps_greedy);
  }

  static constexpr const char* _type_key = "meta_schedule.EvolutionarySearch";
  TVM_DECLARE_FINAL_OBJECT_INFO(EvolutionarySearchNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& context) final {
    CHECK(context.defined()) << "TuneContext must be defined!";
    CHECK(context->num_threads > 0) << "Number of threads has to be larger than 0.";
    CHECK(context->target.defined()) << "Target must be defined!";
    this->context_ = context.get();
    this->target_ = context->target.value();
    this->args_info_ = ArgInfo::FromPrimFunc(FindEntryFunc(context->mod.value()));
    this->mutator_probs_ = context->mutator_probs;
    this->postprocs_ = context->postprocs;
    this->num_threads_ = context->num_threads;
    this->rand_state_ = ForkSeed(&context->rand_state);
    this->cost_model_ = context->task_scheduler->cost_model.value();
    this->database_ = context->task_scheduler->database;
    this->token_ = this->database_->CommitWorkload(context->mod.value());
    this->per_thread_data_.resize(this->num_threads_);
    for (const auto& kv : this->mutator_probs_) {
      double mass = kv.second->value;
      TVM_META_SCHEDULE_CHECK_PROB_RANGE(mass, "mutator_probs");
    }
    for (PerThreadData& data : this->per_thread_data_) {
      data.mod = DeepCopyIRModule(context->mod.value());
      data.rand_state = ForkSeed(&this->rand_state_);
    }
    this->state_.reset();
  }

  void PreTuning(const Array<Schedule>& design_spaces) final {
    ICHECK(!design_spaces.empty());
    ICHECK(this->state_ == nullptr);
    // Change to traces
    Array<tir::Trace> design_space_traces;
    design_space_traces.reserve(design_spaces.size());
    for (const Schedule& space : design_spaces) {
      design_space_traces.push_back(space->trace().value()->Simplified(true));
    }
    this->state_ = std::make_unique<State>(this, design_space_traces);
  }

  void PostTuning() final {
    ICHECK(this->state_ != nullptr);
    this->state_.reset();
  }

  Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    ICHECK(this->state_ != nullptr);
    return this->state_->GenerateMeasureCandidates();
  }

  void NotifyRunnerResults(const TuneContext& context,
                           const Array<MeasureCandidate>& measure_candidates,
                           const Array<RunnerResult>& results) final {
    ICHECK(this->state_ != nullptr);
    this->state_->NotifyRunnerResults(context, measure_candidates, results);
  }
};

std::vector<Schedule> EvolutionarySearchNode::State::PickBestFromDatabase(int num) {
  std::vector<tir::Trace> measured_traces;
  measured_traces.reserve(num);
  Array<TuningRecord> top_records = self->database_->GetTopK(self->token_, num);
  for (TuningRecord record : top_records) {
    measured_traces.push_back(record->trace);
  }
  int actual_num = measured_traces.size();
  ThreadedTraceApply pp(self->postprocs_);
  std::vector<Schedule> results(actual_num, Schedule{nullptr});
  auto f_proc_measured = [this, &measured_traces, &results, &pp](int thread_id,
                                                                 int trace_id) -> void {
    PerThreadData& data = self->per_thread_data_.at(thread_id);
    TRandState* rand_state = &data.rand_state;
    const IRModule& mod = data.mod;
    tir::Trace trace = measured_traces.at(trace_id);
    Schedule& result = results.at(trace_id);
    ICHECK(!result.defined());
    if (Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
      result = sch.value();
    } else {
      LOG(FATAL) << "ValueError: Cannot postprocess the trace:\n" << trace;
      throw;
    }
  };
  support::parallel_for_dynamic(0, actual_num, self->num_threads_, f_proc_measured);
  return results;
}

std::vector<Schedule> EvolutionarySearchNode::State::SampleInitPopulation(int num) {
  ThreadedTraceApply pp(self->postprocs_);
  std::vector<Schedule> out_schs;
  while (static_cast<int>(out_schs.size()) < self->init_min_unmeasured) {
    std::vector<Schedule> results(num, Schedule{nullptr});
    auto f_proc_unmeasured = [this, &results, &pp](int thread_id, int trace_id) -> void {
      PerThreadData& data = self->per_thread_data_.at(thread_id);
      TRandState* rand_state = &data.rand_state;
      const IRModule& mod = data.mod;
      Schedule& result = results.at(trace_id);
      ICHECK(!result.defined());
      int design_space_index = tir::SampleInt(rand_state, 0, design_spaces.size());
      tir::Trace trace(design_spaces[design_space_index]->insts, {});
      if (Optional<Schedule> sch = pp.Apply(mod, trace, rand_state)) {
        result = sch.value();
      }
    };
    support::parallel_for_dynamic(0, num, self->num_threads_, f_proc_unmeasured);
    for (int i = 0; i < num; i++) {
      if (results[i].defined()) {
        out_schs.push_back(results[i]);
      }
    }
    LOG(INFO) << "Sample-Init-Population summary:\n" << pp.SummarizeFailures();
  }
  return out_schs;
}

std::vector<Schedule> EvolutionarySearchNode::State::EvolveWithCostModel(
    std::vector<Schedule> population, int num) {
  ICHECK_GT(num, 0);
  // The heap to record best schedule, we do not consider schedules that are already measured
  // Also we use `in_heap` to make sure items in the heap are de-duplicated
  SizedHeap heap(num);
  for (int iter = 0;; ++iter) {
    // Predict normalized score with the cost model,
    std::vector<double> scores = PredictNormalizedScore(population,                           //
                                                        GetRef<TuneContext>(self->context_),  //
                                                        self->cost_model_,                    //
                                                        self->args_info_);
    ICHECK_EQ(scores.size(), population.size());
    for (int i = 0, n = population.size(); i < n; ++i) {
      Schedule sch = population.at(i);
      IRModule mod = sch->mod();
      double score = scores.at(i);
      if (!self->database_->HasWorkload(mod)) {
        heap.Push(sch, mod, score);
      }
    }
    // Discontinue once it reaches end of search
    if (iter == self->genetic_num_iters) {
      break;
    }
    // Set threaded samplers, with probability from predicated normalized throughputs
    for (PerThreadData& data : self->per_thread_data_) {
      data.Set(scores, self->genetic_mutate_prob, self->mutator_probs_);
    }
    ThreadedTraceApply pp(self->postprocs_);
    ConcurrentBitmask cbmask(self->population_size);
    std::vector<Schedule> next_population(self->population_size, Schedule{nullptr});
    // The worker function
    auto f_find_candidate = [&cbmask, &population, &next_population, &pp, this](int thread_id,
                                                                                int trace_id) {
      // Prepare samplers
      PerThreadData& data = self->per_thread_data_.at(thread_id);
      TRandState* rand_state = &data.rand_state;
      const IRModule& mod = data.mod;
      std::function<int()>& trace_sampler = data.trace_sampler;
      std::function<Optional<Mutator>()>& mutator_sampler = data.mutator_sampler;
      Schedule& result = next_population.at(trace_id);
      int sampled_trace_id = -1;
      // Loop until success
      for (int fail_count = 0; fail_count <= self->genetic_max_fail_count; ++fail_count) {
        sampled_trace_id = trace_sampler();
        tir::Trace trace = population.at(sampled_trace_id)->trace().value();
        if (Optional<Mutator> opt_mutator = mutator_sampler()) {
          // Decision: mutate
          Mutator mutator = opt_mutator.value();
          if (Optional<tir::Trace> new_trace = mutator->Apply(trace, rand_state)) {
            if (Optional<Schedule> sch = pp.Apply(mod, new_trace.value(), rand_state)) {
              // note that sch's trace is different from new_trace
              // because it contains post-processing information
              result = sch.value();
              break;
            }
          }
        } else if (cbmask.QueryAndMark(sampled_trace_id)) {
          // Decision: do not mutate
          break;
        }
      }
      // if retry count exceeds the limit, reuse an old sample
      if (!result.defined()) {
        result = population.at(sampled_trace_id);
      }
    };
    support::parallel_for_dynamic(0, self->population_size, self->num_threads_, f_find_candidate);
    population.swap(next_population);
    LOG(INFO) << "Evolve iter #" << iter << " done. Summary:\n" << pp.SummarizeFailures();
  }
  // Return the best states from the heap, sorting from higher score to lower ones
  std::sort(heap.heap.begin(), heap.heap.end());
  std::vector<Schedule> results;
  results.reserve(num);
  for (const SizedHeap::Item& item : heap.heap) {
    results.push_back(item.sch);
  }

  constexpr int kNumScoresPerLine = 16;
  std::ostringstream os;
  int n = heap.heap.size();
  for (int st = 0; st < n; st += kNumScoresPerLine) {
    os << std::endl;
    int ed = std::min(st + kNumScoresPerLine, n);
    os << "[" << (st + 1) << " : " << ed << "]:\t";
    for (int i = st; i < ed; ++i) {
      if (i != st) {
        os << "  ";
      }
      os << std::fixed << std::setprecision(4) << heap.heap.at(i).score;
    }
  }
  LOG(INFO) << "Scores of the best " << n << " candidates:" << os.str();
  return results;
}

std::vector<Schedule> EvolutionarySearchNode::State::PickWithEpsGreedy(
    const std::vector<Schedule>& unmeasured, const std::vector<Schedule>& bests, int num) {
  int num_rands = num * self->eps_greedy;
  int num_bests = num - num_rands;
  std::vector<int> rands =
      tir::SampleWithoutReplacement(&self->rand_state_, unmeasured.size(), unmeasured.size());
  std::vector<Schedule> results;
  results.reserve(num);
  for (int i = 0, i_bests = 0, i_rands = 0; i < num; ++i) {
    bool has_best = i_bests < static_cast<int>(bests.size());
    bool has_rand = i_rands < static_cast<int>(rands.size());
    // Pick a schedule
    Schedule sch{nullptr};
    // If needs `bests`, then prefer `bests`
    if (i < num_bests) {
      if (has_best) {
        sch = bests[i_bests++];
      } else if (has_rand) {
        sch = unmeasured[rands[i_rands++]];
      } else {
        break;
      }
    } else {
      // Else prefer `rands`
      if (has_rand) {
        sch = unmeasured[rands[i_rands++]];
      } else if (has_best) {
        sch = bests[i_bests++];
      } else {
        break;
      }
    }
    results.push_back(sch);
  }
  return results;
}

Optional<Array<MeasureCandidate>> EvolutionarySearchNode::State::GenerateMeasureCandidates() {
  if (st >= self->num_trials_total) {
    return NullOpt;
  }
  int sample_num = self->num_trials_per_iter;
  if (ed > self->num_trials_total) {
    sample_num = self->num_trials_total - st;
    ed = self->num_trials_total;
  }
  ICHECK_LT(st, ed);
  int pop = self->population_size;
  std::vector<Schedule> inits;
  inits.reserve(pop);

  LOG(INFO) << "Generating candidates......";
  std::vector<Schedule> measured = PickBestFromDatabase(pop * self->init_measured_ratio);
  LOG(INFO) << "Picked top " << measured.size() << " candidate(s) from database";
  std::vector<Schedule> unmeasured = SampleInitPopulation(pop - measured.size());
  LOG(INFO) << "Sampled " << unmeasured.size() << " candidate(s)";
  inits.insert(inits.end(), measured.begin(), measured.end());
  inits.insert(inits.end(), unmeasured.begin(), unmeasured.end());
  std::vector<Schedule> bests = EvolveWithCostModel(inits, sample_num);
  LOG(INFO) << "Got " << bests.size() << " candidate(s) with evolutionary search";
  std::vector<Schedule> picks = PickWithEpsGreedy(unmeasured, bests, sample_num);
  LOG(INFO) << "Sending " << picks.size() << " candidates(s) for measurement";
  return AssembleCandidates(picks, self->args_info_);
}

void EvolutionarySearchNode::State::NotifyRunnerResults(
    const TuneContext& context, const Array<MeasureCandidate>& measure_candidates,
    const Array<RunnerResult>& results) {
  st += results.size();
  ed += results.size();
}

SearchStrategy SearchStrategy::EvolutionarySearch(int num_trials_per_iter,     //
                                                  int num_trials_total,        //
                                                  int population_size,         //
                                                  double init_measured_ratio,  //
                                                  int init_min_unmeasured,     //
                                                  int genetic_num_iters,       //
                                                  double genetic_mutate_prob,  //
                                                  int genetic_max_fail_count,  //
                                                  double eps_greedy) {
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(init_measured_ratio, "Initial measured ratio");
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(genetic_mutate_prob, "Mutation probability");
  TVM_META_SCHEDULE_CHECK_PROB_RANGE(eps_greedy, "Greedy pick probability");
  ObjectPtr<EvolutionarySearchNode> n = make_object<EvolutionarySearchNode>();
  n->num_trials_per_iter = num_trials_per_iter;
  n->num_trials_total = num_trials_total;
  n->population_size = population_size;
  n->init_measured_ratio = init_measured_ratio;
  n->init_min_unmeasured = init_min_unmeasured;
  n->genetic_num_iters = genetic_num_iters;
  n->genetic_max_fail_count = genetic_max_fail_count;
  n->genetic_mutate_prob = genetic_mutate_prob;
  n->eps_greedy = eps_greedy;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(EvolutionarySearchNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyEvolutionarySearch")
    .set_body_typed(SearchStrategy::EvolutionarySearch);

}  // namespace meta_schedule
}  // namespace tvm
