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
namespace meta_schedule {

/*! \brief A search strategy that generates measure candidates using trace and random decisions. */
class ReplayTraceNode : public SearchStrategyNode {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    ReplayTraceNode* self;
    /*! \brief The design spaces. */
    Array<tir::Trace> design_spaces;
    /*! \brief The number of total trials. */
    int max_trials;
    /*! \brief The number of trials per iteration. */
    int num_trials_per_iter;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    /*! \brief The module to be tuned. */
    Array<IRModule> per_thread_mod_{nullptr};

    explicit State(ReplayTraceNode* self, Array<tir::Trace> design_spaces, int max_trials,
                   int num_trials_per_iter)
        : self(self),
          design_spaces(design_spaces),
          max_trials(max_trials),
          num_trials_per_iter(num_trials_per_iter),
          st(0),
          ed(num_trials_per_iter) {
      IRModule mod = self->mod_.value();
      this->per_thread_mod_.reserve(self->num_threads_);
      for (int i = 0; i < self->num_threads_; i++) {
        this->per_thread_mod_.push_back(DeepCopyIRModule(mod));
      }
    }

    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const Array<RunnerResult>& results);
  };

  /*! \brief The max number of failures during trace replaying. */
  int max_fail_count;

  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The IRModule to be scheduled from TuneContext. */
  Optional<IRModule> mod_ = NullOpt;
  /*! \brief The number of threads to be used. */
  int num_threads_ = -1;
  /*! \brief The postprocessors. */
  Array<Postproc> postprocs_ = {};
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_fail_count", &max_fail_count);
    // `rand_state_` is not visited
    // `mod_` is not visited
    // `num_threads_` is not visited
    // `postprocs_` is not visited
    // `state_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.ReplayTrace";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayTraceNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& ctx) final {
    CHECK(ctx->mod.defined()) << "ValueError: TuneContext.mod is not defined";
    CHECK(ctx->space_generator.defined())
        << "ValueError: TuneContext.space_generator is not defined";
    if (!ctx->space_generator.value()->postprocs.defined()) {
      TVM_PY_LOG(WARNING, ctx->logger)
          << "`postprocs` is not defined in " << ctx->space_generator.value()
          << ". Please explicitly set `postprocs` to an empty list if you don't want to "
             "apply any post-processing.";
    }
    this->rand_state_ = ForkSeed(&ctx->rand_state);
    this->mod_ = ctx->mod;
    this->num_threads_ = ctx->num_threads;
    this->postprocs_ = ctx->space_generator.value()->postprocs.value_or({});
    this->state_.reset();
  }

  void PreTuning(int max_trials, int num_trials_per_iter, const Array<tir::Schedule>& design_spaces,
                 const Optional<Database>& database, const Optional<CostModel>& cost_model) final {
    ICHECK(!design_spaces.empty());
    CHECK(this->state_ == nullptr)
        << "ValueError: `PreTuning` is already invoked without corresponding `PostTuning`.";
    Array<tir::Trace> design_space_traces;
    design_space_traces.reserve(design_spaces.size());
    for (const tir::Schedule& space : design_spaces) {
      design_space_traces.push_back(space->trace().value()->Simplified(true));
    }
    this->state_ =
        std::make_unique<State>(this, design_space_traces, max_trials, num_trials_per_iter);
  }

  void PostTuning() final {
    ICHECK(this->state_ != nullptr);
    this->state_.reset();
  }

  Optional<Array<MeasureCandidate>> GenerateMeasureCandidates() final {
    ICHECK(this->state_ != nullptr);
    return this->state_->GenerateMeasureCandidates();
  }

  void NotifyRunnerResults(const Array<MeasureCandidate>& measure_candidates,
                           const Array<RunnerResult>& results) final {
    ICHECK(this->state_ != nullptr);
    this->state_->NotifyRunnerResults(results);
  }

  SearchStrategy Clone() const final {
    ObjectPtr<ReplayTraceNode> n = make_object<ReplayTraceNode>();
    n->max_fail_count = this->max_fail_count;
    n->rand_state_ = this->rand_state_;
    n->state_ = nullptr;  // cleared the state
    return SearchStrategy(n);
  }
};

inline Optional<Array<MeasureCandidate>> ReplayTraceNode::State::GenerateMeasureCandidates() {
  if (st >= max_trials) {
    return NullOpt;
  }
  ed = std::min(ed, max_trials);
  ICHECK_LT(st, ed);
  std::vector<TRandState> per_thread_rand_state = ForkSeed(&self->rand_state_, self->num_threads_);
  Array<MeasureCandidate> per_task_result(ed - st, MeasureCandidate{nullptr});
  ThreadedTraceApply pp(self->postprocs_);
  auto f_worker = [this, &per_thread_rand_state, &per_task_result, &pp](int thread_id,
                                                                        int task_id) -> void {
    TRandState& rand_state = per_thread_rand_state[thread_id];
    IRModule mod = this->per_thread_mod_[thread_id];

    for (int fail_count = 0; fail_count < self->max_fail_count; fail_count++) {
      int design_space_index = tir::SampleInt(&rand_state, 0, design_spaces.size());
      tir::Trace trace = design_spaces[design_space_index];
      tir::Trace new_trace = tir::Trace(trace->insts, {});
      if (Optional<tir::Schedule> opt_sch = pp.Apply(mod, new_trace, &rand_state)) {
        tir::Schedule sch = opt_sch.value();
        Array<ArgInfo> args_info = ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true);
        per_task_result.Set(task_id, MeasureCandidate(sch, args_info));
        break;
      }
    }
  };
  support::parallel_for_dynamic(0, ed - st, self->num_threads_, f_worker);
  Array<MeasureCandidate> filtered;
  filtered.reserve(ed - st);
  for (MeasureCandidate result : per_task_result)
    if (result.defined()) {
      filtered.push_back(result);
    }
  return filtered;
}

inline void ReplayTraceNode::State::NotifyRunnerResults(const Array<RunnerResult>& results) {
  st += num_trials_per_iter;
  ed += num_trials_per_iter;
}

SearchStrategy SearchStrategy::ReplayTrace(int max_fail_count) {
  ObjectPtr<ReplayTraceNode> n = make_object<ReplayTraceNode>();
  n->max_fail_count = max_fail_count;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(ReplayTraceNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyReplayTrace")
    .set_body_typed(SearchStrategy::ReplayTrace);

}  // namespace meta_schedule
}  // namespace tvm
