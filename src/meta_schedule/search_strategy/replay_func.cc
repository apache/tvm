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

/*! \brief A search strategy that generates measure candidates using space generator. */
class ReplayFuncNode : public SearchStrategyNode {
 public:
  /*! \brief The state of the search strategy. */
  struct State {
    /*! \brief The search strategy itself */
    ReplayFuncNode* self;
    /*! \brief The number of total trials. */
    int max_trials;
    /*! \brief The number of trials per iteration. */
    int num_trials_per_iter;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(ReplayFuncNode* self, int max_trials, int num_trials_per_iter)
        : self(self),
          max_trials(max_trials),
          num_trials_per_iter(num_trials_per_iter),
          st(0),
          ed(num_trials_per_iter) {
      CHECK(self->mod_.defined() && self->space_generator_.defined())
          << "ValueError: The search strategy has not been initialized.";
    }

    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const Array<RunnerResult>& results);
  };

  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The IRModule to be scheduled from TuneContext. */
  Optional<IRModule> mod_ = NullOpt;
  /*! \brief The space generator from TuneContext. */
  Optional<SpaceGenerator> space_generator_ = NullOpt;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.ReplayFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayFuncNode, SearchStrategyNode);

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
    this->space_generator_ = ctx->space_generator;
    this->state_.reset();
  }

  void PreTuning(int max_trials, int num_trials_per_iter, const Array<tir::Schedule>& design_spaces,
                 const Optional<Database>& database, const Optional<CostModel>& cost_model) final {
    CHECK(this->state_ == nullptr)
        << "ValueError: `PreTuning` is already invoked without corresponding `PostTuning`.";
    this->state_ = std::make_unique<State>(this, max_trials, num_trials_per_iter);
  }

  void PostTuning() final {
    CHECK(this->state_ != nullptr) << "ValueError: `PostTuning` is invoked without corresponding "
                                      "`PreTuning`, or `PostTuning` is already invoked.";
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
    ObjectPtr<ReplayFuncNode> n = make_object<ReplayFuncNode>();
    n->rand_state_ = -1;
    n->mod_ = NullOpt;
    n->space_generator_ = NullOpt;
    n->state_ = nullptr;
    return SearchStrategy(n);
  }
};

inline Optional<Array<MeasureCandidate>> ReplayFuncNode::State::GenerateMeasureCandidates() {
  if (st >= max_trials) {
    return NullOpt;
  }
  ed = std::min(ed, max_trials);
  Array<MeasureCandidate> result;
  IRModule mod = self->mod_.value();
  Array<Postproc> postprocs = self->space_generator_.value()->postprocs.value_or({});
  for (int i = st; i < ed; i++) {
    for (;;) {
      Array<tir::Schedule> schs = self->space_generator_.value()->GenerateDesignSpace(mod);
      int design_space_index = tir::SampleInt(&self->rand_state_, 0, schs.size());
      tir::Schedule sch = schs[design_space_index];
      sch->EnterPostproc();
      bool failed = false;
      for (const Postproc& proc : postprocs) {
        if (!proc->Apply(sch)) {
          failed = true;
          break;
        }
      }
      if (!failed) {
        Array<ArgInfo> args_info = ArgInfo::FromEntryFunc(sch->mod(), /*remove_preproc=*/true);
        result.push_back(MeasureCandidate(sch, args_info));
        break;
      }
    }
  }
  return result;
}

inline void ReplayFuncNode::State::NotifyRunnerResults(const Array<RunnerResult>& results) {
  st += num_trials_per_iter;
  ed += num_trials_per_iter;
}

SearchStrategy SearchStrategy::ReplayFunc() {
  ObjectPtr<ReplayFuncNode> n = make_object<ReplayFuncNode>();
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(ReplayFuncNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyReplayFunc")
    .set_body_typed(SearchStrategy::ReplayFunc);

}  // namespace meta_schedule
}  // namespace tvm
