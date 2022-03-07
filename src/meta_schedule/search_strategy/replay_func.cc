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
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int st;
    /*! \brief `[st, ed)` are the indices of the next batch of candidates. */
    int ed;

    explicit State(ReplayFuncNode* self) : self(self), st(0), ed(self->num_trials_per_iter) {}

    inline Optional<Array<MeasureCandidate>> GenerateMeasureCandidates();
    inline void NotifyRunnerResults(const Array<RunnerResult>& results);
  };

  /*! \brief The number of trials per iteration. */
  int num_trials_per_iter;
  /*! \brief The number of total trials. */
  int num_trials_total;

  /*! \brief The module to be tuned. */
  IRModule mod_{nullptr};
  /*! \brief The metadata of the function arguments. */
  Array<ArgInfo> args_info_{nullptr};
  /*! \brief The post processors */
  Array<Postproc> postprocs_{nullptr};
  /*! \brief The space generator for measure candidates generation. */
  SpaceGenerator space_generator_{nullptr};
  /*! \brief The random state. -1 means using random number. */
  TRandState rand_state_ = -1;
  /*! \brief The state of the search strategy. */
  std::unique_ptr<State> state_ = nullptr;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_trials_per_iter", &num_trials_per_iter);
    v->Visit("num_trials_total", &num_trials_total);
    // `space_generator_` is not visited
    // `mod_` is not visited
    // `args_info_` is not visited
    // `num_threads_` is not visited
    // `rand_state_` is not visited
    // `state_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.ReplayFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(ReplayFuncNode, SearchStrategyNode);

  void InitializeWithTuneContext(const TuneContext& context) final {
    this->space_generator_ = context->space_generator.value();
    this->mod_ = context->mod.value();
    this->args_info_ = ArgInfo::FromPrimFunc(FindEntryFunc(context->mod.value()));
    this->postprocs_ = context->postprocs;
    this->rand_state_ = ForkSeed(&context->rand_state);
    this->state_.reset();
  }

  void PreTuning(const Array<tir::Schedule>& design_spaces) final {
    ICHECK(this->state_ == nullptr);
    this->state_ = std::make_unique<State>(this);
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
    this->state_->NotifyRunnerResults(results);
  }
};

inline Optional<Array<MeasureCandidate>> ReplayFuncNode::State::GenerateMeasureCandidates() {
  if (st >= self->num_trials_total) {
    return NullOpt;
  }
  ed = std::min(ed, self->num_trials_total);
  Array<MeasureCandidate> result;
  for (int i = st; i < ed; i++) {
    for (;;) {
      Array<tir::Schedule> schs = self->space_generator_->GenerateDesignSpace(self->mod_);
      int design_space_index = tir::SampleInt(&self->rand_state_, 0, schs.size());
      tir::Schedule sch = schs[design_space_index];
      sch->EnterPostproc();
      bool failed = false;
      for (const Postproc& proc : self->postprocs_) {
        if (!proc->Apply(sch)) {
          failed = true;
          break;
        }
      }
      if (!failed) {
        result.push_back(MeasureCandidate(sch, self->args_info_));
        break;
      }
    }
  }
  return result;
}

inline void ReplayFuncNode::State::NotifyRunnerResults(const Array<RunnerResult>& results) {
  st += self->num_trials_per_iter;
  ed += self->num_trials_per_iter;
}

SearchStrategy SearchStrategy::ReplayFunc(int num_trials_per_iter, int num_trials_total) {
  ObjectPtr<ReplayFuncNode> n = make_object<ReplayFuncNode>();
  n->num_trials_per_iter = num_trials_per_iter;
  n->num_trials_total = num_trials_total;
  return SearchStrategy(n);
}

TVM_REGISTER_NODE_TYPE(ReplayFuncNode);
TVM_REGISTER_GLOBAL("meta_schedule.SearchStrategyReplayFunc")
    .set_body_typed(SearchStrategy::ReplayFunc);

}  // namespace meta_schedule
}  // namespace tvm
