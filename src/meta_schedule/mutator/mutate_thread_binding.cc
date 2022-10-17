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

using tir::Instruction;
using tir::InstructionKind;
using tir::Trace;

/*! \brief A mutator that mutates the thread binding factor decision of SampleCategorical */
class MutateThreadBindingNode : public MutatorNode {
 public:
  /*! \brief JSON representation of the workload */
  std::string json_mod_;

  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.MutateThreadBinding";
  TVM_DECLARE_FINAL_OBJECT_INFO(MutateThreadBindingNode, MutatorNode);

 public:
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {
    this->json_mod_ = SaveJSON(context->mod.value());
  }
  // Inherit from `MutatorNode`
  Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorNode`
  Mutator Clone() const final {
    ObjectPtr<MutateThreadBindingNode> n = make_object<MutateThreadBindingNode>(*this);
    return Mutator(n);
  }

 private:
  struct Candidate {
    /*! \brief The sampling instruction to be mutated */
    Instruction inst;
    /*! \brief The probability */
    std::vector<double> probs;
    /*! \brief The decision made */
    int decision;

    explicit Candidate(Instruction inst, std::vector<double> probs, int decision)
        : inst(std::move(inst)), probs(std::move(probs)), decision(std::move(decision)) {}
  };

  std::vector<Candidate> FindCandidates(const Trace& trace, TRandState* rand_state);
};

/*!
 * \brief Find Candidate with the following pattern:
 * \code
 * v = sch.sample_categorical(...)
 * l1, l2 = sch.split(loop=l0, factors=[None, v])
 * sch.bind(loop=l2, thread_axis="threadIdx.x")
 * \endcode
 *
 * \param trace The trace from which to find the instructions
 * \return All the candidate instructions
 */
std::vector<MutateThreadBindingNode::Candidate> MutateThreadBindingNode::FindCandidates(
    const Trace& trace, TRandState* rand_state) {
  using tir::InstructionNode;

  static InstructionKind inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  static InstructionKind inst_split = InstructionKind::Get("Split");
  static InstructionKind inst_bind = InstructionKind::Get("Bind");

  std::vector<MutateThreadBindingNode::Candidate> candidates;
  std::unordered_map<const PrimExprNode*, const tir::InstructionNode*> sample_insts;
  std::unordered_map<const tir::LoopRVNode*, const tir::InstructionNode*> sampled_split_insts;
  std::vector<const InstructionNode*> bind_insts;

  auto is_split_by_sample = [&sample_insts](const Instruction& inst) -> bool {
    if (!inst->kind.same_as(inst_split)) {
      return false;
    }
    // Only consider cases with 2 factors and the first one is None
    if (inst->inputs.size() != 3 || inst->inputs[1].defined()) return false;
    ICHECK(inst->inputs[2].defined());

    return sample_insts.find(Downcast<PrimExpr>(inst->inputs[2]).get()) != sample_insts.end();
  };

  auto is_thread_binding_by_sample = [&sampled_split_insts](const Instruction& inst) -> bool {
    if (!inst->kind.same_as(inst_bind)) {
      return false;
    }
    ICHECK_EQ(inst->inputs.size(), 1);
    ICHECK_EQ(inst->attrs.size(), 1);
    if (Downcast<String>(inst->attrs[0]) != "threadIdx.x") return false;

    return sampled_split_insts.find(Downcast<tir::LoopRV>(inst->inputs[0]).get()) !=
           sampled_split_insts.end();
  };

  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_sample_categorical)) {
      ICHECK_EQ(inst->outputs.size(), 1);
      const PrimExprNode* var_rv = TVM_TYPE_AS(inst->outputs[0], PrimExprNode);
      sample_insts[var_rv] = inst.get();
    } else if (is_split_by_sample(inst)) {
      CHECK_EQ(inst->outputs.size(), 2);
      // Only consider the inner loop, which can be bound to threadIdx.x
      const tir::LoopRVNode* var_rv = TVM_TYPE_AS(inst->outputs[1], tir::LoopRVNode);
      sampled_split_insts[var_rv] = inst.get();
    } else if (is_thread_binding_by_sample(inst)) {
      bind_insts.push_back(inst.get());
    }
  }

  for (const InstructionNode* bind_inst : bind_insts) {
    const auto* loop_rv = TVM_TYPE_AS(bind_inst->inputs[0], tir::LoopRVNode);
    auto split_it = sampled_split_insts.find(loop_rv);
    ICHECK(split_it != sampled_split_insts.end());
    const InstructionNode* split_inst = split_it->second;

    const auto* expr_rv = TVM_TYPE_AS(split_inst->inputs[2], PrimExprNode);
    auto sample_it = sample_insts.find(expr_rv);
    ICHECK(sample_it != sample_insts.end());
    const InstructionNode* sample_inst = sample_it->second;

    int decision = Downcast<Integer>(trace->decisions[GetRef<Instruction>(sample_inst)])->value;

    std::vector<double> probs =
        support::AsVector<FloatImm, double>(Downcast<Array<FloatImm>>(sample_inst->attrs[1]));

    candidates.emplace_back(GetRef<Instruction>(sample_inst), probs, decision);
  }
  return candidates;
}

Optional<Trace> MutateThreadBindingNode::Apply(const Trace& trace, TRandState* rand_state) {
  std::vector<Candidate> candidates = FindCandidates(trace, rand_state);
  if (candidates.empty()) {
    return NullOpt;
  }
  Candidate candidate = candidates[tir::SampleInt(rand_state, 0, candidates.size())];
  // Remove the current decision
  candidate.probs.erase(candidate.probs.begin() + candidate.decision);
  int result = tir::MakeMultinomialSampler(rand_state, candidate.probs)();
  if (result >= candidate.decision) {
    result += 1;
  }
  return trace->WithDecision(candidate.inst, Integer(result), /*remove_postproc=*/true);
}

Mutator Mutator::MutateThreadBinding() { return Mutator(make_object<MutateThreadBindingNode>()); }

TVM_REGISTER_NODE_TYPE(MutateThreadBindingNode);
TVM_REGISTER_GLOBAL("meta_schedule.MutateThreadBinding")
    .set_body_typed(Mutator::MutateThreadBinding);

}  // namespace meta_schedule
}  // namespace tvm
