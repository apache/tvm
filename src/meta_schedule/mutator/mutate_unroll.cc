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
 * \brief Check if an instruction is annotate with
 * `meta_schedule_unroll_explicit` or `meta_schedule_unroll_implicit`
 * \param inst The instruction to be checked
 * \return Whether the instruction is annotated
 */
bool IsAnnotateWithUnroll(const Instruction& inst) {
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_annotate)) {
    return false;
  }
  ICHECK_EQ(inst->attrs.size(), 1);
  String ann_key = Downcast<String>(inst->attrs[0]);
  return ann_key == attr::meta_schedule_unroll_explicit ||
         ann_key == attr::meta_schedule_unroll_implicit;
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::Instruction;
using tir::Trace;

/*! \brief Create a Mutator that mutates auto unroll step */
class MutateUnrollNode : public MutatorNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.MutateUnroll";
  TVM_DECLARE_FINAL_OBJECT_INFO(MutateUnrollNode, MutatorNode);

 public:
  struct Candidate;
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherit from `MutatorNode`
  Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorNode`
  Mutator Clone() const final {
    ObjectPtr<MutateUnrollNode> n = make_object<MutateUnrollNode>(*this);
    return Mutator(n);
  }
};

/*! \brief A candidate to be mutated */
struct MutateUnrollNode::Candidate {
  /*! \brief The sampling instruction to be mutated */
  Instruction inst;
  /*! \brief The probability */
  std::vector<double> probs;
  /*! \brief The decision made */
  int decision;
};

/*!
 * \brief Find the Sample-Categorical instruction to be mutated that affects the maximal unroll step
 * \param trace The trace to be mutated
 * \param rand_state The random state
 * \param candidates The mutation candidate
 * \return Whether a decision is found
 */
bool FindUnrollDecision(const Trace& trace, TRandState* rand_state,
                        MutateUnrollNode::Candidate* candidate) {
  using tir::InstructionKind;
  using tir::InstructionNode;
  static const InstructionKind& inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  std::unordered_map<const PrimExprNode*, const InstructionNode*> sample_insts;
  std::vector<const InstructionNode*> ann_insts;
  sample_insts.reserve(trace->insts.size());
  ann_insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_sample_categorical)) {
      ICHECK_EQ(inst->outputs.size(), 1);
      const PrimExprNode* var_rv = TVM_TYPE_AS(inst->outputs[0], PrimExprNode);
      sample_insts[var_rv] = inst.get();
    } else if (IsAnnotateWithUnroll(inst)) {
      ann_insts.push_back(inst.get());
    }
  }
  int n_ann_insts = ann_insts.size();
  if (n_ann_insts == 0) {
    return false;
  }
  const InstructionNode* ann_inst = ann_insts[tir::SampleInt(rand_state, 0, n_ann_insts)];
  ICHECK_EQ(ann_inst->inputs.size(), 2);
  const auto* var_rv = TVM_TYPE_AS(ann_inst->inputs[1], PrimExprNode);
  ICHECK(sample_insts.count(var_rv));
  const InstructionNode* sample_inst = sample_insts.at(var_rv);
  ICHECK_EQ(sample_inst->attrs.size(), 2);
  candidate->inst = GetRef<Instruction>(sample_inst);
  candidate->decision =
      Downcast<Integer>(trace->decisions[GetRef<Instruction>(sample_inst)])->value;
  candidate->probs =
      support::AsVector<FloatImm, double>(Downcast<Array<FloatImm>>(sample_inst->attrs[1]));
  return true;
}

Optional<Trace> MutateUnrollNode::Apply(const Trace& trace, TRandState* rand_state) {
  Candidate candidate;
  if (!FindUnrollDecision(trace, rand_state, &candidate)) {
    return NullOpt;
  }
  if (candidate.probs.size() == 0) {
    return NullOpt;
  }
  candidate.probs.erase(candidate.probs.begin() + candidate.decision);
  int result = tir::MakeMultinomialSampler(rand_state, candidate.probs)();
  if (result >= candidate.decision) {
    result += 1;
  }
  return trace->WithDecision(candidate.inst, Integer(result), /*remove_postproc=*/true);
}

Mutator Mutator::MutateUnroll() { return Mutator(make_object<MutateUnrollNode>()); }

TVM_REGISTER_NODE_TYPE(MutateUnrollNode);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorMutateUnroll").set_body_typed(Mutator::MutateUnroll);

}  // namespace meta_schedule
}  // namespace tvm
