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
#include <algorithm>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Check if the instruction is annotation with `meta_schedule_parallel`
 * \param inst The instruction to be checked
 * \return Whether the instruction is annotation with `meta_schedule_parallel`
 */
bool IsAnnotateWithParallel(const Instruction& inst) {
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  if (!inst->kind.same_as(inst_annotate)) {
    return false;
  }
  ICHECK_EQ(inst->attrs.size(), 1);
  String ann_key = Downcast<String>(inst->attrs[0]);
  return ann_key == attr::meta_schedule_parallel;
}

/*!
 * \brief Replace the annotation value
 * \param inst The instruction to be replaced
 * \param ann_val The new annotation value
 * \return The replaced instruction
 */
Instruction ReplaceAnnValue(Instruction inst, int64_t ann_val) {
  ICHECK_EQ(inst->inputs.size(), 2);
  return Instruction(/*kind=*/inst->kind,                             //
                     /*inputs=*/{inst->inputs[0], Integer(ann_val)},  //
                     /*attrs=*/inst->attrs,
                     /*outputs=*/inst->outputs);
}

/*!
 * \brief Get the output of the instruction Get-Block
 * \param inst The instruction to be checked
 * \return The output of the instruction Get-Block
 */
const BlockRVNode* GetInstGetBlockOutput(const Instruction& inst) {
  static const InstructionKind& inst_get_block = InstructionKind::Get("GetBlock");
  if (!inst->kind.same_as(inst_get_block)) {
    return nullptr;
  }
  ICHECK_EQ(inst->outputs.size(), 1);
  const BlockRVNode* block = TVM_TYPE_AS(inst->outputs[0], BlockRVNode);
  return block;
}

/*!
 * \brief Analyze the parallel structure
 * \param self The schedule state
 * \param block_name The name of the root block
 * \param func_name The name of the PrimFunc
 * \param limit The uplimit of the parallelism
 * \return The parallel structure
 */
std::vector<std::vector<int64_t>> AnalyzeParallel(const ScheduleState& self,
                                                  const String& block_name, const String& func_name,
                                                  int64_t limit) {
  Array<StmtSRef> block_srefs =
      tir::GetBlocks(self, block_name, self->mod->GetGlobalVar(func_name));
  ICHECK_EQ(block_srefs.size(), 1);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_srefs[0]);
  ScopeBlockLoopInfo info = GetScopeBlockLoopInfo(GetRef<Block>(block));
  std::vector<std::vector<int64_t>> results;
  results.reserve(info.realizes.size());
  for (const BlockRealize& realize : info.realizes) {
    // Step 1. Extract static loop extents for spatial loops
    std::vector<int64_t> loop_extents;
    const ForNode* loop = nullptr;
    for (const StmtSRefNode* loop_sref = self->stmt2ref.at(realize->block.get())->parent;
         (loop = loop_sref->StmtAs<ForNode>()) != nullptr;  //
         loop_sref = loop_sref->parent) {
      int64_t loop_extent = -1;
      if (const auto* ext = GetLoopIntExtent(loop)) {
        if (!info.non_spatial_vars.count(loop->loop_var.get())) {
          loop_extent = *ext;
        }
      }
      if (loop_extent != -1) {
        loop_extents.push_back(loop_extent);
      } else {
        loop_extents.clear();
      }
    }
    // Step 2. Take the prefix product of loop extents
    if (!loop_extents.empty()) {
      results.emplace_back();
      std::vector<int64_t>& result = results.back();
      result.reserve(loop_extents.size());
      int64_t prod_extent = 1;
      for (auto it = loop_extents.rbegin(); it != loop_extents.rend(); ++it) {
        result.push_back(prod_extent *= *it);
        if (prod_extent >= limit) {
          break;
        }
      }
    }
  }
  return results;
}

/*!
 * \brief Get the number of parallelizable loops for each subtree
 * \param loop_extent_prods The parallel structure for each subtree
 * \param limit The uplimit of the parallelism
 * \return The number of parallelizable loops for each subtree
 */
std::vector<int> GetNumFusedLoops(const std::vector<std::vector<int64_t>>& loop_extent_prods,
                                  int64_t limit) {
  std::vector<int> results;
  results.reserve(loop_extent_prods.size());
  for (const std::vector<int64_t>& prods : loop_extent_prods) {
    int n = prods.size();
    int i = std::upper_bound(prods.begin(), prods.end(), limit) - prods.begin();
    if (i > 0 && prods[i - 1] == limit) {
      --i;
    }
    if (i != n) {
      ++i;
    }
    results.push_back(i);
  }
  return results;
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

using tir::Instruction;
using tir::Trace;

/*! \brief Create a Mutator that mutates the parallel extent */
class MutateParallelNode : public MutatorNode {
 public:
  /*!
   * \brief The maximum number of jobs to be launched per CPU core.
   * It sets the uplimit of CPU parallelism, i.e. `num_cores * max_jobs_per_core`.
   * Use -1 to disable parallelism.
   */
  int64_t max_jobs_per_core;
  /*! \brief The number of cores in CPU. */
  int max_parallel_extent_;
  /*! \brief JSON representation of the workload */
  std::string json_mod_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("max_jobs_per_core", &max_jobs_per_core);
    // `max_parallel_extent_` is not visited.
    // `json_mod` is not visited.
  }

  static constexpr const char* _type_key = "meta_schedule.MutateParallel";
  TVM_DECLARE_FINAL_OBJECT_INFO(MutateParallelNode, MutatorNode);

 public:
  struct Candidate;
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {
    Target target = context->target.value();
    this->max_parallel_extent_ = GetTargetNumCores(target) * this->max_jobs_per_core;
    this->json_mod_ = SaveJSON(context->mod.value());
  }
  // Inherit from `MutatorNode`
  Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorNode`
  Mutator Clone() const final {
    ObjectPtr<MutateParallelNode> n = make_object<MutateParallelNode>(*this);
    return Mutator(n);
  }
};

/*! \brief The candidate to be mutated */
struct MutateParallelNode::Candidate {
  /*! \brief The annotation instruction */
  Instruction inst;
  /*! \brief The current parallel extent */
  int64_t parallel_extent;
  /*! \brief The name of the root block */
  String block_name;
  /*! \brief The name of the PrimFunc */
  String func_name;
};

/*!
 * \brief Get an instruction that annotates the maximum parallel extent
 * \param trace The trace to be mutated
 * \param rand_state The random state
 * \param candidate The candidate to be mutated
 * \return Whether a decision is found
 */
bool FindParallelDecision(const Trace& trace, TRandState* rand_state,
                          MutateParallelNode::Candidate* candidate) {
  using tir::BlockRVNode;
  using tir::InstructionNode;
  std::unordered_map<const BlockRVNode*, const InstructionNode*> get_block_insts;
  std::vector<const InstructionNode*> ann_insts;
  get_block_insts.reserve(trace->insts.size());
  ann_insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (tir::IsAnnotateWithParallel(inst)) {
      ann_insts.push_back(inst.get());
    }
    if (const BlockRVNode* block_rv = tir::GetInstGetBlockOutput(inst)) {
      get_block_insts[block_rv] = inst.get();
    }
  }
  int n_ann_insts = ann_insts.size();
  if (n_ann_insts == 0) {
    return false;
  }
  const InstructionNode* ann_inst = ann_insts[tir::SampleInt(rand_state, 0, n_ann_insts)];
  ICHECK_EQ(ann_inst->inputs.size(), 2);
  const InstructionNode* get_block_inst =
      get_block_insts.at(Downcast<tir::BlockRV>(ann_inst->inputs[0]).get());
  ICHECK_EQ(get_block_inst->attrs.size(), 2);
  candidate->inst = GetRef<Instruction>(ann_inst);
  candidate->parallel_extent = Downcast<IntImm>(ann_inst->inputs[1])->value;
  candidate->block_name = Downcast<String>(get_block_inst->attrs[0]);
  candidate->func_name = Downcast<String>(get_block_inst->attrs[1]);
  return true;
}

Optional<Trace> MutateParallelNode::Apply(const Trace& trace, TRandState* rand_state) {
  // Step 1. Find a parallel decision.
  Candidate candidate;
  if (!FindParallelDecision(trace, rand_state, &candidate)) {
    return NullOpt;
  }
  // Step 2. Replay the instructions to recover loop extents
  tir::Schedule sch = tir::Schedule::Traced(                  //
      /*mod=*/Downcast<IRModule>(LoadJSON(this->json_mod_)),  //
      /*rand_state=*/ForkSeed(rand_state),                    //
      /*debug_mode=*/0,
      /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);
  trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
  // Step 3. Find all possible parallel plans
  std::vector<std::vector<int64_t>> loop_extent_prods = tir::AnalyzeParallel(
      sch->state(), candidate.block_name, candidate.func_name, this->max_parallel_extent_);
  std::unordered_map<int64_t, std::vector<int>> limit2plan;
  std::map<std::vector<int>, int64_t> plan2limit;
  for (const std::vector<int64_t>& prods : loop_extent_prods) {
    for (int64_t limit : prods) {
      if (limit <= this->max_parallel_extent_ && !limit2plan.count(limit)) {
        std::vector<int> plan = tir::GetNumFusedLoops(loop_extent_prods, limit);
        limit2plan[limit] = plan;
        plan2limit[plan] = limit;
      }
    }
  }
  // Step 4. Remove the original plan and remove it
  std::vector<int> original_plan =
      tir::GetNumFusedLoops(loop_extent_prods, candidate.parallel_extent);
  auto it = plan2limit.find(original_plan);
  if (it != plan2limit.end()) {
    plan2limit.erase(it);
  }
  // Step 5. Pick a new plan
  int n_plans = plan2limit.size();
  if (n_plans == 0) {
    return NullOpt;
  }
  it = plan2limit.begin();
  for (int i = 0, n = tir::SampleInt(rand_state, 0, n_plans); i < n; ++i) {
    ++it;
  }
  int64_t limit = it->second;
  // Step 6. Assemble a new trace
  Array<Instruction> insts;
  insts.reserve(trace->insts.size());
  for (const Instruction& inst : trace->insts) {
    if (inst.same_as(candidate.inst)) {
      insts.push_back(tir::ReplaceAnnValue(candidate.inst, limit));
    } else if (inst->kind->IsPostproc()) {
      break;
    } else {
      insts.push_back(inst);
    }
  }
  return Trace(insts, trace->decisions);
}

Mutator Mutator::MutateParallel(int64_t max_jobs_per_core) {
  ObjectPtr<MutateParallelNode> n = make_object<MutateParallelNode>();
  n->max_jobs_per_core = max_jobs_per_core;
  return Mutator(n);
}

TVM_REGISTER_NODE_TYPE(MutateParallelNode);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorMutateParallel").set_body_typed(Mutator::MutateParallel);

}  // namespace meta_schedule
}  // namespace tvm
