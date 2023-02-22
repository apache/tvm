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
#include <mutex>
#include <unordered_map>

#include "../utils.h"

namespace tvm {
namespace meta_schedule {

using tir::Instruction;
using tir::InstructionKind;
using tir::Trace;

/*!
 * \brief Downcast the decision of Sample-Perfect-Tile to an array of integers
 * \param decision The decision of Sample-Perfect-Tile
 * \return The result of downcast
 */
std::vector<int64_t> DowncastTilingDecision(const ObjectRef& decision) {
  const auto* arr = TVM_TYPE_AS(decision, runtime::ArrayNode);
  return support::AsVector<ObjectRef, int64_t>(GetRef<Array<ObjectRef>>(arr));
}

/*!
 * \brief Calculate the product of elements in an array
 * \param array The array
 * \return The product of elements in the array
 */
int64_t Product(const std::vector<int64_t>& array) {
  int64_t result = 1;
  for (int64_t x : array) {
    result *= x;
  }
  return result;
}

/*! \brief A mutator that mutates the tile size */
class MutateTileSizeNode : public MutatorNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}
  static constexpr const char* _type_key = "meta_schedule.MutateTileSize";
  TVM_DECLARE_FINAL_OBJECT_INFO(MutateTileSizeNode, MutatorNode);

 public:
  // Inherit from `MutatorNode`
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherit from `MutatorNode`
  Optional<Trace> Apply(const Trace& trace, TRandState* rand_state) final;
  // Inherit from `MutatorNode`
  Mutator Clone() const final {
    ObjectPtr<MutateTileSizeNode> n = make_object<MutateTileSizeNode>(*this);
    return Mutator(n);
  }
};

/*!
 * \brief Find a sample-perfect-tile decision in the trace
 * \param trace The trace
 * \param rand_state The random state
 * \param inst The instruction selected
 * \param decision The decision selected
 * \return Whether a decision is found
 */
void FindSamplePerfectTile(const Trace& trace, std::vector<Instruction>* inst,
                           std::vector<std::vector<int64_t>>* decision) {
  static const InstructionKind& inst_sample_perfect_tile =
      InstructionKind::Get("SamplePerfectTile");
  std::vector<Instruction>& instructions = *inst;
  std::vector<std::vector<int64_t>>& decisions = *decision;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    const ObjectRef& decision = kv.second;
    if (inst->kind.same_as(inst_sample_perfect_tile)) {
      std::vector<int64_t> tiles = DowncastTilingDecision(decision);
      if (tiles.size() >= 2 && Product(tiles) >= 2) {
        instructions.push_back(inst);
        decisions.push_back(tiles);
      }
    }
  }
}

void FindSampleVectorize(const Trace& trace, std::vector<Instruction>* inst,
                         std::vector<int64_t>* decision) {
  static const InstructionKind& inst_sample_categorical = InstructionKind::Get("SampleCategorical");
  static const InstructionKind& inst_annotate = InstructionKind::Get("Annotate");
  std::vector<Instruction>& instructions = *inst;
  std::vector<int64_t>& decisions = *decision;
  std::unordered_set<const Object*> annotated;
  instructions.reserve(trace->decisions.size());
  decisions.reserve(trace->decisions.size());
  annotated.reserve(trace->decisions.size());
  // Find annotation with `meta_schedule_cooperative_fetch`
  for (const Instruction& inst : trace->insts) {
    if (inst->kind.same_as(inst_annotate)) {
      ICHECK_EQ(inst->attrs.size(), 1);
      ICHECK_EQ(inst->inputs.size(), 2);
      if (Downcast<String>(inst->attrs[0]) == tir::attr::meta_schedule_cooperative_fetch) {
        const auto* ann_val = inst->inputs[1].as<tir::ExprRVNode>();
        ICHECK(ann_val);
        annotated.insert(ann_val);
      }
    }
  }
  // Find sampling instruction that generates the annotation
  for (const auto& kv : trace->decisions) {
    const Instruction& inst = kv.first;
    const ObjectRef& decision = kv.second;
    if (inst->kind.same_as(inst_sample_categorical)) {
      ICHECK_EQ(inst->outputs.size(), 1);
      if (annotated.count(inst->outputs[0].get())) {
        ICHECK_EQ(inst->attrs.size(), 2);
        std::vector<double> probs =
            support::AsVector<FloatImm, double>(Downcast<Array<FloatImm>>(inst->attrs[1]));
        if (probs.size() == 1) {
          // Skip mutating the sampling instructions who have only single candidate.
          continue;
        }
        const auto* d = TVM_TYPE_AS(decision, IntImmNode);
        instructions.push_back(inst);
        decisions.push_back(d->value);
      }
    }
  }
}

struct FactorMemo {
  static std::vector<int> Factorize(int n) {
    if (const std::vector<int>* result = Global()->Query(n)) {
      return *result;
    }
    std::vector<int> result;
    for (int64_t i = 1; i * i <= n; ++i) {
      if (n % i == 0) {
        result.push_back(i);
        if (i * i != n) {
          result.push_back(n / i);
        }
      }
    }
    std::sort(result.begin(), result.end());
    Global()->Add(n, result);
    return result;
  }

 private:
  const std::vector<int>* Query(int n) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = memo_.find(n);
    if (it != memo_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  void Add(int n, std::vector<int> result) {
    std::unique_lock<std::mutex> lock(mutex_);
    memo_.emplace(n, std::move(result));
  }

  static FactorMemo* Global() {
    static FactorMemo singleton;
    return &singleton;
  }

  std::unordered_map<int, std::vector<int>> memo_;
  std::mutex mutex_;
};

Optional<Trace> MutateSampleTileSize(const Trace& trace, Instruction inst,
                                     std::vector<int64_t> tiles, TRandState* rand_state) {
  int n_splits = tiles.size();
  // Step 1. Choose two loops, `x` and `y`
  int x, y;
  // select source
  while (true) {
    x = tir::SampleInt(rand_state, 0, n_splits);
    if (tiles[x] <= 1) {
      continue;
    }
    y = tir::SampleInt(rand_state, 0, n_splits - 1);
    if (y >= x) {
      ++y;
    }
    std::vector<int> factors = FactorMemo::Factorize(tiles[x]);
    // Step 2. Choose the divide factor
    int64_t divide_factor;
    if (y != n_splits - 1) {
      divide_factor = factors[tir::SampleInt(rand_state, 1, factors.size())];
    } else {
      int64_t limit = Downcast<Integer>(inst->attrs[1])->value;
      int max_factor_index = static_cast<int>(factors.size()) - 1;
      for (; max_factor_index >= 1; max_factor_index--) {
        if (factors[max_factor_index] * tiles[y] <= limit) {
          break;
        }
      }
      if (max_factor_index == 0) {
        if (n_splits <= 2) {
          return NullOpt;
        }
        // Failed on this dst_idx, try next one.
        continue;
      }
      divide_factor = factors[tir::SampleInt(rand_state, 1, max_factor_index + 1)];
    }
    tiles[x] /= divide_factor;
    tiles[y] *= divide_factor;
    return trace->WithDecision(inst, support::AsArray<int64_t, ObjectRef>(tiles),
                               /*remove_postproc=*/true);
  }
}

Optional<Trace> MutateSampleVectorize(const Trace& trace, Instruction inst,
                                      int64_t original_decision, TRandState* rand_state) {
  ICHECK_EQ(inst->attrs.size(), 2);
  std::vector<double> probs =
      support::AsVector<FloatImm, double>(Downcast<Array<FloatImm>>(inst->attrs[1]));
  probs.erase(probs.begin() + original_decision);
  int result = tir::MakeMultinomialSampler(rand_state, probs)();
  if (result >= original_decision) {
    result += 1;
  }
  return trace->WithDecision(inst, Integer(result), /*remove_postproc=*/true);
}

Optional<Trace> MutateTileSizeNode::Apply(const Trace& trace, TRandState* rand_state) {
  std::vector<Instruction> sample_perfect_tile_insts;
  std::vector<Instruction> sample_vectorize_insts;
  std::vector<std::vector<int64_t>> sample_perfect_tile_tiles;
  std::vector<int64_t> sample_vectorize_decisions;
  FindSamplePerfectTile(trace, &sample_perfect_tile_insts, &sample_perfect_tile_tiles);
  FindSampleVectorize(trace, &sample_vectorize_insts, &sample_vectorize_decisions);
  int size_a = sample_perfect_tile_insts.size();
  int size_b = sample_vectorize_insts.size();
  if (size_a == 0 && size_b == 0) {
    return NullOpt;
  }
  int n = tir::SampleInt(rand_state, 0, size_a + size_b);
  if (n < size_a) {
    return MutateSampleTileSize(trace, sample_perfect_tile_insts[n], sample_perfect_tile_tiles[n],
                                rand_state);
  } else {
    n -= size_a;
    return MutateSampleVectorize(trace, sample_vectorize_insts[n], sample_vectorize_decisions[n],
                                 rand_state);
  }
}

Mutator Mutator::MutateTileSize() { return Mutator(make_object<MutateTileSizeNode>()); }

TVM_REGISTER_NODE_TYPE(MutateTileSizeNode);
TVM_REGISTER_GLOBAL("meta_schedule.MutatorMutateTileSize").set_body_typed(Mutator::MutateTileSize);

}  // namespace meta_schedule
}  // namespace tvm
