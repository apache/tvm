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
#include "./concrete_schedule.h"

#include <random>

namespace tvm {
namespace tir {

Schedule Schedule::Concrete(IRModule mod, support::LinearCongruentialEngine::TRandState seed,
                            int debug_mask, ScheduleErrorRenderLevel error_render_level) {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mask);
  n->error_render_level_ = error_render_level;
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  support::LinearCongruentialEngine(&n->rand_state_).Seed(seed);
  return Schedule(std::move(n));
}

/******** Copy ********/

/*! \brief Helper class to perform a deep copy of the sref tree */
class ScheduleCopier {
  using TSymbolTable = ConcreteScheduleNode::TSymbolTable;
  template <class K, class V>
  using UMap = std::unordered_map<K, V>;
  template <class K, class V>
  using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

 public:
  static void Copy(const ConcreteScheduleNode* self, ScheduleState* new_state,
                   TSymbolTable* new_symbol_table) {
    const ScheduleState& src_state = self->state_;
    ScheduleCopier copier(src_state);
    ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
    n->mod = src_state->mod;
    n->block_info = copier.Copy(src_state->block_info);
    n->stmt2ref = copier.Copy(src_state->stmt2ref);
    n->debug_mask = src_state->debug_mask;
    *new_state = ScheduleState(std::move(n));
    *new_symbol_table = copier.Copy(self->symbol_table_);
  }

 private:
  /*! \brief Create the copier and properly set up the `old2new_` table */
  explicit ScheduleCopier(const ScheduleState& state) {
    // Create SRef tree without parents
    for (const auto& kv : state->stmt2ref) {
      const StmtSRefNode* sref = kv.second.operator->();
      old2new_.emplace(sref,                          // the old StmtSRef
                       StmtSRef(/*stmt=*/sref->stmt,  // the new StmtSRef
                                /*parent=*/nullptr,   // parent is not set yet
                                /*seq_index=*/sref->seq_index));
    }
    // Fill in the parent field
    // Find out the root along the way
    for (auto& kv : old2new_) {
      const StmtSRefNode* parent = kv.first->parent;
      StmtSRef& sref = kv.second;
      sref->parent = parent ? old2new_.at(parent).get() : nullptr;
    }
  }

  /*! \brief Copy StmtSRef */
  StmtSRef Copy(const StmtSRef& sref) { return old2new_.at(sref.operator->()); }

  /*! \brief Copy StmtSRefNode */
  StmtSRef Copy(const StmtSRefNode* sref) {
    if (old2new_.count(sref)) {
      return old2new_.at(sref);
    }
    // Handle expired sref
    return old2new_[sref] = StmtSRef(nullptr, nullptr, -1);
  }

  /*! \brief Copy Array<StmtSRef> */
  Array<StmtSRef> Copy(const Array<StmtSRef>& list) {
    Array<StmtSRef> result;
    result.reserve(list.size());
    for (const StmtSRef& elem : list) {
      result.push_back(Copy(elem));
    }
    return result;
  }

  /*! \brief Copy Array<Dependency> */
  Array<Dependency> Copy(const Array<Dependency>& list) {
    Array<Dependency> result;
    result.reserve(list.size());
    for (const Dependency& elem : list) {
      result.push_back(Dependency(Copy(elem->src), Copy(elem->dst), elem->kind));
    }
    return result;
  }

  /*! \brief Copy SMap<StmtSRef, Array<Dependency>> */
  SMap<StmtSRef, Array<Dependency>> Copy(const SMap<StmtSRef, Array<Dependency>>& map) {
    SMap<StmtSRef, Array<Dependency>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[Copy(kv.first)] = Copy(kv.second);
    }
    return result;
  }

  /*! \brief Copy SMap<Buffer, Array<StmtSRef>> */
  SMap<Buffer, Array<StmtSRef>> Copy(const SMap<Buffer, Array<StmtSRef>>& map) {
    SMap<Buffer, Array<StmtSRef>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[kv.first] = Copy(kv.second);
    }
    return result;
  }

  /*! \brief Copy SMap<StmtSRef, Scope> */
  SMap<StmtSRef, BlockInfo> Copy(const SMap<StmtSRef, BlockInfo>& scopes) {
    SMap<StmtSRef, BlockInfo> result;
    for (const auto& kv : scopes) {
      const StmtSRef& old_sref = kv.first;
      const BlockInfo& old_info = kv.second;
      BlockInfo new_info = old_info;
      ObjectPtr<BlockScopeNode> scope = make_object<BlockScopeNode>();
      scope->src2deps = Copy(old_info.scope->src2deps);
      scope->dst2deps = Copy(old_info.scope->dst2deps);
      scope->buffer_writers = Copy(old_info.scope->buffer_writers);
      scope->stage_pipeline = old_info.scope->stage_pipeline;
      new_info.scope = BlockScope(std::move(scope));
      result[Copy(old_sref)] = std::move(new_info);
    }
    return result;
  }

  /*! \brief Copy the stmt2ref */
  UMap<const StmtNode*, StmtSRef> Copy(const UMap<const StmtNode*, StmtSRef>& stmt2ref) {
    UMap<const StmtNode*, StmtSRef> result;
    result.reserve(stmt2ref.size());
    for (const auto& kv : stmt2ref) {
      const StmtNode* stmt = kv.first;
      const StmtSRef& sref = kv.second;
      result.emplace(stmt, Copy(sref));
    }
    return result;
  }

  /*! \brief Copy the symbol table */
  TSymbolTable Copy(const TSymbolTable& tab) {
    TSymbolTable result;
    for (const auto& kv : tab) {
      ObjectRef entry = kv.second;
      if (const auto* sref = entry.as<StmtSRefNode>()) {
        entry = Copy(sref);
      }
      result.Set(kv.first, entry);
    }
    return result;
  }

 private:
  std::unordered_map<const StmtSRefNode*, StmtSRef> old2new_;
};

void ConcreteScheduleNode::Copy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const {
  ScheduleCopier::Copy(this, new_state, new_symbol_table);
  new_state->get()->DebugVerify();
}

Schedule ConcreteScheduleNode::Copy() const {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->error_render_level_ = this->error_render_level_;
  ConcreteScheduleNode::Copy(&n->state_, &n->symbol_table_);
  n->analyzer_ = std::make_unique<arith::Analyzer>();  // new analyzer needed because it is stateful
  return Schedule(std::move(n));
}

/*! \brief Macro that guards the beginning of each invocation of TensorIR schedule primitive */
#define TVM_TIR_SCHEDULE_BEGIN() try {
/*!
 * \brief Macro that pairs with `TVM_TIR_SCHEDULE_BEGIN`, handling potential errors and error
 * message rendering
 * \param level An ScheduleErrorRenderLevel enum, level of error rendering
 * \sa ScheduleErrorRenderLevel
 */
#define TVM_TIR_SCHEDULE_END(primitive, level)                    \
  }                                                               \
  catch (const ScheduleError& error) {                            \
    if ((level) == ScheduleErrorRenderLevel::kDetail) {           \
      throw tvm::runtime::Error(error.RenderReport(primitive));   \
    } else if ((level) == ScheduleErrorRenderLevel::kFast) {      \
      throw tvm::runtime::Error(error.FastErrorString());         \
    } else if ((level) == ScheduleErrorRenderLevel::kNone) {      \
      throw tvm::runtime::Error("ScheduleError: (not rendered)"); \
    }                                                             \
  }

/******** Schedule: Schedule: Sampling ********/

void ConcreteScheduleNode::Seed(support::LinearCongruentialEngine::TRandState seed) {
  if (seed == -1) {
    seed = std::random_device()();
  }
  support::LinearCongruentialEngine(&rand_state_).Seed(seed);
}

support::LinearCongruentialEngine::TRandState ConcreteScheduleNode::ForkSeed() {
  // In order for reproducibility, we computer the new seed using RNG's random state and a different
  // set of parameters. Note that both 32767 and 1999999973 are prime numbers.
  return (support::LinearCongruentialEngine(&rand_state_)() * 32767) % 1999999973;
}

ExprRV ConcreteScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                               const Array<FloatImm>& probs,
                                               Optional<Integer> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV(tir::SampleCategorical(&this->rand_state_, candidates, probs, &decision));
  TVM_TIR_SCHEDULE_END("sample-categorical", this->error_render_level_);
  throw;
}

/******** Schedule: Get blocks & loops ********/

BlockRV ConcreteScheduleNode::GetBlock(const String& name, const String& func_name) {
  class NotSingleResult : public ScheduleError {
   public:
    explicit NotSingleResult(String name, IRModule mod, const Array<StmtSRef>& blocks)
        : name_(name), mod_(mod), blocks_{} {
      blocks_.reserve(blocks.size());
      for (const StmtSRef& block_sref : blocks) {
        const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
        blocks_.push_back(GetRef<Block>(block));
      }
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {blocks_.begin(), blocks_.end()}; }

    String DetailRenderTemplate() const final {
      if (blocks_.empty()) {
        return "Cannot find a block with the name: " + name_;
      } else {
        return "Found  " + std::to_string(blocks_.size()) + " blocks with the name: " + name_;
      }
    }

    String FastErrorString() const final {
      if (blocks_.empty()) {
        return "ScheduleError: Cannot find a block with the specified name";
      } else {
        return "ScheduleError: Found multiple blocks with the specified name";
      }
    }

    String name_;
    IRModule mod_;
    Array<Block> blocks_;
  };
  Array<StmtSRef> blocks = tir::GetBlocks(this->state_, name, func_name);
  if (blocks.size() != 1) {
    TVM_TIR_SCHEDULE_BEGIN();
    throw NotSingleResult(name, this->state_->mod, blocks);
    TVM_TIR_SCHEDULE_END("get-block", this->error_render_level_);
  }
  return CreateRV<BlockRV>(blocks[0]);
}

Array<LoopRV> ConcreteScheduleNode::GetLoops(const BlockRV& block_rv) {
  return CreateRV<LoopRV>(tir::GetLoops(this->GetSRef(block_rv)));
}

/******** Schedule: Transform loops ********/

LoopRV ConcreteScheduleNode::Fuse(const Array<LoopRV>& loop_rvs) {
  CHECK(!loop_rvs.empty()) << "ValueError: 'fuse' requires at least 1 loop(s)";
  Array<StmtSRef> loop_srefs = this->GetSRefs(loop_rvs);
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::Fuse(state_, loop_srefs);
  TVM_TIR_SCHEDULE_END("fuse", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(result);
}

Array<LoopRV> ConcreteScheduleNode::Split(const LoopRV& loop_rv,
                                          const Array<Optional<ExprRV>>& factor_rvs) {
  class NotSingleInferFactorError : public ScheduleError {
   public:
    explicit NotSingleInferFactorError(IRModule mod) : mod_(mod) {}

    String FastErrorString() const final {
      return "ScheduleError: only one factor can be specified as -1 or none";
    }

    String DetailRenderTemplate() const final {
      return "Only one factor can be specified as -1 or none";
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {}; }

    IRModule mod_;
  };

  class WrongFactorProductError : public ScheduleError {
   public:
    explicit WrongFactorProductError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

    String FastErrorString() const final {
      return "ScheduleError: The product of factors is not larger than or equal to the extent of "
             "loop";
    }

    String DetailRenderTemplate() const final {
      return "The product of factors is not larger than or equal to the extent of loop {0}";
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

    IRModule mod_;
    For loop_;
  };
  // Prepare for the splitting
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  Array<PrimExpr> factors;
  factors.reserve(factor_rvs.size());
  int infer_index = -1;
  PrimExpr tot_length = 1;
  Array<StmtSRef> results;
  TVM_TIR_SCHEDULE_BEGIN();
  // infer factor if needed and check validity of factors
  for (size_t i = 0; i < factor_rvs.size(); i++) {
    if (!factor_rvs[i].defined()) {
      factors.push_back(Integer(-1));
      if (infer_index == -1) {
        infer_index = i;
      } else {
        throw NotSingleInferFactorError(state_->mod);
      }
    } else {
      PrimExpr factor = this->Get(factor_rvs[i].value());
      factors.push_back(factor);
      tot_length *= factor;
    }
  }
  if (infer_index != -1) {
    factors.Set(infer_index,
                this->analyzer_->Simplify(floordiv(loop->extent + tot_length - 1, tot_length)));
  } else if (!this->analyzer_->CanProve(tot_length >= loop->extent)) {
    throw WrongFactorProductError(state_->mod, GetRef<For>(loop));
  }
  results = tir::Split(state_, loop_sref, factors);
  TVM_TIR_SCHEDULE_END("split", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(results);
}

void ConcreteScheduleNode::Reorder(const Array<LoopRV>& ordered_loop_rvs) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Reorder(state_, GetSRefs(ordered_loop_rvs));
  TVM_TIR_SCHEDULE_END("reorder", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Manipulate ForKind ********/

void ConcreteScheduleNode::Parallel(const LoopRV& loop_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Parallel(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("parallel", this->error_render_level_);
}

void ConcreteScheduleNode::Vectorize(const LoopRV& loop_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Vectorize(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("vectorize", this->error_render_level_);
}

void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const String& thread_axis) {
  if (thread_axis == "vthread") {
    LOG(WARNING) << "`vthread` is legacy behavior and is going to be deprecated. Please use "
                    "`vthread.x`, `vthread.y` and `vthread.z` instead";
  }
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Bind(state_, this->GetSRef(loop_rv),
            IterVar(/*dom=*/Range(nullptr), /*var=*/Var(thread_axis), /*iter_type=*/kThreadIndex,
                    /*thread_tag=*/thread_axis));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("bind", this->error_render_level_);
}

void ConcreteScheduleNode::Unroll(const LoopRV& loop_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Unroll(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("unroll", this->error_render_level_);
}

/******** Schedule: Insert cache stages ********/

BlockRV ConcreteScheduleNode::CacheRead(const BlockRV& block_rv, int read_buffer_index,
                                        const String& storage_scope) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::CacheRead(state_, this->GetSRef(block_rv), read_buffer_index, storage_scope);
  TVM_TIR_SCHEDULE_END("cache-read", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::CacheWrite(const BlockRV& block_rv, int write_buffer_index,
                                         const String& storage_scope) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::CacheWrite(state_, this->GetSRef(block_rv), write_buffer_index, storage_scope);
  TVM_TIR_SCHEDULE_END("cache-write", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

/******** Schedule: Compute location ********/

void ConcreteScheduleNode::ComputeInline(const BlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::ComputeInline(state_, this->GetSRef(block_rv));
  TVM_TIR_SCHEDULE_END("compute-inline", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::ReverseComputeInline(state_, this->GetSRef(block_rv));
  TVM_TIR_SCHEDULE_END("reverse-compute-inline", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Block Annotation ********/

void ConcreteScheduleNode::StorageAlign(const BlockRV& block_rv, int buffer_index, int axis,
                                        int factor, int offset) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::StorageAlign(state_, this->GetSRef(block_rv), buffer_index, axis, factor, offset);
  TVM_TIR_SCHEDULE_END("storage-align", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Reduction ********/

BlockRV ConcreteScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::RFactor(state_, this->GetSRef(loop_rv), factor_axis);
  TVM_TIR_SCHEDULE_END("rfactor", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

/******** Schedule: Blockize & Tensorize ********/
/******** Schedule: Annotation ********/
/******** Schedule: Misc ********/

}  // namespace tir
}  // namespace tvm
