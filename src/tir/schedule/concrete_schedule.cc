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
                            int debug_mask, ScheduleErrorRenderLevel error_render_level,
                            bool enable_check) {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mask, enable_check);
  n->error_render_level_ = error_render_level;
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  n->Seed(seed);
  GlobalVar gv = NullValue<GlobalVar>();
  if (FindEntryFunc(mod, &gv) != nullptr) {
    n->func_working_on_ = gv;
  } else {
    n->func_working_on_ = NullOpt;
  }
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
    n->enable_check = src_state->enable_check;
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

void ConcreteScheduleNode::WorkOn(const String& func_name) {
  this->func_working_on_ = this->state_->mod->GetGlobalVar(func_name);
}

void ConcreteScheduleNode::Copy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const {
  ScheduleCopier::Copy(this, new_state, new_symbol_table);
  new_state->get()->DebugVerify();
}

Schedule ConcreteScheduleNode::Copy() {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->func_working_on_ = this->func_working_on_;
  n->error_render_level_ = this->error_render_level_;
  ConcreteScheduleNode::Copy(&n->state_, &n->symbol_table_);
  n->analyzer_ = std::make_unique<arith::Analyzer>();  // new analyzer needed because it is stateful
  n->rand_state_ = ForkSeed();
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
#define TVM_TIR_SCHEDULE_END(primitive, level)                                                \
  }                                                                                           \
  catch (const ScheduleError& error) {                                                        \
    if ((level) == ScheduleErrorRenderLevel::kDetail) {                                       \
      throw tvm::runtime::Error(error.RenderReport(primitive) + "\n" + runtime::Backtrace()); \
    } else if ((level) == ScheduleErrorRenderLevel::kFast) {                                  \
      throw tvm::runtime::Error(error.FastErrorString());                                     \
    } else if ((level) == ScheduleErrorRenderLevel::kNone) {                                  \
      throw tvm::runtime::Error("ScheduleError: (not rendered)");                             \
    }                                                                                         \
  }

/******** Schedule: Schedule: Sampling ********/

void ConcreteScheduleNode::Seed(support::LinearCongruentialEngine::TRandState seed) {
  this->rand_state_ = support::LinearCongruentialEngine::NormalizeSeed(seed);
}

support::LinearCongruentialEngine::TRandState ConcreteScheduleNode::ForkSeed() {
  return support::LinearCongruentialEngine(&rand_state_).ForkSeed();
}

ExprRV ConcreteScheduleNode::SampleCategorical(const Array<runtime::Int>& candidates,
                                               const Array<runtime::Float>& probs,
                                               Optional<runtime::Int> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV(tir::SampleCategorical(&this->rand_state_, candidates, probs, &decision));
  TVM_TIR_SCHEDULE_END("sample-categorical", this->error_render_level_);
  throw;
}

Array<ExprRV> ConcreteScheduleNode::SamplePerfectTile(const LoopRV& loop_rv, int n,
                                                      int max_innermost_factor,
                                                      Optional<Array<Integer>> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  // use None RV object to denotes auto-infer tile factors.
  return CreateRV(tir::SamplePerfectTile(&this->rand_state_, this->GetSRef(loop_rv), n,
                                         max_innermost_factor, &decision),
                  /*convert_negone_to_none=*/true);
  TVM_TIR_SCHEDULE_END("sample-perfect-tile", this->error_render_level_);
  throw;
}

Array<ExprRV> ConcreteScheduleNode::SamplePartitionedTile(const LoopRV& loop_rv, int n,
                                                          int partition_pos, int innerpart_factor,
                                                          Optional<Array<Integer>> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV(tir::SamplePartitionedTile(&this->rand_state_, this->GetSRef(loop_rv), n,
                                             partition_pos, innerpart_factor, &decision));
  TVM_TIR_SCHEDULE_END("sample-partitioned-tile", this->error_render_level_);
  throw;
}

LoopRV ConcreteScheduleNode::SampleComputeLocation(const BlockRV& block_rv,
                                                   Optional<Integer> decision) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<LoopRV>(
      tir::SampleComputeLocation(state_, &this->rand_state_, this->GetSRef(block_rv), &decision));
  TVM_TIR_SCHEDULE_END("sample-compute-location", this->error_render_level_);
  throw;
}

/******** Schedule: Get blocks & loops ********/

BlockRV ConcreteScheduleNode::GetBlock(const String& name, const Optional<String>& func_name) {
  class NotSingleResult : public ScheduleError {
   public:
    explicit NotSingleResult(String name, IRModule mod, const Array<StmtSRef>& blocks)
        : name_(name), mod_(mod), blocks_{} {
      blocks_.reserve(blocks.size());
      for (const StmtSRef& block_sref : blocks) {
        const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
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
  GlobalVar gv = NullValue<GlobalVar>();
  if (func_name.defined()) {
    gv = state_->mod->GetGlobalVar(func_name.value());
  } else if (func_working_on_.defined()) {
    gv = this->func_working_on_.value();
  } else {
    LOG(FATAL) << "ValueError: `get_block` does not know which function to be working on. Please "
                  "specify the function name explicitly, or call `work_on` to specify the function "
                  "before using `get_block`.";
  }
  Array<StmtSRef> blocks = tir::GetBlocks(this->state_, name, gv);
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

Array<BlockRV> ConcreteScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  Array<BlockRV> result;
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<BlockRV>(tir::GetChildBlocks(state_, this->GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("get-child-blocks", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

Array<BlockRV> ConcreteScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  Array<BlockRV> result;
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<BlockRV>(tir::GetChildBlocks(state_, this->GetSRef(loop_rv)));
  TVM_TIR_SCHEDULE_END("get-child-blocks", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

Array<BlockRV> ConcreteScheduleNode::GetProducers(const BlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<BlockRV>(tir::GetProducers(state_, this->GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("get-producers", this->error_render_level_);
  throw;
}

Array<BlockRV> ConcreteScheduleNode::GetConsumers(const BlockRV& block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<BlockRV>(tir::GetConsumers(state_, this->GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("get-consumers", this->error_render_level_);
  throw;
}

Array<BlockRV> ConcreteScheduleNode::GetOutputBlocks(const BlockRV& scope_block_rv) {
  TVM_TIR_SCHEDULE_BEGIN();
  return CreateRV<BlockRV>(tir::GetOutputBlocks(state_, this->GetSRef(scope_block_rv)));
  TVM_TIR_SCHEDULE_END("get-output-blocks", this->error_render_level_);
  throw;
}

/******** Schedule: Transform loops ********/

LoopRV ConcreteScheduleNode::Merge(const Array<LoopRV>& loop_rvs) {
  CHECK(loop_rvs.size() > 1) << "ValueError: 'merge' requires at least 2 loop(s)";
  Array<StmtSRef> loop_srefs = this->GetSRefs(loop_rvs);
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::Merge(state_, loop_srefs);
  TVM_TIR_SCHEDULE_END("merge", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(result);
}

LoopRV ConcreteScheduleNode::Fuse(const Array<LoopRV>& loop_rvs, bool preserve_unit_iters) {
  CHECK(!loop_rvs.empty()) << "ValueError: 'fuse' requires at least 1 loop(s)";
  Array<StmtSRef> loop_srefs = this->GetSRefs(loop_rvs);
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::Fuse(state_, loop_srefs, preserve_unit_iters);
  TVM_TIR_SCHEDULE_END("fuse", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(result);
}

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

class WrongFactorError : public ScheduleError {
 public:
  explicit WrongFactorError(IRModule mod, For loop, bool product)
      : mod_(mod), loop_(std::move(loop)), product_(product) {}

  String FastErrorString() const final {
    if (product_)
      return "ScheduleError: The product of factors is not larger than or equal to the extent of "
             "loop";
    else
      return "ScheduleError: The sum of factors is larger than or equal to the extent of loop";
  }

  String DetailRenderTemplate() const final {
    if (product_)
      return "The product of factors is not larger than or equal to the extent of loop {0}";
    else
      return "The sum of factors is larger than or equal to the extent of loop {0}";
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

  IRModule mod_;
  For loop_;
  bool product_;
};

class NonPositiveFactorError : public ScheduleError {
 public:
  explicit NonPositiveFactorError(IRModule mod, int64_t factor, size_t idx)
      : mod_(std::move(mod)), factor_(factor), idx_(idx) {}

  String FastErrorString() const final {
    return "ScheduleError: All the constant factors are required to be positive. However, some "
           "constant input factor is zero or negative.";
  }
  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "All the constant factors are required to be positive. However, the factor at position "
       << idx_ << " is " << factor_;
    return os.str();
  }
  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {}; }

 private:
  IRModule mod_;
  int64_t factor_;
  size_t idx_;
};

Array<LoopRV> ConcreteScheduleNode::Split(const LoopRV& loop_rv,
                                          const Array<Optional<ExprRV>>& factor_rvs,
                                          bool preserve_unit_iters, bool disable_predication) {
  // Prepare for the splitting
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
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
      if (infer_index != -1) {
        throw NotSingleInferFactorError(state_->mod);
      }
      infer_index = i;
    } else {
      PrimExpr factor = this->Get(factor_rvs[i].value());
      if (is_const_int(factor) && !is_positive_const(factor)) {
        throw NonPositiveFactorError(state_->mod, factor.as<IntImmNode>()->value, i);
      }
      if (factor.dtype().bits() > loop->extent.dtype().bits()) {
        factor = cast(loop->extent.dtype(), factor);
      }
      factors.push_back(factor);
      tot_length *= factor;
    }
  }
  if (infer_index != -1) {
    factors.Set(infer_index,
                this->analyzer_->Simplify(floordiv(loop->extent + tot_length - 1, tot_length)));
  } else if (!this->analyzer_->CanProve(tot_length >= loop->extent)) {
    throw WrongFactorError(state_->mod, GetRef<For>(loop), true);
  }
  results = tir::Split(state_, loop_sref, factors, preserve_unit_iters, disable_predication);
  TVM_TIR_SCHEDULE_END("split", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(results);
}

Array<LoopRV> ConcreteScheduleNode::LoopPartition(const LoopRV& loop_rv,
                                                  const Array<Optional<ExprRV>>& factor_rvs,
                                                  bool preserve_unit_iters) {
  class SymbolicShapeError : public ScheduleError {
   public:
    explicit SymbolicShapeError(IRModule mod, For loop) : mod_(mod), loop_(std::move(loop)) {}

    String FastErrorString() const final {
      return "ScheduleError: The min and extent values of the loop are required to be known at "
             "compile time. However, dynamic shape has been detected.";
    }

    String DetailRenderTemplate() const final {
      return "Detected dynamic shape in either min or extent of a loop {0}";
    }

    IRModule mod() const final { return mod_; }
    Array<ObjectRef> LocationsOfInterest() const final { return {loop_}; }

    IRModule mod_;
    For loop_;
  };

  // Prepare for the loop_partitioning
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);
  Array<PrimExpr> factors;
  factors.reserve(factor_rvs.size());
  int infer_index = -1;
  PrimExpr tot_length = 0;
  Array<StmtSRef> results;
  TVM_TIR_SCHEDULE_BEGIN();
  if (!is_const_number(loop->min) || !is_const_number(loop->extent)) {
    throw SymbolicShapeError(state_->mod, GetRef<For>(loop));
  }
  // infer factor if needed and check validity of factors
  for (size_t i = 0; i < factor_rvs.size(); i++) {
    if (!factor_rvs[i].defined()) {
      factors.push_back(Integer(-1));
      if (infer_index != -1) {
        throw NotSingleInferFactorError(state_->mod);
      }
      infer_index = i;
    } else {
      PrimExpr factor = this->Get(factor_rvs[i].value());
      if (is_const_int(factor) && !is_positive_const(factor)) {
        throw NonPositiveFactorError(state_->mod, factor.as<IntImmNode>()->value, i);
      }
      if (factor.dtype().bits() > loop->extent.dtype().bits()) {
        factor = cast(loop->extent.dtype(), factor);
      }
      factors.push_back(factor);
      tot_length += factor;
    }
  }
  if (this->analyzer_->CanProve(tot_length >= loop->extent)) {
    throw WrongFactorError(state_->mod, GetRef<For>(loop), false);
  }
  if (infer_index != -1) {
    // if there is a 'None' in the factor list, 'None' becomes the difference between the extent and
    // the sum of the factors excluding 'None' specified in the partition directive.
    factors.Set(infer_index, loop->extent - tot_length);
  }
  for (size_t i = 1; i < factor_rvs.size(); i++) {
    factors.Set(i, factors[i] + factors[i - 1]);
  }
  if (infer_index == -1) {
    factors.push_back(loop->extent);
  }
  results = tir::LoopPartition(state_, loop_sref, factors, preserve_unit_iters);
  TVM_TIR_SCHEDULE_END("loop_partition", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<LoopRV>(results);
}

void ConcreteScheduleNode::Reorder(const Array<LoopRV>& ordered_loop_rvs) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Reorder(state_, GetSRefs(ordered_loop_rvs));
  TVM_TIR_SCHEDULE_END("reorder", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::ReorderBlockIterVar(const BlockRV& block_rv,
                                               const Array<Integer> new_order) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::ReorderBlockIterVar(state_, GetSRef(block_rv), new_order);
  TVM_TIR_SCHEDULE_END("reorder_block_iter_var", this->error_render_level_);
  this->state_->DebugVerify();
}

LoopRV ConcreteScheduleNode::AddUnitLoop(const BlockRV& block_rv) {
  LoopRV result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<LoopRV>(tir::AddUnitLoop(state_, GetSRef(block_rv)));
  TVM_TIR_SCHEDULE_END("add-unit-loop", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
}

LoopRV ConcreteScheduleNode::AddUnitLoop(const LoopRV& loop_rv) {
  LoopRV result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = CreateRV<LoopRV>(tir::AddUnitLoop(state_, GetSRef(loop_rv)));
  TVM_TIR_SCHEDULE_END("add-unit-loop", this->error_render_level_);
  this->state_->DebugVerify();
  return result;
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
  tir::Bind(state_, this->GetSRef(loop_rv), thread_axis);
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
                                        const String& storage_scope,
                                        const Array<BlockRV> consumer_blocks) {
  StmtSRef result{nullptr};
  // Create a new array of SRefs from the consumer block list.
  Array<StmtSRef> consumer_block_refs = {};
  for (BlockRV block : consumer_blocks) {
    consumer_block_refs.push_back(this->GetSRef(block));
  }
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::CacheRead(state_, this->GetSRef(block_rv), read_buffer_index, storage_scope,
                          consumer_block_refs);
  TVM_TIR_SCHEDULE_END("cache-read", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::CacheWrite(const BlockRV& block_rv, int write_buffer_index,
                                         const String& storage_scope,
                                         const Array<BlockRV> consumer_blocks) {
  StmtSRef result{nullptr};
  // Create a new array of SRefs from the consumer block list.
  Array<StmtSRef> consumer_block_refs = {};
  for (BlockRV block : consumer_blocks) {
    consumer_block_refs.push_back(this->GetSRef(block));
  }
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::CacheWrite(state_, this->GetSRef(block_rv), write_buffer_index, storage_scope,
                           consumer_block_refs);
  TVM_TIR_SCHEDULE_END("cache-write", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::ReindexCacheRead(const BlockRV& block_rv, int read_buffer_index,
                                               const String& storage_scope,
                                               const IndexMap& index_map) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::ReindexCacheRead(state_, this->GetSRef(block_rv), read_buffer_index, storage_scope,
                                 index_map);
  TVM_TIR_SCHEDULE_END("reverse-cache-read", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::ReindexCacheWrite(const BlockRV& block_rv, int write_buffer_index,
                                                const String& storage_scope,
                                                const IndexMap& index_map) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::ReindexCacheWrite(state_, this->GetSRef(block_rv), write_buffer_index,
                                  storage_scope, index_map);
  TVM_TIR_SCHEDULE_END("reverse-cache-write", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

Array<BlockRV> ConcreteScheduleNode::CacheInplace(const BlockRV& block_rv, int write_buffer_index,
                                                  const String& storage_scope) {
  Array<StmtSRef> results;
  TVM_TIR_SCHEDULE_BEGIN();
  results = tir::CacheInplace(state_, this->GetSRef(block_rv), write_buffer_index, storage_scope);
  TVM_TIR_SCHEDULE_END("cache-buffer", this->error_render_level_);
  this->state_->DebugVerify();
  Array<BlockRV> return_blocks;
  return_blocks.push_back(CreateRV<BlockRV>(results[0]));
  return_blocks.push_back(CreateRV<BlockRV>(results[1]));
  return return_blocks;
}

Array<BlockRV> ConcreteScheduleNode::CacheIndex(const BlockRV& block_rv,
                                                const String& storage_scope, int cse_thresh) {
  Array<StmtSRef> result;
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::CacheIndex(state_, this->GetSRef(block_rv), storage_scope, cse_thresh);
  TVM_TIR_SCHEDULE_END("cache-index", this->error_render_level_);
  this->state_->DebugVerify();
  Array<BlockRV> return_blocks;
  for (const StmtSRef& blockrv : result) {
    return_blocks.push_back(CreateRV<BlockRV>(blockrv));
  }
  return return_blocks;
}

BlockRV ConcreteScheduleNode::ReIndex(const BlockRV& block_rv, int buffer_index,
                                      BufferIndexType buffer_index_type) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::ReIndex(state_, this->GetSRef(block_rv), buffer_index, buffer_index_type);
  TVM_TIR_SCHEDULE_END("reindex", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

/******** Schedule: Data movement ********/

BlockRV ConcreteScheduleNode::ReadAt(const LoopRV& loop_rv, const BlockRV& block_rv,
                                     int read_buffer_index, const String& storage_scope) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::ReadAt(state_, this->GetSRef(loop_rv), this->GetSRef(block_rv), read_buffer_index,
                       storage_scope);
  TVM_TIR_SCHEDULE_END("read-at", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::WriteAt(const LoopRV& loop_rv, const BlockRV& block_rv,
                                      int write_buffer_index, const String& storage_scope) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::WriteAt(state_, this->GetSRef(loop_rv), this->GetSRef(block_rv), write_buffer_index,
                        storage_scope);
  TVM_TIR_SCHEDULE_END("write-at", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

/******** Schedule: Compute location ********/

void ConcreteScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                     bool preserve_unit_loops, int index) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    TVM_TIR_SCHEDULE_BEGIN();
    tir::ComputeInline(state_, this->GetSRef(block_rv));
    TVM_TIR_SCHEDULE_END("compute-at", this->error_render_level_);
  } else {
    TVM_TIR_SCHEDULE_BEGIN();
    tir::ComputeAt(state_, this->GetSRef(block_rv), loop_sref, preserve_unit_loops, index);
    TVM_TIR_SCHEDULE_END("compute-at", this->error_render_level_);
  }
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                            bool preserve_unit_loops, int index) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    TVM_TIR_SCHEDULE_BEGIN();
    tir::ReverseComputeInline(state_, this->GetSRef(block_rv));
    TVM_TIR_SCHEDULE_END("reverse-compute-at", this->error_render_level_);
  } else {
    TVM_TIR_SCHEDULE_BEGIN();
    tir::ReverseComputeAt(state_, this->GetSRef(block_rv), loop_sref, preserve_unit_loops, index);
    TVM_TIR_SCHEDULE_END("reverse-compute-at", this->error_render_level_);
  }
  this->state_->DebugVerify();
}

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

void ConcreteScheduleNode::SetScope(const BlockRV& block_rv, int buffer_index,
                                    const String& storage_scope) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::SetScope(state_, this->GetSRef(block_rv), buffer_index, storage_scope);
  TVM_TIR_SCHEDULE_END("set-scope", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::UnsafeSetDType(const BlockRV& block_rv, int buffer_index,
                                          const String& dtype) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::UnsafeSetDType(state_, this->GetSRef(block_rv), buffer_index, dtype);
  TVM_TIR_SCHEDULE_END("set-dtype", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Reduction ********/

BlockRV ConcreteScheduleNode::DecomposeReduction(const BlockRV& block_rv, const LoopRV& loop_rv) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::DecomposeReduction(state_, this->GetSRef(block_rv), this->GetSRef(loop_rv));
  TVM_TIR_SCHEDULE_END("decompose-reduction", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::RFactor(state_, this->GetSRef(loop_rv), factor_axis);
  TVM_TIR_SCHEDULE_END("rfactor", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

/******** Schedule: Blockize & Tensorize ********/
BlockRV ConcreteScheduleNode::Blockize(const LoopRV& loop_rv, bool preserve_unit_iters) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::Blockize(state_, this->GetSRef(loop_rv), preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("blockize", this->error_render_level_);
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::Blockize(const Array<BlockRV>& blocks, bool preserve_unit_iters) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::Blockize(state_, this->GetSRefs(blocks), preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("blockize", this->error_render_level_);
  return CreateRV<BlockRV>(result);
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin,
                                     bool preserve_unit_iters) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Tensorize(state_, this->GetSRef(loop_rv), tir::TensorIntrin::Get(intrin).value(),
                 preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("tensorize", this->error_render_level_);
}

void ConcreteScheduleNode::Tensorize(const BlockRV& block_rv, const String& intrin,
                                     bool preserve_unit_iters) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Tensorize(state_, this->GetSRef(block_rv), tir::TensorIntrin::Get(intrin).value(),
                 preserve_unit_iters);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("tensorize", this->error_render_level_);
}

/******** Schedule: Annotation ********/

ObjectRef ConcreteScheduleNode::CheckAndGetAnnotationValue(const ObjectRef& ann_val) {
  if (ann_val.as<StringObj>()) {
    return ann_val;
  }
  if (auto* runtime_int = ann_val.as<runtime::Int::ContainerType>()) {
    return IntImm(DataType::Int(32), runtime_int->value);
  } else if (auto* runtime_float = ann_val.as<runtime::Float::ContainerType>()) {
    return FloatImm(DataType::Float(32), runtime_float->value);
  } else if (auto* runtime_bool = ann_val.as<runtime::Bool::ContainerType>()) {
    return Bool(runtime_bool->value);
  }

  if (const auto* expr = ann_val.as<PrimExprNode>()) {
    ICHECK(!ann_val->IsInstance<StringImmNode>())
        << "TypeError: runtime::String is expected, but gets StringImm";
    return this->Get(GetRef<PrimExpr>(expr));
  }
  if (const auto* arr = ann_val.as<ArrayNode>()) {
    Array<ObjectRef> result;
    result.reserve(arr->size());
    for (size_t i = 0; i < arr->size(); i++) {
      result.push_back(CheckAndGetAnnotationValue(arr->at(i)));
    }
    return std::move(result);
  }
  if (const auto* dict = ann_val.as<MapNode>()) {
    Map<String, ObjectRef> result;
    for (auto it = dict->begin(); it != dict->end(); ++it) {
      const auto& key = it->first;
      auto value = CheckAndGetAnnotationValue(it->second);
      if (const StringImmNode* imm = key.as<StringImmNode>()) {
        result.Set(imm->value, value);
      } else if (key->IsInstance<StringObj>()) {
        result.Set(Downcast<String>(key), value);
      } else {
        LOG(FATAL) << "TypeError: annotation dict key expect to be String or StringImm";
      }
    }
    return std::move(result);
  }
  LOG(FATAL)
      << "TypeError: Only strings, integers, floats, ExprRVs and Arrays are supported for now, but "
      << "gets: " << ann_val->GetTypeKey();
  throw;
}

void ConcreteScheduleNode::Annotate(const LoopRV& loop_rv, const String& ann_key,
                                    const ObjectRef& ann_val) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Annotate(state_, this->GetSRef(loop_rv), ann_key, this->CheckAndGetAnnotationValue(ann_val));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("annotate", this->error_render_level_);
}

void ConcreteScheduleNode::Unannotate(const LoopRV& loop_rv, const String& ann_key) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Unannotate(state_, this->GetSRef(loop_rv), ann_key);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("unannotate", this->error_render_level_);
}

void ConcreteScheduleNode::Annotate(const BlockRV& block_rv, const String& ann_key,
                                    const ObjectRef& ann_val) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Annotate(state_, this->GetSRef(block_rv), ann_key,
                this->CheckAndGetAnnotationValue(ann_val));
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("annotate", this->error_render_level_);
}

void ConcreteScheduleNode::Unannotate(const BlockRV& block_rv, const String& ann_key) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::Unannotate(state_, this->GetSRef(block_rv), ann_key);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("unannotate", this->error_render_level_);
}

/******** Schedule: Layout transformation ********/
void ConcreteScheduleNode::TransformLayout(const BlockRV& block_rv, int buffer_index,
                                           BufferIndexType buffer_index_type,
                                           const IndexMap& index_map,
                                           const Optional<IndexMap>& pad_value,
                                           bool assume_injective_transform) {
  TVM_TIR_SCHEDULE_BEGIN();
  auto f_subst = [&](const Var& var) -> Optional<PrimExpr> {
    return Downcast<Optional<PrimExpr>>(symbol_table_.Get(var));
  };
  auto new_index_map = Substitute(index_map, f_subst);
  tir::TransformLayout(state_, this->GetSRef(block_rv), buffer_index, buffer_index_type,
                       new_index_map, pad_value, assume_injective_transform);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("transform_layout", this->error_render_level_);
}

void ConcreteScheduleNode::TransformBlockLayout(const BlockRV& block_rv,
                                                const IndexMap& index_map) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::TransformBlockLayout(state_, this->GetSRef(block_rv), index_map);
  this->state_->DebugVerify();
  TVM_TIR_SCHEDULE_END("transform_block_layout", this->error_render_level_);
}

void ConcreteScheduleNode::SetAxisSeparator(const BlockRV& block_rv, int buffer_index,
                                            BufferIndexType buffer_index_type,
                                            const Array<IntImm>& axis_separators) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::SetAxisSeparator(state_, this->GetSRef(block_rv), buffer_index, buffer_index_type,
                        axis_separators);
  TVM_TIR_SCHEDULE_END("set-axis-separator", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Padding ********/

BlockRV ConcreteScheduleNode::DecomposePadding(const BlockRV& block_rv, const LoopRV& loop_rv) {
  StmtSRef result{nullptr};
  TVM_TIR_SCHEDULE_BEGIN();
  result = tir::DecomposePadding(state_, this->GetSRef(block_rv), this->GetSRef(loop_rv));
  TVM_TIR_SCHEDULE_END("decompose-padding", this->error_render_level_);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

void ConcreteScheduleNode::PadEinsum(const BlockRV& block_rv, const Array<Integer>& padding) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::PadEinsum(state_, this->GetSRef(block_rv), padding);
  TVM_TIR_SCHEDULE_END("pad-einsum", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Buffer Transformation ********/

void ConcreteScheduleNode::RollingBuffer(const BlockRV& block_rv, int write_buffer_index) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::RollingBuffer(state_, this->GetSRef(block_rv), write_buffer_index);
  TVM_TIR_SCHEDULE_END("rolling-buffer", this->error_render_level_);
  this->state_->DebugVerify();
}

/******** Schedule: Misc ********/

void ConcreteScheduleNode::UnsafeHideBufferAccess(const BlockRV& block_rv, const String& buf_type,
                                                  const Array<IntImm>& buf_index_array) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::UnsafeHideBufferAccess(state_, this->GetSRef(block_rv), buf_type, buf_index_array);
  TVM_TIR_SCHEDULE_END("hide-buffer-access", this->error_render_level_);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::AnnotateBufferAccess(const BlockRV& block_rv, int buffer_index,
                                                BufferIndexType buffer_index_type,
                                                const IndexMap& index_map) {
  TVM_TIR_SCHEDULE_BEGIN();
  tir::AnnotateBufferAccess(state_, this->GetSRef(block_rv), buffer_index, buffer_index_type,
                            index_map);
  TVM_TIR_SCHEDULE_END("annotate-buffer-access", this->error_render_level_);
  this->state_->DebugVerify();
}

}  // namespace tir
}  // namespace tvm
