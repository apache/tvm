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

namespace tvm {
namespace tir {

Schedule Schedule::Concrete(IRModule mod, int debug_mode,
                            ScheduleErrorRenderLevel error_render_level) {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mode);
  n->error_render_level_ = error_render_level;
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
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
    n->debug_mode = src_state->debug_mode;
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
  this->Copy(&n->state_, &n->symbol_table_);
  n->analyzer_ = std::make_unique<arith::Analyzer>();
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

/******** Block/Loop relation ********/

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

/******** Schedule: loops manipulation ********/
/******** Schedule: compute location ********/

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

/******** Schedule: loop binding/annotation ********/
/******** Schedule: cache read/write ********/
/******** Schedule: reduction ********/
/******** Schedule: blockize & tensorize ********/

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(ConcreteScheduleNode);

}  // namespace tir
}  // namespace tvm
