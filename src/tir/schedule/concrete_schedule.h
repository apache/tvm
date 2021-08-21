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
#ifndef TVM_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
#define TVM_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_

#include <memory>
#include <utility>

#include "./utils.h"

namespace tvm {
namespace tir {

class ConcreteScheduleNode : public ScheduleNode {
  friend class Schedule;
  friend class ScheduleCopier;

 public:
  using TSymbolTable = Map<ObjectRef, ObjectRef>;

 protected:
  /*! \brief The internal state of scheduling */
  ScheduleState state_;
  /*! \brief The level of error rendering */
  ScheduleErrorRenderLevel error_render_level_;
  /*! \brief A symbol table that maps random variables to concrete StmtSRef/Integers */
  TSymbolTable symbol_table_;
  /*! \brief A persistent stateless arithmetic analyzer. */
  std::unique_ptr<arith::Analyzer> analyzer_;
  /*! \brief The value of random state for sampling. */
  support::LinearCongruentialEngine::TRandState rand_state_;

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    // `state_` is not visited
    // `error_render_level_` is not visited
    // `symbol_table_` is not visited
    // `analyzer_` is not visited
    // `rand_state_` is not visited
  }

  virtual ~ConcreteScheduleNode() = default;

 public:
  ScheduleState state() const final { return state_; }
  Optional<Trace> trace() const override { return NullOpt; }
  Schedule Copy() const override;

 public:
  /******** Lookup random variables ********/
  inline Block Get(const BlockRV& block_rv) const final;
  inline For Get(const LoopRV& loop_rv) const final;
  inline PrimExpr Get(const ExprRV& expr_rv) const final;
  inline StmtSRef GetSRef(const BlockRV& block_rv) const final;
  inline StmtSRef GetSRef(const LoopRV& loop_rv) const final;
  inline Array<StmtSRef> GetSRefs(const Array<BlockRV>& rvs) const;
  inline Array<StmtSRef> GetSRefs(const Array<LoopRV>& rvs) const;
  void RemoveRV(const BlockRV& block_rv) final { RemoveFromSymbolTable(block_rv); }
  void RemoveRV(const LoopRV& loop_rv) final { RemoveFromSymbolTable(loop_rv); }
  void RemoveRV(const ExprRV& expr_rv) final { RemoveFromSymbolTable(expr_rv); }
  using ScheduleNode::GetSRef;

 public:
  /******** Schedule: Sampling ********/
  /*!
   * \brief Seed the randomness
   * \param seed The new random seed, -1 if use device random, otherwise non-negative
   */
  void Seed(support::LinearCongruentialEngine::TRandState seed = -1) final;
  /*! \brief Fork the random state */
  support::LinearCongruentialEngine::TRandState ForkSeed() final;
  /*!
   * \brief Sample an integer given the probability distribution
   * \param candidates The candidates
   * \param probs The probability distribution of the candidates
   * \param decision The sampling decision
   * \return The random variable sampled from candidates
   */
  ExprRV SampleCategorical(const Array<Integer>& candidates, const Array<FloatImm>& probs,
                           Optional<Integer> decision = NullOpt) override;
  /******** Schedule: Get blocks & loops ********/
  BlockRV GetBlock(const String& name, const String& func_name = "main") override;
  Array<LoopRV> GetLoops(const BlockRV& block_rv) override;
  /******** Schedule: Transform loops ********/
  LoopRV Fuse(const Array<LoopRV>& loop_rvs) override;
  Array<LoopRV> Split(const LoopRV& loop_rv, const Array<Optional<ExprRV>>& factors) override;
  void Reorder(const Array<LoopRV>& ordered_loop_rvs) override;
  /******** Schedule: Manipulate ForKind ********/
  void Parallel(const LoopRV& loop_rv) override;
  void Vectorize(const LoopRV& loop_rv) override;
  void Bind(const LoopRV& loop_rv, const String& thread_axis) override;
  void Unroll(const LoopRV& loop_rv) override;
  /******** Schedule: Insert cache stages ********/
  /******** Schedule: Compute location ********/
  void ComputeInline(const BlockRV& block) override;
  void ReverseComputeInline(const BlockRV& block) override;
  /******** Schedule: Reduction ********/
  BlockRV RFactor(const LoopRV& loop_rv, int factor_axis) override;
  /******** Schedule: Block annotation ********/
  void StorageAlign(const BlockRV& block_rv, int buffer_index, int axis, int factor,
                    int offset) override;
  /******** Schedule: Blockize & Tensorize ********/
  /******** Schedule: Annotation ********/
  /******** Schedule: Misc ********/
  void EnterPostproc() override {}

 protected:
  /******** Utility functions ********/
  /*!
   * \brief Copy the schedule state, as well as the symbol table
   * \param new_state The ScheduleState copied
   * \param new_symbol_table The symbol table copied
   */
  void Copy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const;
  /*!
   * \brief Add srefs as random variables into the symbol table
   * \tparam T The type of the random variables
   * \param srefs The srefs to be added to the symbol table
   * \return The new random variables created
   */
  template <class T>
  inline Array<T> CreateRV(const Array<StmtSRef>& srefs);
  /*!
   * \brief Add an sref as a random variable into the symbol table
   * \tparam T The type of the random variable
   * \param sref The sref to be added to the symbol table
   * \return The new random variable created
   */
  template <class T>
  inline T CreateRV(const StmtSRef& sref);
  /*!
   * \brief Add an expr as a random variable into the symbol table
   * \param expr The expr to be added to the symbol table
   * \return The new random variable created
   */
  inline ExprRV CreateRV(const PrimExpr& expr);
  /*!
   * \brief Add expr as random variables into the symbol table
   * \param exprs The expr to be added to the symbol table
   * \return The new random variables created
   */
  inline Array<ExprRV> CreateRV(const Array<PrimExpr>& exprs);
  /*! \brief Remove a random variable from the symbol table */
  inline void RemoveFromSymbolTable(const ObjectRef& rv);
};

// implementations

/******** Lookup random variables ********/

inline Block ConcreteScheduleNode::Get(const BlockRV& block_rv) const {
  StmtSRef sref = this->GetSRef(block_rv);
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, sref);
  return GetRef<Block>(block);
}

inline For ConcreteScheduleNode::Get(const LoopRV& loop_rv) const {
  StmtSRef sref = this->GetSRef(loop_rv);
  const ForNode* loop = TVM_SREF_TO_FOR(loop, sref);
  return GetRef<For>(loop);
}

inline PrimExpr ConcreteScheduleNode::Get(const ExprRV& expr_rv) const {
  PrimExpr transformed = Substitute(expr_rv, [this](const Var& var) -> Optional<PrimExpr> {
    auto it = this->symbol_table_.find(var);
    if (it == this->symbol_table_.end()) {
      LOG(FATAL) << "IndexError: Cannot find corresponding ExprRV: " << var;
    }
    const ObjectRef& obj = (*it).second;
    const auto* int_imm = TVM_TYPE_AS(int_imm, obj, IntImmNode);
    return Integer(int_imm->value);
  });
  return this->analyzer_->Simplify(transformed);
}

inline StmtSRef ConcreteScheduleNode::GetSRef(const BlockRV& block_rv) const {
  auto it = this->symbol_table_.find(block_rv);
  if (it == this->symbol_table_.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding BlockRV: " << block_rv;
  }
  const ObjectRef& obj = (*it).second;
  const auto* sref = obj.as<StmtSRefNode>();
  if (sref == nullptr) {
    LOG(FATAL) << "ValueError: BlockRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt == nullptr) {
    LOG(FATAL) << "ValueError: The StmtSRef has expired";
  }
  return GetRef<StmtSRef>(sref);
}

inline StmtSRef ConcreteScheduleNode::GetSRef(const LoopRV& loop_rv) const {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  auto it = this->symbol_table_.find(loop_rv);
  if (it == this->symbol_table_.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding LoopRV: " << loop_rv;
  }
  const ObjectRef& obj = (*it).second;
  if (obj.same_as(inline_mark)) {
    return inline_mark;
  }
  if (obj.same_as(root_mark)) {
    return root_mark;
  }
  const auto* sref = obj.as<StmtSRefNode>();
  if (sref == nullptr) {
    LOG(FATAL) << "ValueError: LoopRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt == nullptr) {
    LOG(FATAL) << "ValueError: The StmtSRef has expired";
  }
  return GetRef<StmtSRef>(sref);
}

template <class T>
inline Array<StmtSRef> GetSRefsHelper(const ConcreteScheduleNode* sch, const Array<T>& rvs) {
  Array<StmtSRef> result;
  result.reserve(rvs.size());
  for (const T& rv : rvs) {
    result.push_back(sch->GetSRef(rv));
  }
  return result;
}

inline Array<StmtSRef> ConcreteScheduleNode::GetSRefs(const Array<BlockRV>& rvs) const {
  return GetSRefsHelper(this, rvs);
}

inline Array<StmtSRef> ConcreteScheduleNode::GetSRefs(const Array<LoopRV>& rvs) const {
  return GetSRefsHelper(this, rvs);
}

/******** Adding/Removing elements in the symbol table ********/

template <class T>
inline Array<T> ConcreteScheduleNode::CreateRV(const Array<StmtSRef>& srefs) {
  Array<T> result;
  result.reserve(srefs.size());
  for (const StmtSRef& sref : srefs) {
    T rv;
    this->symbol_table_.Set(rv, sref);
    result.push_back(rv);
  }
  return result;
}

template <class T>
inline T ConcreteScheduleNode::CreateRV(const StmtSRef& sref) {
  T rv;
  this->symbol_table_.Set(rv, sref);
  return std::move(rv);
}

inline ExprRV ConcreteScheduleNode::CreateRV(const PrimExpr& expr) {
  ExprRV rv;
  this->symbol_table_.Set(rv, expr);
  return std::move(rv);
}

inline Array<ExprRV> ConcreteScheduleNode::CreateRV(const Array<PrimExpr>& exprs) {
  Array<ExprRV> result;
  result.reserve(exprs.size());
  for (const PrimExpr& expr : exprs) {
    ExprRV rv;
    this->symbol_table_.Set(rv, expr);
    result.push_back(rv);
  }
  return result;
}

inline void ConcreteScheduleNode::RemoveFromSymbolTable(const ObjectRef& obj) {
  auto it = this->symbol_table_.find(obj);
  if (it != this->symbol_table_.end()) {
    this->symbol_table_.erase(obj);
  } else {
    LOG(FATAL) << "IndexError: Cannot find the object in the symbol table: " << obj;
    throw;
  }
}

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_CONCRETE_SCHEDULE_H_
