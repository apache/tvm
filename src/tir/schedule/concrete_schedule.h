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

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    // `error_render_level_` is not visited
    // `state_` is not visited
    // `symbol_table_` is not visited
    // `analyzer_` is not visitied
  }

  virtual ~ConcreteScheduleNode() = default;

  static constexpr const char* _type_key = "tir.ConcreteSchedule";
  TVM_DECLARE_BASE_OBJECT_INFO(ConcreteScheduleNode, ScheduleNode);

 public:
  ScheduleState state() const final { return state_; }
  Schedule Copy() const override;

 public:
  /******** Lookup random variables ********/
  inline Block Get(const BlockRV& block_rv) const final;
  inline For Get(const LoopRV& loop_rv) const final;
  inline PrimExpr Get(const ExprRV& expr_rv) const final;
  inline StmtSRef GetSRef(const BlockRV& block_rv) const final;
  inline StmtSRef GetSRef(const LoopRV& loop_rv) const final;
  void RemoveRV(const BlockRV& block_rv) final { RemoveFromSymbolTable(block_rv); }
  void RemoveRV(const LoopRV& loop_rv) final { RemoveFromSymbolTable(loop_rv); }
  void RemoveRV(const ExprRV& expr_rv) final { RemoveFromSymbolTable(expr_rv); }
  using ScheduleNode::GetSRef;

 public:
  /******** Block/Loop relation ********/
  BlockRV GetBlock(const String& name, const String& func_name = "main") override;
  Array<LoopRV> GetLoops(const BlockRV& block_rv) override;
  /******** Schedule: loops manipulation ********/
  /******** Schedule: compute location ********/
  void ComputeInline(const BlockRV& block) override;
  void ReverseComputeInline(const BlockRV& block) override;
  /******** Schedule: loop binding/annotation ********/
  /******** Schedule: cache read/write ********/
  /******** Schedule: reduction ********/
  /******** Schedule: blockize & tensorize ********/

  /******** Utility functions ********/
 protected:
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
  const auto* block = TVM_SREF_TO_BLOCK(block, sref);
  return GetRef<Block>(block);
}

inline For ConcreteScheduleNode::Get(const LoopRV& loop_rv) const {
  StmtSRef sref = this->GetSRef(loop_rv);
  const auto* loop = TVM_SREF_TO_FOR(loop, sref);
  return GetRef<For>(loop);
}

inline PrimExpr ConcreteScheduleNode::Get(const ExprRV& expr_rv) const {
  auto it = this->symbol_table_.find(expr_rv);
  if (it == this->symbol_table_.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding ExprRV: " << expr_rv;
  }
  const ObjectRef& obj = (*it).second;
  const auto* expr_node = obj.as<PrimExprNode>();
  if (expr_node == nullptr) {
    LOG(FATAL) << "ValueError: ExprRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  return GetRef<PrimExpr>(expr_node);
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
