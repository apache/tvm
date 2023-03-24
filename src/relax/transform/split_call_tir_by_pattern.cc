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
/*!
 * \file src/relax/transform/to_non_dataflow.cc
 * \brief Transform all dataflow structure to non-dataflow version.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/tir_pattern.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "../../tir/schedule/ir_comparator.h"

namespace tvm {

static const constexpr char* kLibraryKernel = "library_kernel";
static const constexpr char* kCSource = "c_source";
static const constexpr char* kCSourceFmt = "c_source_fmt";
static const constexpr char* kCSourceFmtCuda = "cu";

namespace tir {

using relax::FCodegen;
using relax::MatchResult;
using relax::TIRPattern;

/*! \brief helper to match a for stmt to a pattern*/
class ForMatcher : public TensorizeComparator {
 public:
  using SymbolMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
  explicit ForMatcher(const tir::PrimFunc& pattern, const Array<Var>& pattern_vars)
      : TensorizeComparator(IRModule({{GlobalVar(""), pattern}}), false), pattern_(pattern) {
    for (const auto& pattern_var : pattern_vars) {
      this->pattern_vars_.insert(pattern_var);
    }
    this->evaluated_symbols.push_back(SymbolMap());
  }

  bool Match(const For& top) {
    const ForNode* pattern_top = pattern_->body.as<BlockRealizeNode>()->block->body.as<ForNode>();
    ICHECK(pattern_top) << "Invalid pattern function";
    if (!VisitStmt(top, GetRef<Stmt>(pattern_top))) {
      return false;
    }
    // Get evaluated symbols, buffers from the pattern.
    for (const auto& arg : pattern_->params) {
      auto it = pattern_->buffer_map.find(arg);
      if (it != pattern_->buffer_map.end()) {
        auto itt = rhs_buffer_map_.find((*it).second);
        ICHECK(itt != rhs_buffer_map_.end());
        evaluated_buffers.push_back(itt->second);
      }
    }
    return true;
  }

  std::vector<SymbolMap> evaluated_symbols;
  std::vector<Buffer> evaluated_buffers;

 private:
  using ExprComparator::VisitExpr_;

  Optional<PrimExpr> QueryEvaluatedSymbols(const Var& var) {
    for (const SymbolMap& symbol_map : evaluated_symbols) {
      auto it = symbol_map.find(var);
      if (it != symbol_map.end()) {
        return it->second;
      }
    }
    return NullOpt;
  }

  bool VisitExpr(const PrimExpr& lhs, const PrimExpr& rhs) final {
    if (const auto* op = rhs.as<VarNode>()) {
      if (pattern_vars_.count(GetRef<Var>(op))) {
        // special case for pattern vars
        const auto* lhs_ptr = lhs.as<VarNode>();
        if (lhs_ptr == nullptr) {
          if (lhs->IsInstance<tir::IntImmNode>() || lhs->IsInstance<tir::FloatImmNode>()) {
            Optional<PrimExpr> value = QueryEvaluatedSymbols(GetRef<Var>(op));
            if (value.defined()) {
              if (!analyzer_.CanProveEqual(lhs, value.value())) return false;
            } else {
              evaluated_symbols.back()[GetRef<Var>(op)] = lhs;
            }
            return true;
          } else {
            return false;
          }
        }
      }
    }
    // pattern_var * expr
    if (const auto* rhs_ptr = rhs.as<MulNode>()) {
      const auto* operand_a = rhs_ptr->a.as<VarNode>();
      const auto* operand_b = rhs_ptr->b.as<VarNode>();
      if (operand_a != nullptr && pattern_vars_.count(GetRef<Var>(operand_a))) {
        // pattern var is on the left
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->b);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_a)] = MakeConstScalar(rhs_ptr->b.dtype(), 1);
          return true;
        }
      }
      if (operand_b != nullptr && pattern_vars_.count(GetRef<Var>(operand_b))) {
        // pattern var is on the right
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->a);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_b)] = MakeConstScalar(rhs_ptr->a.dtype(), 1);
          return true;
        }
      }
    }
    // pattern_Var + expr
    if (const auto* rhs_ptr = rhs.as<AddNode>()) {
      const auto* operand_a = rhs_ptr->a.as<VarNode>();
      const auto* operand_b = rhs_ptr->b.as<VarNode>();
      if (operand_a != nullptr && pattern_vars_.count(GetRef<Var>(operand_a))) {
        // pattern var is on the left
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->b);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_a)] = MakeConstScalar(rhs_ptr->b.dtype(), 0);
          return true;
        }
      }
      if (operand_b != nullptr && pattern_vars_.count(GetRef<Var>(operand_b))) {
        // pattern var is on the right
        evaluated_symbols.push_back(SymbolMap());
        bool match = VisitExpr(lhs, rhs_ptr->a);
        SymbolMap symbol_map = std::move(evaluated_symbols.back());
        evaluated_symbols.pop_back();
        if (match) {
          evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
          evaluated_symbols.back()[GetRef<Var>(operand_b)] = MakeConstScalar(rhs_ptr->a.dtype(), 0);
          return true;
        }
      }
    }
    return TensorizeComparator::VisitExpr(lhs, rhs);
  }

  bool VisitExpr_(const tir::AddNode* add, const PrimExpr& other) final {
    const auto* rhs = other.as<AddNode>();
    if (rhs == nullptr) return false;
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(add->a, rhs->a) && VisitExpr(add->b, rhs->b);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(add->a, rhs->b) && VisitExpr(add->b, rhs->a);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    return false;
  }

  bool VisitExpr_(const tir::MulNode* mul, const PrimExpr& other) final {
    const auto* rhs = other.as<MulNode>();
    if (rhs == nullptr) return false;
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(mul->a, rhs->a) && VisitExpr(mul->b, rhs->b);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    {
      this->evaluated_symbols.push_back(SymbolMap());
      bool match = VisitExpr(mul->a, rhs->b) && VisitExpr(mul->b, rhs->a);
      SymbolMap symbol_map = std::move(evaluated_symbols.back());
      this->evaluated_symbols.pop_back();
      if (match) {
        this->evaluated_symbols.back().insert(symbol_map.begin(), symbol_map.end());
        return true;
      }
    }
    return false;
  }

  bool VisitExpr_(const tir::CallNode* call, const PrimExpr& other) final {
    const auto* rhs = other.as<CallNode>();
    if (rhs == nullptr) return false;
    const auto* lhs_op = call->op.as<OpNode>();
    const auto* rhs_op = rhs->op.as<OpNode>();
    if (lhs_op == nullptr || rhs_op == nullptr) return false;
    if (lhs_op->name != rhs_op->name) return false;
    if (call->args.size() != rhs->args.size()) return false;
    for (size_t i = 0; i < call->args.size(); ++i) {
      if (!VisitExpr(call->args[i], rhs->args[i])) return false;
    }
    return true;
  }

  bool VisitStmt_(const tir::ForNode* op, const Stmt& other) final {
    const auto* rhs = other.as<ForNode>();
    loop_stack_lhs_.push_back(GetRef<For>(op));
    loop_stack_rhs_.push_back(GetRef<For>(rhs));
    // The body of loop must be loop or BlockRealize
    if (!op->body->IsInstance<BlockRealizeNode>() && !op->body->IsInstance<ForNode>()) {
      return false;
    }
    if (!rhs->body->IsInstance<BlockRealizeNode>() && !rhs->body->IsInstance<ForNode>()) {
      return false;
    }
    // Build mapping between the loop vars
    if (!DefEqual(op->loop_var, rhs->loop_var)) return false;
    // Only handle the case where the loop start from 0
    if (!is_zero(op->min) || !is_zero(rhs->min)) return false;
    if (op->thread_binding.defined() || rhs->thread_binding.defined()) return false;
    if (op->kind != ForKind::kSerial || op->kind != rhs->kind) return false;
    if (!op->annotations.empty() || !rhs->annotations.empty()) return false;
    // Match the extents of loops
    if (!VisitExpr(op->extent, rhs->extent)) return false;
    return VisitStmt(op->body, rhs->body);
  }

  bool VisitStmt_(const tir::BlockNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockNode>();
    // Check block equality.
    // All iter vars and buffer regions including the order should match.
    // When checking iter vars, DefEqual is used to remap variables.
    if (!CompareArray(op->iter_vars, rhs->iter_vars, &ForMatcher::CompareIterVar)) {
      return false;
    }
    // disallow alloc buffers inside the block
    if (!op->alloc_buffers.empty() || !rhs->alloc_buffers.empty()) return false;
    if (!CompareArray(op->writes, rhs->writes, &ForMatcher::CompareBufferRegion)) {
      return false;
    }
    if (!CompareArray(op->reads, rhs->reads, &ForMatcher::CompareBufferRegion)) {
      return false;
    }
    // The body of the block has to be BufferStore
    if (!op->body->IsInstance<BufferStoreNode>() || !rhs->body->IsInstance<BufferStoreNode>()) {
      return false;
    }
    // Handle init block
    if (op->init.defined() && !rhs->init.defined()) return false;
    if (!op->init.defined() && rhs->init.defined()) return false;
    if (op->init.defined() && rhs->init.defined()) {
      if (!VisitStmt(op->init.value(), rhs->init.value())) return false;
    }
    return VisitStmt(op->body, rhs->body);
  }

  bool VisitStmt_(const BlockRealizeNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockRealizeNode>();
    // Only allow trivial bindings
    for (size_t i = 0; i < op->iter_values.size(); ++i) {
      if (!op->iter_values[i].same_as(loop_stack_lhs_[i]->loop_var)) return false;
    }
    for (size_t i = 0; i < rhs->iter_values.size(); ++i) {
      if (!rhs->iter_values[i].same_as(loop_stack_rhs_[i]->loop_var)) return false;
    }
    // Disallow predicates now
    if (!is_one(op->predicate) || !is_one(rhs->predicate)) return false;
    return VisitStmt(op->block, rhs->block);
  }

  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) {
    const auto* rhs = other.as<BufferStoreNode>();
    return CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
  }

  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) {
    const auto* rhs = other.as<BufferLoadNode>();
    return CompareBufferAccess(op, rhs);
  }

  bool CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
    if (lhs.same_as(rhs)) return true;
    auto it = rhs_buffer_map_.find(rhs);
    bool equal;
    if (it != rhs_buffer_map_.end()) {
      equal = (*it).second.same_as(lhs);
    } else {
      // Compare shape
      if (lhs->shape.size() != rhs->shape.size()) return false;
      for (size_t i = 0; i < lhs->shape.size(); ++i) {
        if (!VisitExpr(lhs->shape[i], rhs->shape[i])) return false;
      }
      // Remap both buffer itself and buffer data
      equal =
          DefEqual(lhs->data, rhs->data) && lhs->dtype == rhs->dtype && lhs.scope() == rhs.scope();
      if (equal) {
        rhs_buffer_map_[rhs] = lhs;
      }
    }
    return equal;
  }

  bool CompareBufferRegion(const BufferRegion& lhs, const BufferRegion& rhs) {
    if (!CompareBuffer(lhs->buffer, rhs->buffer)) {
      return false;
    }
    return CompareArray(lhs->region, rhs->region, &ForMatcher::CompareRange);
  }

  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs) {
    if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
    return CompareArray(lhs->indices, rhs->indices, &ForMatcher::VisitExpr);
  }

  template <typename T, typename Self, typename F>
  bool CompareArray(const Array<T>& lhs, const Array<T>& rhs, F Self::*cmp) {
    if (lhs.same_as(rhs)) return true;
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!(static_cast<Self*>(this)->*cmp)(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  arith::Analyzer analyzer_;
  std::vector<For> loop_stack_lhs_, loop_stack_rhs_;
  tir::PrimFunc pattern_;
  std::unordered_set<Var, ObjectHash, ObjectEqual> pattern_vars_;
};

/*! \brief Analyze the function and match it with a list of patterns */
class TIRPatternMatcher {
 public:
  static Array<MatchResult> Match(Array<TIRPattern> patterns, Stmt body) {
    TIRPatternMatcher matcher(patterns);
    matcher.OpMatternMatch(body);
    if (matcher.fail_) return {};
    return matcher.match_results_;
  }

 private:
  explicit TIRPatternMatcher(Array<TIRPattern> patterns) : patterns_(patterns) {}

  // Find an op that matches this block
  bool BlockPatternMatch(const For& top) {
    for (const TIRPattern& pattern : patterns_) {
      tir::PrimFunc pattern_func = pattern;
      Array<Var> pattern_symbolic_vars;
      int buffer_count = pattern_func->buffer_map.size();
      for (int i = buffer_count; i < static_cast<int>(pattern_func->params.size()); i++) {
        pattern_symbolic_vars.push_back(pattern_func->params[i]);
      }
      ForMatcher block_matcher(pattern_func, pattern_symbolic_vars);
      if (block_matcher.Match(top)) {
        // We have found a match
        Array<PrimExpr> symbol_values;
        for (int i = buffer_count; i < static_cast<int>(pattern_func->params.size()); i++) {
          symbol_values.push_back(block_matcher.evaluated_symbols.back()[pattern_func->params[i]]);
        }
        match_results_.push_back(
            MatchResult(pattern, symbol_values, block_matcher.evaluated_buffers));
        return true;
      }
    }
    // The block fails to match any pattern
    return false;
  }

  // For each block in the body, try to find its corresponding pattern one by one
  void OpMatternMatch(const Stmt& body) {
    Array<Stmt> blocks;
    if (body->IsInstance<ForNode>()) {
      // {for}
      blocks = {body};
    } else if (const SeqStmtNode* seq = body.as<SeqStmtNode>()) {
      blocks = seq->seq;
    } else {
      fail_ = true;
      return;
    }
    for (const Stmt& stmt : blocks) {
      const ForNode* loop = stmt.as<ForNode>();
      if (loop == nullptr || !BlockPatternMatch(GetRef<For>(loop))) {
        break;
      }
    }
    if (match_results_.empty()) {
      fail_ = true;
    }
  }
  /*! \brief Indicate whether we fail to match.*/
  bool fail_ = false;
  /*! \brief The patterns we match the target stmt to.*/
  Array<TIRPattern> patterns_;
  /*! \brief The results of the matching process.*/
  Array<MatchResult> match_results_;
};

/*! \brief helper class to partition a function into 2 parts. Return function information which we
 * can use to construct the two partitioned parts.*/
class FunctionPartitioner : public StmtExprVisitor {
 public:
  explicit FunctionPartitioner(int num_matched_ops) : num_matched_ops_(num_matched_ops) {}
  /*! \brief alloc_buffers for the first function */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs1;
  /*! \brief alloc_buffers for the second function */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs2;
  /*! \brief whether the current block is in the first function */
  Map<Block, Bool> block_partition;
  /*! \brief input buffers for the first function */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input1;
  /*! \brief input buffers for the second function */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input2;
  /*! \brief The output buffer for the first function, which is also the input buffer for the second
  function */
  Buffer intermediate_buffer;
  /*! \brief Indicate whether we have failed. If failed, we will not do any further analysis and
  directly return the original one. */
  bool fail = false;

 private:
  void VisitStmt_(const BlockNode* op) final {
    block_counter_++;
    bool is_matching_ = block_counter_ <= num_matched_ops_;
    if (block_counter_ == num_matched_ops_) {
      allocs1.erase(intermediate_buffer);
    }
    for (const auto& read : op->reads) {
      if (is_matching_) {
        input1.insert(read->buffer);
      } else {
        input2.insert(read->buffer);
      }
    }
    for (const auto& write : op->writes) {
      if (is_matching_) {
        allocs1.insert(write->buffer);
      } else if (allocs1.count(write->buffer)) {
        fail = true;
        return;
      } else {
        allocs2.insert(write->buffer);
      }
      if (is_matching_) {
        intermediate_buffer = write->buffer;
      } else {
        input2.insert(write->buffer);
      }
    }
    block_partition.Set(GetRef<Block>(op), Bool(is_matching_));
  }
  // The number of matched ops in the function
  size_t num_matched_ops_;
  size_t block_counter_ = 0;
};

/*! \brief remove parts according to block partition, and update the alloc_buffers for blocks */
class BlockRemover : public StmtExprMutator {
 public:
  static Stmt RemoveBlockByPartition(
      Stmt stmt, const Map<Block, Bool>& block_partition,
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& allocs,
      bool is_library_part) {
    BlockRemover remover(block_partition, allocs, is_library_part);
    return remover(stmt);
  }

 private:
  BlockRemover(const Map<Block, Bool>& block_partition,
               const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& allocs,
               bool is_library_part)
      : block_partition(block_partition), allocs_(allocs), is_library_part_(is_library_part) {}

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    ObjectPtr<BlockNode> n = make_object<BlockNode>(*block.operator->());
    if (op->name_hint != "root") {
      ICHECK(block_partition.count(GetRef<Block>(op)));
      bool block_is_library = block_partition[GetRef<Block>(op)]->value;
      if (!(is_library_part_ ^ block_is_library)) {
        n->body = block->body;
      } else {
        erased_ = true;
      }
    }
    Array<Buffer> alloc_buffers;
    for (const Buffer& b : block->alloc_buffers) {
      if (allocs_.count(b)) {
        alloc_buffers.push_back(b);
      }
    }
    n->alloc_buffers = alloc_buffers;
    return Block(n);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Array<Stmt> seq;
    for (const Stmt& s : op->seq) {
      Stmt new_s = VisitStmt(s);
      if (erased_) {
        erased_ = false;
      } else {
        seq.push_back(new_s);
      }
    }
    return SeqStmt::Flatten(seq);
  }

  bool erased_ = false;
  Map<Block, Bool> block_partition;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocs_;
  bool is_library_part_ = false;
};

/*!
 * \brief Split the input function into two functions, one for the library kernel and one for the
 * rest.
 * \param func The input function.
 * \param arg_partition The input arg for the functions after split.
 * \param patterns The patterns to match.
 * \param f_codegen The function to generate the code for the library kernel.
 * \return A pair of functions, the first one is the library kernel and the second one is the
 * rest.
 */
std::pair<PrimFunc, Optional<PrimFunc>> SplitFunctions(PrimFunc func,
                                                       std::vector<std::vector<int>>* arg_partition,
                                                       Array<TIRPattern> patterns,
                                                       FCodegen f_codegen) {
  // Step 1. Find the library kernel and the rest.
  Stmt body = func->body.as<BlockRealizeNode>()->block->body;
  Array<MatchResult> match_results =
      TIRPatternMatcher::Match(patterns, func->body.as<BlockRealizeNode>()->block->body);
  if (match_results.empty()) {
    return {func, NullOpt};
  }
  Array<ObjectRef> codegen_result = f_codegen(match_results);
  ICHECK(codegen_result.size() == 3);
  String library_code = Downcast<String>(codegen_result[0]);
  int num_matched_ops = Downcast<Integer>(codegen_result[1])->value;
  Array<Buffer> func1_args = Downcast<Array<Buffer>>(codegen_result[2]);
  if (num_matched_ops == 0) {
    return {func, NullOpt};
  }
  FunctionPartitioner partitioner(num_matched_ops);
  partitioner(body);
  if (partitioner.fail) {
    return {func, NullOpt};
  }
  bool has_second_func = false;
  for (const auto& pr : partitioner.block_partition) {
    if (!pr.second->value) {
      has_second_func = true;
      break;
    }
  }
  if (!has_second_func) {
    // No need to split the function.
    return {WithAttr(func, kLibraryKernel, library_code), NullOpt};
  }
  // Step 2. Split the function into two functions.
  Stmt body1 = BlockRemover::RemoveBlockByPartition(func->body, partitioner.block_partition,
                                                    partitioner.allocs1, true);
  Stmt body2 = BlockRemover::RemoveBlockByPartition(func->body, partitioner.block_partition,
                                                    partitioner.allocs2, false);
  // Step 3. Craft the first function.
  Array<Var> new_params1;
  std::vector<int> arg_partition1;
  ICHECK_LE(func1_args.size(), partitioner.input1.size());
  for (const auto& buffer : func1_args) {
    ICHECK(partitioner.input1.find(buffer) != partitioner.input1.end());
    for (size_t i = 0; i < func->params.size(); i++) {
      if (func->buffer_map[func->params[i]].same_as(buffer)) {
        new_params1.push_back(func->params[i]);
        arg_partition1.push_back(i);
        break;
      }
    }
  }
  arg_partition->push_back(arg_partition1);
  new_params1.push_back(Var("output", DataType::Handle()));
  Map<Var, Buffer> new_buffer_map1;
  for (const auto& kv : func->buffer_map) {
    if (partitioner.input1.count(kv.second)) {
      new_buffer_map1.Set(kv.first, kv.second);
    }
  }
  new_buffer_map1.Set(new_params1.back(), partitioner.intermediate_buffer);
  PrimFunc func1 = PrimFunc(new_params1, body1, func->ret_type, new_buffer_map1, func->attrs);
  func1 = WithAttr(func1, kLibraryKernel, library_code);
  // Step 4. Craft the second function.
  Array<Var> new_params2;
  std::vector<int> arg_partition2;
  new_params2.push_back(Var("input", DataType::Handle()));
  for (int i = 0; i < static_cast<int>(func->params.size()); i++) {
    Var param = func->params[i];
    if (partitioner.input2.count(func->buffer_map[param])) {
      new_params2.push_back(param);
      if (i != static_cast<int>(func->params.size()) - 1) {
        arg_partition2.push_back(i);
      }
    }
  }
  arg_partition->push_back(arg_partition2);
  Map<Var, Buffer> new_buffer_map2;
  new_buffer_map2.Set(new_params2[0], partitioner.intermediate_buffer);
  for (const auto& kv : func->buffer_map) {
    if (partitioner.input2.count(kv.second)) {
      new_buffer_map2.Set(kv.first, kv.second);
    }
  }
  PrimFunc func2 = PrimFunc(new_params2, body2, func->ret_type, new_buffer_map2, func->attrs);
  return {func1, func2};
}
}  // namespace tir

namespace relax {
void StringReplace(std::string* subject, const std::string& search, const std::string& replace) {
  for (size_t pos = 0; (pos = subject->find(search, pos)) != std::string::npos;
       pos += replace.length()) {
    subject->replace(pos, search.length(), replace);
  }
}

tvm::BaseFunc CodegenWithLibrary(const tir::PrimFuncNode* pf, String global_symbol) {
  using namespace tvm::tir;
  Optional<runtime::String> library_code = pf->attrs.GetAttr<runtime::String>(kLibraryKernel);
  if (!library_code.defined()) {
    return GetRef<tir::PrimFunc>(pf);
  }
  std::string source = library_code.value();
  StringReplace(&source, "{global_symbol}", global_symbol);
  ExternFunc ret(global_symbol);
  ret = WithAttrs(std::move(ret), Map<String, ObjectRef>{
                                      {String(kCSource), String(source)},
                                      {String(kCSourceFmt), String(kCSourceFmtCuda)},
                                  });
  return ret;
}

/*! \brief Emit 2 calls to the library kernel and the rest of the function. */
class SplitMutator : public ExprMutator {
 public:
  SplitMutator(const tvm::IRModule& mod, Array<TIRPattern> patterns, FCodegen fcodegen)
      : ExprMutator(mod), mod_(mod), patterns_(patterns), fcodegen_(fcodegen) {}
  static IRModule Transform(const IRModule& mod, Array<TIRPattern> patterns, FCodegen fcodegen) {
    SplitMutator mutator(mod, patterns, fcodegen);
    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionNode>()) {
        Function new_func = Downcast<Function>(mutator(GetRef<Function>(func)));
        mutator.builder_->UpdateFunction(kv.first, new_func);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  using ExprMutator::VisitExpr_;

  inline Array<Expr> GetCallTIRArgs(Expr args) {
    if (args.as<TupleNode>()) {
      return args.as<TupleNode>()->fields;
    } else {
      return {args};
    }
  }

  Expr VisitExpr_(const CallNode* op) final {
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(op));
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    static const Op& call_dps_packed_ = Op::Get("relax.call_dps_packed");
    if (!call->op.same_as(call_tir_op_)) return call;
    // the first argument is the function to be called
    const auto* gv_ptr = call->args[0].as<GlobalVarNode>();
    if (gv_ptr == nullptr) return call;
    GlobalVar gv = GetRef<GlobalVar>(gv_ptr);
    // retrieve the function from the module and split it
    tir::PrimFunc func = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
    std::vector<std::vector<int>> arg_partition;
    // split the function into two functions, one for the library kernel and one for the rest.
    std::pair<tir::PrimFunc, Optional<tir::PrimFunc>> split_funcs =
        tir::SplitFunctions(func, &arg_partition, patterns_, fcodegen_);
    if (!split_funcs.second.defined()) {
      // no need to split, the function itself a library kernel
      tvm::BaseFunc lib_func = CodegenWithLibrary(split_funcs.first.get(), gv->name_hint);
      if (lib_func->IsInstance<tir::PrimFuncNode>()) return GetRef<Call>(op);
      // Update the function in the module with the library kernel
      ICHECK(lib_func->IsInstance<ExternFuncNode>());
      builder_->UpdateFunction(gv, lib_func);
      // emit the call to the library kernel
      ObjectPtr<CallNode> new_call = make_object<CallNode>(*call.operator->());
      new_call->op = this->call_dps_packed_;
      new_call->args = {lib_func, call->args[1]};
      return Call(new_call);
    }
    tir::PrimFunc func1 = tir::RenewDefs(split_funcs.first);
    tir::PrimFunc func2 = tir::RenewDefs(split_funcs.second.value());
    ICHECK(arg_partition.size() == 2);
    // emit the first call to the library kernel
    Array<Expr> args1;
    for (int p : arg_partition[0]) {
      args1.push_back(GetCallTIRArgs(call->args[1])[p]);
    }
    // replace the function in the module with the library kernel
    tvm::BaseFunc lib_func = CodegenWithLibrary(func1.get(), gv->name_hint);
    if (lib_func->IsInstance<tir::PrimFuncNode>()) return GetRef<Call>(op);
    ICHECK(lib_func->IsInstance<ExternFuncNode>());
    builder_->UpdateFunction(gv, lib_func);
    tir::Buffer intermediate_buffer = func1->buffer_map.at(func1->params.back());
    DataType dtype = intermediate_buffer->dtype;
    Call call1(call_dps_packed_, {lib_func, Tuple(args1)}, call->attrs,
               {TensorStructInfo(ShapeExpr(intermediate_buffer->shape), dtype)});
    Var call_var1 = builder_->Emit(call1);
    // emit the second call to the rest of the function
    Array<Expr> args2;
    args2.push_back(call_var1);
    for (int p : arg_partition[1]) {
      args2.push_back(GetCallTIRArgs(call->args[1])[p]);
    }
    GlobalVar gv2 = builder_->AddFunction(func2, "unfused_epilogue");
    Call call2(call_tir_op_, {gv2, Tuple(args2)}, call->attrs, call->sinfo_args);
    builder_->UpdateFunction(gv, WithoutAttr(func, "global_symbol"));
    return call2;
  }

  const Op& call_dps_packed_ = Op::Get("relax.call_dps_packed");
  tvm::IRModule mod_;
  Array<TIRPattern> patterns_;
  FCodegen fcodegen_;
};

namespace transform {
Pass SplitCallTIRByPattern(Array<TIRPattern> patterns, FCodegen fcodegen) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return SplitMutator::Transform(m, patterns, fcodegen); };
  return CreateModulePass(/*pass_function=*/pass_func,            //
                          /*opt_level=*/0,                        //
                          /*pass_name=*/"SplitCallTIRByPattern",  //
                          /*required=*/{});
}
TVM_REGISTER_GLOBAL("relax.transform.SplitCallTIRByPattern").set_body_typed(SplitCallTIRByPattern);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
