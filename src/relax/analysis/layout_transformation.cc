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
 * \file relax/analysis/layout_transormation.cc
 * \brief Analyze the PrimFunc and suggest layout transformation on it's blocks and buffers based on
 * the user provided layout transformations on it's outputs.
 */
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>

#include <queue>

#include "../../tir/schedule/analysis.h"

namespace tvm {
namespace relax {

using namespace tir;

using IterSetTy = Optional<tir::Var>;
using ArrayIterSetTy = Array<IterSetTy>;
using VarSet = std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual>;

static inline Array<PrimExpr> RegionToIndices(const Region& region) {
  ICHECK(region.size() != 0);
  Array<PrimExpr> indices;
  for (const auto& r : region) {
    if (!is_one(r->extent)) return {};
    indices.push_back(r->min);
  }
  return indices;
}

template <typename T>
static inline Array<T> list_to_array(std::list<T> l) {
  Array<T> array;
  for (auto& v : l) {
    array.push_back(v);
  }
  return array;
}

template <typename T>
static inline std::list<T> array_to_list(Array<T> array) {
  std::list<T> l;
  for (auto& v : array) {
    l.push_back(v);
  }
  return l;
}

// Gather all tir::VarNodes in an expr
static inline void GatherVars(const PrimExpr& expr, std::unordered_set<const tir::VarNode*>* vars) {
  tir::PostOrderVisit(expr, [&vars](const ObjectRef& node) {
    if (const tir::VarNode* op = node.as<tir::VarNode>()) {
      vars->insert(op);
    }
  });
}
static inline bool HasUndefinedVars(const PrimExpr& expr, const VarSet& defs) {
  bool has_undefined_vars = false;
  tir::PostOrderVisit(expr, [&](const ObjectRef& node) {
    if (const tir::VarNode* op = node.as<tir::VarNode>()) {
      if (defs.count(GetRef<tir::Var>(op)) == 0) {
        has_undefined_vars = true;
      }
    }
  });
  return has_undefined_vars;
}

class IterMapAnalyzer : public ExprVisitor {
 public:
  Array<tir::Var> Analyze(const PrimExpr& expr) {
    VisitExpr(expr);
    return iterators_;
  }

 private:
  /*! \brief Override VisitExpr for iter expr type processing */
  void VisitExpr(const PrimExpr& expr) override {
    if (const auto* op = expr.as<arith::IterSumExprNode>()) {
      Visit_(GetRef<arith::IterSumExpr>(op));
      return;
    }
    if (const auto* op = expr.as<arith::IterSplitExprNode>()) {
      Visit_(GetRef<arith::IterSplitExpr>(op));
      return;
    }
    return ExprVisitor::VisitExpr(expr);
  }

  void Visit_(const arith::IterMark& op) {
    if (const auto* var = op->source.as<tir::VarNode>()) {
      iterators_.push_back(GetRef<tir::Var>(var));
      return;
    }
    VisitExpr(op->source);
    VisitExpr(op->extent);
  }
  void Visit_(const arith::IterSplitExpr& op) {
    Visit_(op->source);
    VisitExpr(op->lower_factor);
    VisitExpr(op->extent);
    VisitExpr(op->scale);
    return;
  }
  void Visit_(const arith::IterSumExpr& op) {
    for (const auto& arg : op->args) {
      Visit_(arg);
    }
    VisitExpr(op->base);
    return;
  }

 private:
  Array<tir::Var> iterators_;
};

class PrimFuncAnalyzer : public StmtExprVisitor {
 public:
  explicit PrimFuncAnalyzer(const PrimFunc& func, IndexMap write_transformation)
      : can_be_transformed_(true), write_transformation_(write_transformation) {
    for (const tir::Var& param : func->params) {
      Optional<Buffer> param_buf = func->buffer_map.Get(param);
      if (param_buf.defined()) {
        param_buffers_.insert(param_buf.value());
      }
    }
  }

 private:
  bool IterVarsToDomain(const Array<IterVar>& itervars) {
    for (const IterVar& v : itervars) {
      // Only support
      if (v->iter_type == kDataPar)
        spatial_dom_.Set(v->var, v->dom);
      else if (v->iter_type == kCommReduce)
        reduction_dom_.Set(v->var, v->dom);
      else
        return false;
    }
    return true;
  }

  // Helper to break down the indices of buffer access
  void DetectRegionIterMap(const BufferRegion& buffer_region, arith::Analyzer* analyzer) {
    auto indices = RegionToIndices(buffer_region->region);
    if (indices.empty()) {
      LOG(WARNING) << "Buffer: " << buffer_region
                   << " cannot be transformed because extent of indices is not 1";
      return;
    }
    auto result = arith::DetectIterMap(
        /*indices=*/indices, /*input_iters*/ spatial_dom_,
        /*predicate*/ tir::const_true(),
        /*check_level*/ arith::IterMapLevel::Surjective, /*analyzer*/ analyzer);
    if (result->indices.empty()) {
      LOG(WARNING) << "Buffer: " << buffer_region << " cannot be transformed because "
                   << result->errors;
      return;
    }
    auto [is_supported, iterators] = GetIteratorAccess(result);
    if (!is_supported) return;
    buffer_to_iter_map_result_[buffer_region] = std::move(iterators);
  }

  // Get the iterator access
  std::tuple<bool, ArrayIterSetTy> GetIteratorAccess(const arith::IterMapResult& iter_map_result) {
    ICHECK(!iter_map_result->indices.empty());
    ArrayIterSetTy result;
    for (const arith::IterSumExpr& index : iter_map_result->indices) {
      IterMapAnalyzer index_analyzer;
      auto iter_vars = index_analyzer.Analyze(index);
      if (iter_vars.size() >= 2) return {false, {}};
      if (iter_vars.empty()) {
        result.push_back({});
        continue;
      }
      // iter_vars.size() == 1
      result.push_back(iter_vars[0]);
    }
    return {true, result};
  }

  bool IsSequentialAccess(const ArrayIterSetTy& iterators,
                          const std::unordered_map<const tir::VarNode*, int>& iter_to_block_index) {
    int last_value = -1;
    for (const auto& i : iterators) {
      if (!i.defined()) continue;
      int blk_index = iter_to_block_index.at(i.value().get());
      if (blk_index <= last_value) return false;
      last_value = blk_index;
    }
    return true;
  }

  IndexMap GetTransformation(const ArrayIterSetTy& src_iterators,
                             const IndexMap& src_transformation,
                             const ArrayIterSetTy& tgt_iterators) {
    // Copy over the src transformation intial and final indices
    auto initial_indices = array_to_list(src_transformation->initial_indices);
    auto final_indices = array_to_list(src_transformation->final_indices);

    // Remove any indices on both sides
    VarSet tgt_var_set;
    for (const auto& i : tgt_iterators) {
      if (i.defined()) tgt_var_set.insert(i.value());
    }

    // Drop vars that do not occur in tgt iterators
    auto initial_indices_it = initial_indices.begin();
    VarSet defs;
    for (const auto& i : src_iterators) {
      ICHECK(i.defined());
      if (tgt_var_set.count(i.value())) {
        defs.insert(*initial_indices_it);
        initial_indices_it++;
        continue;
      }
      initial_indices_it = initial_indices.erase(initial_indices_it);
    }

    // Erase any expressions in final indices that have undefined vars
    auto final_indices_it = final_indices.begin();
    while (final_indices_it != final_indices.end()) {
      if (!HasUndefinedVars(*final_indices_it, defs)) {
        final_indices_it++;
        continue;
      }
      final_indices_it = final_indices.erase(final_indices_it);
    }

    VarSet src_var_set;
    for (const auto& i : src_iterators) {
      ICHECK(i.defined());
      src_var_set.insert(i.value());
    }

    initial_indices_it = initial_indices.begin();
    final_indices_it = final_indices.begin();
    for (const auto& i : tgt_iterators) {
      if (i.defined() && src_var_set.count(i.value())) {
        initial_indices_it++;
        if (final_indices_it != final_indices.end()) final_indices_it++;
        continue;
      }

      auto v = tir::Var("dim");
      initial_indices.insert(initial_indices_it, v);
      final_indices.insert(final_indices_it, v);
    }

    auto ret = IndexMap(list_to_array(initial_indices), list_to_array(final_indices));
    return ret;
  }
  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      // Skip the root block
      StmtVisitor::VisitStmt_(op);
      return;
    }
    // For now we can only transform functions with a single block.
    if (block_.defined()) {
      can_be_transformed_ = false;
      return;
    }

    Block block = GetRef<Block>(op);
    block_ = block;
    if (!IterVarsToDomain(block->iter_vars)) {
      can_be_transformed_ = false;
      return;
    }

    // Only handle blocks which write to a single param buffer.
    if (block->writes.size() != 1) {
      can_be_transformed_ = false;
    }

    // All the buffer access are quasi-affine expressions on spatial domain.
    tvm::arith::Analyzer analyzer;
    for (const auto& w : block->writes) DetectRegionIterMap(w, &analyzer);
    if (can_be_transformed_ == false) return;
    for (const auto& r : block->reads) DetectRegionIterMap(r, &analyzer);

    // Get loop ordering in block iterators
    std::unordered_map<const tir::VarNode*, IterVarType> iter_var_to_type;
    int index = 0;
    for (const auto& iter_var : block->iter_vars) {
      iter_var_to_block_index_[iter_var->var.get()] = index++;
      iter_var_to_type[iter_var->var.get()] = iter_var->iter_type;
    }

    // Map from write indices to blk indices
    // Write has sequential access
    // This would mean that the mapping between spatial dom vars and their use in the buffer
    // access is sequential.
    auto write_region = block->writes[0];
    auto it = buffer_to_iter_map_result_.find(write_region);
    if (it == buffer_to_iter_map_result_.end()) {
      can_be_transformed_ = false;
      return;
    }
    auto write_access_iterators = it->second;
    if (!IsSequentialAccess(write_access_iterators, iter_var_to_block_index_)) {
      can_be_transformed_ = false;
      return;
    }
    // Propose transformation of block
    ArrayIterSetTy block_iterators;
    for (const auto& i : block->iter_vars) {
      block_iterators.push_back(i->var);
    }
    block_transformation_ =
        GetTransformation(write_access_iterators, write_transformation_, block_iterators);

    // Map from read indices to blk indices
    for (const auto& r : block->reads) {
      auto it = buffer_to_iter_map_result_.find(r);
      if (it == buffer_to_iter_map_result_.end()) continue;
      auto read_access_iterators = it->second;
      if (!IsSequentialAccess(read_access_iterators, iter_var_to_block_index_)) continue;
      if (read_buffer_transformation_.count(r->buffer)) continue;

      auto read_transformation =
          GetTransformation(write_access_iterators, write_transformation_, read_access_iterators);
      read_buffer_transformation_.Set(r->buffer, read_transformation);
    }
  }

 private:
  bool can_be_transformed_;
  /*! \brief The buffers from function params. I.e. the input and output buffers. */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> param_buffers_;
  std::unordered_map<BufferRegion, ArrayIterSetTy, ObjectPtrHash, ObjectPtrEqual>
      buffer_to_iter_map_result_;
  Map<tir::Var, Range> spatial_dom_;
  Map<tir::Var, Range> reduction_dom_;

  std::unordered_map<const tir::VarNode*, int> iter_var_to_block_index_;
  IndexMap write_transformation_;

 public:
  Optional<Block> block_;
  IndexMap block_transformation_;
  Map<Buffer, IndexMap> read_buffer_transformation_;

 public:
  bool CanBeTransformed() { return can_be_transformed_; }
  Block GetBlock() {
    ICHECK(block_.defined());
    return block_.value();
  }
};

Map<tir::Block, Map<ObjectRef, tir::IndexMap>> SuggestLayoutTransforms(
    const PrimFunc& prim_func, Array<IndexMap> write_buffer_transformations) {
  Map<tir::Block, Map<ObjectRef, tir::IndexMap>> result;
  // No changes to the PrimFunc are required if no transformations on output buffers.
  if (write_buffer_transformations.empty()) return result;

  if (write_buffer_transformations.size() > 1) {
    LOG(WARNING) << "PrimFunc with more than one outputs is not supported yet";
    return result;
  }
  ICHECK(write_buffer_transformations.size() == 1);

  PrimFuncAnalyzer analyzer(prim_func, write_buffer_transformations[0]);
  analyzer(prim_func->body);
  auto can_be_transformed = analyzer.CanBeTransformed();
  if (!can_be_transformed) {
    LOG(WARNING) << "Requested PrimFunc cannot be transformed";
    return result;
  }

  Map<ObjectRef, IndexMap> block_result;

  Block block = analyzer.GetBlock();
  block_result.Set(block, analyzer.block_transformation_);
  block_result.Set(block->writes[0]->buffer, write_buffer_transformations[0]);

  for (const auto& read_buffer_region : block->reads) {
    if (analyzer.read_buffer_transformation_.count(read_buffer_region->buffer) == 0) continue;
    const auto& buffer_transformation =
        analyzer.read_buffer_transformation_[read_buffer_region->buffer];
    block_result.Set(read_buffer_region->buffer, buffer_transformation);
  }
  result.Set(block, block_result);
  return result;
}

TVM_REGISTER_GLOBAL(("relax.analysis.suggest_layout_transforms"))
    .set_body_typed([](PrimFunc fn, Array<tir::IndexMap> write_buffer_transformations) {
      return SuggestLayoutTransforms(fn, write_buffer_transformations);
    });

}  // namespace relax
}  // namespace tvm
