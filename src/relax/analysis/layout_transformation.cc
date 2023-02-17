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

using VarSet = std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual>;
using SpatialLayout = Array<Optional<tir::Var>>;
using VarToBlockIndexMap = std::unordered_map<tir::Var, int, ObjectPtrHash, ObjectPtrEqual>;
using VarToIterTypeMap = std::unordered_map<tir::Var, IterVarType, ObjectPtrHash, ObjectPtrEqual>;

/********** Helper Functions **********/

/*! \brief Converts a list to an Array */
template <typename T>
static inline Array<T> list_to_array(std::list<T> l) {
  Array<T> array;
  for (auto& v : l) {
    array.push_back(v);
  }
  return array;
}

/*! \brief Converts an Array to a list */
template <typename T>
static inline std::list<T> array_to_list(Array<T> array) {
  std::list<T> l;
  for (auto& v : array) {
    l.push_back(v);
  }
  return l;
}

/*! \brief Checks the PrimExpr for any vars not defined in `defs` */
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

/*! \brief Analyzes the indices returned from DetectIterMap analysis. It collects the spatial
 * iterator vars that are used in indices. This is important to get which spatial iter vars are
 * accessed in each index of buffer access.
 */
class IndexAnalyzer : public ExprVisitor {
 public:
  Array<tir::Var> Analyze(const arith::IterSumExpr& expr) {
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

/*! \brief Analyzes IterMapResult to get the spatial layout */
static inline SpatialLayout GetSpatialLayout(const arith::IterMapResult& iter_map_result) {
  ICHECK(!iter_map_result->indices.empty());
  SpatialLayout result;
  for (const arith::IterSumExpr& index : iter_map_result->indices) {
    IndexAnalyzer index_analyzer;
    Array<tir::Var> iter_vars = index_analyzer.Analyze(index);
    if (iter_vars.size() >= 2) return {};
    if (iter_vars.empty()) {
      result.push_back({});
      continue;
    }
    result.push_back(iter_vars[0]);
  }
  return result;
}

/*! \brief Checks if the two spatial layouts are identical. Two empty spatial layouts are treated as
 * unequal.*/
static inline bool AreIdenticalSpatialAccess(const SpatialLayout& s0, const SpatialLayout& s1) {
  if (s0.empty() || s1.empty()) return false;
  if (s0.size() != s1.size()) return false;
  for (size_t i = 0; i < s0.size(); ++i) {
    if ((!s0[i].defined() && s1[i].defined()) || (s0[i].defined() && !s1[i].defined()))
      return false;
    if (!s0[i].same_as(s1[i])) return false;
  }
  return true;
}

/*! \brief Given the spatial layout and t*/
static bool IsSequentialAccess(const SpatialLayout& iterators,
                               const VarToBlockIndexMap& iter_to_block_index) {
  int last_value = -1;
  for (const auto& i : iterators) {
    if (!i.defined()) continue;
    auto it = iter_to_block_index.find(i.value());
    ICHECK(it != iter_to_block_index.end());
    int blk_index = it->second;
    if (blk_index <= last_value) return false;
    last_value = blk_index;
  }
  return true;
}

/*! \brief Checks if two IndexMap transforms are identical*/
static inline bool AreIdenticalTransforms(const IndexMap& t0, const IndexMap& t1) {
  LOG(INFO) << "Comparing index maps: " << t0 << " :: " << t1;
  if (t0->initial_indices.size() != t1->initial_indices.size()) return false;
  if (t0->final_indices.size() != t1->final_indices.size()) return false;
  if (t0->initial_indices.empty()) return false;

  // Create a new shape expression.
  Array<PrimExpr> indices;
  for (size_t i = 0; i < t0->initial_indices.size(); ++i) indices.push_back(tir::Var("d"));
  LOG(INFO) << indices;
  auto t0_output = t0->MapIndices(indices);
  auto t1_output = t1->MapIndices(indices);
  for (size_t i = 0; i < t0_output.size(); ++i) {
    LOG(INFO) << "Comparing expr: " << t0_output[i] << " :: " << t1_output[i];
    if (!t0_output[i].same_as(t1_output[i])) {
      LOG(INFO) << "Not identical";
      return false;
    }
  }
  return true;
}

static IndexMap GetTransformation(const SpatialLayout& src_spatial_layout,
                                  const IndexMap& src_transformation,
                                  const SpatialLayout& tgt_spatial_layout) {
  // Copy over the src transformation intial and final indices
  auto initial_indices = array_to_list(src_transformation->initial_indices);
  auto final_indices = array_to_list(src_transformation->final_indices);

  // Remove any indices on both sides
  VarSet tgt_var_set;
  for (const auto& i : tgt_spatial_layout) {
    if (i.defined()) tgt_var_set.insert(i.value());
  }

  // Drop vars that do not occur in tgt iterators
  auto initial_indices_it = initial_indices.begin();
  VarSet defs;
  for (const auto& i : src_spatial_layout) {
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
  for (const auto& i : src_spatial_layout) {
    ICHECK(i.defined());
    src_var_set.insert(i.value());
  }

  initial_indices_it = initial_indices.begin();
  final_indices_it = final_indices.begin();
  for (const auto& i : tgt_spatial_layout) {
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
class BlockAnalyzer : public StmtExprVisitor {
 public:
  explicit BlockAnalyzer(const Block& block, IndexMap write_transformation)
      : can_transform_block_(true), write_transformation_(write_transformation), block_(block) {
    ICHECK(block_->writes.size() == 1);
    ComputeDomain();

    // Get load/store access patterns
    VisitStmt(block_->body);

    // While visiting the load/store patterns it is possible we see an unexpected pattern, such as
    // nested block or write access to multiple buffers. In such a case, we can return early as we
    // would not be making any layout suggesstions.
    if (!can_transform_block_) {
      LOG(WARNING) << "Unable to transform block " << block->name_hint;
      return;
    }

    // Get loop ordering in block iterators
    VarToIterTypeMap iter_var_to_type;
    VarToBlockIndexMap iter_var_to_block_index;
    SpatialLayout block_spatial_layout;
    int index = 0;
    for (const auto& iter_var : block->iter_vars) {
      auto var = iter_var->var;
      iter_var_to_block_index[var] = index++;
      iter_var_to_type[var] = iter_var->iter_type;
      block_spatial_layout.push_back(var);
    }

    auto get_spatial_layout = [&](Buffer b) -> SpatialLayout {
      auto it = buffer_access_info_.find(b);
      if (it == buffer_access_info_.end()) {
        return {};
      }
      auto access_info = it->second;
      return access_info.GetValidSpatialLayout();
    };

    auto write_buffer = block_->writes[0]->buffer;

    const auto& write_spatial_layout = get_spatial_layout(write_buffer);
    if (write_spatial_layout.empty()) {
      can_transform_block_ = false;
      return;
    }
    if (!IsSequentialAccess(write_spatial_layout, iter_var_to_block_index)) {
      can_transform_block_ = false;
      return;
    }

    block_transformation_ =
        GetTransformation(write_spatial_layout, write_transformation_, block_spatial_layout);
    Array<Range> block_iter_ranges;
    for (const auto& iter_var : block_->iter_vars) {
      block_iter_ranges.push_back(iter_var->dom);
    }
    Array<Range> block_ranges = block_->iter_vars.Map([](const IterVar& i) { return i->dom; });
    try {
      block_transformation_.Inverse(block_ranges);
    } catch (...) {
      LOG(WARNING) << "Inferred block transformation is not bijective affine";
      can_transform_block_ = false;
      return;
    }

    for (const auto& r : block->reads) {
      const auto& read_spatial_layout = get_spatial_layout(r->buffer);
      if (read_spatial_layout.empty()) continue;
      if (!IsSequentialAccess(read_spatial_layout, iter_var_to_block_index)) continue;
      auto read_transformation =
          GetTransformation(write_spatial_layout, write_transformation_, read_spatial_layout);
      read_buffer_transformations_.Set(r->buffer, read_transformation);
    }
  }

 private:
  class BufferAccessInfo {
   public:
    BufferAccessInfo() : is_valid_(true) {}
    void Update(SpatialLayout s) {
      if (!IsValid()) return;
      if (spatial_layout_.empty()) spatial_layout_ = s;
      if (!AreIdenticalSpatialAccess(s, spatial_layout_)) {
        Invalidate();
        return;
      }
    }
    bool IsValid() { return is_valid_; }
    void Invalidate() { is_valid_ = false; }
    SpatialLayout GetValidSpatialLayout() {
      if (!IsValid()) return {};
      return spatial_layout_;
    }

   private:
    bool is_valid_;
    SpatialLayout spatial_layout_;
  };
  void VisitStmt_(const BlockNode* op) final {
    // Blocks with nested blocks cannot be handled yet.
    LOG(WARNING) << "Found nested block";
    can_transform_block_ = false;
  }
  void VisitStmt_(const BufferStoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);

    BufferAccessInfo& access_info = buffer_access_info_[op->buffer];

    // Fast path to ignore further analysis if we know that the buffer access is invalid.
    if (!access_info.IsValid()) return;

    // Only single write buffer is supported for each block.
    if (!op->buffer.same_as(block_->writes[0]->buffer)) {
      access_info.Invalidate();
      LOG(WARNING) << "unexpected write access to a different buffer";

      can_transform_block_ = false;
      return;
    }

    // If the write buffer access cannot be analyzed, no transformation to the block will be made.
    auto detected_spatial_layout = DetectBufferAccessIterMap(op->indices);
    if (detected_spatial_layout.empty()) {
      access_info.Invalidate();
      return;
    }

    // Check if we have access info for this buffer, if present, the two accesses must be
    // identical.
    access_info.Update(detected_spatial_layout);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    Buffer read_buffer = op->buffer;
    BufferAccessInfo& access_info = buffer_access_info_[op->buffer];

    auto detected_spatial_layout = DetectBufferAccessIterMap(op->indices);

    if (detected_spatial_layout.empty()) {
      access_info.Invalidate();
      return;
    }
    access_info.Update(detected_spatial_layout);
  }

  // Helper to break down the indices of buffer access.
  SpatialLayout DetectBufferAccessIterMap(Array<PrimExpr> indices) {
    auto result = arith::DetectIterMap(
        /*indices=*/indices, /*input_iters*/ spatial_dom_,
        /*predicate*/ tir::const_true(),
        /*check_level*/ arith::IterMapLevel::Surjective, /*analyzer*/ &arith_analyzer_);
    if (result->indices.empty()) {
      LOG(WARNING) << "Failed to analyze indices " << indices << ", error: " << result->errors;
      return {};
    }
    return GetSpatialLayout(result);
  }

  void ComputeDomain() {
    for (const IterVar& v : block_->iter_vars) {
      if (v->iter_type == kDataPar) {
        spatial_dom_.Set(v->var, v->dom);
        continue;
      }
      if (v->iter_type == kCommReduce) {
        reduction_dom_.Set(v->var, v->dom);
        continue;
      }
      // Unknown iter type
      LOG(WARNING) << "Unable to compute domain of block";
      can_transform_block_ = false;
      return;
    }
  }

 public:
  bool CanBeTransformed() { return can_transform_block_; }
  IndexMap GetBlockTransformation() { return block_transformation_; }
  Map<Buffer, IndexMap> GetReadBufferTransformations() { return read_buffer_transformations_; }

 private:
  bool can_transform_block_;
  IndexMap write_transformation_;
  Map<tir::Var, Range> spatial_dom_;
  Map<tir::Var, Range> reduction_dom_;
  arith::Analyzer arith_analyzer_;

  Block block_;
  IndexMap block_transformation_;

  Map<Buffer, IndexMap> read_buffer_transformations_;
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual> buffer_access_info_;
};  // namespace relax

class PrimFuncAnalyzer : public StmtExprVisitor {
 public:
  explicit PrimFuncAnalyzer(const PrimFunc& func, Array<IndexMap> write_transformations) {
    ICHECK(write_transformations.size() <= func->params.size())
        << "Incompatible PrimFunc and write_transformations";

    size_t first_write_index = func->params.size() - write_transformations.size();
    for (size_t i = 0; i < write_transformations.size(); ++i) {
      auto param = func->params[first_write_index + i];
      Optional<Buffer> param_buf = func->buffer_map.Get(param);
      ICHECK(param_buf.defined());
      buffer_transformations_.Set(param_buf.value(), write_transformations[i]);
    }
    VisitStmt(func->body);
  }
  Map<Block, Map<ObjectRef, IndexMap>> GetSuggestedTransforms() { return suggested_transforms_; }

 private:
  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      // Skip the root block
      StmtVisitor::VisitStmt_(op);
      return;
    }

    Block block = GetRef<Block>(op);
    // Get block write buffer transformation.
    if (block->writes.size() != 1) return;
    auto write_buffer = block->writes[0]->buffer;
    BlockAnalyzer block_analyzer(block, buffer_transformations_[write_buffer]);

    if (!block_analyzer.CanBeTransformed()) return;
    // Collect the suggested transformations
    Map<ObjectRef, IndexMap> suggested_block_transormations;
    suggested_block_transormations.Set(block, block_analyzer.GetBlockTransformation());

    for (const auto& [buffer, index_map] : block_analyzer.GetReadBufferTransformations()) {
      suggested_block_transormations.Set(buffer, index_map);
    }
    suggested_block_transormations.Set(write_buffer, buffer_transformations_[write_buffer]);
    suggested_transforms_.Set(block, suggested_block_transormations);
  }

 private:
  Map<Buffer, IndexMap> buffer_transformations_;
  Map<Buffer, Block> buffer_to_block_;
  Map<Block, Map<ObjectRef, IndexMap>> suggested_transforms_;
};  // namespace relax

Map<tir::Block, Map<ObjectRef, tir::IndexMap>> SuggestLayoutTransforms(
    const PrimFunc& prim_func, Array<IndexMap> write_buffer_transformations) {
  // No changes to the PrimFunc are required if no transformations on output buffers.
  if (write_buffer_transformations.empty()) return {};

  PrimFuncAnalyzer analyzer(prim_func, write_buffer_transformations);
  return analyzer.GetSuggestedTransforms();
}

TVM_REGISTER_GLOBAL(("relax.analysis.suggest_layout_transforms"))
    .set_body_typed([](PrimFunc fn, Array<tir::IndexMap> write_buffer_transformations) {
      return SuggestLayoutTransforms(fn, write_buffer_transformations);
    });

}  // namespace relax
}  // namespace tvm
