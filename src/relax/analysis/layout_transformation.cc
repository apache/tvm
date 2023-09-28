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
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/relax/analysis.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>

#include "../../support/array.h"

namespace tvm {
namespace relax {

using namespace tir;

/********** Helper Functions **********/

/*! \brief Checks if a transformation is bijective affine over the given ranges */
static bool IsBijectiveAffine(const IndexMap& m, const Array<Range>& ranges) {
  Map<tir::Var, Range> input_iters;
  ICHECK_EQ(m->initial_indices.size(), ranges.size());
  for (size_t i = 0; i < ranges.size(); i++) {
    input_iters.Set(m->initial_indices[i], ranges[i]);
  }
  arith::Analyzer analyzer;
  auto iter_map_result = DetectIterMap(m->final_indices, input_iters, /* predicate = */ 1,
                                       /*check_level=*/arith::IterMapLevel::Bijective, &analyzer,
                                       /*simplify_trivial_iterators=*/true);
  return !iter_map_result->indices.empty();
}

/*!
 * \brief Analyzer to collect iterators from IterSumExpr.
 * \details Analyzes the indices from DetectIterMap analysis to collect the spatial iterators that
 * are used in it. This is important to get which spatial iterators are accessed in each index
 * of buffer access.
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
      for (const auto& arg : op->args) VisitExpr(arg);
      VisitExpr(op->base);
      return;
    }
    if (const auto* op = expr.as<arith::IterSplitExprNode>()) {
      VisitIterMark(op->source);
      VisitExpr(op->lower_factor);
      VisitExpr(op->extent);
      VisitExpr(op->scale);
      return;
    }
    return ExprVisitor::VisitExpr(expr);
  }

  void VisitIterMark(const arith::IterMark& op) {
    if (const auto* var = op->source.as<tir::VarNode>())
      iterators_.push_back(GetRef<tir::Var>(var));
    else
      VisitExpr(op->source);
    VisitExpr(op->extent);
  }

 private:
  Array<tir::Var> iterators_;
};

/*!
 * \brief Analyzes IterMapResult to get the Spatial Layout of buffer access.
 * \details We define Spatial Layout of a buffer access as an array of length equal to the
 * dimensions of the buffer. i-th element of Spatial Layout contains spatial iter var used from the
 * block iteration domain. For indices, where no spatial iter vars are used, the spatial layout
 * element is empty. If any of the buffer access indices use multiple spatial iter vars, the spatial
 * layout is undefined.
 *
 * Here are a few examples of inferred spatial layout from buffer access. si denotes i-th spatial
 * iter var, and ri denotes i-th reduction iter var.
 *
 * SpatialLayout(A[s0*constant, s1]) = {s0, s1}
 * SpatialLayout(A[s0, constant, r0, s1]) = {s0, null, null, s1}
 * SpatialLayout(A[s0 * c + s1]) = undefined
 */
using SpatialLayout = Array<Optional<tir::Var>>;
static SpatialLayout GetSpatialLayout(const arith::IterMapResult& iter_map_result) {
  ICHECK(!iter_map_result->indices.empty());
  SpatialLayout result;
  for (const arith::IterSumExpr& index : iter_map_result->indices) {
    IndexAnalyzer index_analyzer;
    Array<tir::Var> iter_vars = index_analyzer.Analyze(index);
    if (iter_vars.size() >= 2) {
      LOG(WARNING) << "[LayoutInference] Unable to get spatial layout of access: "
                   << arith::NormalizeIterMapToExpr(index);
      return {};
    }
    if (iter_vars.empty()) {
      result.push_back({});
      continue;
    }
    result.push_back(iter_vars[0]);
  }
  return result;
}

/*!
 * \brief Checks if the two spatial layouts are identical. Two empty spatial layouts are treated as
 * unequal.
 */
static bool AreIdenticalSpatialAccess(const SpatialLayout& s0, const SpatialLayout& s1) {
  if (s0.empty() || s1.empty()) return false;
  if (s0.size() != s1.size()) return false;
  for (size_t i = 0; i < s0.size(); ++i) {
    if ((!s0[i].defined() && s1[i].defined()) || (s0[i].defined() && !s1[i].defined()))
      return false;
    if (!s0[i].same_as(s1[i])) return false;
  }
  return true;
}

/*!
 * \brief Checks if the block accesses a buffer sequentially in terms of spatial dimensions
 * (ignoring reduction dimensions). It checks that the order of spatial iter vars in spatial layout
 * of a buffer access is same as the order of spatial iter vars in block domain.
 */
using VarToBlockIndexMap = std::unordered_map<tir::Var, int, ObjectPtrHash, ObjectPtrEqual>;
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

/*! \brief Checks if two IndexMaps represent identical transforms */
static bool AreIdenticalTransforms(const IndexMap& t0, const IndexMap& t1) {
  if (t0->initial_indices.size() != t1->initial_indices.size()) return false;
  if (t0->final_indices.size() != t1->final_indices.size()) return false;

  // Create a new shape expression.
  Array<PrimExpr> t1_initial_indices =
      t1->initial_indices.Map([](tir::Var i) -> PrimExpr { return i; });
  arith::Analyzer analyzer;
  auto t0_output = t0->MapIndices(t1_initial_indices, &analyzer);
  for (size_t i = 0; i < t0_output.size(); ++i) {
    if (!analyzer.CanProveEqual(t0_output[i], t1->final_indices[i])) return false;
  }
  return true;
}

/*!
 * \brief Returns the layout transformation for a target spatial layout from the source spatial
 * layout and transformation.
 * \details Given the source buffer spatial layout \p src_spatial_layout and its transformation \p
 * src_transformation, this function constructs the transformation for the target buffer whose
 * spatial layout is given as \p tgt_spatial_layout.
 *
 * The algorithm is explained below using an example:
 *
 * Let's say the source transformation is lambda N, C, H, W -> (N, H, W, C // 4, C %
 * 4), source spatial layout is 'NCHW' and target spatial layout is 'KCHW'.
 *
 * Step 1: Copy over the source transformation initial & final indices to target transformation
 * initial and final indices.
 * target transformation = lambda N, C, H, W -> (N, H, W, C // 4, C %4)
 *
 * Step 2: Drop any vars from initial indices which do not occur in target buffer using source and
 * target spatial layouts.
 * target transformation = lambda C, H, W -> (N, H, W, C // 4, C %4)
 *
 * Step 3: Erase any expression from final indices which is dependent on a var not present in
 * initial indices.
 * target transformation = lambda C, H, W -> (H, W, C // 4, C %4)
 *
 * Step 4: Go over the target spatial layout and add any missing dims to both initial and final
 * indices. This is done by checking if any iterator in target spatial layout is not present in
 * source spatial layout.
 * target transformation = lambda dim, C, H, W -> (dim, H, W, C // 4, C %4)
 */
using VarSet = std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual>;
static Optional<IndexMap> InferLayoutTransformation(const SpatialLayout& src_spatial_layout,
                                                    const IndexMap& src_transformation,
                                                    const SpatialLayout& tgt_spatial_layout) {
  // Copy over the src transformation intial and final indices
  auto initial_indices = support::AsList(src_transformation->initial_indices);
  auto final_indices = support::AsList(src_transformation->final_indices);

  // Get the iterator var set used in target spatial layout.
  VarSet tgt_var_set;
  for (const auto& i : tgt_spatial_layout) {
    if (i.defined()) tgt_var_set.insert(i.value());
  }

  // Erase initial indices corresponding to iter vars that do not occur in target spatial layout.
  // Also compute the var set of initial indices.
  auto initial_indices_it = initial_indices.begin();
  VarSet initial_indices_var_set;
  for (const auto& i : src_spatial_layout) {
    ICHECK(i.defined());
    if (tgt_var_set.count(i.value())) {
      initial_indices_var_set.insert(*initial_indices_it);
      initial_indices_it++;
      continue;
    }
    initial_indices_it = initial_indices.erase(initial_indices_it);
  }

  // Erase any expressions in final indices that have undefined vars
  auto final_indices_it = final_indices.begin();
  while (final_indices_it != final_indices.end()) {
    // Collect all the vars used in this final index.
    Array<tir::Var> used_vars = tir::UndefinedVars(*final_indices_it);
    ICHECK(!used_vars.empty())
        << "IndexMap expression must always contain tir::Var nodes but found none in: "
        << *final_indices_it;

    bool has_undefined_vars = std::any_of(used_vars.begin(), used_vars.end(),
                                          [&initial_indices_var_set](const tir::Var& v) {
                                            return initial_indices_var_set.count(v) == 0;
                                          });

    // If all vars are from initial indices, nothing to do for this final index.
    if (!has_undefined_vars) {
      final_indices_it++;
      continue;
    }
    // We are about to drop this expr from final indices since it has undefined vars. Check if it is
    // dependent on any of the initial indices. If it is dependent, this cannot be dropped and we
    // bail by returning null.
    // This captures the scenario where the source transformation is unpacking a dimension (e.g,
    // "H4h" -> "H*4+h" ) and the buffer we are trying to infer the transformation of has 'h'
    // dimension, but not 'H'. So, it is dependent on undefined var 'H' and defined var 'h'.
    bool depends_on_initial_indices = std::any_of(used_vars.begin(), used_vars.end(),
                                                  [&initial_indices_var_set](const tir::Var& v) {
                                                    return initial_indices_var_set.count(v) != 0;
                                                  });
    if (depends_on_initial_indices) {
      LOG(WARNING)
          << "[LayoutInference] Buffer access is dependent on both defined and undefined vars";
      return {};
    }
    // It is ok to erase this final index expression as it only depends on undefined vars.
    final_indices_it = final_indices.erase(final_indices_it);
  }

  // Go over the target spatial layout and add any missing dims to both initial and final indices.
  // This is done by checking if any iterator in target spatial layout is not present in source
  // spatial layout.
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

    auto new_dim = tir::Var("d");
    initial_indices.insert(initial_indices_it, new_dim);
    final_indices.insert(final_indices_it, new_dim);
  }

  return IndexMap(support::AsArray(initial_indices), support::AsArray(final_indices));
}

/*!
 * \brief Analyzes the Block and given output buffer transformations to propose
 * transformations of block and read buffers.
 * \details It does a best effort analysis to propose transformations which would preserve
 * sequential access to buffers (especially output buffers). Since this is best effort, it is
 * possible that the Block is too complex for analysis. In such a case, no transformations are
 * proposed. Limitations:
 * 1. Expects exactly one write buffer in the block whose transformation is given by
 * `write_transformation`.
 * 2. Expects write buffer access to be affine and only use spatial iterators of the block.
 * 3. Proposes transformations to a read buffer if all access to it are affine.
 */
class BlockAnalyzer : public StmtExprVisitor {
 public:
  explicit BlockAnalyzer(const Block& block, const Map<Buffer, IndexMap>& transformation_cache,
                         IndexMap write_transformation)
      : can_transform_block_(true),
        write_transformation_(write_transformation),
        block_(block),
        buffer_transformation_cache_(transformation_cache) {
    ICHECK(block_->writes.size() == 1);
    auto write_buffer = block_->writes[0]->buffer;

    ComputeBlockSpatialDomain();

    // Visit the block body to collect load/store access patterns of different buffers.
    VisitStmt(block_->body);

    // While visiting the load/store accesses it is possible we see an unexpected pattern, such as
    // nested block or write access to multiple buffers. In such a case, we can return early as we
    // would not be making any layout suggesstions.
    if (!can_transform_block_) {
      LOG(WARNING) << "[LayoutInference] Unable to transform block " << block->name_hint;
      return;
    }

    // Get iterator ordering and it's spatial layout.
    VarToBlockIndexMap iter_var_to_block_index;
    SpatialLayout block_spatial_layout;
    int index = 0;
    for (const auto& iter_var : block->iter_vars) {
      auto var = iter_var->var;
      iter_var_to_block_index[var] = index++;
      block_spatial_layout.push_back(var);
    }

    // Helper to get the spatial layout of buffer from buffer access map.
    auto get_spatial_layout = [&](Buffer b) -> SpatialLayout {
      auto it = buffer_access_info_.find(b);
      if (it == buffer_access_info_.end()) {
        return {};
      }
      auto access_info = it->second;
      return access_info.GetValidSpatialLayout();
    };

    // Check that write has sequential access within the block.
    SpatialLayout write_spatial_layout = get_spatial_layout(write_buffer);
    if (write_spatial_layout.empty()) {
      can_transform_block_ = false;
      return;
    }
    if (!IsSequentialAccess(write_spatial_layout, iter_var_to_block_index)) {
      can_transform_block_ = false;
      return;
    }

    // Infer Block transformation from write buffer transformation.
    auto maybe_block_transformation = InferLayoutTransformation(
        write_spatial_layout, write_transformation_, block_spatial_layout);
    if (!maybe_block_transformation.defined()) {
      can_transform_block_ = false;
      return;
    }
    block_transformation_ = maybe_block_transformation.value();

    Array<Range> block_ranges = block_->iter_vars.Map([](const IterVar& i) { return i->dom; });
    if (!IsBijectiveAffine(block_transformation_, block_ranges)) {
      can_transform_block_ = false;
      LOG(WARNING) << "[LayoutInference] Inferred block transformation is not bijective affine, "
                      "transformation: ("
                   << block_transformation_ << ") over range (" << block_ranges << ")";
      return;
    }

    // Infer read buffer transformations from write buffer transformation.
    for (const auto& r : block->reads) {
      SpatialLayout read_spatial_layout = get_spatial_layout(r->buffer);
      if (read_spatial_layout.empty()) continue;
      if (!IsSequentialAccess(read_spatial_layout, iter_var_to_block_index)) continue;

      auto maybe_read_transformation = InferLayoutTransformation(
          write_spatial_layout, write_transformation_, read_spatial_layout);
      if (!maybe_read_transformation.defined()) continue;
      IndexMap read_transformation = maybe_read_transformation.value();
      if (buffer_transformation_cache_.count(r->buffer) != 0) {
        if (!AreIdenticalTransforms(read_transformation, buffer_transformation_cache_[r->buffer]))
          LOG(WARNING) << "[LayoutInference] Buffer: " << r->buffer
                       << " has conflicting transform proposals -- (preferred) "
                       << buffer_transformation_cache_[r->buffer] << " vs. " << read_transformation;
        continue;
      }
      read_buffer_transformations_.Set(r->buffer, read_transformation);
    }
  }

 private:
  // Helper class to keep track of spatial layout of buffer as we visit multiple accesses to this
  // buffer within the block.
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

  // Helper to break down the indices of buffer access.
  SpatialLayout DetectBufferAccessIterMap(Array<PrimExpr> indices) {
    auto result = arith::DetectIterMap(
        /*indices=*/indices, /*input_iters*/ spatial_dom_,
        /*predicate*/ 1, /*check_level*/ arith::IterMapLevel::NoCheck, &arith_analyzer_);
    if (result->indices.empty()) {
      DLOG(INFO) << "[LayoutInference] Failed to analyze indices " << indices
                 << ", error: " << result->errors;
      return {};
    }
    return GetSpatialLayout(result);
  }

  // Compute the spatial domain map of block
  void ComputeBlockSpatialDomain() {
    for (const IterVar& v : block_->iter_vars) {
      if (v->iter_type == kDataPar) {
        spatial_dom_.Set(v->var, v->dom);
        continue;
      }
      if (v->iter_type == kCommReduce) continue;
      LOG(WARNING) << "[LayoutInference] Cannot compute block spatial domain in presence of "
                      "unknown block iter_type : "
                   << v->iter_type;
      can_transform_block_ = false;
      return;
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    // Blocks with nested blocks cannot be handled yet.
    LOG(WARNING) << "[LayoutInference] Nested blocks are not supported for layout inference yet";
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
      LOG(WARNING) << "[LayoutInference] Exactly one write buffer is supported for layout "
                      "inference, found two: "
                   << op->buffer << " and " << block_->writes[0]->buffer;
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

 public:
  bool CanBeTransformed() { return can_transform_block_; }
  IndexMap GetBlockTransformation() { return block_transformation_; }
  Map<Buffer, IndexMap> GetReadBufferTransformations() { return read_buffer_transformations_; }

 private:
  bool can_transform_block_;
  IndexMap write_transformation_;
  Map<tir::Var, Range> spatial_dom_;
  arith::Analyzer arith_analyzer_;

  Block block_;
  IndexMap block_transformation_;

  Map<Buffer, IndexMap> read_buffer_transformations_;
  const Map<Buffer, IndexMap>& buffer_transformation_cache_;
  std::unordered_map<Buffer, BufferAccessInfo, ObjectPtrHash, ObjectPtrEqual> buffer_access_info_;
};

/*!
 * \brief Analyzes the PrimFunc and user provided output buffer transformations to propose
 * transformations of block and buffers within the PrimFunc.
 * \details It does a best effort analysis to propose transformations which would preserve
 * sequential access to buffers (especially output buffers). Since this is best effort, it is
 * possible that the PrimFunc is too complex for analysis. In such a case, no transformations are
 * proposed.
 */
class PrimFuncAnalyzer : public StmtExprVisitor {
 public:
  explicit PrimFuncAnalyzer(const PrimFunc& func, Array<IndexMap> write_transformations) {
    ICHECK_LE(write_transformations.size(), func->params.size())
        << "Incompatible PrimFunc and write_transformations";

    size_t first_write_index = func->params.size() - write_transformations.size();
    for (size_t i = 0; i < write_transformations.size(); ++i) {
      auto param = func->params[first_write_index + i];
      Optional<Buffer> param_buf = func->buffer_map.Get(param);
      ICHECK(param_buf.defined());
      ICHECK_EQ(param_buf.value()->shape.size(), write_transformations[i]->initial_indices.size())
          << "Mismatch between output buffer shape and index map";
      buffer_transformation_cache_.Set(param_buf.value(), write_transformations[i]);
    }
    VisitStmt(func->body);
  }
  Map<Block, Map<ObjectRef, IndexMap>> GetSuggestedTransforms() {
    Map<Block, Map<ObjectRef, IndexMap>> result;
    for (const auto& [block, index_map] : block_transformations_) {
      Map<ObjectRef, IndexMap> block_transformations;
      block_transformations.Set(block, index_map);
      for (const auto& buffer : block_to_buffer_[block]) {
        block_transformations.Set(buffer, buffer_transformation_cache_[buffer]);
      }
      result.Set(block, block_transformations);
    }
    return result;
  }

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
    block_to_buffer_[block].push_back(write_buffer);
    BlockAnalyzer block_analyzer(block, buffer_transformation_cache_,
                                 buffer_transformation_cache_[write_buffer]);

    if (!block_analyzer.CanBeTransformed()) return;
    // Collect the suggested transformations
    block_transformations_.Set(block, block_analyzer.GetBlockTransformation());

    for (const auto& [buffer, index_map] : block_analyzer.GetReadBufferTransformations()) {
      // BlockAnalyzer makes sure that it does not propose transformation for a buffer for which a
      // transformation has already been proposed by other blocks or by write_transformations which
      // are input to this analysis.
      ICHECK_EQ(buffer_transformation_cache_.count(buffer), 0);
      buffer_transformation_cache_.Set(buffer, index_map);
      block_to_buffer_[block].push_back(buffer);
    }
  }

 private:
  Map<Buffer, IndexMap> buffer_transformation_cache_;
  Map<Block, IndexMap> block_transformations_;
  std::unordered_map<Block, Array<Buffer>, ObjectPtrHash, ObjectPtrEqual> block_to_buffer_;
};

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
